"""`vizdoom-turbo`-shaped vector API backed by device tensors."""

from __future__ import annotations

import math
import operator
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from .actions import DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS, normalize_action_table
from .engine import DEVICE_SIGNAL_NAMES, TorchDeathmatchEngine
from .scenario import CompiledScenario, compile_deathmatch_scenario

_DEFAULT_SIGNALS = (
    "killcount",
    "deathcount",
    "hitcount",
    "damagecount",
    "health",
    "armor",
    "selected_weapon",
    "selected_weapon_ammo",
    "weapon1",
    "weapon2",
    "weapon3",
    "weapon4",
    "weapon5",
    "weapon6",
    "ammo1",
    "ammo2",
    "ammo3",
    "ammo4",
    "ammo5",
    "ammo6",
)
_DERIVED_SIGNALS = ("episode_time", "episode_return", "player_dead", "pending_reset")
_COMPILED_ENGINE_PHASES = (
    "_begin_decision",
    "_game_tic_bookkeeping",
    "_select_weapons",
    "_move_player",
    "_vertical_player_tick",
    "_player_attack",
    "_hitscan_puff_tick",
    "_post_player_attack_bookkeeping",
    "_projectile_tick",
    "_enemy_tick",
    "_enemy_projectile_tick",
    "_collect_items",
    "_finish_transition",
    "render_frame",
    "_finish_observation",
)
_LIMITED_FUSION_PHASES = frozenset(
    {
        "render_frame",
        "_enemy_tick",
        "_enemy_projectile_tick",
        "_collect_items",
        "_begin_decision",
        "_game_tic_bookkeeping",
        "_post_player_attack_bookkeeping",
        "_finish_transition",
        "_finish_observation",
    }
)
_LIMITED_FUSION_OPTIONS = {"max_fusion_size": 8}
_RENDER_FUSION_OPTIONS = {"max_fusion_size": 32}
_ENEMY_FUSION_OPTIONS = {"max_fusion_size": 8}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(result)


def _normalize_seed(
    seed: int | Sequence[int | None] | None,
    num_envs: int,
) -> list[int | None]:
    if seed is None:
        return [None] * num_envs
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
        result = [None if value is None else int(value) for value in seed]
        if len(result) != num_envs:
            raise ValueError("seed sequence length must match num_envs")
        return result
    base = int(seed)
    return [base + lane for lane in range(num_envs)]


def _validate_deathmatch_request(game: Any, scenario: Any) -> None:
    requested = scenario if scenario not in (None, "scenario") else game
    if requested is None:
        return
    candidate = Path(str(requested)).expanduser()
    if candidate.is_file():
        if candidate.suffix.casefold() not in {".cfg", ".wad"}:
            raise ValueError("GraDOOM scenarios must be a ViZDoom .cfg or Doom .wad file")
        return
    alias = str(requested).strip().casefold().removesuffix(".cfg")
    if alias not in {"deathmatch", "vizdoomdeathmatch-v1"}:
        raise ValueError(
            f"unsupported GraDOOM game/scenario {requested!r}; "
            "only VizdoomDeathmatch-v1 is supported"
        )


def _resolve_scenario_wad(game: Any, scenario: Any) -> Path:
    requested = scenario if scenario not in (None, "scenario") else game
    candidate = Path(str(requested)).expanduser() if requested is not None else None
    if candidate is not None and candidate.is_file():
        if candidate.suffix.casefold() == ".wad":
            return candidate.resolve()
        if candidate.suffix.casefold() == ".cfg":
            match = re.search(
                r"(?im)^\s*doom_scenario_path\s*=\s*([^#\r\n]+)",
                candidate.read_text(encoding="utf-8"),
            )
            if match is None:
                raise ValueError(f"scenario config {candidate} has no doom_scenario_path")
            wad = (candidate.parent / match.group(1).strip()).resolve()
            if not wad.is_file():
                raise FileNotFoundError(
                    f"scenario WAD referenced by {candidate} does not exist: {wad}"
                )
            return wad
    configured = os.environ.get("GRADOOM_DEATHMATCH_WAD")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
    try:
        import vizdoom as vzd

        path = Path(vzd.scenarios_path) / "deathmatch.wad"
        if path.is_file():
            return path.resolve()
    except ImportError:
        pass
    raise FileNotFoundError(
        "cannot locate deathmatch.wad; pass scenario=... or set GRADOOM_DEATHMATCH_WAD"
    )


def scenario_buttons(
    game: str | Path | None = "VizdoomDeathmatch-v1",
    *,
    scenario: str | Path | None = None,
) -> tuple[str, ...]:
    _validate_deathmatch_request(game, scenario)
    return DEATHMATCH_BUTTONS


@dataclass(frozen=True)
class DeviceTransition:
    """Allocation-light transition consumed directly by GPU-native learners."""

    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    signals: torch.Tensor
    info_histories: torch.Tensor


@dataclass(frozen=True)
class DeviceAutoResetTransition:
    """One device step plus masked reset, retaining terminal tensors for bootstrapping."""

    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    signals: torch.Tensor
    info_histories: torch.Tensor
    final_observations: torch.Tensor
    final_signals: torch.Tensor
    final_info_histories: torch.Tensor


class GraDoomVecEnv(VectorEnv):
    """Device-resident vector deathmatch environment.

    Torch tensors are the only transition transport, including reset selectors
    and state indices. NumPy is limited to diagnostic RGB rendering.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "autoreset_mode": AutoresetMode.DISABLED,
        "render_modes": ["rgb_array"],
        "render_fps": 35,
        "turbo_api_version": 2,
        "transition_transport": "torch",
        "gradoom_device_api_version": 1,
    }
    supports_live_snapshots = False
    live_snapshots_deterministic = False
    parity_certified = False
    device_signal_names = DEVICE_SIGNAL_NAMES

    def __init__(
        self,
        game: str | Path,
        state: Any = None,
        scenario: str | Path | None = None,
        info: Any = None,
        use_restricted_actions: Any = "default",
        record: bool = False,
        players: int = 1,
        inttype: Any = "stable",
        obs_type: Any = "image",
        render_mode: str | None = None,
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        rom_path: str | None = None,
        transport: Literal["default", "torch"] = "default",
        obs_copy: Literal["copy", "safe_view", "unsafe_view"] = "safe_view",
        obs_resize: tuple[int, int] | None = (84, 84),
        obs_crop: tuple[int, int, int, int] | None = None,
        obs_crop_mode: Literal["remove", "mask"] = "remove",
        obs_crop_fill: int = 0,
        obs_grayscale: bool = True,
        obs_resize_algorithm: Literal["nearest", "bilinear", "area"] = "area",
        obs_layout: Literal["hwc", "chw"] = "chw",
        frame_skip: int = 4,
        frame_stack: int = 4,
        maxpool_last_two: bool = False,
        noop_reset_max: int = 0,
        use_fire_reset: bool = False,
        sticky_action_prob: float = 0.0,
        reward_clip: bool | tuple[float, float] = False,
        info_filter: str | Mapping[str, Any] = "all",
        info_frame_stack_keys: Sequence[str] | None = None,
        state_catalog: Sequence[Any] | None = None,
        device: str | torch.device | None = None,
        doom_map: str | None = None,
        doom_skill: int | None = None,
        wall_contact_damage_scale: float = 1.0,
        game_args: str | None = None,
        game_variables: Sequence[str] | None = None,
        enemy_variants: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
        surface_variants: Mapping[str, Sequence[str]] | None = None,
        treat_episode_timeout_as_truncation: bool = True,
        vizdoom_config: Mapping[str, Any] | None = None,
        compiled_scenario: CompiledScenario | None = None,
        require_pinned_scenario: bool = True,
        compile_engine: bool = False,
    ) -> None:
        if transport == "default":
            transport = "torch"
        if isinstance(use_restricted_actions, str) and use_restricted_actions == "default":
            use_restricted_actions = DEATHMATCH_ACTIONS
        _validate_deathmatch_request(game, scenario)
        if info not in (None, "data"):
            raise ValueError("info must be None or 'data'")
        if record:
            raise ValueError("record=True is not supported on the device path")
        if players != 1:
            raise ValueError("the deathmatch-p1-v1 profile supports players=1")
        if str(obs_type).split(".")[-1].casefold() != "image":
            raise ValueError("GraDoomVecEnv supports image observations only")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if state is not None and state_catalog is not None:
            raise ValueError("state and state_catalog are mutually exclusive")
        configured_catalog = ("default",) if state_catalog is None else tuple(state_catalog)
        if not configured_catalog:
            raise ValueError("state_catalog must not be empty")
        if len(set(configured_catalog)) != len(configured_catalog):
            raise ValueError("state_catalog must contain unique states")
        if state not in (None, "default") or configured_catalog != ("default",):
            raise ValueError("the first device profile supports only the default initial state")
        if any(
            value is not None for value in (doom_map, game_args, enemy_variants, surface_variants)
        ):
            raise ValueError("custom maps, game args, and variants are not yet supported")
        if doom_skill not in (None, 1, 3):
            raise ValueError("deathmatch-p1-v1 supports Doom skill 1 or 3")
        resolved_doom_skill = 3 if doom_skill is None else doom_skill
        self.doom_skill = resolved_doom_skill
        if (
            not math.isfinite(wall_contact_damage_scale)
            or wall_contact_damage_scale < 0.0
            or wall_contact_damage_scale > 1.0
        ):
            raise ValueError("wall_contact_damage_scale must be finite and in [0, 1]")
        self.wall_contact_damage_scale = float(wall_contact_damage_scale)
        if use_fire_reset or noop_reset_max or sticky_action_prob:
            raise ValueError(
                "fire reset, no-op reset, and sticky actions are not in deathmatch-p1-v1"
            )
        if maxpool_last_two:
            raise ValueError("deathmatch-p1-v1 does not max-pool consecutive frames")
        if obs_resize != (84, 84) or not obs_grayscale or frame_stack != 4:
            raise ValueError("deathmatch-p1-v1 requires 84x84 grayscale frame-stack 4")
        if obs_crop not in (None, (0, 32, 0, 0), [0, 32, 0, 0]):
            raise ValueError("deathmatch-p1-v1 supports no crop or the pinned bottom-32 mask")
        if obs_crop is not None and (obs_crop_mode != "mask" or obs_crop_fill != 0):
            raise ValueError("the pinned deathmatch crop is a zero-filled mask")
        if obs_crop is None and obs_crop_mode not in {"remove", "mask"}:
            raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
        if obs_crop_fill != 0:
            raise ValueError("deathmatch-p1-v1 requires obs_crop_fill=0")
        if obs_layout not in {"chw", "hwc"}:
            raise ValueError("obs_layout must be 'chw' or 'hwc'")
        if obs_resize_algorithm != "area":
            raise ValueError("deathmatch-p1-v1 supports obs_resize_algorithm='area' only")
        if transport != "torch":
            raise ValueError("transport must be 'torch'; NumPy transition transport is unsupported")
        if obs_copy not in {"copy", "safe_view", "unsafe_view"}:
            raise ValueError("obs_copy must be copy, safe_view, or unsafe_view")
        inttype_name = getattr(inttype, "name", inttype)
        if str(inttype_name).split(".")[-1].casefold() != "stable":
            raise ValueError("inttype must select the Stable integration")
        if num_threads is not None:
            raise ValueError("num_threads is unsupported and must be None on the device path")

        self.num_envs = _positive_int(num_envs, "num_envs")
        self.num_threads = None
        self.frame_skip = _positive_int(frame_skip, "frame_skip")
        self.frame_stack = frame_stack
        if info_frame_stack_keys is None:
            history_names: tuple[str, ...] = ()
        else:
            if isinstance(info_frame_stack_keys, (str, bytes, bytearray)) or not isinstance(
                info_frame_stack_keys, Sequence
            ):
                raise TypeError("info_frame_stack_keys must be a sequence of signal names or None")
            if any(not isinstance(name, str) for name in info_frame_stack_keys):
                raise TypeError("info_frame_stack_keys must contain only strings")
            history_names = tuple(name.casefold() for name in info_frame_stack_keys)
        self.info_frame_stack_keys = history_names
        self.device_info_history_names = history_names
        if len(self.device_info_history_names) != len(set(self.device_info_history_names)):
            raise ValueError("info_frame_stack_keys must not contain duplicates")
        unknown_history_names = set(self.device_info_history_names) - set(DEVICE_SIGNAL_NAMES)
        if unknown_history_names:
            raise ValueError(
                f"unknown device info frame-stack signals: {sorted(unknown_history_names)}"
            )
        self.obs_layout = obs_layout
        self.obs_copy = obs_copy
        self.observation_ownership = {
            "copy": "owned",
            "safe_view": "safe_view",
            "unsafe_view": "unsafe_view",
        }[obs_copy]
        self.observation_buffer_depth = {"copy": None, "safe_view": 2, "unsafe_view": 1}[obs_copy]
        self.render_mode = render_mode
        self.autoreset_mode = AutoresetMode.DISABLED
        self.closed = False
        self.game = str(game or "VizdoomDeathmatch-v1")
        self.transport = transport
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if compile_engine and self.device.type != "cuda":
            raise ValueError("compile_engine=True requires a CUDA device")
        self.state_catalog = configured_catalog
        self._device_state_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self._resident_device = self._device_state_indices.device
        self._pending_actions: torch.Tensor | None = None
        self._initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._seed_values: list[int | None] = [None] * self.num_envs
        self._reset_rngs = [np.random.default_rng(lane) for lane in range(self.num_envs)]
        self._reward_clip = self._normalize_reward_clip(reward_clip)
        self.treat_episode_timeout_as_truncation = bool(treat_episode_timeout_as_truncation)
        if not self.treat_episode_timeout_as_truncation:
            raise ValueError("deathmatch-p1-v1 treats its native timeout as truncation")

        iwad_value = rom_path or os.environ.get("GRADOOM_IWAD")
        if compiled_scenario is None:
            if not iwad_value:
                raise FileNotFoundError("pass rom_path=... or set GRADOOM_IWAD")
            scenario_path = _resolve_scenario_wad(game, scenario)
            compiled_scenario = compile_deathmatch_scenario(
                scenario_path,
                iwad_value,
                require_pinned_scenario=require_pinned_scenario,
            )
        self.compiled_scenario = compiled_scenario
        self.scenario_sha256 = compiled_scenario.scenario_sha256
        self.iwad_sha256 = compiled_scenario.iwad_sha256
        vizdoom_options = dict(vizdoom_config or {})
        unknown_vizdoom_options = set(vizdoom_options) - {
            "episode_timeout",
            "render_screen_flashes",
        }
        if unknown_vizdoom_options:
            raise ValueError(f"unsupported vizdoom_config keys: {sorted(unknown_vizdoom_options)}")
        episode_timeout = _positive_int(
            vizdoom_options.get("episode_timeout", 4200),
            "episode_timeout",
        )
        render_screen_flashes = vizdoom_options.get("render_screen_flashes", False)
        if not isinstance(render_screen_flashes, bool):
            raise ValueError("render_screen_flashes must be a boolean")
        self._engine = TorchDeathmatchEngine(
            compiled_scenario,
            self.num_envs,
            device=self.device,
            frame_skip=self.frame_skip,
            frame_stack=self.frame_stack,
            episode_timeout=episode_timeout,
            doom_skill=resolved_doom_skill,
            wall_contact_damage_scale=self.wall_contact_damage_scale,
            mask_hud=obs_crop is not None,
            render_screen_flashes=render_screen_flashes,
        )
        signal_indices = {name: index for index, name in enumerate(DEVICE_SIGNAL_NAMES)}
        self._info_history_indices = torch.tensor(
            [signal_indices[name] for name in self.device_info_history_names],
            device=self.device,
            dtype=torch.int64,
        )
        self._info_history = torch.zeros(
            (self.num_envs, len(self.device_info_history_names), self.frame_stack),
            device=self.device,
            dtype=torch.float32,
        )
        obs_tensor_shape = (4, 84, 84) if obs_layout == "chw" else (84, 84, 4)
        self._safe_obs_buffers = (
            tuple(
                torch.empty(
                    (self.num_envs, *obs_tensor_shape),
                    device=self.device,
                    dtype=torch.uint8,
                )
                for _ in range(2)
            )
            if obs_copy == "safe_view"
            else ()
        )
        self._safe_obs_index = 0
        self.compile_engine = bool(compile_engine)
        self.engine_backend = "torch-compiled-cudagraph" if self.compile_engine else "torch-eager"
        if self.compile_engine:
            for phase_name in _COMPILED_ENGINE_PHASES:
                compile_options: dict[str, Any] = {
                    "backend": "inductor",
                    "fullgraph": True,
                    "dynamic": False,
                }
                if phase_name == "render_frame":
                    compile_options["options"] = _RENDER_FUSION_OPTIONS
                elif phase_name == "_enemy_tick":
                    compile_options["options"] = _ENEMY_FUSION_OPTIONS
                elif phase_name in _LIMITED_FUSION_PHASES:
                    compile_options["options"] = _LIMITED_FUSION_OPTIONS
                setattr(
                    self._engine,
                    phase_name,
                    torch.compile(getattr(self._engine, phase_name), **compile_options),
                )
        self._step_engine = self._engine.step
        self._reset_engine = (
            torch.compile(
                self._engine.reset,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
                options=_LIMITED_FUSION_OPTIONS,
            )
            if self.compile_engine
            else self._engine.reset
        )
        self._latest_reset_seeds = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.int64,
        )
        self._cuda_graph: torch.cuda.CUDAGraph | None = None
        self._cuda_graph_actions = torch.empty(
            self.num_envs,
            device=self.device,
            dtype=torch.int64,
        )
        self._cuda_graph_reset_seeds = torch.empty_like(self._cuda_graph_actions)
        self._cuda_graph_transition: DeviceAutoResetTransition | None = None

        self.buttons = DEATHMATCH_BUTTONS
        if isinstance(use_restricted_actions, str):
            raise ValueError(
                "deathmatch-p1-v1 requires the pinned custom 17-action table; "
                "string action modes are not supported"
            )
        action_value = use_restricted_actions
        uses_default_action_preset = action_value is DEATHMATCH_ACTIONS
        self.action_table, self.action_meanings, self.action_table_hash = normalize_action_table(
            action_value,
            buttons=self.buttons,
        )
        self.action_mode = "custom_discrete"
        self.action_preset = "deathmatch-p1-v1" if uses_default_action_preset else None
        self.use_restricted_actions = use_restricted_actions
        action_matrix = torch.zeros(
            (len(self.action_table), len(self.buttons)), device=self.device, dtype=torch.bool
        )
        button_index = {name: index for index, name in enumerate(self.buttons)}
        for action_index, labels in enumerate(self.action_table):
            for label in labels:
                action_matrix[action_index, button_index[label]] = True
        self._action_matrix = action_matrix
        self.single_action_space = gym.spaces.Discrete(len(self.action_table))
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(self.num_envs, len(self.action_table), dtype=np.int64)
        )
        single_shape = (4, 84, 84) if obs_layout == "chw" else (84, 84, 4)
        self.single_observation_space = gym.spaces.Box(0, 255, single_shape, dtype=np.uint8)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        if isinstance(game_variables, (str, bytes, bytearray)):
            raise TypeError("game_variables must be a sequence of signal names or None")
        self.game_variable_names = tuple(
            str(value).casefold() for value in (game_variables or _DEFAULT_SIGNALS)
        )
        if len(self.game_variable_names) != len(set(self.game_variable_names)):
            raise ValueError("game_variables must not contain duplicates")
        unknown = set(self.game_variable_names) - set(_DEFAULT_SIGNALS)
        if unknown:
            raise ValueError(f"unknown deathmatch game variables: {sorted(unknown)}")
        self._configure_info_filter(info_filter)
        history_not_selected = set(self.device_info_history_names) - set(self._info_keys)
        if history_not_selected:
            raise ValueError(
                "info_frame_stack_keys must be included by info_filter: "
                f"{sorted(history_not_selected)}"
            )
        if self.device_info_history_names and self._info_mode != "all":
            raise ValueError("info_frame_stack_keys must be available on reset and every step")
        signal_schema = (
            {
                name: MappingProxyType(
                    {
                        "dtype": "float64",
                        "shape": (),
                        "available_on_reset": self._info_mode == "all",
                        "available_on_step": self._info_mode != "none",
                    }
                )
                for name in self._info_keys
            }
            if self._info_mode != "none"
            else {}
        )
        for name in self.device_info_history_names:
            signal_schema[f"{name}_frame_stack"] = MappingProxyType(
                {
                    "dtype": "float64",
                    "shape": (self.frame_stack,),
                    "available_on_reset": True,
                    "available_on_step": True,
                }
            )
        self.signal_schema = MappingProxyType(signal_schema)
        self.capabilities = MappingProxyType(
            {
                "supported_action_modes": ("custom_discrete",),
                "supported_observation_layouts": ("chw", "hwc"),
                "supported_observation_color_modes": ("grayscale",),
                "supported_resize_algorithms": ("area",),
                "supported_crop_modes": ("remove", "mask"),
                "supported_observation_copy_modes": ("copy", "safe_view", "unsafe_view"),
                "supported_transition_transports": ("torch",),
                "supports_async_step": True,
                "supports_branching": False,
                "supports_device_api": True,
                "supports_emulator_ram": False,
                "supports_enemy_variants": False,
                "supports_fire_reset": False,
                "supports_info_frame_stack": True,
                "supports_live_snapshots": False,
                "supports_maxpool_last_two": False,
                "supports_noop_reset": False,
                "supports_per_lane_rgb": render_mode == "rgb_array",
                "supports_reward_clipping": True,
                "supports_snapshot_codec": False,
                "supports_state_catalog": True,
                "supports_sticky_action_prob": False,
                "supports_surface_variants": False,
            }
        )

    @staticmethod
    def _normalize_reward_clip(value: Any) -> tuple[float, float] | None:
        if value is False:
            return None
        if value is True:
            return (-1.0, 1.0)
        try:
            low, high = (float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise ValueError("reward_clip must be a bool or a (low, high) pair") from exc
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError("reward_clip bounds must be finite with low <= high")
        return low, high

    def _configure_info_filter(self, value: str | Mapping[str, Any]) -> None:
        signal_names = (*self.game_variable_names, *_DERIVED_SIGNALS)
        if isinstance(value, Mapping):
            unknown_options = set(value) - {"mode", "keys"}
            if unknown_options:
                raise ValueError(f"unknown info_filter keys: {sorted(unknown_options)}")
            mode = str(value.get("mode", "all"))
            keys = value.get("keys")
            if isinstance(keys, (str, bytes, bytearray)):
                raise TypeError("info_filter keys must be a sequence of signal names")
            selected = signal_names if keys is None else tuple(str(key).casefold() for key in keys)
        else:
            mode = str(value)
            selected = signal_names
        if mode not in {"all", "terminal", "none"}:
            raise ValueError("info_filter mode must be 'all', 'terminal', or 'none'")
        if len(selected) != len(set(selected)):
            raise ValueError("info_filter keys must not contain duplicates")
        unknown = set(selected) - set(signal_names)
        if unknown:
            raise ValueError(f"unknown info signals: {sorted(unknown)}")
        self._info_mode = mode
        self._info_keys = tuple(selected)

    def _observations(self, values: torch.Tensor) -> torch.Tensor:
        if self.obs_layout == "hwc":
            values = values.permute(0, 2, 3, 1)
        if self.obs_copy == "copy":
            return values.clone()
        if self.obs_copy == "safe_view":
            buffer = self._safe_obs_buffers[self._safe_obs_index]
            self._safe_obs_index = (self._safe_obs_index + 1) % len(self._safe_obs_buffers)
            buffer.copy_(values)
            return buffer
        return values

    def _device_observations(self, values: torch.Tensor) -> torch.Tensor:
        return values if self.obs_layout == "chw" else values.permute(0, 2, 3, 1)

    def _infos(self, availability: torch.Tensor | None = None) -> dict[str, Any]:
        if self._info_mode == "none":
            return {}
        if availability is None:
            availability = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        if self._info_mode == "terminal":
            availability = availability & self._engine.pending_reset
        signals = self._engine.signals()
        result: dict[str, Any] = {}
        for name in self._info_keys:
            result[name] = signals[name]
            result[f"_{name}"] = availability.clone()
        for index, name in enumerate(self.device_info_history_names):
            history_name = f"{name}_frame_stack"
            result[history_name] = self._info_history[:, index].to(torch.float64)
            result[f"_{history_name}"] = availability.clone()
        return result

    def _require_device_tensor(
        self,
        value: Any,
        name: str,
        *,
        shape: tuple[int, ...],
        dtypes: tuple[torch.dtype, ...],
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a Torch tensor")
        if value.device != self._resident_device:
            raise TypeError(f"{name} must be on device {self._resident_device}")
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if value.dtype not in dtypes:
            expected = ", ".join(str(dtype) for dtype in dtypes)
            raise TypeError(f"{name} must have dtype {expected}")
        return value

    def _reset_info_histories(self, mask: torch.Tensor) -> None:
        if not self.device_info_history_names:
            return
        current = self._engine.signal_buffer.index_select(1, self._info_history_indices)
        reset_values = current[:, :, None].expand(-1, -1, self.frame_stack)
        self._info_history.copy_(torch.where(mask[:, None, None], reset_values, self._info_history))

    def _advance_info_histories(self) -> None:
        if not self.device_info_history_names:
            return
        history = torch.roll(self._info_history, shifts=-1, dims=2)
        history[:, :, -1].copy_(
            self._engine.signal_buffer.index_select(1, self._info_history_indices)
        )
        self._info_history.copy_(history)

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._seed_values = _normalize_seed(seed, self.num_envs)
        return list(self._seed_values)

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        if self.closed:
            raise RuntimeError("cannot reset a closed environment")
        reset_options = dict(options or {})
        raw_mask = reset_options.pop("reset_mask", None)
        if raw_mask is None:
            mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            mask = self._require_device_tensor(
                raw_mask,
                "options['reset_mask']",
                shape=(self.num_envs,),
                dtypes=(torch.bool,),
            )
        if not bool(torch.any(mask)):
            raise ValueError("options['reset_mask'] must select at least one lane")
        state_indices = reset_options.pop("state_indices", None)
        if state_indices is not None:
            selected_states = self._require_device_tensor(
                state_indices,
                "options['state_indices']",
                shape=(self.num_envs,),
                dtypes=(torch.int32,),
            )
            if bool(torch.any(selected_states[mask] != 0)):
                raise ValueError("deathmatch-p1-v1 has only state index 0")
        if reset_options:
            raise ValueError(f"unsupported reset options: {sorted(reset_options)}")
        seed_values = (
            _normalize_seed(seed, self.num_envs) if seed is not None else self._seed_values
        )
        game_seeds = [0] * self.num_envs
        for lane in torch.nonzero(mask, as_tuple=False).flatten().to("cpu").tolist():
            lane_seed = seed_values[lane]
            if lane_seed is not None:
                self._reset_rngs[lane] = np.random.default_rng(lane_seed)
            game_seeds[lane] = int(
                self._reset_rngs[lane].integers(
                    0,
                    np.iinfo(np.uint32).max + 1,
                    dtype=np.uint32,
                )
            )
        seeds = torch.tensor(game_seeds, device=self.device, dtype=torch.int64)
        self._seed_values = [None] * self.num_envs
        observations = self._engine.reset(mask, seeds)
        self._initialized |= mask
        self._reset_info_histories(mask)
        infos = self._infos(mask)
        infos["state_index"] = self._device_state_indices.clone()
        infos["_state_index"] = mask.clone()
        infos["start_source"] = torch.zeros(self.num_envs, device=self.device, dtype=torch.int8)
        infos["_start_source"] = mask.clone()
        infos["noop_reset_count"] = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int64
        )
        infos["_noop_reset_count"] = mask.clone()
        return self._observations(observations), infos

    def step(self, actions: torch.Tensor):
        if self.closed:
            raise RuntimeError("cannot step a closed environment")
        if not bool(torch.all(self._initialized)):
            raise RuntimeError("all lanes must be reset before the first step")
        if bool(torch.any(self._engine.pending_reset)):
            lanes = torch.nonzero(self._engine.pending_reset).flatten().to("cpu").tolist()
            raise RuntimeError(f"terminal lanes must be reset before step: {lanes}")
        indices = self._require_device_tensor(
            actions,
            "actions",
            shape=(self.num_envs,),
            dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
        ).to(torch.int64)
        if bool(torch.any((indices < 0) | (indices >= len(self.action_table)))):
            raise ValueError("actions fall outside the declared action space")
        transition = self.step_device(indices)
        return (
            self._observations(transition.observations),
            transition.rewards,
            transition.terminated,
            transition.truncated,
            self._infos(),
        )

    def step_async(self, actions: torch.Tensor) -> None:
        if self._pending_actions is not None:
            raise RuntimeError("an asynchronous step is already pending")
        self._pending_actions = actions

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("no asynchronous step is pending")
        actions = self._pending_actions
        self._pending_actions = None
        return self.step(actions)

    def step_device(self, actions: torch.Tensor) -> DeviceTransition:
        """Advance all lanes and return only device tensors, with no host synchronization."""

        if self.closed:
            raise RuntimeError("cannot step a closed environment")
        indices = self._require_device_tensor(
            actions,
            "actions",
            shape=(self.num_envs,),
            dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
        ).to(torch.int64)
        buttons = self._action_matrix[indices]
        observations, rewards, terminated, truncated = self._step_engine(buttons)
        self._advance_info_histories()
        if self._reward_clip is not None:
            raw_rewards = rewards.clone()
            rewards.clamp_(self._reward_clip[0], self._reward_clip[1])
            self._engine.episode_return.add_(rewards - raw_rewards)
            self._engine.signal_buffer[:, 18].copy_(self._engine.episode_return)
        return DeviceTransition(
            observations=self._device_observations(observations),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            signals=self._engine.signal_buffer,
            info_histories=self._info_history,
        )

    def reset_device(
        self, mask: torch.Tensor, seeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reset selected lanes entirely on device and return observation/signal views."""

        if self.closed:
            raise RuntimeError("cannot reset a closed environment")
        device_mask = self._require_device_tensor(
            mask,
            "mask",
            shape=(self.num_envs,),
            dtypes=(torch.bool,),
        )
        device_seeds = self._require_device_tensor(
            seeds,
            "seeds",
            shape=(self.num_envs,),
            dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
        ).to(torch.int64)
        observations = self._reset_engine(
            device_mask,
            device_seeds,
        )
        self._latest_reset_seeds.copy_(
            torch.where(device_mask, device_seeds, self._latest_reset_seeds)
        )
        self._initialized |= device_mask
        self._reset_info_histories(device_mask)
        return self._device_observations(observations), self._engine.signal_buffer

    def _step_and_reset_device_impl(
        self,
        actions: torch.Tensor,
        reset_seeds: torch.Tensor,
    ) -> DeviceAutoResetTransition:
        transition = self.step_device(actions)
        final_observations = transition.observations.clone()
        final_signals = transition.signals.clone()
        final_info_histories = transition.info_histories.clone()
        done = transition.terminated | transition.truncated
        observations, signals = self.reset_device(done, reset_seeds)
        return DeviceAutoResetTransition(
            observations=observations,
            rewards=transition.rewards,
            terminated=transition.terminated,
            truncated=transition.truncated,
            signals=signals,
            info_histories=self._info_history,
            final_observations=final_observations,
            final_signals=final_signals,
            final_info_histories=final_info_histories,
        )

    def _capture_step_and_reset_graph(
        self,
        actions: torch.Tensor,
        reset_seeds: torch.Tensor,
    ) -> None:
        """Warm compiled phases, restore the lanes, and capture one fixed-shape transaction."""

        initial_seeds = self._latest_reset_seeds.clone()
        all_lanes = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._cuda_graph_actions.copy_(actions)
        self._cuda_graph_reset_seeds.copy_(reset_seeds)

        self._step_and_reset_device_impl(
            self._cuda_graph_actions,
            self._cuda_graph_reset_seeds,
        )
        torch.cuda.synchronize(self.device)
        self.reset_device(all_lanes, initial_seeds)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stepped = self.step_device(self._cuda_graph_actions)
            final_observations = stepped.observations.clone()
            final_signals = stepped.signals.clone()
            final_info_histories = stepped.info_histories.clone()
            done = stepped.terminated | stepped.truncated
            reset_requested = torch.any(done)
            graph.begin_capture_to_if_node(reset_requested)
            observations, signals = self.reset_device(
                done,
                self._cuda_graph_reset_seeds,
            )
            graph.end_capture_to_conditional_node()
            transition = DeviceAutoResetTransition(
                observations=observations,
                rewards=stepped.rewards,
                terminated=stepped.terminated,
                truncated=stepped.truncated,
                signals=signals,
                info_histories=self._info_history,
                final_observations=final_observations,
                final_signals=final_signals,
                final_info_histories=final_info_histories,
            )
        torch.cuda.synchronize(self.device)

        # Capture executes the transaction once. Restore the exact initial lane
        # states so the caller's first replay remains its first environment step.
        self.reset_device(all_lanes, initial_seeds)
        torch.cuda.synchronize(self.device)
        self._cuda_graph = graph
        self._cuda_graph_transition = transition

    def step_and_reset_device(
        self,
        actions: torch.Tensor,
        reset_seeds: torch.Tensor,
    ) -> DeviceAutoResetTransition:
        """Step and reset terminal lanes without synchronizing or leaving the device."""

        if not self.compile_engine:
            return self._step_and_reset_device_impl(actions, reset_seeds)
        device_actions = self._require_device_tensor(
            actions,
            "actions",
            shape=(self.num_envs,),
            dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
        ).to(torch.int64)
        device_reset_seeds = self._require_device_tensor(
            reset_seeds,
            "reset_seeds",
            shape=(self.num_envs,),
            dtypes=(torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
        ).to(torch.int64)
        if self._cuda_graph is None:
            self._capture_step_and_reset_graph(device_actions, device_reset_seeds)
        self._cuda_graph_actions.copy_(device_actions)
        self._cuda_graph_reset_seeds.copy_(device_reset_seeds)
        self._cuda_graph.replay()
        if self._cuda_graph_transition is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("CUDA graph capture did not produce a transition")
        return self._cuda_graph_transition

    def device_signals(self) -> torch.Tensor:
        return self._engine.signal_buffer

    def device_info_histories(self) -> torch.Tensor:
        return self._info_history

    def active_state_indices(self) -> torch.Tensor:
        return self._device_state_indices

    def render_lane(self, lane: int) -> np.ndarray | None:
        if self.closed:
            raise RuntimeError("cannot render a closed environment")
        if isinstance(lane, (bool, np.bool_)):
            raise TypeError("lane must be an integer")
        lane_index = operator.index(lane)
        if not 0 <= lane_index < self.num_envs:
            raise IndexError(f"lane must be in [0, {self.num_envs - 1}]")
        if self.render_mode != "rgb_array":
            return None
        return (
            self._engine.render_native_frame(include_hud=True)[lane_index]
            .detach()
            .to("cpu")
            .numpy()
            .copy()
        )

    def render(self) -> np.ndarray | None:
        return self.render_lane(0)

    def get_images(self) -> list[np.ndarray | None]:
        if self.closed:
            raise RuntimeError("cannot render a closed environment")
        if self.render_mode != "rgb_array":
            return [None for _ in range(self.num_envs)]
        frames = self._engine.render_native_frame(include_hud=True).detach().to("cpu").numpy()
        return [frame.copy() for frame in frames]

    def close(self) -> None:
        self._pending_actions = None
        self.closed = True


VizdoomGpuVecEnv = GraDoomVecEnv

__all__ = [
    "DeviceAutoResetTransition",
    "DeviceTransition",
    "GraDoomVecEnv",
    "VizdoomGpuVecEnv",
    "scenario_buttons",
]
