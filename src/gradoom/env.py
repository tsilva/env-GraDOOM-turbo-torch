"""`vizdoom-turbo`-shaped vector API backed by device tensors."""

from __future__ import annotations

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
    del game, scenario
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

    Torch tensors are the certified transport. NumPy transport exists for CPU
    contract tests and diagnostics and is not part of the performance path.
    """

    metadata: ClassVar[dict[str, Any]] = {
        "autoreset_mode": AutoresetMode.DISABLED,
        "render_modes": ["rgb_array"],
        "render_fps": 35,
        "turbo_api_version": 1,
        "gradoom_device_api_version": 1,
    }
    supports_live_snapshots = False
    live_snapshots_deterministic = False
    parity_certified = False
    device_signal_names = DEVICE_SIGNAL_NAMES

    def __init__(
        self,
        game: str | Path | None = "VizdoomDeathmatch-v1",
        state: Any = "default",
        scenario: str | Path | None = None,
        info: Any = None,
        use_restricted_actions: Any = DEATHMATCH_ACTIONS,
        record: bool = False,
        players: int = 1,
        inttype: Any = "stable",
        obs_type: Any = "image",
        render_mode: str = "rgb_array",
        *,
        num_envs: int = 1,
        num_threads: int | None = None,
        rom_path: str | None = None,
        device: str | torch.device | None = None,
        transport: Literal["torch", "numpy"] = "torch",
        obs_copy: Literal["copy", "safe_view", "unsafe_view"] = "unsafe_view",
        obs_resize: tuple[int, int] | None = (84, 84),
        obs_crop: tuple[int, int, int, int] | None = (0, 32, 0, 0),
        obs_crop_mode: Literal["remove", "mask"] = "mask",
        obs_crop_fill: int = 0,
        obs_grayscale: bool = True,
        obs_resize_algorithm: Literal["nearest", "bilinear", "area"] = "area",
        obs_layout: Literal["hwc", "chw"] = "chw",
        frame_skip: int = 2,
        frame_stack: int = 4,
        maxpool_last_two: bool = False,
        noop_reset_max: int = 0,
        use_fire_reset: bool = False,
        sticky_action_prob: float = 0.0,
        reward_clip: bool | tuple[float, float] = False,
        info_filter: str | Mapping[str, Any] = "all",
        info_frame_stack_keys: Sequence[str] | None = None,
        state_catalog: Sequence[Any] | None = None,
        doom_map: str | None = None,
        doom_skill: int | None = None,
        game_args: str | None = None,
        game_variables: Sequence[str] | None = None,
        enemy_variants: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
        surface_variants: Mapping[str, Sequence[str]] | None = None,
        treat_episode_timeout_as_truncation: bool = True,
        vizdoom_config: Mapping[str, Any] | None = None,
        compiled_scenario: CompiledScenario | None = None,
        require_pinned_scenario: bool = True,
        compile_engine: bool = False,
        **unsupported: Any,
    ) -> None:
        if unsupported:
            raise TypeError(f"unsupported option(s): {', '.join(sorted(unsupported))}")
        if info not in (None, "data"):
            raise ValueError("info must be None or 'data'")
        if record:
            raise ValueError("record=True is not supported on the device path")
        if players != 1:
            raise ValueError("the deathmatch-p1-v1 profile supports players=1")
        if str(obs_type).split(".")[-1].casefold() != "image":
            raise ValueError("GraDoomVecEnv supports image observations only")
        if render_mode != "rgb_array":
            raise ValueError("render_mode must be 'rgb_array'")
        if state not in (None, "default") or state_catalog not in (None, ("default",), ["default"]):
            raise ValueError("the first device profile supports only the default initial state")
        if any(
            value is not None for value in (doom_map, game_args, enemy_variants, surface_variants)
        ):
            raise ValueError("custom maps, game args, and variants are not yet supported")
        if doom_skill not in (None, 1, 3):
            raise ValueError("deathmatch-p1-v1 supports the configured Doom skill only")
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
        if obs_layout not in {"chw", "hwc"}:
            raise ValueError("obs_layout must be 'chw' or 'hwc'")
        if obs_resize_algorithm not in {"nearest", "bilinear", "area"}:
            raise ValueError("unknown resize algorithm")
        if transport not in {"torch", "numpy"}:
            raise ValueError("transport must be 'torch' or 'numpy'")
        if obs_copy not in {"copy", "safe_view", "unsafe_view"}:
            raise ValueError("obs_copy must be copy, safe_view, or unsafe_view")
        del inttype

        self.num_envs = _positive_int(num_envs, "num_envs")
        self.num_threads = 0 if num_threads is None else _positive_int(num_threads, "num_threads")
        self.frame_skip = _positive_int(frame_skip, "frame_skip")
        self.frame_stack = frame_stack
        self.device_info_history_names = tuple(
            str(name).casefold() for name in (info_frame_stack_keys or ())
        )
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
        if transport == "numpy" and self.device.type != "cpu":
            raise ValueError("NumPy transport is diagnostic-only and requires device='cpu'")
        self.state_catalog = ("default",)
        self._active_state_indices = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        self._seed_base = 0
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
        episode_timeout = int((vizdoom_config or {}).get("episode_timeout", 4200))
        self._engine = TorchDeathmatchEngine(
            compiled_scenario,
            self.num_envs,
            device=self.device,
            frame_skip=self.frame_skip,
            frame_stack=self.frame_stack,
            episode_timeout=episode_timeout,
            mask_hud=obs_crop is not None,
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
        self.compile_engine = bool(compile_engine)
        self.engine_backend = "torch-compiled" if self.compile_engine else "torch-eager"
        self._step_engine = (
            torch.compile(
                self._engine.step,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )
            if self.compile_engine
            else self._engine.step
        )

        self.buttons = DEATHMATCH_BUTTONS
        if isinstance(use_restricted_actions, str):
            raise ValueError(
                "deathmatch-p1-v1 requires the pinned custom 17-action table; "
                "string action modes are not supported"
            )
        action_value = use_restricted_actions
        self.action_table, self.action_meanings, self.action_table_hash = normalize_action_table(
            action_value,
            buttons=self.buttons,
        )
        self.action_mode = "custom_discrete"
        self.action_preset = "deathmatch-p1-v1" if action_value == DEATHMATCH_ACTIONS else None
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

        self.game_variable_names = tuple(
            str(value).casefold() for value in (game_variables or _DEFAULT_SIGNALS)
        )
        unknown = set(self.game_variable_names) - set(_DEFAULT_SIGNALS)
        if unknown:
            raise ValueError(f"unknown deathmatch game variables: {sorted(unknown)}")
        self._configure_info_filter(info_filter)
        if self.device_info_history_names and self._info_mode != "all":
            raise ValueError("device info frame stacks require info_filter mode='all'")
        signal_schema = {
            name: MappingProxyType(
                {
                    "dtype": np.dtype(np.float64),
                    "shape": (),
                    "available_on_reset": self._info_mode == "all",
                    "available_on_step": self._info_mode != "none",
                }
            )
            for name in self._info_keys
        }
        for name in self.device_info_history_names:
            signal_schema[f"{name}_frame_stack"] = MappingProxyType(
                {
                    "dtype": np.dtype(np.float64),
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
                "supported_resize_algorithms": ("area", "nearest", "bilinear"),
                "supported_observation_copy_modes": ("copy", "safe_view", "unsafe_view"),
                "supports_maxpool_last_two": False,
                "supports_sticky_action_prob": False,
                "supports_reward_clipping": True,
                "supports_noop_reset": False,
                "supports_state_catalog": False,
                "supports_live_snapshots": False,
                "supports_per_lane_rgb": True,
                "supports_enemy_variants": False,
                "supports_surface_variants": False,
                "supports_info_frame_stack": True,
            }
        )

    @staticmethod
    def _normalize_reward_clip(value: Any) -> tuple[float, float] | None:
        if value is False:
            return None
        if value is True:
            return (-1.0, 1.0)
        low, high = (float(item) for item in value)
        if low > high:
            raise ValueError("reward clip low must not exceed high")
        return low, high

    def _configure_info_filter(self, value: str | Mapping[str, Any]) -> None:
        signal_names = (*self.game_variable_names, *_DERIVED_SIGNALS)
        if isinstance(value, Mapping):
            mode = str(value.get("mode", "all"))
            keys = value.get("keys")
            selected = signal_names if keys is None else tuple(str(key).casefold() for key in keys)
        else:
            mode = str(value)
            selected = signal_names
        if mode not in {"all", "terminal", "none"}:
            raise ValueError("info_filter mode must be all, terminal, or none")
        unknown = set(selected) - set((*_DEFAULT_SIGNALS, *_DERIVED_SIGNALS))
        if unknown:
            raise ValueError(f"unknown info signals: {sorted(unknown)}")
        self._info_mode = mode
        self._info_keys = tuple(selected)

    def _observations(self, values: torch.Tensor):
        if self.obs_layout == "hwc":
            values = values.permute(0, 2, 3, 1)
        if self.obs_copy in {"copy", "safe_view"}:
            values = values.clone()
        if self.transport == "numpy":
            return values.numpy()
        return values

    def _device_observations(self, values: torch.Tensor) -> torch.Tensor:
        return values if self.obs_layout == "chw" else values.permute(0, 2, 3, 1)

    def _value(self, value: torch.Tensor):
        if self.transport == "numpy":
            return value.numpy()
        return value

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
            result[name] = self._value(signals[name])
            result[f"_{name}"] = self._value(availability.clone())
        for index, name in enumerate(self.device_info_history_names):
            history_name = f"{name}_frame_stack"
            result[history_name] = self._value(self._info_history[:, index])
            result[f"_{history_name}"] = self._value(availability.clone())
        return result

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
            mask = torch.as_tensor(raw_mask, device=self.device)
            if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
                raise TypeError("options['reset_mask'] must be bool with shape (num_envs,)")
        state_indices = reset_options.pop("state_indices", None)
        if state_indices is not None and bool(torch.any(torch.as_tensor(state_indices) != 0)):
            raise ValueError("deathmatch-p1-v1 has only state index 0")
        if reset_options:
            raise ValueError(f"unsupported reset options: {sorted(reset_options)}")
        if seed is None:
            seeds = torch.arange(
                self._seed_base,
                self._seed_base + self.num_envs,
                device=self.device,
                dtype=torch.int64,
            )
            self._seed_base += self.num_envs
        elif isinstance(seed, Sequence) and not isinstance(seed, (str, bytes, bytearray)):
            if len(seed) != self.num_envs:
                raise ValueError("seed sequence length must match num_envs")
            seeds = torch.tensor(
                [
                    self._seed_base + index if value is None else int(value)
                    for index, value in enumerate(seed)
                ],
                device=self.device,
                dtype=torch.int64,
            )
            self._seed_base += self.num_envs
        else:
            seeds = torch.arange(int(seed), int(seed) + self.num_envs, device=self.device)
        observations = self._engine.reset(mask, seeds)
        self._reset_info_histories(mask)
        infos = self._infos(mask)
        infos["state_index"] = self._value(self._active_state_indices.clone())
        infos["_state_index"] = self._value(mask.clone())
        infos["start_source"] = np.full(self.num_envs, "environment", dtype=object)
        infos["_start_source"] = self._value(mask.clone())
        return self._observations(observations), infos

    def step(self, actions: Any):
        if self.closed:
            raise RuntimeError("cannot step a closed environment")
        indices = torch.as_tensor(actions, device=self.device, dtype=torch.int64)
        if indices.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if self.device.type == "cpu" and bool(
            torch.any((indices < 0) | (indices >= len(self.action_table)))
        ):
            raise ValueError("actions fall outside the declared action space")
        transition = self.step_device(indices)
        return (
            self._observations(transition.observations),
            self._value(transition.rewards),
            self._value(transition.terminated),
            self._value(transition.truncated),
            self._infos(),
        )

    def step_device(self, actions: torch.Tensor) -> DeviceTransition:
        """Advance all lanes and return only device tensors, with no host synchronization."""

        if self.closed:
            raise RuntimeError("cannot step a closed environment")
        indices = actions.to(device=self.device, dtype=torch.int64)
        if indices.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        buttons = self._action_matrix[indices]
        observations, rewards, terminated, truncated = self._step_engine(buttons)
        self._advance_info_histories()
        if self._reward_clip is not None:
            rewards.clamp_(self._reward_clip[0], self._reward_clip[1])
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
        device_mask = mask.to(device=self.device, dtype=torch.bool)
        observations = self._engine.reset(
            device_mask,
            seeds.to(device=self.device, dtype=torch.int64),
        )
        self._reset_info_histories(device_mask)
        return self._device_observations(observations), self._engine.signal_buffer

    def step_and_reset_device(
        self,
        actions: torch.Tensor,
        reset_seeds: torch.Tensor,
    ) -> DeviceAutoResetTransition:
        """Step and reset terminal lanes without synchronizing or leaving the device."""

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

    def device_signals(self) -> torch.Tensor:
        return self._engine.signal_buffer

    def device_info_histories(self) -> torch.Tensor:
        return self._info_history

    def active_state_indices(self):
        values = self._active_state_indices.detach().to("cpu").numpy().copy()
        values.setflags(write=False)
        return values

    def render_lane(self, lane: int) -> np.ndarray:
        if self.closed:
            raise RuntimeError("cannot render a closed environment")
        lane_index = operator.index(lane)
        if not 0 <= lane_index < self.num_envs:
            raise IndexError(f"lane must be in [0, {self.num_envs - 1}]")
        frame = self._engine.frames[lane_index, -1].detach().to("cpu").numpy()
        return np.repeat(frame[..., None], 3, axis=-1)

    def render(self) -> np.ndarray:
        return self.render_lane(0)

    def get_images(self) -> list[np.ndarray]:
        return [self.render_lane(lane) for lane in range(self.num_envs)]

    def close(self) -> None:
        self.closed = True


VizdoomGpuVecEnv = GraDoomVecEnv

__all__ = [
    "DeviceAutoResetTransition",
    "DeviceTransition",
    "GraDoomVecEnv",
    "VizdoomGpuVecEnv",
    "scenario_buttons",
]
