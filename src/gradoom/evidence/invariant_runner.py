from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradoom.actions import DEATHMATCH_ACTION_MEANINGS, DEATHMATCH_ACTIONS

RUNNER_PROTOCOL_VERSION = 1
_SIGNALS = ("health", "killcount", "player_killcount", "episode_return")


class _FixtureTurboEnv:
    def __init__(
        self,
        game: str,
        state: Any = None,
        scenario: Any = None,
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
        transport: str = "default",
        obs_copy: str = "safe_view",
        obs_resize: tuple[int, int] | None = (84, 84),
        obs_crop: tuple[int, int, int, int] | None = None,
        obs_crop_mode: str = "remove",
        obs_crop_fill: int = 0,
        obs_grayscale: bool = True,
        obs_resize_algorithm: str = "area",
        obs_layout: str = "chw",
        frame_skip: int = 4,
        frame_stack: int = 4,
        maxpool_last_two: bool = False,
        noop_reset_max: int = 0,
        use_fire_reset: bool = False,
        sticky_action_prob: float = 0.0,
        reward_clip: bool = False,
        info_filter: Any = "all",
        info_frame_stack_keys: Any = None,
        state_catalog: Any = None,
        fixture_transport: str = "numpy",
    ) -> None:
        del (
            game,
            state,
            scenario,
            info,
            use_restricted_actions,
            record,
            players,
            inttype,
            obs_type,
            render_mode,
            num_threads,
            rom_path,
            transport,
            obs_copy,
            obs_resize,
            obs_crop,
            obs_crop_mode,
            obs_crop_fill,
            obs_grayscale,
            obs_resize_algorithm,
            obs_layout,
            frame_skip,
            frame_stack,
            maxpool_last_two,
            noop_reset_max,
            use_fire_reset,
            sticky_action_prob,
            reward_clip,
            info_filter,
            info_frame_stack_keys,
            state_catalog,
        )
        self.num_envs = num_envs
        self.fixture_transport = fixture_transport
        self.device = torch.device("cpu")
        self.action_meanings = DEATHMATCH_ACTION_MEANINGS
        self._initialized = np.zeros(num_envs, dtype=np.bool_)
        self._pending_reset = np.zeros(num_envs, dtype=np.bool_)
        self._state = np.zeros(num_envs, dtype=np.uint8)
        self._killcount = np.zeros(num_envs, dtype=np.float64)
        self._player_killcount = np.zeros(num_envs, dtype=np.float64)
        self._episode_return = np.zeros(num_envs, dtype=np.float64)

    def _array(self, value: np.ndarray) -> Any:
        return torch.from_numpy(value.copy()) if self.fixture_transport == "torch" else value.copy()

    def _observation(self) -> Any:
        value = np.broadcast_to(self._state[:, None, None, None], (self.num_envs, 4, 84, 84))
        return self._array(value)

    def _infos(self) -> dict[str, Any]:
        return {
            "health": self._array(np.full(self.num_envs, 100.0, dtype=np.float64)),
            "killcount": self._array(self._killcount),
            "player_killcount": self._array(self._player_killcount),
            "episode_return": self._array(self._episode_return),
        }

    def reset(
        self,
        *,
        seed: Any = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        del seed
        raw_mask = (options or {}).get("reset_mask")
        if raw_mask is None:
            mask = np.ones(self.num_envs, dtype=np.bool_)
        elif isinstance(raw_mask, torch.Tensor):
            mask = raw_mask.detach().cpu().numpy().astype(np.bool_, copy=False)
        else:
            mask = np.asarray(raw_mask, dtype=np.bool_)
        self._initialized[mask] = True
        self._pending_reset[mask] = False
        self._state[mask] = 0
        self._killcount[mask] = 0
        self._player_killcount[mask] = 0
        self._episode_return[mask] = 0
        return self._observation(), self._infos()

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        if not bool(np.all(self._initialized)):
            raise RuntimeError("all lanes must be reset before step")
        if bool(np.any(self._pending_reset)):
            raise RuntimeError("terminal lanes must be reset before step")
        if isinstance(actions, torch.Tensor):
            indices = actions.detach().cpu().numpy()
        else:
            indices = np.asarray(actions)
        self._state += 1
        rewards = np.arange(self.num_envs, dtype=np.float32)
        terminated = indices == 1
        truncated = indices == 2
        player_kills = indices == 3
        enemy_kills = indices == 4
        self._killcount += player_kills + enemy_kills
        self._player_killcount += player_kills
        self._episode_return += rewards
        self._pending_reset |= terminated | truncated
        return (
            self._observation(),
            self._array(rewards),
            self._array(terminated.astype(np.bool_)),
            self._array(truncated.astype(np.bool_)),
            self._infos(),
        )

    def close(self) -> None:
        pass


def _json_default(value: object) -> object:
    if value is inspect.Parameter.empty:
        return "required"
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "transport": "torch",
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
        }
    array = np.asarray(value)
    return {
        "transport": "numpy",
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "device": "cpu",
    }


def _values(value: Any) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return [float(item) for item in np.asarray(value).tolist()]


def _mask(provider: str, values: list[bool]) -> Any:
    return torch.tensor(values, dtype=torch.bool) if provider == "gradoom" else np.asarray(values)


def _actions(provider: str, values: list[int]) -> Any:
    return torch.tensor(values, dtype=torch.int64) if provider == "gradoom" else np.asarray(values)


def _capture_contract(
    *,
    provider: str,
    revision: str,
    env_class: type[Any],
    factory: Callable[[], Any],
    fixture_case: str,
    kill_signal_reader: Callable[..., dict[str, float]] | None = None,
) -> dict[str, Any]:
    signature = inspect.signature(env_class).parameters
    common = list(signature.values())[:32]
    env = factory()
    try:
        try:
            env.step(_actions(provider, [0, 0]))
        except RuntimeError:
            step_before_reset_rejected = True
        else:
            step_before_reset_rejected = False
        reset_observation, reset_infos = env.reset(seed=[7, 8])
        step_result = env.step(_actions(provider, [0, 0]))
        step_observation, rewards, terminated, truncated, step_infos = step_result
        before_masked = (
            step_observation.clone()
            if isinstance(step_observation, torch.Tensor)
            else step_observation.copy()
        )
        masked_observation, _masked_infos = env.reset(
            options={"reset_mask": _mask(provider, [True, False])}
        )
        masked_selected_only = bool(
            _values(masked_observation[:, 0, 0, 0]) == [0.0, _values(before_masked[:, 0, 0, 0])[1]]
        )
    finally:
        env.close()

    terminal_env = factory()
    try:
        terminal_env.reset(seed=[9, 10])
        terminal = terminal_env.step(_actions(provider, [1, 0]))
        try:
            terminal_env.step(_actions(provider, [0, 0]))
        except RuntimeError:
            termination_requires_reset = True
        else:
            termination_requires_reset = False
    finally:
        terminal_env.close()

    truncation_env = factory()
    try:
        truncation_env.reset(seed=[11, 12])
        truncation = truncation_env.step(_actions(provider, [0, 2]))
        try:
            truncation_env.step(_actions(provider, [0, 0]))
        except RuntimeError:
            truncation_requires_reset = True
        else:
            truncation_requires_reset = False
    finally:
        truncation_env.close()

    kill_env = factory()
    try:
        kill_env.reset(seed=[13, 14])
        player_infos = kill_env.step(_actions(provider, [3, 0]))[4]
        kill_env.reset(options={"reset_mask": _mask(provider, [True, False])})
        enemy_infos = kill_env.step(_actions(provider, [4, 0]))[4]
    finally:
        kill_env.close()

    signal_shapes = {
        operation: {name: _descriptor(infos[name]) for name in _SIGNALS}
        for operation, infos in (("reset", reset_infos), ("step", step_infos))
    }
    reward_values = _values(rewards)
    if fixture_case == "reward_mismatch" and provider == "env-vizdoom-turbo":
        reward_values = [9.0, 9.0]
    if kill_signal_reader is None:
        player_kill_delta = int(_values(player_infos["player_killcount"])[0])
        enemy_player_kill_delta = int(_values(enemy_infos["player_killcount"])[0])
        enemy_compatibility_kill_delta = int(_values(enemy_infos["killcount"])[0])
    else:
        player_signals = kill_signal_reader(player_infos, lane=0)
        enemy_signals = kill_signal_reader(enemy_infos, lane=0)
        player_kill_delta = int(player_signals["player_killcount"])
        enemy_player_kill_delta = int(enemy_signals["player_killcount"])
        enemy_compatibility_kill_delta = int(enemy_signals["compatibility_killcount"])
    behaviors = {
        "constructor": {
            "accepted": True,
            "parameters": [parameter.name for parameter in common],
            "defaults": [_json_default(parameter.default) for parameter in common],
            "kinds": [parameter.kind.name for parameter in common],
        },
        "action_meanings": list(env.action_meanings),
        "observation_shapes": {
            "reset": _descriptor(reset_observation),
            "step": _descriptor(step_observation),
        },
        "signal_shapes": signal_shapes,
        "rewards": {**_descriptor(rewards), "sample": reward_values},
        "reset": {"returns_observation_and_signals": len((reset_observation, reset_infos)) == 2},
        "step": {"returns_five_tuple": len(step_result) == 5},
        "masked_reset": {"supported": True, "selected_lane_only": masked_selected_only},
        "termination": {
            "reported_separately": _values(terminal[2]) == [1.0, 0.0]
            and _values(terminal[3]) == [0.0, 0.0],
            "requires_reset": termination_requires_reset,
        },
        "truncation": {
            "reported_separately": _values(truncation[2]) == [0.0, 0.0]
            and _values(truncation[3]) == [0.0, 1.0],
            "requires_reset": truncation_requires_reset,
        },
        "episode": {"step_before_reset_rejected": step_before_reset_rejected, "autoreset": False},
        "player_killcount": {
            "present": "player_killcount" in player_infos,
            "player_kill_delta": player_kill_delta,
        },
        "player_killcount.enemy_on_enemy_exclusion": {
            "enemy_on_enemy_delta": enemy_player_kill_delta,
            "compatibility_kill_delta": enemy_compatibility_kill_delta,
        },
    }
    contract: dict[str, Any] = {
        "schema_version": 1,
        "provider": provider,
        "revision": revision,
        "behaviors": behaviors,
    }
    if provider == "gradoom":
        contract["tensor_device"] = {
            "declared_device": str(reset_observation.device),
            "reset_mask_input": _descriptor(_mask(provider, [True, False])),
            "step_action_input": _descriptor(_actions(provider, [0, 0])),
            "reset_outputs": {
                "observation": _descriptor(reset_observation),
                "signals": {name: _descriptor(reset_infos[name]) for name in _SIGNALS},
            },
            "step_outputs": {
                "observation": _descriptor(step_observation),
                "reward": _descriptor(rewards),
                "terminated": _descriptor(terminated),
                "truncated": _descriptor(truncated),
                "signals": {name: _descriptor(step_infos[name]) for name in _SIGNALS},
            },
        }
    return contract


def _fixture_contracts(case: str) -> list[dict[str, Any]]:
    if case not in {"pass", "reward_mismatch"}:
        raise ValueError(f"unsupported fixture_case {case!r}")
    contracts = []
    for provider, transport in (("gradoom", "torch"), ("env-vizdoom-turbo", "numpy")):
        contracts.append(
            _capture_contract(
                provider=provider,
                revision=f"fixture-{provider}-revision",
                env_class=_FixtureTurboEnv,
                factory=lambda transport=transport: _FixtureTurboEnv(
                    "VizdoomDeathmatch-v1",
                    num_envs=2,
                    fixture_transport=transport,
                ),
                fixture_case=case,
            )
        )
    return contracts


def _real_contracts(request: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    try:
        from gradoom import GraDoomVecEnv
        from gradoom.evidence.reference_provider import load_reference_provider

        provider = load_reference_provider()
    except (ImportError, RuntimeError) as error:
        return [], [{"code": "provider_unavailable", "message": str(error)}]
    configuration = request.get("real_configuration")
    if not isinstance(configuration, dict):
        return [], [
            {
                "code": "real_probe_configuration_unavailable",
                "message": (
                    "Real invariant execution requires matched provider assets and configuration."
                ),
            }
        ]
    required_paths = ("iwad_path", "pwad_path", "reference_scenario_config_path")
    if any(not Path(str(configuration.get(name, ""))).is_file() for name in required_paths):
        return [], [
            {
                "code": "real_probe_assets_unavailable",
                "message": "Real invariant execution assets are unavailable.",
            }
        ]
    common = {
        "use_restricted_actions": DEATHMATCH_ACTIONS,
        "num_envs": 2,
        "obs_resize": (84, 84),
        "obs_crop": (0, 32, 0, 0),
        "obs_crop_mode": "mask",
        "obs_crop_fill": 0,
        "obs_grayscale": True,
        "obs_layout": "chw",
        "obs_resize_algorithm": "area",
        "frame_skip": 2,
        "frame_stack": 4,
        "reward_clip": False,
        "info": "data",
        "info_filter": {"mode": "all", "keys": list(_SIGNALS)},
        "game_variables": tuple(name.upper() for name in _SIGNALS),
        "treat_episode_timeout_as_truncation": True,
        "vizdoom_config": {"episode_timeout": 4200},
    }

    def gradoom_factory() -> Any:
        return GraDoomVecEnv(
            "VizdoomDeathmatch-v1",
            scenario=configuration["pwad_path"],
            rom_path=configuration["iwad_path"],
            device=configuration.get("device", "cpu"),
            **common,
        )

    def reference_factory() -> Any:
        return provider.make_env(
            configuration["reference_scenario_config_path"],
            rom_path=configuration["iwad_path"],
            num_threads=2,
            **common,
        )

    try:
        contracts = [
            _capture_contract(
                provider="gradoom",
                revision=request["gradoom_revision"],
                env_class=GraDoomVecEnv,
                factory=gradoom_factory,
                fixture_case="pass",
            ),
            _capture_contract(
                provider="env-vizdoom-turbo",
                revision=provider.revision,
                env_class=provider.env_class,
                factory=reference_factory,
                fixture_case="pass",
                kill_signal_reader=provider.episode_kill_signals,
            ),
        ]
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return [], [{"code": "real_probe_unavailable", "message": str(error)}]
    return contracts, []


def _runner_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict) or request.get("protocol_version") != 1:
            raise ValueError("runner request requires protocol_version 1")
        challenge = request.get("challenge")
        if not isinstance(challenge, str) or len(challenge) != 64:
            raise ValueError("runner request challenge is invalid")
        mode = request.get("mode")
        if mode == "fixture":
            contracts = _fixture_contracts(str(request.get("fixture_case", "pass")))
            unavailable: list[dict[str, str]] = []
        elif mode == "real":
            contracts, unavailable = _real_contracts(request)
        else:
            raise ValueError(f"unsupported runner mode {mode!r}")
        response = {
            "protocol_version": RUNNER_PROTOCOL_VERSION,
            "challenge": challenge,
            "runner_sha256": _runner_sha256(),
            "status": "unavailable" if unavailable else "complete",
            "contracts": contracts,
            "unavailable_reasons": unavailable,
        }
        print(json.dumps(response, allow_nan=False, sort_keys=True))
        return 0
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"invariant runner: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RUNNER_PROTOCOL_VERSION", "main"]
