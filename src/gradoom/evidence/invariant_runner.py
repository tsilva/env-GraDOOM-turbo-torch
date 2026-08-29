from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradoom.actions import DEATHMATCH_ACTION_MEANINGS, DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS

RUNNER_PROTOCOL_VERSION = 1
_SIGNALS = ("health", "killcount", "player_killcount", "episode_return")
_NATIVE_GAME_VARIABLES = ("health", "killcount", "player_killcount")


@dataclass(frozen=True)
class _AttributionProof:
    attacker: str
    target: str
    evidence_sha256: str


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
        fixture_missing_signal: str | None = None,
        fixture_masked_reset: str = "respect",
        fixture_terminal_reset: str = "respect",
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
        self.fixture_missing_signal = fixture_missing_signal
        self.fixture_masked_reset = fixture_masked_reset
        self.fixture_terminal_reset = fixture_terminal_reset
        self.device = torch.device("cpu")
        self.action_meanings = DEATHMATCH_ACTION_MEANINGS
        self._initialized = np.zeros(num_envs, dtype=np.bool_)
        self._pending_reset = np.zeros(num_envs, dtype=np.bool_)
        self._state = np.zeros(num_envs, dtype=np.uint8)
        self._killcount = np.zeros(num_envs, dtype=np.float64)
        self._player_killcount = np.zeros(num_envs, dtype=np.float64)
        self._episode_return = np.zeros(num_envs, dtype=np.float64)
        self._last_attributions: list[list[dict[str, str]]] = [[] for _ in range(num_envs)]

    def _array(self, value: np.ndarray) -> Any:
        return torch.from_numpy(value.copy()) if self.fixture_transport == "torch" else value.copy()

    def _observation(self) -> Any:
        value = np.broadcast_to(self._state[:, None, None, None], (self.num_envs, 4, 84, 84))
        return self._array(value)

    def _infos(self) -> dict[str, Any]:
        infos = {
            "health": self._array(np.full(self.num_envs, 100.0, dtype=np.float64)),
            "killcount": self._array(self._killcount),
            "player_killcount": self._array(self._player_killcount),
            "episode_return": self._array(self._episode_return),
        }
        if self.fixture_missing_signal is not None:
            infos.pop(self.fixture_missing_signal)
        return infos

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
        forge_reset = raw_mask is not None and (
            (self.fixture_masked_reset == "forge" and not bool(np.any(self._pending_reset & mask)))
            or (
                self.fixture_terminal_reset == "broken" and bool(np.any(self._pending_reset & mask))
            )
        )
        saved = None
        if forge_reset:
            saved = tuple(
                value.copy()
                for value in (
                    self._initialized,
                    self._pending_reset,
                    self._state,
                    self._killcount,
                    self._player_killcount,
                    self._episode_return,
                )
            )
        if (
            raw_mask is not None
            and self.fixture_masked_reset == "ignore"
            and not bool(np.any(self._pending_reset & mask))
        ):
            mask = np.zeros(self.num_envs, dtype=np.bool_)
        elif raw_mask is not None and self.fixture_masked_reset == "leak":
            mask = np.ones(self.num_envs, dtype=np.bool_)
        self._initialized[mask] = True
        self._pending_reset[mask] = False
        self._state[mask] = 0
        self._killcount[mask] = 0
        self._player_killcount[mask] = 0
        self._episode_return[mask] = 0
        result = self._observation(), self._infos()
        if saved is not None:
            (
                self._initialized,
                self._pending_reset,
                self._state,
                self._killcount,
                self._player_killcount,
                self._episode_return,
            ) = saved
        return result

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
        self._last_attributions = [
            (
                [{"attacker": "player", "target": "enemy"}]
                if bool(player_kills[lane])
                else [{"attacker": "enemy", "target": "enemy"}]
                if bool(enemy_kills[lane])
                else []
            )
            for lane in range(self.num_envs)
        ]
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


def _mask(provider: str, values: list[bool], device: torch.device | None = None) -> Any:
    return (
        torch.tensor(values, dtype=torch.bool, device=device)
        if provider == "gradoom"
        else np.asarray(values)
    )


def _actions(provider: str, values: list[int], device: torch.device | None = None) -> Any:
    return (
        torch.tensor(values, dtype=torch.int64, device=device)
        if provider == "gradoom"
        else np.asarray(values)
    )


def _probe_error(error: BaseException | str) -> dict[str, str]:
    return {"probe_error": str(error)}


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return np.array(value, copy=True)


def _snapshot_signals(infos: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in _SIGNALS if name not in infos]
    if missing:
        raise RuntimeError(f"reset or step signals are missing {missing}")
    return {name: _snapshot(infos[name]) for name in _SIGNALS}


def _lane_equal(left: Any, right: Any, lane: int) -> bool:
    if isinstance(left, torch.Tensor):
        left = left.detach().cpu().numpy()
    if isinstance(right, torch.Tensor):
        right = right.detach().cpu().numpy()
    return bool(np.array_equal(np.asarray(left)[lane], np.asarray(right)[lane]))


def _lane_signals_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    lane: int,
) -> bool:
    return all(_lane_equal(left[name], right[name], lane) for name in _SIGNALS)


def _kill_signals(
    infos: Mapping[str, Any],
    *,
    kill_signal_reader: Callable[..., dict[str, float]] | None,
    lane: int,
) -> tuple[float, float]:
    if kill_signal_reader is not None:
        signals = kill_signal_reader(infos, lane=lane)
        return signals["player_killcount"], signals["compatibility_killcount"]
    return (
        _values(infos["player_killcount"])[lane],
        _values(infos["killcount"])[lane],
    )


def _fixture_attribution_oracle(
    env: Any,
    *,
    lane: int,
    behavior: str,
    **_evidence: Any,
) -> _AttributionProof:
    events = env._last_attributions[lane]
    if len(events) != 1:
        raise RuntimeError("staged fixture did not observe exactly one attribution event")
    event = dict(events[0])
    evidence_sha256 = hashlib.sha256(
        json.dumps(
            {"behavior": behavior, "event": event, "lane": lane},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    return _AttributionProof(
        attacker=event["attacker"],
        target=event["target"],
        evidence_sha256=evidence_sha256,
    )


def _observation_lane_sha256(value: Any, *, lane: int) -> str:
    if isinstance(value, torch.Tensor):
        value = value.detach().to("cpu").contiguous().numpy()
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim < 2 or not 0 <= lane < array.shape[0]:
        raise RuntimeError("staged attribution requires public uint8 vector observations")
    selected = np.ascontiguousarray(array[lane])
    descriptor = json.dumps(
        {"dtype": str(selected.dtype), "shape": list(selected.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(descriptor + b"\0" + selected.tobytes()).hexdigest()


def _real_attribution_oracle(
    provider: str,
    binding: Mapping[str, Any],
) -> Callable[..., _AttributionProof]:
    if provider not in {"gradoom", "env-vizdoom-turbo"}:
        raise ValueError(f"unsupported attribution provider {provider!r}")
    expected_assets = {
        asset: (Path(str(binding[f"{asset}_path"])), str(binding[f"{asset}_sha256"]))
        for asset in ("iwad", "pwad")
    }

    def observe(
        env: Any,
        *,
        lane: int,
        behavior: str,
        initial_observation: Any,
        event_observation: Any,
        action_history: list[list[int]],
        event_step: int,
    ) -> _AttributionProof:
        for asset, (path, expected_sha256) in expected_assets.items():
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256
            ):
                raise RuntimeError(
                    f"staged attribution {asset.upper()} binding changed during execution"
                )
        if provider == "gradoom" and (
            getattr(env, "iwad_sha256", None) != expected_assets["iwad"][1]
            or getattr(env, "scenario_sha256", None) != expected_assets["pwad"][1]
        ):
            raise RuntimeError(
                "staged attribution environment does not expose the bound GraDOOM assets"
            )
        if event_step < 0 or len(action_history) != event_step + 1:
            raise RuntimeError("staged attribution action history is incomplete")
        try:
            lane_actions = [row[lane] for row in action_history]
            action_buttons = [DEATHMATCH_ACTIONS[index] for index in lane_actions]
        except (IndexError, TypeError) as error:
            raise RuntimeError("staged attribution action history is invalid") from error
        if behavior == "player_killcount":
            attacker = "player"
            if "ATTACK" not in action_buttons[-1]:
                raise RuntimeError(
                    "staged player attribution did not observe a pinned player attack action"
                )
        elif behavior == "player_killcount.enemy_on_enemy_exclusion":
            attacker = "enemy"
            if any("ATTACK" in buttons for buttons in action_buttons):
                raise RuntimeError(
                    "staged enemy attribution observed a player attack before the event"
                )
        else:
            raise RuntimeError(f"unsupported staged attribution behavior {behavior!r}")
        initial_sha256 = _observation_lane_sha256(initial_observation, lane=lane)
        event_sha256 = _observation_lane_sha256(event_observation, lane=lane)
        if initial_sha256 == event_sha256:
            raise RuntimeError(
                "staged attribution requires an observed public state transition; "
                "counters alone are insufficient"
            )
        evidence = {
            "action_history": lane_actions,
            "attacker": attacker,
            "behavior": behavior,
            "event_observation_sha256": event_sha256,
            "event_step": event_step,
            "initial_observation_sha256": initial_sha256,
            "iwad_sha256": expected_assets["iwad"][1],
            "lane": lane,
            "provider": provider,
            "pwad_sha256": expected_assets["pwad"][1],
            "target": "enemy",
        }
        evidence_sha256 = hashlib.sha256(
            json.dumps(evidence, separators=(",", ":"), sort_keys=True).encode("ascii")
        ).hexdigest()
        return _AttributionProof(
            attacker=attacker,
            target="enemy",
            evidence_sha256=evidence_sha256,
        )

    return observe


def _verified_attribution(
    oracle: Callable[..., _AttributionProof] | None,
    env: Any,
    *,
    lane: int,
    behavior: str,
    initial_observation: Any,
    event_observation: Any,
    action_history: list[list[int]],
    event_step: int,
) -> dict[str, str]:
    if oracle is None:
        raise RuntimeError(
            "kill semantics require an independent actor/target attribution oracle; "
            "compatibility counters alone are insufficient"
        )
    event = oracle(
        env,
        lane=lane,
        behavior=behavior,
        initial_observation=initial_observation,
        event_observation=event_observation,
        action_history=action_history,
        event_step=event_step,
    )
    if (
        not isinstance(event, _AttributionProof)
        or re.fullmatch(r"[0-9a-f]{64}", event.evidence_sha256) is None
    ):
        raise RuntimeError("actor/target attribution oracle returned an invalid proof")
    expected_attacker = "player" if behavior == "player_killcount" else "enemy"
    if event.attacker != expected_attacker or event.target != "enemy":
        raise RuntimeError(
            f"actor/target attribution oracle did not observe {expected_attacker}-to-enemy"
        )
    return {"attacker": event.attacker, "target": event.target}


def _semantic_probe(
    *,
    behavior: str,
    provider: str,
    factory: Callable[[], Any],
    probe: Mapping[str, Any],
    requested_device: torch.device | None,
    kill_signal_reader: Callable[..., dict[str, float]] | None,
    attribution_oracle: Callable[..., _AttributionProof] | None = None,
) -> dict[str, Any]:
    env = factory()
    try:
        initial_observation, initial_infos = env.reset(seed=probe["seeds"])
        initial_observation = _snapshot(initial_observation)
        initial_kills = [
            _kill_signals(initial_infos, kill_signal_reader=kill_signal_reader, lane=lane)
            for lane in range(2)
        ]
        observed: (
            tuple[
                int,
                int,
                tuple[Any, Any, Any, Any, Mapping[str, Any]],
            ]
            | None
        ) = None
        actions = probe["actions"]
        for step_index in range(probe["max_steps"]):
            transition = env.step(
                _actions(provider, actions[step_index % len(actions)], requested_device)
            )
            if not isinstance(transition, tuple) or len(transition) != 5:
                raise RuntimeError("step did not return the public five-tuple")
            terminated_values = _values(transition[2])
            truncated_values = _values(transition[3])
            infos = transition[4]
            if not isinstance(infos, Mapping):
                raise RuntimeError("step signals are not a mapping")
            for lane in range(2):
                if behavior == "termination":
                    matched = bool(terminated_values[lane]) and not bool(truncated_values[lane])
                elif behavior == "truncation":
                    matched = bool(truncated_values[lane]) and not bool(terminated_values[lane])
                else:
                    player, compatibility = _kill_signals(
                        infos,
                        kill_signal_reader=kill_signal_reader,
                        lane=lane,
                    )
                    player_delta = player - initial_kills[lane][0]
                    compatibility_delta = compatibility - initial_kills[lane][1]
                    matched = (
                        player_delta > 0
                        if behavior == "player_killcount"
                        else compatibility_delta > 0 and player_delta == 0
                    )
                if matched:
                    observed = (lane, step_index, transition)
                    break
            if observed is not None:
                break
            if any(bool(value) for value in (*terminated_values, *truncated_values)):
                break
        if observed is None:
            raise RuntimeError(f"{behavior} event was not observed through public step results")
        lane, event_step, transition = observed
        if behavior in {"termination", "truncation"}:
            try:
                env.step(_actions(provider, actions[0], requested_device))
            except RuntimeError:
                requires_reset = True
            else:
                requires_reset = False
            terminal_mask = [
                bool(terminated_values[index]) or bool(truncated_values[index])
                for index in range(2)
            ]
            reset_seeds = [
                probe["seeds"][index] if selected else None
                for index, selected in enumerate(terminal_mask)
            ]
            try:
                reset_result = env.reset(
                    seed=reset_seeds,
                    options={"reset_mask": _mask(provider, terminal_mask, requested_device)},
                )
                if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                    raise RuntimeError("terminal reset did not return observation and signals")
                resumed = env.step(_actions(provider, actions[0], requested_device))
                if not isinstance(resumed, tuple) or len(resumed) != 5:
                    raise RuntimeError("resumed step did not return the public five-tuple")
            except Exception as error:
                raise RuntimeError(
                    f"terminal lane reset did not resume stepping: {error}"
                ) from error
            return {
                "reported_separately": True,
                "requires_reset": requires_reset,
                "terminal_lane_reset": True,
                "stepping_resumed": True,
            }
        attribution = _verified_attribution(
            attribution_oracle,
            env,
            lane=lane,
            behavior=behavior,
            initial_observation=initial_observation,
            event_observation=_snapshot(transition[0]),
            action_history=[list(actions[index % len(actions)]) for index in range(event_step + 1)],
            event_step=event_step,
        )
        player, compatibility = _kill_signals(
            transition[4],
            kill_signal_reader=kill_signal_reader,
            lane=lane,
        )
        player_delta = player - initial_kills[lane][0]
        compatibility_delta = compatibility - initial_kills[lane][1]
        if behavior == "player_killcount":
            return {
                "present": True,
                "player_kill_delta": player_delta,
                "attribution": attribution,
            }
        return {
            "enemy_on_enemy_delta": player_delta,
            "compatibility_kill_delta": compatibility_delta,
            "attribution": attribution,
        }
    finally:
        env.close()


def _capture_contract(
    *,
    provider: str,
    revision: str,
    env_class: type[Any],
    factory: Callable[[], Any],
    fixture_case: str,
    semantic_probes: Mapping[str, Mapping[str, Any]],
    requested_device: torch.device | None = None,
    kill_signal_reader: Callable[..., dict[str, float]] | None = None,
    attribution_oracle: Callable[..., _AttributionProof] | None = None,
) -> dict[str, Any]:
    signature = inspect.signature(env_class).parameters
    common = list(signature.values())[:32]
    behaviors: dict[str, Any] = {
        "constructor": {
            "accepted": False,
            "parameters": [parameter.name for parameter in common],
            "defaults": [_json_default(parameter.default) for parameter in common],
            "kinds": [parameter.kind.name for parameter in common],
        },
        "action_meanings": _probe_error("provider construction did not complete"),
        "observation_shapes": _probe_error("reset and step did not complete"),
        "signal_shapes": _probe_error("reset and step did not complete"),
        "rewards": _probe_error("step did not complete"),
        "reset": _probe_error("reset did not complete"),
        "step": _probe_error("step did not complete"),
        "masked_reset": _probe_error("masked reset did not complete"),
        "termination": _probe_error("termination probe did not complete"),
        "truncation": _probe_error("truncation probe did not complete"),
        "episode": _probe_error("episode probe did not complete"),
        "player_killcount": _probe_error("player kill probe did not complete"),
        "player_killcount.enemy_on_enemy_exclusion": _probe_error(
            "enemy-on-enemy probe did not complete"
        ),
    }
    reset_observation = reset_infos = step_observation = rewards = None
    terminated = truncated = step_infos = None
    try:
        reset_mask = _mask(provider, [True, False], requested_device)
        step_actions = _actions(provider, [0, 0], requested_device)
    except Exception as error:
        reset_mask = step_actions = None
        behaviors["constructor"]["probe_error"] = str(error)
        for behavior in behaviors:
            if behavior != "constructor":
                behaviors[behavior] = _probe_error(error)
    try:
        try:
            env = factory()
            behaviors["constructor"]["accepted"] = True
            behaviors["action_meanings"] = list(env.action_meanings)
            try:
                env.step(step_actions)
            except RuntimeError:
                step_before_reset_rejected = True
            else:
                step_before_reset_rejected = False
            reset_result = env.reset(seed=[7, 8])
            if not isinstance(reset_result, tuple) or len(reset_result) != 2:
                raise RuntimeError("reset did not return observation and signals")
            reset_observation, reset_infos = reset_result
            if not isinstance(reset_infos, Mapping):
                raise RuntimeError("reset signals are not a mapping")
            reset_observation_snapshot = _snapshot(reset_observation)
            reset_signal_snapshots = _snapshot_signals(reset_infos)
            behaviors["reset"] = {"returns_observation_and_signals": True}
            step_result = env.step(step_actions)
            if not isinstance(step_result, tuple) or len(step_result) != 5:
                raise RuntimeError("step did not return the public five-tuple")
            step_observation, rewards, terminated, truncated, step_infos = step_result
            behaviors["step"] = {"returns_five_tuple": True}
            behaviors["observation_shapes"] = {
                "reset": _descriptor(reset_observation),
                "step": _descriptor(step_observation),
            }
            if not isinstance(step_infos, Mapping):
                raise RuntimeError("step signals are not a mapping")
            step_observation_snapshot = _snapshot(step_observation)
            step_signal_snapshots = _snapshot_signals(step_infos)
            control_env = factory()
            try:
                control_env.reset(seed=[7, 8])
                control_first = control_env.step(step_actions)
                control_second = control_env.step(step_actions)
                control_first_observation = _snapshot(control_first[0])
                control_first_signals = _snapshot_signals(control_first[4])
                control_second_observation = _snapshot(control_second[0])
                control_second_signals = _snapshot_signals(control_second[4])
            finally:
                control_env.close()
            behaviors["signal_shapes"] = {
                operation: {name: _descriptor(infos[name]) for name in _SIGNALS if name in infos}
                for operation, infos in (("reset", reset_infos), ("step", step_infos))
            }
            reward_values = _values(rewards)
            if fixture_case == "reward_mismatch" and provider == "env-vizdoom-turbo":
                reward_values = [9.0, 9.0]
            behaviors["rewards"] = {**_descriptor(rewards), "sample": reward_values}
            masked_result = env.reset(
                seed=[7, None],
                options={"reset_mask": reset_mask},
            )
            if not isinstance(masked_result, tuple) or len(masked_result) != 2:
                raise RuntimeError("masked reset did not return observation and signals")
            masked_observation, masked_infos = masked_result
            if not isinstance(masked_infos, Mapping):
                raise RuntimeError("masked reset signals are not a mapping")
            masked_observation_snapshot = _snapshot(masked_observation)
            masked_signal_snapshots = _snapshot_signals(masked_infos)
            selected_lane_was_advanced = not (
                _lane_equal(reset_observation_snapshot, step_observation_snapshot, 0)
                and _lane_signals_equal(reset_signal_snapshots, step_signal_snapshots, 0)
            )
            selected_lane_reset = (
                selected_lane_was_advanced
                and _lane_equal(reset_observation_snapshot, masked_observation_snapshot, 0)
                and _lane_signals_equal(reset_signal_snapshots, masked_signal_snapshots, 0)
            )
            unselected_lane_was_advanced = not (
                _lane_equal(reset_observation_snapshot, step_observation_snapshot, 1)
                and _lane_signals_equal(reset_signal_snapshots, step_signal_snapshots, 1)
            )
            unselected_lane_unchanged = (
                unselected_lane_was_advanced
                and _lane_equal(step_observation_snapshot, masked_observation_snapshot, 1)
                and _lane_signals_equal(step_signal_snapshots, masked_signal_snapshots, 1)
            )
            continued_result = env.step(step_actions)
            if not isinstance(continued_result, tuple) or len(continued_result) != 5:
                raise RuntimeError("post-masked-reset step did not return the public five-tuple")
            continued_observation = _snapshot(continued_result[0])
            if not isinstance(continued_result[4], Mapping):
                raise RuntimeError("post-masked-reset step signals are not a mapping")
            continued_signals = _snapshot_signals(continued_result[4])
            selected_lane_continues_from_reset = _lane_equal(
                control_first_observation, continued_observation, 0
            ) and _lane_signals_equal(control_first_signals, continued_signals, 0)
            unselected_lane_continues = _lane_equal(
                control_second_observation, continued_observation, 1
            ) and _lane_signals_equal(control_second_signals, continued_signals, 1)
            behaviors["masked_reset"] = {
                "supported": True,
                "selected_lane_state_and_signals_reset": selected_lane_reset,
                "unselected_lane_state_and_signals_unchanged": unselected_lane_unchanged,
                "selected_lane_continues_from_reset_state": selected_lane_continues_from_reset,
                "unselected_lane_continues_without_reset": unselected_lane_continues,
            }
            behaviors["episode"] = {
                "step_before_reset_rejected": step_before_reset_rejected,
                "autoreset": False,
            }
        except Exception as error:
            for behavior in (
                "observation_shapes",
                "signal_shapes",
                "rewards",
                "reset",
                "step",
                "masked_reset",
                "episode",
            ):
                if isinstance(behaviors[behavior], dict) and "probe_error" in behaviors[behavior]:
                    behaviors[behavior] = _probe_error(error)
        finally:
            if "env" in locals():
                env.close()
        for behavior in (
            "termination",
            "truncation",
            "player_killcount",
            "player_killcount.enemy_on_enemy_exclusion",
        ):
            try:
                behaviors[behavior] = _semantic_probe(
                    behavior=behavior,
                    provider=provider,
                    factory=factory,
                    probe=semantic_probes[behavior],
                    requested_device=requested_device,
                    kill_signal_reader=kill_signal_reader,
                    attribution_oracle=attribution_oracle,
                )
            except Exception as error:
                behaviors[behavior] = _probe_error(error)
    except Exception as error:
        behaviors["constructor"]["probe_error"] = str(error)
        for behavior in behaviors:
            if behavior != "constructor":
                behaviors[behavior] = _probe_error(error)

    contract: dict[str, Any] = {
        "schema_version": 1,
        "provider": provider,
        "revision": revision,
        "behaviors": behaviors,
    }
    if provider == "gradoom":
        declared_device = str(requested_device) if requested_device is not None else "cpu"
        contract["tensor_device"] = {
            "declared_device": declared_device,
            "reset_mask_input": _descriptor(reset_mask) if reset_mask is not None else {},
            "step_action_input": _descriptor(step_actions) if step_actions is not None else {},
            "reset_outputs": {
                "observation": _descriptor(reset_observation)
                if reset_observation is not None
                else {},
                "signals": {
                    name: _descriptor(reset_infos[name])
                    for name in _SIGNALS
                    if isinstance(reset_infos, Mapping) and name in reset_infos
                },
            },
            "step_outputs": {
                "observation": _descriptor(step_observation)
                if step_observation is not None
                else {},
                "reward": _descriptor(rewards) if rewards is not None else {},
                "terminated": _descriptor(terminated) if terminated is not None else {},
                "truncated": _descriptor(truncated) if truncated is not None else {},
                "signals": {
                    name: _descriptor(step_infos[name])
                    for name in _SIGNALS
                    if isinstance(step_infos, Mapping) and name in step_infos
                },
            },
        }
    return contract


def _fixture_contracts(case: str) -> list[dict[str, Any]]:
    if case not in {
        "pass",
        "reward_mismatch",
        "missing_player_killcount",
        "missing_termination",
        "ignored_masked_reset",
        "leaky_masked_reset",
        "forged_masked_reset",
        "broken_terminal_reset",
        "counter_only_kills",
    }:
        raise ValueError(f"unsupported fixture_case {case!r}")
    contracts = []
    semantic_probes = {
        "termination": {
            "seeds": [9, 10],
            "actions": [[0, 0]] if case == "missing_termination" else [[1, 0]],
            "max_steps": 1,
        },
        "truncation": {"seeds": [11, 12], "actions": [[0, 2]], "max_steps": 1},
        "player_killcount": {"seeds": [13, 14], "actions": [[3, 0]], "max_steps": 1},
        "player_killcount.enemy_on_enemy_exclusion": {
            "seeds": [15, 16],
            "actions": [[4, 0]],
            "max_steps": 1,
        },
    }
    for provider, transport in (("gradoom", "torch"), ("env-vizdoom-turbo", "numpy")):
        fixture_masked_reset = (
            "ignore"
            if case == "ignored_masked_reset" and provider == "env-vizdoom-turbo"
            else "leak"
            if case == "leaky_masked_reset" and provider == "env-vizdoom-turbo"
            else "forge"
            if case == "forged_masked_reset" and provider == "env-vizdoom-turbo"
            else "respect"
        )
        fixture_terminal_reset = (
            "broken"
            if case == "broken_terminal_reset" and provider == "env-vizdoom-turbo"
            else "respect"
        )

        def factory(
            transport: str = transport,
            fixture_masked_reset: str = fixture_masked_reset,
            fixture_terminal_reset: str = fixture_terminal_reset,
            fixture_missing_signal: str | None = (
                "player_killcount" if case == "missing_player_killcount" else None
            ),
        ) -> _FixtureTurboEnv:
            return _FixtureTurboEnv(
                "VizdoomDeathmatch-v1",
                num_envs=2,
                fixture_transport=transport,
                fixture_missing_signal=fixture_missing_signal,
                fixture_masked_reset=fixture_masked_reset,
                fixture_terminal_reset=fixture_terminal_reset,
            )

        contracts.append(
            _capture_contract(
                provider=provider,
                revision=f"fixture-{provider}-revision",
                env_class=_FixtureTurboEnv,
                factory=factory,
                fixture_case=case,
                semantic_probes=semantic_probes,
                requested_device=torch.device("cpu") if provider == "gradoom" else None,
                attribution_oracle=(
                    None
                    if case == "counter_only_kills" and provider == "env-vizdoom-turbo"
                    else _fixture_attribution_oracle
                ),
            )
        )
    return contracts


def _real_contracts(request: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    configuration = request.get("real_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("real invariant execution requires bound configuration")
    provider_bindings = configuration.get("providers")
    if not isinstance(provider_bindings, dict) or set(provider_bindings) != {
        "gradoom",
        "env-vizdoom-turbo",
    }:
        raise ValueError("real invariant execution requires both WAD provider bindings")
    for provider_id, binding in provider_bindings.items():
        if not isinstance(binding, dict):
            raise ValueError(f"{provider_id} WAD provider binding is invalid")
        for asset in ("iwad", "pwad"):
            path = Path(str(binding.get(f"{asset}_path", "")))
            expected = binding.get(f"{asset}_sha256")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{provider_id} validated {asset.upper()} binding changed")
    gradoom_binding = provider_bindings["gradoom"]
    reference_binding = provider_bindings["env-vizdoom-turbo"]
    if any(
        gradoom_binding[f"{asset}_sha256"] != reference_binding[f"{asset}_sha256"]
        for asset in ("iwad", "pwad")
    ):
        raise ValueError("real invariant providers are not bound to byte-identical WADs")
    reference_config = Path(str(configuration.get("reference_scenario_config_path", "")))
    reference_pwad = Path(reference_binding["pwad_path"])
    try:
        reference_consumes_bound_pwad = reference_config.with_name("deathmatch.wad").samefile(
            reference_pwad
        )
    except OSError:
        reference_consumes_bound_pwad = False
    if not reference_config.is_file() or not reference_consumes_bound_pwad:
        raise ValueError("reference scenario configuration does not consume the bound PWAD")
    try:
        from gradoom import GraDoomVecEnv
        from gradoom.evidence.reference_provider import (
            ReferenceProviderError,
            load_reference_provider,
        )
    except ImportError as error:
        return [], [{"code": "provider_unavailable", "message": str(error)}]
    try:
        provider = load_reference_provider()
    except ReferenceProviderError as error:
        message = str(error)
        if "is not installed" in message:
            return [], [{"code": "provider_unavailable", "message": message}]
        behavior = (
            "env-vizdoom-turbo.revision"
            if "revision" in message or "direct_url.json" in message
            else "env-vizdoom-turbo.provider_api"
        )
        return [], [
            {
                "code": "provider_contract_failure",
                "provider": "env-vizdoom-turbo",
                "behavior": behavior,
                "message": message,
            }
        ]
    profile_configuration = gradoom_binding.get("configuration")
    if profile_configuration != reference_binding.get("configuration") or not isinstance(
        profile_configuration, dict
    ):
        raise ValueError("real invariant provider configurations do not match")
    scenario_configuration = profile_configuration["scenario"]
    _validate_reference_scenario_config(reference_config, profile_configuration)
    observation = profile_configuration["observation"]
    crop = observation["crop_or_mask"]
    resize = observation["resize"]
    common = {
        "use_restricted_actions": DEATHMATCH_ACTIONS,
        "num_envs": 2,
        "obs_resize": tuple(resize["shape"]),
        "obs_crop": tuple(crop["edges"]),
        "obs_crop_mode": crop["kind"],
        "obs_crop_fill": crop["fill"],
        "obs_grayscale": observation["grayscale"]["enabled"],
        "obs_layout": observation["layout"],
        "obs_resize_algorithm": resize["algorithm"],
        "frame_skip": profile_configuration["frame_skip"],
        "frame_stack": observation["frame_stack"],
        "reward_clip": False,
        "info": "data",
        "info_filter": {"mode": "all", "keys": list(_SIGNALS)},
        "game_variables": tuple(name.upper() for name in _NATIVE_GAME_VARIABLES),
        "treat_episode_timeout_as_truncation": scenario_configuration[
            "episode_timeout_as_truncation"
        ],
        "doom_skill": profile_configuration["skill"],
    }
    requested_device = torch.device(configuration["device"])
    semantic_probes = configuration["semantic_probes"]

    def gradoom_factory() -> Any:
        return GraDoomVecEnv(
            scenario_configuration["game"],
            scenario=gradoom_binding["pwad_path"],
            rom_path=gradoom_binding["iwad_path"],
            device=requested_device,
            vizdoom_config={
                "episode_timeout": profile_configuration["episode_horizon_tics"],
                "render_screen_flashes": scenario_configuration["render_screen_flashes"],
            },
            **common,
        )

    def reference_factory() -> Any:
        return provider.make_env(
            str(reference_config),
            rom_path=reference_binding["iwad_path"],
            num_threads=2,
            doom_map=profile_configuration["map"],
            vizdoom_config={
                "episode_timeout": profile_configuration["episode_horizon_tics"],
                "render_hud": scenario_configuration["render_hud"],
                "render_screen_flashes": scenario_configuration["render_screen_flashes"],
            },
            **common,
        )

    try:
        gradoom_revision = _installed_gradoom_revision()
    except RuntimeError as error:
        return [], [
            {
                "code": "provider_contract_failure",
                "provider": "gradoom",
                "behavior": "gradoom.revision",
                "message": str(error),
            }
        ]
    contracts = [
        _capture_contract(
            provider="gradoom",
            revision=gradoom_revision,
            env_class=GraDoomVecEnv,
            factory=gradoom_factory,
            fixture_case="pass",
            semantic_probes=semantic_probes,
            requested_device=requested_device,
            attribution_oracle=_real_attribution_oracle("gradoom", gradoom_binding),
        ),
        _capture_contract(
            provider="env-vizdoom-turbo",
            revision=provider.revision,
            env_class=provider.env_class,
            factory=reference_factory,
            fixture_case="pass",
            semantic_probes=semantic_probes,
            kill_signal_reader=provider.episode_kill_signals,
            attribution_oracle=_real_attribution_oracle("env-vizdoom-turbo", reference_binding),
        ),
    ]
    return contracts, []


def _installed_gradoom_revision() -> str:
    repository = Path(__file__).resolve().parents[3]
    relative_source = Path(__file__).resolve().relative_to(repository)
    try:
        top_level = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", str(repository), "ls-files", "--error-unmatch", str(relative_source)],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        source_status = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                "src/gradoom",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot prove the executed GraDOOM checkout revision") from error
    if (
        Path(top_level).resolve() != repository
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or source_status
    ):
        raise RuntimeError("cannot prove the executed GraDOOM checkout revision")
    return revision


def _validate_reference_scenario_config(
    path: Path,
    profile_configuration: Mapping[str, Any],
) -> None:
    text = re.sub(r"(?m)#.*$", "", path.read_text(encoding="utf-8"))
    assignment_pattern = re.compile(
        r"\s*([A-Za-z_]+)\s*=\s*(?:\{([^}]*)\}|([^\r\n{]+))",
    )
    position = 0
    assignments: dict[str, str | tuple[str, ...]] = {}
    while position < len(text):
        match = assignment_pattern.match(text, position)
        if match is None:
            if not text[position:].strip():
                break
            raise ValueError(
                "reference scenario config is not an exact validated scenario configuration"
            )
        key = match.group(1).replace("_", "").casefold()
        if key in assignments:
            raise ValueError(
                "reference scenario config is not an exact validated scenario configuration"
            )
        block, scalar = match.group(2), match.group(3)
        assignments[key] = (
            tuple(token.casefold() for token in block.split())
            if block is not None
            else scalar.strip().casefold()
        )
        position = match.end()
    scenario = profile_configuration["scenario"]
    expected: dict[str, str | tuple[str, ...]] = {
        "doomscenariopath": "deathmatch.wad",
        "doomskill": str(profile_configuration["skill"]),
        "screenresolution": "res_" + "x".join(map(str, scenario["screen_resolution"])),
        "renderhud": str(scenario["render_hud"]).casefold(),
        "renderscreenflashes": str(scenario["render_screen_flashes"]).casefold(),
        "episodestarttime": str(scenario["episode_start_time"]),
        "episodetimeout": str(profile_configuration["episode_horizon_tics"]),
        "availablebuttons": tuple(name.casefold() for name in DEATHMATCH_BUTTONS),
        "availablegamevariables": tuple(name.casefold() for name in _NATIVE_GAME_VARIABLES),
        "mode": str(scenario["mode"]).casefold(),
    }
    if assignments != expected:
        raise ValueError(
            "reference scenario config is not an exact validated scenario configuration"
        )


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
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"invariant runner: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RUNNER_PROTOCOL_VERSION", "main"]
