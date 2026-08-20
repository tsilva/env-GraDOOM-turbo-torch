"""Evaluate a standalone env-Doom-turbo-torch checkpoint unchanged in reference ViZDoom.

This optional transfer gate depends on ``env-vizdoom-turbo``. The root trainer does
not import it and remains independent of ViZDoom, GradLab, and Stable-Baselines3.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

UINT32_MASK = (1 << 32) - 1
DEFAULT_EPISODES = 100
DEFAULT_NUM_ENVS = 16
REFERENCE_KILLS_TARGET = 31.78
REFERENCE_RENDER_HUD = True
TRACE_GAME_VARIABLES = (
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "CAMERA_POSITION_Z",
    "ANGLE",
)
TRACE_INFO_NAMES = tuple(name.casefold() for name in TRACE_GAME_VARIABLES)
SURVIVAL_GAME_VARIABLES = ("HITS_TAKEN", "DAMAGE_TAKEN")
SURVIVAL_INFO_NAMES = tuple(name.casefold() for name in SURVIVAL_GAME_VARIABLES)
ENV_DOOM_TURBO_TORCH_ONLY_SIGNAL_NAMES = frozenset({"player_killcount"})


def _reference_signal_names(names: Sequence[str]) -> tuple[str, ...]:
    """Remove env-Doom-turbo-torch-only diagnostics from a ViZDoom provider contract."""

    return tuple(
        name for name in names if str(name).casefold() not in ENV_DOOM_TURBO_TORCH_ONLY_SIGNAL_NAMES
    )


def _load_standalone_train() -> ModuleType:
    path = Path(__file__).parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("env_doom_turbo_torch_standalone_train", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"cannot load standalone trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _provider_seed(run_seed: int, lane: int, episode_index: int) -> int:
    """Reproduce GradLab BatchRuntime's provider seed for one lane episode."""

    if episode_index == 0:
        return int(run_seed) + int(lane)
    sequence = np.random.SeedSequence([int(run_seed), int(lane), int(episode_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _game_seed(provider_seed: int) -> int:
    """Reproduce env-vizdoom-turbo's provider-seed to game-seed conversion."""

    generator = np.random.default_rng(int(provider_seed))
    return int(generator.integers(0, UINT32_MASK + 1, dtype=np.uint32))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an unchanged standalone checkpoint in reference ViZDoom.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iwad", type=Path, required=True)
    parser.add_argument("--scenario-config", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument(
        "--compile-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stochastic-actions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample policy actions (default) or select argmax actions for parity diagnostics.",
    )
    parser.add_argument(
        "--include-action-traces",
        action="store_true",
        help="Embed each completed episode's restricted-action indices in the result.",
    )
    parser.add_argument(
        "--include-survival-diagnostics",
        action="store_true",
        help=(
            "Embed cumulative post-armor incoming damage/hit counters and summarize "
            "their rate per 1000 decisions."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace, train: ModuleType) -> None:
    for name in ("episodes", "num_envs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if int(args.num_envs) > int(args.episodes):
        raise ValueError("num-envs cannot exceed episodes")
    if not 0 <= int(args.seed) <= UINT32_MASK:
        raise ValueError(f"seed must be in [0, {UINT32_MASK}]")
    args.checkpoint = train._checkpoint_destination(args.checkpoint)
    args.iwad = args.iwad.expanduser().resolve()
    args.scenario_config = args.scenario_config.expanduser().resolve()
    for label in ("checkpoint", "iwad", "scenario_config"):
        path = getattr(args, label)
        if not path.is_file():
            raise FileNotFoundError(f"{label.replace('_', '-')} does not exist: {path}")


def _info_values(
    infos: Mapping[str, Any],
    names: Sequence[str],
    num_envs: int,
) -> np.ndarray:
    values = np.stack([np.asarray(infos[name]) for name in names], axis=1)
    if values.shape != (num_envs, len(names)):
        raise RuntimeError(f"expected info values {(num_envs, len(names))}, got {values.shape}")
    return values.astype(np.float32, copy=False)


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kills = [float(record["kills"]) for record in records]
    if not kills:
        raise ValueError("zero-shot evaluation requires completed episodes")
    mean_kills = statistics.fmean(kills)
    summary = {
        "evaluation/episode/count": len(records),
        "evaluation/kills/mean": mean_kills,
        "evaluation/kills/median": statistics.median(kills),
        "evaluation/kills/std": statistics.pstdev(kills),
        "evaluation/kills/min": min(kills),
        "evaluation/kills/max": max(kills),
        "evaluation/episode/length/mean": statistics.fmean(
            float(record["length"]) for record in records
        ),
        "evaluation/target/kills/mean": REFERENCE_KILLS_TARGET,
        "evaluation/target/passed": mean_kills >= REFERENCE_KILLS_TARGET,
    }
    if all("damage_taken" in record and "hits_taken" in record for record in records):
        damage_taken = [float(record["damage_taken"]) for record in records]
        hits_taken = [float(record["hits_taken"]) for record in records]
        total_decisions = sum(float(record["length"]) for record in records)
        summary.update(
            {
                "evaluation/survival/damage_taken/mean": statistics.fmean(damage_taken),
                "evaluation/survival/damage_taken/per_1000_decisions": (
                    1000.0 * sum(damage_taken) / total_decisions
                ),
                "evaluation/survival/hits_taken/mean": statistics.fmean(hits_taken),
                "evaluation/survival/hits_taken/per_1000_decisions": (
                    1000.0 * sum(hits_taken) / total_decisions
                ),
                "evaluation/survival/truncation_rate": statistics.fmean(
                    float(bool(record["truncated"])) for record in records
                ),
            }
        )
    observed_names = (
        "observed_health_loss",
        "observed_health_gain",
        "observed_armor_loss",
        "observed_armor_gain",
    )
    if all(all(name in record for name in observed_names) for record in records):
        total_decisions = sum(float(record["length"]) for record in records)
        for name in observed_names:
            values = [float(record[name]) for record in records]
            metric = name.removeprefix("observed_")
            summary[f"evaluation/survival/{metric}/mean"] = statistics.fmean(values)
            summary[f"evaluation/survival/{metric}/per_1000_decisions"] = (
                1000.0 * sum(values) / total_decisions
            )
    if all("actions" in record for record in records):
        counts = np.zeros(17, dtype=np.int64)
        for record in records:
            counts += np.bincount(
                np.asarray(record["actions"], dtype=np.int64),
                minlength=len(counts),
            )
        total_actions = int(counts.sum())
        if total_actions <= 0:
            raise ValueError("zero-shot action traces must contain at least one action")
        for index, count in enumerate(counts.tolist()):
            summary[f"evaluation/actions/{index}/count"] = int(count)
            summary[f"evaluation/actions/{index}/fraction"] = float(count / total_actions)
    return summary


def _evaluate(args: argparse.Namespace, train: ModuleType) -> dict[str, Any]:
    try:
        from vizdoom_turbo import VizdoomTurboVecEnv
    except ImportError as exc:
        raise RuntimeError(
            "zero-shot evaluation requires env-vizdoom-turbo in the selected Python runtime"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for standalone checkpoint policy inference")

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda")
    num_envs = int(args.num_envs)
    quotas = train._episode_quotas(int(args.episodes), num_envs)
    scenario_wad = args.scenario_config.with_name("deathmatch.wad")
    if not scenario_wad.is_file():
        raise FileNotFoundError(f"scenario WAD does not exist beside config: {scenario_wad}")

    extra_game_variables = (TRACE_GAME_VARIABLES if args.include_action_traces else ()) + (
        SURVIVAL_GAME_VARIABLES if args.include_survival_diagnostics else ()
    )
    extra_info_names = (TRACE_INFO_NAMES if args.include_action_traces else ()) + (
        SURVIVAL_INFO_NAMES if args.include_survival_diagnostics else ()
    )
    game_variables = (
        *_reference_signal_names(train.GAME_VARIABLES),
        *extra_game_variables,
    )
    info_keys = (
        *_reference_signal_names(train.INFO_SIGNALS),
        *extra_info_names,
    )
    env = VizdoomTurboVecEnv(
        str(args.scenario_config),
        use_restricted_actions=train.RESTRICTED_ACTIONS,
        rom_path=str(args.iwad),
        num_envs=num_envs,
        num_threads=num_envs,
        obs_resize=(84, 84),
        obs_crop=(0, 32, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_grayscale=True,
        obs_layout="chw",
        obs_copy="safe_view",
        obs_resize_algorithm="area",
        frame_skip=train.REFERENCE_RECIPE.frame_skip,
        frame_stack=train.FRAME_STACK,
        maxpool_last_two=False,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        info="data",
        info_filter={"mode": "all", "keys": list(info_keys)},
        doom_skill=train.REFERENCE_RECIPE.doom_skill,
        game_variables=game_variables,
        treat_episode_timeout_as_truncation=True,
        vizdoom_config={
            "episode_timeout": train.REFERENCE_RECIPE.episode_timeout,
            "render_hud": REFERENCE_RENDER_HUD,
        },
    )
    started = time.perf_counter()
    try:
        if tuple(env.action_table or ()) != train.RESTRICTED_ACTIONS:
            raise RuntimeError("reference action table differs from checkpoint contract")
        loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if (
            not isinstance(loaded, Mapping)
            or loaded.get("format") != "standalone-env_doom_turbo_torch-ppo-v1"
        ):
            raise ValueError(f"unsupported standalone checkpoint: {args.checkpoint}")
        checkpoint_config = loaded.get("config", {})
        effective_recipe = (
            checkpoint_config.get("effective_recipe", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        observation_invariance = (
            checkpoint_config.get("observation_invariance", {})
            if isinstance(checkpoint_config, Mapping)
            else {}
        )
        observation_blur_kernel = int(
            effective_recipe.get(
                "observation_blur_kernel",
                observation_invariance.get("observation_blur_kernel", 1),
            )
        )
        policy = train.NatureActorCritic(observation_blur_kernel=observation_blur_kernel).to(device)
        policy.load_state_dict(loaded["policy_state_dict"])
        policy.eval()
        calls = train.PolicyCalls(policy, compile_policy=bool(args.compile_policy))
        precision = train.Precision("fp32", device)
        context_encoder = train.CombatContextEncoder(train.MODEL_HISTORY_SIGNALS, device)

        episode_indices = np.zeros(num_envs, dtype=np.int64)
        provider_seeds = [_provider_seed(int(args.seed), lane, 0) for lane in range(num_envs)]
        observations, infos = env.reset(seed=provider_seeds)
        initial_poses = (
            [
                {name: float(np.asarray(infos[name])[lane]) for name in TRACE_INFO_NAMES}
                for lane in range(num_envs)
            ]
            if args.include_action_traces
            else []
        )
        initial_values = _info_values(infos, train.MODEL_HISTORY_SIGNALS, num_envs)
        histories = (
            torch.as_tensor(initial_values, device=device)[..., None]
            .expand(
                -1,
                -1,
                train.FRAME_STACK,
            )
            .clone()
        )
        episode_returns = np.zeros(num_envs, dtype=np.float64)
        episode_lengths = np.zeros(num_envs, dtype=np.int64)
        episode_health_loss = np.zeros(num_envs, dtype=np.float64)
        episode_health_gain = np.zeros(num_envs, dtype=np.float64)
        episode_armor_loss = np.zeros(num_envs, dtype=np.float64)
        episode_armor_gain = np.zeros(num_envs, dtype=np.float64)
        previous_health = np.asarray(infos["health"], dtype=np.float64).copy()
        previous_armor = np.asarray(infos["armor"], dtype=np.float64).copy()
        action_traces: list[list[int]] = [[] for _ in range(num_envs)]
        records_by_grid: dict[tuple[int, int], dict[str, Any]] = {}
        watchdog = train.REFERENCE_RECIPE.episode_timeout // train.REFERENCE_RECIPE.frame_skip + 1
        maximum_decisions = watchdog * max(quotas)

        for decision in range(maximum_decisions):
            observation_device = torch.as_tensor(observations, device=device)
            context = context_encoder.encode(histories)
            with torch.no_grad(), precision.autocast():
                if args.stochastic_actions:
                    actions, _values, _log_probs = calls.act(observation_device, context)
                else:
                    actions = calls.deterministic_action(observation_device, context)
            action_values = actions.cpu().numpy()
            if args.include_action_traces:
                for lane in range(num_envs):
                    if int(episode_indices[lane]) < quotas[lane]:
                        action_traces[lane].append(int(action_values[lane]))
            next_observations, rewards, terminated, truncated, step_infos = env.step(action_values)
            done = np.asarray(terminated) | np.asarray(truncated)
            episode_returns += np.asarray(rewards, dtype=np.float64)
            episode_lengths += 1
            current_health = np.asarray(step_infos["health"], dtype=np.float64)
            current_armor = np.asarray(step_infos["armor"], dtype=np.float64)
            health_delta = current_health - previous_health
            armor_delta = current_armor - previous_armor
            episode_health_loss += np.maximum(-health_delta, 0.0)
            episode_health_gain += np.maximum(health_delta, 0.0)
            episode_armor_loss += np.maximum(-armor_delta, 0.0)
            episode_armor_gain += np.maximum(armor_delta, 0.0)
            step_values = _info_values(step_infos, train.MODEL_HISTORY_SIGNALS, num_envs)
            histories = torch.roll(histories, shifts=-1, dims=2)
            histories[:, :, -1].copy_(torch.as_tensor(step_values, device=device))

            for lane in np.flatnonzero(done).tolist():
                lane_episode = int(episode_indices[lane])
                if lane_episode < quotas[lane]:
                    provider_seed = _provider_seed(int(args.seed), lane, lane_episode)
                    record = {
                        "lane": lane,
                        "lane_episode": lane_episode,
                        "provider_seed": provider_seed,
                        "game_seed": _game_seed(provider_seed),
                        "kills": float(np.asarray(step_infos["killcount"])[lane]),
                        "return": float(episode_returns[lane]),
                        "length": int(episode_lengths[lane]),
                        "terminated": bool(np.asarray(terminated)[lane]),
                        "truncated": bool(np.asarray(truncated)[lane]),
                        "completion_decision": decision + 1,
                    }
                    if args.include_action_traces:
                        record["actions"] = action_traces[lane].copy()
                        record["initial_pose"] = initial_poses[lane].copy()
                    if args.include_survival_diagnostics:
                        record.update(
                            {
                                name: float(np.asarray(step_infos[name])[lane])
                                for name in SURVIVAL_INFO_NAMES
                            }
                        )
                        record.update(
                            {
                                "observed_health_loss": float(episode_health_loss[lane]),
                                "observed_health_gain": float(episode_health_gain[lane]),
                                "observed_armor_loss": float(episode_armor_loss[lane]),
                                "observed_armor_gain": float(episode_armor_gain[lane]),
                            }
                        )
                    records_by_grid[(lane, lane_episode)] = record
                episode_indices[lane] += 1
                action_traces[lane].clear()

            if len(records_by_grid) == int(args.episodes):
                observations = next_observations
                break
            if np.any(done):
                reset_seeds: list[int | None] = [None] * num_envs
                for lane in np.flatnonzero(done).tolist():
                    reset_seeds[lane] = _provider_seed(
                        int(args.seed),
                        lane,
                        int(episode_indices[lane]),
                    )
                reset_observations, reset_infos = env.reset(
                    seed=reset_seeds,
                    options={
                        "reset_mask": done.astype(np.bool_, copy=False),
                        "state_indices": np.zeros(num_envs, dtype=np.int32),
                    },
                )
                reset_values = _info_values(
                    reset_infos,
                    train.MODEL_HISTORY_SIGNALS,
                    num_envs,
                )
                reset_history = torch.as_tensor(reset_values, device=device)[..., None].expand(
                    -1,
                    -1,
                    train.FRAME_STACK,
                )
                done_device = torch.as_tensor(done, device=device)
                if args.include_action_traces:
                    for lane in np.flatnonzero(done).tolist():
                        initial_poses[lane] = {
                            name: float(np.asarray(reset_infos[name])[lane])
                            for name in TRACE_INFO_NAMES
                        }
                histories.copy_(torch.where(done_device[:, None, None], reset_history, histories))
                episode_returns[done] = 0.0
                episode_lengths[done] = 0
                episode_health_loss[done] = 0.0
                episode_health_gain[done] = 0.0
                episode_armor_loss[done] = 0.0
                episode_armor_gain[done] = 0.0
                previous_health = np.where(
                    done,
                    np.asarray(reset_infos["health"], dtype=np.float64),
                    current_health,
                )
                previous_armor = np.where(
                    done,
                    np.asarray(reset_infos["armor"], dtype=np.float64),
                    current_armor,
                )
                observations = reset_observations
            else:
                previous_health = current_health.copy()
                previous_armor = current_armor.copy()
                observations = next_observations
        else:
            missing = [
                (lane, episode)
                for lane, quota in enumerate(quotas)
                for episode in range(quota)
                if (lane, episode) not in records_by_grid
            ]
            raise RuntimeError(f"zero-shot evaluation watchdog expired: {missing}")

        expected_grid = [
            (lane, episode) for lane, quota in enumerate(quotas) for episode in range(quota)
        ]
        records = [
            {"index": index, **records_by_grid[key]} for index, key in enumerate(expected_grid)
        ]
        torch.cuda.synchronize(device)
        return {
            "type": "evaluation",
            "status": "completed",
            "protocol": "standalone-zero-shot-vizdoom-turbo-v2-fixed-seed-grid",
            "action_sampling": "stochastic" if args.stochastic_actions else "argmax",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": train._file_sha256(args.checkpoint),
            "checkpoint_step": int(loaded.get("step", 0)),
            "checkpoint_config": loaded.get("config"),
            "episodes": records,
            "episode_quotas": list(quotas),
            "evaluation_seed": int(args.seed),
            "evaluation_num_envs": num_envs,
            "evaluation_seconds": time.perf_counter() - started,
            "deterministic_actions": not bool(args.stochastic_actions),
            "survival_diagnostics": bool(args.include_survival_diagnostics),
            "iwad_sha256": train._file_sha256(args.iwad),
            "scenario_config_sha256": train._file_sha256(args.scenario_config),
            "scenario_sha256": train._file_sha256(scenario_wad),
            "doom_skill": train.REFERENCE_RECIPE.doom_skill,
            "environment_contract": {
                "frame_skip": train.REFERENCE_RECIPE.frame_skip,
                "frame_stack": train.FRAME_STACK,
                "obs_crop": [0, 32, 0, 0],
                "obs_crop_mode": "mask",
                "obs_resize": [84, 84],
                "obs_resize_algorithm": "area",
                "render_hud": REFERENCE_RENDER_HUD,
            },
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            **_summary(records),
        }
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    train = _load_standalone_train()
    args = _parser().parse_args(argv)
    _validate_args(args, train)
    result = _evaluate(args, train)
    line = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print(line, flush=True)
    if args.metrics_jsonl is not None:
        destination = args.metrics_jsonl.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
