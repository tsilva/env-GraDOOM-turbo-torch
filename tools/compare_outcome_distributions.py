"""Compare post-spawn outcome distributions under fixed scripted actions.

ViZDoom and env-Doom-turbo-torch intentionally use different stochastic implementations.
This diagnostic therefore aligns only ViZDoom's randomized initial player pose,
then compares aggregate outcomes over independent game streams rather than
requiring trajectory identity.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from env_doom_turbo_torch.actions import DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS
from env_doom_turbo_torch.engine import TorchDeathmatchEngine
from env_doom_turbo_torch.scenario import compile_deathmatch_scenario

UINT32_MASK = (1 << 32) - 1
FIXED_UNIT = 1 << 16
BAM_TO_RADIANS = 2.0 * math.pi / float(1 << 32)
POSE_NAMES = ("position_x", "position_y", "position_z", "camera_position_z", "angle")
GAME_VARIABLES = (
    "KILLCOUNT",
    "HITCOUNT",
    "DAMAGECOUNT",
    "HITS_TAKEN",
    "DAMAGE_TAKEN",
    "HEALTH",
    "ARMOR",
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "CAMERA_POSITION_Z",
    "ANGLE",
)


def _game_seed(provider_seed: int) -> int:
    generator = np.random.default_rng(provider_seed)
    return int(generator.integers(0, UINT32_MASK + 1, dtype=np.uint32))


def _action_index(program: str, decision: int) -> int:
    if program == "noop":
        return 0
    if program == "forward":
        return 2
    if program == "forward-fire":
        return 9
    if program == "spiral":
        return 13 if (decision // 20) % 2 == 0 else 14
    raise ValueError(f"unsupported program: {program}")


def _action_rows(device: torch.device) -> torch.Tensor:
    button_indices = {name: index for index, name in enumerate(DEATHMATCH_BUTTONS)}
    rows = torch.zeros(
        (len(DEATHMATCH_ACTIONS), len(DEATHMATCH_BUTTONS)),
        device=device,
        dtype=torch.bool,
    )
    for action_index, labels in enumerate(DEATHMATCH_ACTIONS):
        for label in labels:
            rows[action_index, button_indices[label]] = True
    return rows


def _align_poses(engine: TorchDeathmatchEngine, infos: dict[str, Any]) -> None:
    x = torch.as_tensor(infos["position_x"], device=engine.device)
    y = torch.as_tensor(infos["position_y"], device=engine.device)
    z = torch.as_tensor(infos["position_z"], device=engine.device)
    camera_z = torch.as_tensor(infos["camera_position_z"], device=engine.device)
    angle_degrees = torch.as_tensor(infos["angle"], device=engine.device)
    x_fixed = torch.round(x * FIXED_UNIT).to(torch.int64)
    y_fixed = torch.round(y * FIXED_UNIT).to(torch.int64)
    angle_bam = torch.bitwise_and(
        torch.round(angle_degrees / 360.0 * (1 << 32)).to(torch.int64),
        UINT32_MASK,
    )
    engine._x_fixed.copy_(x_fixed)
    engine._y_fixed.copy_(y_fixed)
    engine.x.copy_(x_fixed.to(torch.float32) / FIXED_UNIT)
    engine.y.copy_(y_fixed.to(torch.float32) / FIXED_UNIT)
    engine.z.copy_(z)
    engine.view_z.copy_(camera_z)
    engine.view_height.copy_(camera_z - z)
    engine._angle_bam.copy_(angle_bam)
    engine.angle.copy_(angle_bam.to(torch.float32) * BAM_TO_RADIANS)
    sector = engine._sector_at(engine.x, engine.y)
    engine.player_floor_z.copy_(engine.map.sector_heights[sector, 0])
    engine.previous_player_floor_z.copy_(engine.player_floor_z)
    engine.player_ceiling_z.copy_(engine.map.sector_heights[sector, 1])


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "episodes": len(records),
        "kills_mean": statistics.fmean(float(record["kills"]) for record in records),
        "kills_median": statistics.median(float(record["kills"]) for record in records),
        "length_mean": statistics.fmean(float(record["length"]) for record in records),
        "return_mean": statistics.fmean(float(record["return"]) for record in records),
        "terminated_rate": statistics.fmean(float(record["terminated"]) for record in records),
    }
    total_length = sum(float(record["length"]) for record in records)
    for name in (
        "hitcount",
        "damagecount",
        "hits_taken",
        "damage_taken",
        "health_gain",
        "health_loss",
        "armor_gain",
        "armor_loss",
    ):
        total = sum(float(record[name]) for record in records)
        summary[f"{name}_mean"] = total / len(records)
        summary[f"{name}_per_1000_decisions"] = total / total_length * 1_000.0
    return summary


def _normal_summary(values: Sequence[float]) -> dict[str, Any]:
    """Summarize independent outcome samples with an explicit uncertainty bound."""

    count = len(values)
    if count == 0:
        return {
            "count": 0,
            "mean": None,
            "normal_95_ci": None,
            "sample_stddev": None,
            "standard_error": None,
        }
    mean = statistics.fmean(values)
    if count == 1:
        sample_stddev = None
        standard_error = None
        interval = None
    else:
        sample_stddev = statistics.stdev(values)
        standard_error = sample_stddev / math.sqrt(count)
        interval = [mean - 1.96 * standard_error, mean + 1.96 * standard_error]
    return {
        "count": count,
        "mean": mean,
        "normal_95_ci": interval,
        "sample_stddev": sample_stddev,
        "standard_error": standard_error,
    }


def _outcome_values(records: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    values = {
        name: [float(record[name]) for record in records]
        for name in (
            "kills",
            "length",
            "return",
            "terminated",
            "hitcount",
            "damagecount",
            "hits_taken",
            "damage_taken",
            "health_gain",
            "health_loss",
            "armor_gain",
            "armor_loss",
        )
    }
    for name in (
        "hitcount",
        "damagecount",
        "hits_taken",
        "damage_taken",
        "health_gain",
        "health_loss",
        "armor_gain",
        "armor_loss",
    ):
        values[f"{name}_per_1000_decisions"] = [
            1_000.0 * float(record[name]) / float(record["length"]) for record in records
        ]
    return values


def _distribution_comparison(
    reference: Sequence[Mapping[str, Any]],
    env_doom_turbo_torch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare aggregate distributions without treating trajectories as paired."""

    reference_values = _outcome_values(reference)
    env_doom_turbo_torch_values = _outcome_values(env_doom_turbo_torch)
    comparison: dict[str, Any] = {}
    for name in reference_values:
        reference_summary = _normal_summary(reference_values[name])
        env_doom_turbo_torch_summary = _normal_summary(env_doom_turbo_torch_values[name])
        reference_mean = reference_summary["mean"]
        env_doom_turbo_torch_mean = env_doom_turbo_torch_summary["mean"]
        reference_error = reference_summary["standard_error"]
        env_doom_turbo_torch_error = env_doom_turbo_torch_summary["standard_error"]
        if reference_mean is None or env_doom_turbo_torch_mean is None:
            delta = None
            delta_error = None
            interval = None
        else:
            delta = env_doom_turbo_torch_mean - reference_mean
            if reference_error is None or env_doom_turbo_torch_error is None:
                delta_error = None
                interval = None
            else:
                delta_error = math.hypot(reference_error, env_doom_turbo_torch_error)
                interval = [delta - 1.96 * delta_error, delta + 1.96 * delta_error]
        comparison[name] = {
            "env_doom_turbo_torch": env_doom_turbo_torch_summary,
            "env_doom_turbo_torch_minus_vizdoom": delta,
            "env_doom_turbo_torch_minus_vizdoom_normal_95_ci": interval,
            "env_doom_turbo_torch_minus_vizdoom_standard_error": delta_error,
            "vizdoom": reference_summary,
        }
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--episode-timeout", type=int, default=4_200)
    parser.add_argument(
        "--programs",
        choices=("noop", "forward", "forward-fire", "spiral"),
        nargs="+",
        default=("noop", "forward-fire", "spiral"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be positive")

    try:
        from vizdoom_turbo import VizdoomTurboVecEnv
    except ImportError as exc:
        raise RuntimeError("comparison requires env-vizdoom-turbo") from exc

    device = torch.device("cuda")
    scenario = compile_deathmatch_scenario(args.scenario, args.iwad)
    provider_seeds = [args.seed + lane for lane in range(args.num_envs)]
    game_seeds = torch.tensor(
        [_game_seed(seed) for seed in provider_seeds],
        device=device,
        dtype=torch.int64,
    )
    action_rows = _action_rows(device)
    results: list[dict[str, Any]] = []

    for program in args.programs:
        env = VizdoomTurboVecEnv(
            str(args.config),
            use_restricted_actions=DEATHMATCH_ACTIONS,
            rom_path=str(args.iwad),
            num_envs=args.num_envs,
            num_threads=args.num_envs,
            obs_resize=(84, 84),
            obs_crop=(0, 32, 0, 0),
            obs_crop_mode="mask",
            obs_crop_fill=0,
            obs_grayscale=True,
            obs_layout="chw",
            obs_resize_algorithm="area",
            frame_skip=args.frame_skip,
            frame_stack=1,
            maxpool_last_two=False,
            noop_reset_max=0,
            sticky_action_prob=0.0,
            reward_clip=False,
            info="data",
            info_filter={
                "mode": "all",
                "keys": [name.casefold() for name in GAME_VARIABLES],
            },
            doom_skill=1,
            game_variables=GAME_VARIABLES,
            treat_episode_timeout_as_truncation=True,
            vizdoom_config={"episode_timeout": args.episode_timeout},
        )
        engine = TorchDeathmatchEngine(
            scenario,
            args.num_envs,
            device=device,
            frame_skip=args.frame_skip,
            episode_timeout=args.episode_timeout,
            doom_skill=1,
            debug_checks=False,
        )
        blank = torch.zeros((args.num_envs, 84, 84), device=device, dtype=torch.uint8)
        engine.render_frame = lambda active=None, blank=blank: blank
        mask = torch.ones(args.num_envs, device=device, dtype=torch.bool)
        engine.reset(mask, game_seeds)
        _observations, infos = env.reset(seed=provider_seeds)
        _align_poses(engine, infos)
        reference_done = np.zeros(args.num_envs, dtype=np.bool_)
        env_doom_turbo_torch_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
        reference_returns = np.zeros(args.num_envs, dtype=np.float64)
        env_doom_turbo_torch_returns = torch.zeros(args.num_envs, device=device)
        reference_health = np.asarray(infos["health"], dtype=np.float64).copy()
        reference_armor = np.asarray(infos["armor"], dtype=np.float64).copy()
        reference_health_gain = np.zeros(args.num_envs, dtype=np.float64)
        reference_health_loss = np.zeros(args.num_envs, dtype=np.float64)
        reference_armor_gain = np.zeros(args.num_envs, dtype=np.float64)
        reference_armor_loss = np.zeros(args.num_envs, dtype=np.float64)
        env_doom_turbo_torch_health = engine.health.clone()
        env_doom_turbo_torch_armor = engine.armor.clone()
        env_doom_turbo_torch_health_gain = torch.zeros(args.num_envs, device=device)
        env_doom_turbo_torch_health_loss = torch.zeros(args.num_envs, device=device)
        env_doom_turbo_torch_armor_gain = torch.zeros(args.num_envs, device=device)
        env_doom_turbo_torch_armor_loss = torch.zeros(args.num_envs, device=device)
        reference_records: list[dict[str, Any] | None] = [None] * args.num_envs
        env_doom_turbo_torch_records: list[dict[str, Any] | None] = [None] * args.num_envs
        maximum_decisions = math.ceil(args.episode_timeout / args.frame_skip)
        try:
            for decision in range(maximum_decisions):
                action_index = _action_index(program, decision)
                reference_actions = np.full(args.num_envs, action_index, dtype=np.int64)
                _observations, rewards, terminated, truncated, step_infos = env.step(
                    reference_actions
                )
                current_reference_done = np.asarray(terminated) | np.asarray(truncated)
                newly_reference_done = (~reference_done) & current_reference_done
                reference_active = ~reference_done
                reference_returns[reference_active] += np.asarray(rewards)[reference_active]
                next_reference_health = np.asarray(step_infos["health"], dtype=np.float64)
                next_reference_armor = np.asarray(step_infos["armor"], dtype=np.float64)
                health_delta = next_reference_health - reference_health
                armor_delta = next_reference_armor - reference_armor
                reference_health_gain[reference_active] += np.maximum(
                    health_delta[reference_active], 0.0
                )
                reference_health_loss[reference_active] += np.maximum(
                    -health_delta[reference_active], 0.0
                )
                reference_armor_gain[reference_active] += np.maximum(
                    armor_delta[reference_active], 0.0
                )
                reference_armor_loss[reference_active] += np.maximum(
                    -armor_delta[reference_active], 0.0
                )
                reference_health = next_reference_health.copy()
                reference_armor = next_reference_armor.copy()
                for lane in np.flatnonzero(newly_reference_done).tolist():
                    reference_records[lane] = {
                        "armor_gain": float(reference_armor_gain[lane]),
                        "armor_loss": float(reference_armor_loss[lane]),
                        "health_gain": float(reference_health_gain[lane]),
                        "health_loss": float(reference_health_loss[lane]),
                        "hitcount": float(np.asarray(step_infos["hitcount"])[lane]),
                        "damagecount": float(np.asarray(step_infos["damagecount"])[lane]),
                        "hits_taken": float(np.asarray(step_infos["hits_taken"])[lane]),
                        "damage_taken": float(np.asarray(step_infos["damage_taken"])[lane]),
                        "kills": float(np.asarray(step_infos["killcount"])[lane]),
                        "length": decision + 1,
                        "return": float(reference_returns[lane]),
                        "terminated": bool(np.asarray(terminated)[lane]),
                    }
                reference_done |= newly_reference_done
                if np.any(current_reference_done):
                    env.reset(
                        seed=[
                            provider_seeds[lane] if current_reference_done[lane] else None
                            for lane in range(args.num_envs)
                        ],
                        options={
                            "reset_mask": current_reference_done,
                            "state_indices": np.zeros(args.num_envs, dtype=np.int32),
                        },
                    )

                _frames, rewards_device, terminated_device, truncated_device = engine.step(
                    action_rows[action_index].expand(args.num_envs, -1)
                )
                env_doom_turbo_torch_active = ~env_doom_turbo_torch_done
                env_doom_turbo_torch_returns.add_(
                    torch.where(env_doom_turbo_torch_active, rewards_device, 0.0)
                )
                health_delta_device = engine.health - env_doom_turbo_torch_health
                armor_delta_device = engine.armor - env_doom_turbo_torch_armor
                env_doom_turbo_torch_health_gain.add_(
                    torch.where(env_doom_turbo_torch_active, health_delta_device.clamp_min(0), 0.0)
                )
                env_doom_turbo_torch_health_loss.add_(
                    torch.where(
                        env_doom_turbo_torch_active, (-health_delta_device).clamp_min(0), 0.0
                    )
                )
                env_doom_turbo_torch_armor_gain.add_(
                    torch.where(env_doom_turbo_torch_active, armor_delta_device.clamp_min(0), 0.0)
                )
                env_doom_turbo_torch_armor_loss.add_(
                    torch.where(
                        env_doom_turbo_torch_active, (-armor_delta_device).clamp_min(0), 0.0
                    )
                )
                env_doom_turbo_torch_health.copy_(engine.health)
                env_doom_turbo_torch_armor.copy_(engine.armor)
                current_env_doom_turbo_torch_done = terminated_device | truncated_device
                newly_env_doom_turbo_torch_done = (
                    ~env_doom_turbo_torch_done & current_env_doom_turbo_torch_done
                )
                for lane in torch.nonzero(newly_env_doom_turbo_torch_done).flatten().cpu().tolist():
                    env_doom_turbo_torch_records[lane] = {
                        "armor_gain": float(env_doom_turbo_torch_armor_gain[lane]),
                        "armor_loss": float(env_doom_turbo_torch_armor_loss[lane]),
                        "health_gain": float(env_doom_turbo_torch_health_gain[lane]),
                        "health_loss": float(env_doom_turbo_torch_health_loss[lane]),
                        "hitcount": float(engine.player_hitcount[lane]),
                        "damagecount": float(engine.player_damagecount[lane]),
                        "hits_taken": float(engine.player_hits_taken[lane]),
                        "damage_taken": float(engine.player_damage_taken[lane]),
                        "kills": float(engine.killcount[lane]),
                        "length": decision + 1,
                        "return": float(env_doom_turbo_torch_returns[lane]),
                        "terminated": bool(terminated_device[lane]),
                    }
                env_doom_turbo_torch_done |= newly_env_doom_turbo_torch_done
                if torch.any(current_env_doom_turbo_torch_done):
                    engine.reset(current_env_doom_turbo_torch_done, game_seeds)
                if np.all(reference_done) and bool(torch.all(env_doom_turbo_torch_done)):
                    break
        finally:
            env.close()
        if any(record is None for record in reference_records + env_doom_turbo_torch_records):
            raise RuntimeError(f"{program} did not complete every scripted episode")
        reference_complete = [record for record in reference_records if record is not None]
        env_doom_turbo_torch_complete = [
            record for record in env_doom_turbo_torch_records if record is not None
        ]
        results.append(
            {
                "comparison": _distribution_comparison(
                    reference_complete,
                    env_doom_turbo_torch_complete,
                ),
                "program": program,
                "vizdoom": _summary(reference_complete),
                "env_doom_turbo_torch": _summary(env_doom_turbo_torch_complete),
            }
        )

    result = {
        "doom_skill": 1,
        "episode_timeout": args.episode_timeout,
        "frame_skip": args.frame_skip,
        "initial_pose_alignment": True,
        "num_envs": args.num_envs,
        "results": results,
        "schema": "env_doom_turbo_torch.outcome-distributions.aligned-pose.v2",
        "seed": args.seed,
    }
    serialized = json.dumps(result, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
