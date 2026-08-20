"""Replay fixed ViZDoom policy action traces in env-Doom-turbo-torch.

The reference evaluator can record each episode's initial pose and restricted
action sequence.  This diagnostic applies those controls unchanged to
env-Doom-turbo-torch, stopping each lane at the earlier of its
env-Doom-turbo-torch termination or the reference episode horizon. Rendering
and policy inference are therefore
removed from the comparison, leaving post-spawn simulation behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from env_doom_turbo_torch.actions import DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS
from env_doom_turbo_torch.engine import TorchDeathmatchEngine
from env_doom_turbo_torch.scenario import compile_deathmatch_scenario

UINT32_MASK = (1 << 32) - 1
FIXED_UNIT = 1 << 16
BAM_TO_RADIANS = 2.0 * math.pi / float(1 << 32)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-jsonl", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--metrics-jsonl", type=Path)
    return parser


def _load_trace(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    lines = [line for line in resolved.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"trace JSONL is empty: {resolved}")
    document = json.loads(lines[-1])
    if not isinstance(document, dict) or document.get("status") != "completed":
        raise ValueError("trace JSONL does not end in a completed evaluation")
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("trace evaluation contains no episodes")
    for index, record in enumerate(episodes):
        if not isinstance(record, dict):
            raise TypeError(f"trace episode {index} is not an object")
        actions = record.get("actions")
        pose = record.get("initial_pose")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"trace episode {index} has no actions")
        if not isinstance(pose, Mapping):
            raise ValueError(f"trace episode {index} has no initial pose")
        if len(actions) != int(record.get("length", -1)):
            raise ValueError(f"trace episode {index} action count differs from length")
    return document


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


def _align_poses(engine: TorchDeathmatchEngine, episodes: Sequence[Mapping[str, Any]]) -> None:
    poses = [record["initial_pose"] for record in episodes]
    x = torch.tensor([pose["position_x"] for pose in poses], device=engine.device)
    y = torch.tensor([pose["position_y"] for pose in poses], device=engine.device)
    z = torch.tensor([pose["position_z"] for pose in poses], device=engine.device)
    camera_z = torch.tensor(
        [pose["camera_position_z"] for pose in poses],
        device=engine.device,
    )
    angle_degrees = torch.tensor([pose["angle"] for pose in poses], device=engine.device)
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


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    lengths = [float(record["length"]) for record in records]
    total_length = sum(lengths)
    result: dict[str, float | int] = {
        "episodes": len(records),
        "kills_mean": statistics.fmean(float(record["kills"]) for record in records),
        "kills_median": statistics.median(float(record["kills"]) for record in records),
        "length_mean": statistics.fmean(lengths),
        "return_mean": statistics.fmean(float(record["return"]) for record in records),
        "terminated_rate": statistics.fmean(float(record["terminated"]) for record in records),
    }
    for name in ("hits_taken", "damage_taken"):
        total = sum(float(record[name]) for record in records)
        result[f"{name}_mean"] = total / len(records)
        result[f"{name}_per_1000_decisions"] = total / total_length * 1_000.0
    return result


def main() -> int:
    args = _parser().parse_args()
    trace = _load_trace(args.trace_jsonl)
    episodes: list[dict[str, Any]] = trace["episodes"]
    num_envs = len(episodes)
    device = torch.device("cuda")
    scenario = compile_deathmatch_scenario(
        args.scenario.expanduser().resolve(),
        args.iwad.expanduser().resolve(),
    )
    contract = trace.get("environment_contract", {})
    frame_skip = int(contract.get("frame_skip", 2))
    doom_skill = int(trace.get("doom_skill", 1))
    episode_timeout = int(
        trace.get("checkpoint_config", {}).get("effective_recipe", {}).get("episode_timeout", 4_200)
    )
    engine = TorchDeathmatchEngine(
        scenario,
        num_envs,
        device=device,
        frame_skip=frame_skip,
        episode_timeout=episode_timeout,
        doom_skill=doom_skill,
        debug_checks=False,
    )
    blank = torch.zeros((num_envs, 84, 84), device=device, dtype=torch.uint8)
    engine.render_frame = lambda active=None, blank=blank: blank
    all_lanes = torch.ones(num_envs, device=device, dtype=torch.bool)
    game_seeds = torch.tensor(
        [int(record["game_seed"]) for record in episodes],
        device=device,
        dtype=torch.int64,
    )
    engine.reset(all_lanes, game_seeds)
    _align_poses(engine, episodes)

    lengths = torch.tensor(
        [len(record["actions"]) for record in episodes],
        device=device,
        dtype=torch.int64,
    )
    maximum_length = int(lengths.max())
    padded_actions = torch.zeros(
        (maximum_length, num_envs),
        device=device,
        dtype=torch.int64,
    )
    for lane, record in enumerate(episodes):
        padded_actions[: len(record["actions"]), lane] = torch.tensor(
            record["actions"],
            device=device,
            dtype=torch.int64,
        )
    if bool(torch.any((padded_actions < 0) | (padded_actions >= len(DEATHMATCH_ACTIONS)))):
        raise ValueError("trace contains an out-of-range restricted action")

    action_rows = _action_rows(device)
    returns = torch.zeros(num_envs, device=device)
    recorded = torch.zeros(num_envs, device=device, dtype=torch.bool)
    result_kills = torch.zeros(num_envs, device=device)
    result_returns = torch.zeros(num_envs, device=device)
    result_lengths = torch.zeros(num_envs, device=device, dtype=torch.int64)
    result_terminated = torch.zeros(num_envs, device=device, dtype=torch.bool)
    result_hits_taken = torch.zeros(num_envs, device=device)
    result_damage_taken = torch.zeros(num_envs, device=device)

    for decision in range(maximum_length):
        within_horizon = decision < lengths
        actions = action_rows.index_select(0, padded_actions[decision])
        _frames, rewards, terminated, truncated = engine.step(actions)
        returns.add_(torch.where(within_horizon & ~recorded, rewards, 0.0))
        reached_horizon = decision + 1 >= lengths
        finished = ~recorded & (terminated | truncated | reached_horizon)
        result_kills.copy_(torch.where(finished, engine.killcount, result_kills))
        result_returns.copy_(torch.where(finished, returns, result_returns))
        result_lengths.copy_(
            torch.where(
                finished,
                torch.minimum(
                    lengths,
                    torch.full_like(lengths, decision + 1),
                ),
                result_lengths,
            )
        )
        result_terminated |= finished & terminated
        result_hits_taken.copy_(torch.where(finished, engine.player_hits_taken, result_hits_taken))
        result_damage_taken.copy_(
            torch.where(finished, engine.player_damage_taken, result_damage_taken)
        )
        recorded |= finished
        if bool(torch.all(recorded)):
            break
    if not bool(torch.all(recorded)):
        raise RuntimeError("not every action trace produced a result")

    env_doom_turbo_torch_records = [
        {
            "damage_taken": float(result_damage_taken[lane]),
            "hits_taken": float(result_hits_taken[lane]),
            "kills": float(result_kills[lane]),
            "length": int(result_lengths[lane]),
            "return": float(result_returns[lane]),
            "terminated": bool(result_terminated[lane]),
        }
        for lane in range(num_envs)
    ]
    reference_records = [
        {
            "damage_taken": float(record.get("damage_taken", 0.0)),
            "hits_taken": float(record.get("hits_taken", 0.0)),
            "kills": float(record["kills"]),
            "length": int(record["length"]),
            "return": float(record["return"]),
            "terminated": bool(record["terminated"]),
        }
        for record in episodes
    ]
    kill_deltas = [
        env_doom_turbo_torch_records[index]["kills"] - reference_records[index]["kills"]
        for index in range(num_envs)
    ]
    result = {
        "schema": "env_doom_turbo_torch.vizdoom-action-trace-replay.v1",
        "source_trace": str(args.trace_jsonl.expanduser().resolve()),
        "frame_skip": frame_skip,
        "doom_skill": doom_skill,
        "episode_timeout": episode_timeout,
        "initial_pose_alignment": True,
        "reference": _summary(reference_records),
        "env_doom_turbo_torch": _summary(env_doom_turbo_torch_records),
        "paired": {
            "kills_delta_mean": statistics.fmean(kill_deltas),
            "kills_delta_median": statistics.median(kill_deltas),
            "kills_at_least_reference_rate": statistics.fmean(
                float(delta >= 0) for delta in kill_deltas
            ),
            "env_doom_turbo_torch_survived_reference_horizon_rate": statistics.fmean(
                float(not record["terminated"]) for record in env_doom_turbo_torch_records
            ),
        },
    }
    line = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    print(line, flush=True)
    if args.metrics_jsonl is not None:
        destination = args.metrics_jsonl.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
