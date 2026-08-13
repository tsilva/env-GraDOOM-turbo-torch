"""Compare ACS monster-spawn distributions with ViZDoom.

Short runs isolate the first spawn checks before policy feedback can amplify
the intentionally independent random streams.  Long runs expose population
pressure over a full combat trajectory.  ViZDoom object IDs provide cumulative
successful spawns by class; GraDOOM records the corresponding inactive-to-alive
slot transitions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradoom.actions import DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS
from gradoom.engine import TorchDeathmatchEngine
from gradoom.scenario import compile_deathmatch_scenario

UINT32_MASK = (1 << 32) - 1
FIXED_UNIT = 1 << 16
BAM_TO_RADIANS = 2.0 * math.pi / float(1 << 32)
MONSTER_NAMES = (
    "Zombieman",
    "ShotgunGuy",
    "MarineChainsawVzd",
    "ChaingunGuy",
    "Demon",
    "HellKnight",
)
POSE_VARIABLES = (
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "CAMERA_POSITION_Z",
    "ANGLE",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--decisions", type=int, default=125)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--program",
        choices=("noop", "forward-fire"),
        default="noop",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _game_seed(provider_seed: int) -> int:
    generator = np.random.default_rng(provider_seed)
    return int(generator.integers(0, UINT32_MASK + 1, dtype=np.uint32))


def _action_matrix(available: tuple[str, ...]) -> np.ndarray:
    if available != DEATHMATCH_BUTTONS:
        raise RuntimeError(f"reference buttons differ: {available!r}")
    indices = {name: index for index, name in enumerate(available)}
    rows = np.zeros((len(DEATHMATCH_ACTIONS), len(available)), dtype=np.float64)
    for action_index, labels in enumerate(DEATHMATCH_ACTIONS):
        for label in labels:
            rows[action_index, indices[label]] = 1.0
    return rows


def _run_vizdoom_episode(
    *,
    config: Path,
    iwad: Path,
    game_seed: int,
    decisions: int,
    frame_skip: int,
    action_index: int,
) -> dict[str, Any]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("spawn comparison requires vizdoom") from exc

    game = vzd.DoomGame()
    config_directory = tempfile.TemporaryDirectory(prefix="gradoom-vizdoom-spawns-")
    game.load_config(str(config))
    game.set_doom_config_path(str(Path(config_directory.name) / "engine.ini"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_audio_buffer_enabled(False)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_objects_info_enabled(True)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_doom_game_path(str(iwad))
    game.set_doom_skill(1)
    variables = tuple(getattr(vzd.GameVariable, name) for name in POSE_VARIABLES)
    for variable in variables:
        if variable not in game.get_available_game_variables():
            game.add_available_game_variable(variable)
    game.set_seed(game_seed)
    game.init()
    try:
        game.new_episode()
        actions = _action_matrix(tuple(value.name for value in game.get_available_buttons()))
        initial_pose = {
            name.casefold(): float(game.get_game_variable(variable))
            for name, variable in zip(POSE_VARIABLES, variables, strict=True)
        }
        seen: dict[str, set[int]] = {name: set() for name in MONSTER_NAMES}
        first_spawn_time: int | None = None
        executed = 0
        for decision in range(decisions + 1):
            state = game.get_state()
            if state is not None:
                for actor in state.objects:
                    if actor.name in seen:
                        old_count = len(seen[actor.name])
                        seen[actor.name].add(int(actor.id))
                        if first_spawn_time is None and len(seen[actor.name]) > old_count:
                            first_spawn_time = int(game.get_episode_time())
            if decision == decisions or game.is_episode_finished() or game.is_player_dead():
                break
            game.make_action(actions[action_index].tolist(), frame_skip)
            executed += 1
        return {
            "decisions": executed,
            "first_spawn_time": first_spawn_time,
            "game_seed": game_seed,
            "health": float(game.get_game_variable(vzd.GameVariable.HEALTH)),
            "initial_pose": initial_pose,
            "kills": float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)),
            "spawn_counts": {name: len(seen[name]) for name in MONSTER_NAMES},
        }
    finally:
        game.close()
        config_directory.cleanup()


def _align_poses(engine: TorchDeathmatchEngine, records: Sequence[Mapping[str, Any]]) -> None:
    poses = [record["initial_pose"] for record in records]
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


def _run_gradoom(
    *,
    scenario_path: Path,
    iwad: Path,
    reference: Sequence[Mapping[str, Any]],
    decisions: int,
    frame_skip: int,
    action_index: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    num_envs = len(reference)
    scenario = compile_deathmatch_scenario(scenario_path, iwad)
    engine = TorchDeathmatchEngine(
        scenario,
        num_envs,
        device=device,
        frame_skip=frame_skip,
        doom_skill=1,
        debug_checks=False,
    )
    blank = torch.zeros((num_envs, 84, 84), device=device, dtype=torch.uint8)
    engine.render_frame = lambda active=None, blank=blank: blank
    all_lanes = torch.ones(num_envs, device=device, dtype=torch.bool)
    game_seeds = torch.tensor(
        [record["game_seed"] for record in reference],
        device=device,
        dtype=torch.int64,
    )
    engine.reset(all_lanes, game_seeds)
    _align_poses(engine, reference)
    action_rows = torch.as_tensor(
        _action_matrix(DEATHMATCH_BUTTONS),
        device=device,
        dtype=torch.bool,
    )
    previous_alive = engine.enemy_alive.clone()
    spawn_counts = torch.zeros(
        (num_envs, len(MONSTER_NAMES)),
        device=device,
        dtype=torch.int32,
    )
    first_spawn_time = torch.full(
        (num_envs,),
        -1,
        device=device,
        dtype=torch.int32,
    )
    completed_decisions = torch.full(
        (num_envs,),
        decisions,
        device=device,
        dtype=torch.int32,
    )
    done = torch.zeros(num_envs, device=device, dtype=torch.bool)
    for decision in range(decisions):
        _frames, _rewards, terminated, truncated = engine.step(
            action_rows[action_index].expand(num_envs, -1)
        )
        spawned = engine.enemy_alive & ~previous_alive
        spawn_type = engine.enemy_type.clamp(0, len(MONSTER_NAMES) - 1)
        spawn_counts.scatter_add_(
            1,
            spawn_type,
            spawned.to(torch.int32),
        )
        new_lane_spawn = torch.any(spawned, dim=1) & (first_spawn_time < 0)
        first_spawn_time.copy_(torch.where(new_lane_spawn, engine.episode_time, first_spawn_time))
        previous_alive.copy_(engine.enemy_alive)
        newly_done = ~done & (terminated | truncated)
        completed_decisions.copy_(
            torch.where(
                newly_done,
                torch.full_like(completed_decisions, decision + 1),
                completed_decisions,
            )
        )
        done |= newly_done
    return [
        {
            "decisions": int(completed_decisions[lane]),
            "first_spawn_time": (
                None if int(first_spawn_time[lane]) < 0 else int(first_spawn_time[lane])
            ),
            "game_seed": int(game_seeds[lane]),
            "health": float(engine.health[lane]),
            "kills": float(engine.killcount[lane]),
            "spawn_counts": {
                name: int(spawn_counts[lane, enemy_type])
                for enemy_type, name in enumerate(MONSTER_NAMES)
            },
        }
        for lane in range(num_envs)
    ]


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = [sum(record["spawn_counts"].values()) for record in records]
    decisions = [int(record["decisions"]) for record in records]
    total_decisions = sum(decisions)
    spawn_times = [record["first_spawn_time"] for record in records]
    observed_times = [float(value) for value in spawn_times if value is not None]
    return {
        "episodes": len(records),
        "decisions_mean": statistics.fmean(decisions),
        "spawn_count_mean": statistics.fmean(totals),
        "spawn_count_per_1000_decisions": (
            1_000.0 * sum(totals) / total_decisions if total_decisions else None
        ),
        "spawn_count_by_class_mean": {
            name: statistics.fmean(float(record["spawn_counts"][name]) for record in records)
            for name in MONSTER_NAMES
        },
        "spawn_count_by_class_per_1000_decisions": {
            name: (
                1_000.0
                * sum(int(record["spawn_counts"][name]) for record in records)
                / total_decisions
                if total_decisions
                else None
            )
            for name in MONSTER_NAMES
        },
        "first_spawn_observed_rate": len(observed_times) / len(records),
        "first_spawn_time_mean_when_observed": (
            statistics.fmean(observed_times) if observed_times else None
        ),
        "kills_mean": statistics.fmean(float(record["kills"]) for record in records),
        "health_mean": statistics.fmean(float(record["health"]) for record in records),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.episodes <= 0 or args.decisions <= 0 or args.frame_skip <= 0 or args.workers <= 0:
        raise ValueError("episodes, decisions, frame-skip, and workers must be positive")
    config = args.config.expanduser().resolve()
    scenario = args.scenario.expanduser().resolve()
    iwad = args.iwad.expanduser().resolve()
    action_index = 0 if args.program == "noop" else 9
    game_seeds = [_game_seed(args.seed + lane) for lane in range(args.episodes)]
    with ThreadPoolExecutor(max_workers=min(args.workers, args.episodes)) as executor:
        reference = list(
            executor.map(
                lambda game_seed: _run_vizdoom_episode(
                    config=config,
                    iwad=iwad,
                    game_seed=game_seed,
                    decisions=args.decisions,
                    frame_skip=args.frame_skip,
                    action_index=action_index,
                ),
                game_seeds,
            )
        )
    gradoom = _run_gradoom(
        scenario_path=scenario,
        iwad=iwad,
        reference=reference,
        decisions=args.decisions,
        frame_skip=args.frame_skip,
        action_index=action_index,
    )
    result = {
        "schema": "gradoom.spawn-distribution-comparison.v1",
        "seed": args.seed,
        "decisions": args.decisions,
        "frame_skip": args.frame_skip,
        "program": args.program,
        "initial_pose_alignment": True,
        "vizdoom": _summary(reference),
        "gradoom": _summary(gradoom),
    }
    serialized = json.dumps(result, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
