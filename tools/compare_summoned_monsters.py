"""Compare isolated monster behavior from aligned ViZDoom summon states.

The deathmatch ACS loop does not begin until tic 106.  This diagnostic summons
one monster, advances ViZDoom by two tics so the console command materializes,
then initializes one GraDOOM actor from the observed player and monster pose.
Subsequent ACS spawns are disabled in GraDOOM and the comparison stops before
they can occur in ViZDoom.  The random streams are intentionally independent,
so the acceptance signal is the distribution of motion and damage outcomes.
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
VARIABLE_NAMES = (
    "HEALTH",
    "ARMOR",
    "HITS_TAKEN",
    "DAMAGE_TAKEN",
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
    parser.add_argument("--decisions", type=int, default=48)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--classes",
        choices=MONSTER_NAMES,
        nargs="+",
        default=MONSTER_NAMES,
    )
    parser.add_argument(
        "--programs",
        choices=("noop", "forward", "forward-fire", "spiral"),
        nargs="+",
        default=("noop",),
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


def _monster_object(state: Any, monster_name: str) -> Any | None:
    if state is None:
        return None
    matches = [actor for actor in state.objects if actor.name == monster_name]
    if len(matches) > 1:
        raise RuntimeError(f"expected one summoned {monster_name}, found {len(matches)}")
    return None if not matches else matches[0]


def _run_vizdoom_episode(
    *,
    config: Path,
    iwad: Path,
    game_seed: int,
    monster_name: str,
    program: str,
    decisions: int,
    frame_skip: int,
) -> dict[str, Any]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("summoned-monster comparison requires vizdoom") from exc

    game = vzd.DoomGame()
    config_directory = tempfile.TemporaryDirectory(prefix="gradoom-vizdoom-monster-")
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
    variables = tuple(getattr(vzd.GameVariable, name) for name in VARIABLE_NAMES)
    for variable in variables:
        if variable not in game.get_available_game_variables():
            game.add_available_game_variable(variable)
    game.set_seed(game_seed)
    game.init()
    try:
        game.new_episode()
        actions = _action_matrix(tuple(value.name for value in game.get_available_buttons()))
        noop = actions[0].tolist()
        game.send_game_command(f"summon {monster_name}")
        game.make_action(noop, frame_skip)
        state = game.get_state()
        monster = _monster_object(state, monster_name)
        if monster is None:
            raise RuntimeError(f"summoned {monster_name} did not materialize")
        values = {
            name.casefold(): float(game.get_game_variable(variable))
            for name, variable in zip(VARIABLE_NAMES, variables, strict=True)
        }
        initial = {
            "player": values,
            "monster": {
                "angle": float(monster.angle),
                "x": float(monster.position_x),
                "y": float(monster.position_y),
                "z": float(monster.position_z),
            },
        }
        initial_x = float(monster.position_x)
        initial_y = float(monster.position_y)
        initial_player_x = values["position_x"]
        initial_player_y = values["position_y"]
        initial_hits = values["hits_taken"]
        initial_damage = values["damage_taken"]
        first_damage_decision: int | None = None
        first_motion_decision: int | None = None
        last_x = initial_x
        last_y = initial_y
        last_player_x = initial_player_x
        last_player_y = initial_player_y
        minimum_distance = math.hypot(
            initial_x - initial_player_x,
            initial_y - initial_player_y,
        )
        executed = 0
        for decision in range(1, decisions + 1):
            if game.is_episode_finished() or game.is_player_dead():
                break
            game.make_action(
                actions[_action_index(program, decision - 1)].tolist(),
                frame_skip,
            )
            executed = decision
            hits = float(game.get_game_variable(vzd.GameVariable.HITS_TAKEN))
            if first_damage_decision is None and hits > initial_hits:
                first_damage_decision = decision
            last_player_x = float(game.get_game_variable(vzd.GameVariable.POSITION_X))
            last_player_y = float(game.get_game_variable(vzd.GameVariable.POSITION_Y))
            monster = _monster_object(game.get_state(), monster_name)
            if monster is not None:
                last_x = float(monster.position_x)
                last_y = float(monster.position_y)
                minimum_distance = min(
                    minimum_distance,
                    math.hypot(last_x - last_player_x, last_y - last_player_y),
                )
                displacement = math.hypot(last_x - initial_x, last_y - initial_y)
                if first_motion_decision is None and displacement > 0.5:
                    first_motion_decision = decision
        return {
            "damage_taken": float(game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN))
            - initial_damage,
            "decisions": executed,
            "died": bool(game.is_player_dead()),
            "final_displacement": math.hypot(last_x - initial_x, last_y - initial_y),
            "first_damage_decision": first_damage_decision,
            "first_motion_decision": first_motion_decision,
            "game_seed": game_seed,
            "health": float(game.get_game_variable(vzd.GameVariable.HEALTH)),
            "hits_taken": float(game.get_game_variable(vzd.GameVariable.HITS_TAKEN)) - initial_hits,
            "initial": initial,
            "minimum_monster_player_distance": minimum_distance,
            "player_displacement": math.hypot(
                last_player_x - initial_player_x,
                last_player_y - initial_player_y,
            ),
        }
    finally:
        game.close()
        config_directory.cleanup()


def _align_players(
    engine: TorchDeathmatchEngine,
    records: Sequence[Mapping[str, Any]],
) -> None:
    players = [record["initial"]["player"] for record in records]
    x = torch.tensor([player["position_x"] for player in players], device=engine.device)
    y = torch.tensor([player["position_y"] for player in players], device=engine.device)
    z = torch.tensor([player["position_z"] for player in players], device=engine.device)
    camera_z = torch.tensor(
        [player["camera_position_z"] for player in players], device=engine.device
    )
    angle_degrees = torch.tensor([player["angle"] for player in players], device=engine.device)
    x_fixed = torch.round(x * FIXED_UNIT).to(torch.int64)
    y_fixed = torch.round(y * FIXED_UNIT).to(torch.int64)
    angle_bam = torch.bitwise_and(
        torch.round(angle_degrees / 360.0 * (1 << 32)).to(torch.int64), UINT32_MASK
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
    engine.health.copy_(
        torch.tensor([player["health"] for player in players], device=engine.device)
    )
    engine.armor.copy_(torch.tensor([player["armor"] for player in players], device=engine.device))
    sector = engine._sector_at(engine.x, engine.y)
    engine.player_floor_z.copy_(engine.map.sector_heights[sector, 0])
    engine.previous_player_floor_z.copy_(engine.player_floor_z)
    engine.player_ceiling_z.copy_(engine.map.sector_heights[sector, 1])


def _initialize_monsters(
    engine: TorchDeathmatchEngine,
    records: Sequence[Mapping[str, Any]],
    enemy_type: int,
) -> None:
    monsters = [record["initial"]["monster"] for record in records]
    x = torch.tensor([monster["x"] for monster in monsters], device=engine.device)
    y = torch.tensor([monster["y"] for monster in monsters], device=engine.device)
    z = torch.tensor([monster["z"] for monster in monsters], device=engine.device)
    angle = torch.tensor(
        [monster["angle"] * math.pi / 180.0 for monster in monsters],
        device=engine.device,
    )
    spawn = torch.ones(engine.num_envs, device=engine.device, dtype=torch.bool)
    slot = torch.zeros(engine.num_envs, device=engine.device, dtype=torch.int64)
    engine._initialize_enemy_spawn_cuda(enemy_type, spawn, slot, x, y, angle)
    rows = torch.arange(engine.num_envs, device=engine.device)
    # The reference snapshot is captured after the summon command has already
    # advanced two tics.  GraDOOM's spawn helper stores the first A_Look
    # transition as a check-before-decrement countdown, so one less than the
    # observed DECORATE state tic count represents the same next action tic.
    engine.enemy_move_cooldown[rows, slot] = torch.clamp_min(
        engine.enemy_move_cooldown[rows, slot] - 1,
        0,
    )
    sector = engine._sector_at(x, y)
    floor = engine.map.sector_heights[sector, 0]
    ceiling = engine.map.sector_heights[sector, 1]
    z_fixed = torch.round(z * FIXED_UNIT).to(torch.int64)
    engine.enemy_z[rows, slot] = z
    engine._enemy_z_fixed[rows, slot] = z_fixed
    engine._enemy_floor_z_fixed[rows, slot] = torch.round(floor * FIXED_UNIT).to(torch.int64)
    engine._enemy_ceiling_z_fixed[rows, slot] = torch.round(ceiling * FIXED_UNIT).to(torch.int64)
    # A summoned actor observed after two tics is at z=7 with vertical
    # velocity -2; the next two Doom gravity updates place it at z=2.
    engine._enemy_velocity_z_fixed[rows, slot] = torch.where(
        z > floor,
        torch.full_like(z_fixed, -2 * FIXED_UNIT),
        torch.zeros_like(z_fixed),
    )


def _run_gradoom(
    *,
    scenario_path: Path,
    iwad: Path,
    reference: Sequence[Mapping[str, Any]],
    enemy_type: int,
    program: str,
    decisions: int,
    frame_skip: int,
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
    lanes = torch.ones(num_envs, device=device, dtype=torch.bool)
    game_seeds = torch.tensor(
        [record["game_seed"] for record in reference], device=device, dtype=torch.int64
    )
    engine.reset(lanes, game_seeds)
    _align_players(engine, reference)
    _initialize_monsters(engine, reference, enemy_type)
    engine.episode_time.fill_(frame_skip)
    engine.next_spawn_check.fill_(1 << 30)
    initial_x = engine.enemy_x[:, 0].clone()
    initial_y = engine.enemy_y[:, 0].clone()
    initial_player_x = engine.x.clone()
    initial_player_y = engine.y.clone()
    minimum_distance = torch.sqrt(
        (initial_x - initial_player_x) ** 2 + (initial_y - initial_player_y) ** 2
    )
    first_damage = torch.full((num_envs,), -1, device=device, dtype=torch.int32)
    first_motion = torch.full((num_envs,), -1, device=device, dtype=torch.int32)
    done = torch.zeros(num_envs, device=device, dtype=torch.bool)
    completed = torch.zeros(num_envs, device=device, dtype=torch.int32)
    actions = torch.as_tensor(
        _action_matrix(DEATHMATCH_BUTTONS),
        device=device,
        dtype=torch.bool,
    )
    for decision in range(1, decisions + 1):
        action = actions[_action_index(program, decision - 1)].expand(num_envs, -1)
        _frames, _rewards, terminated, truncated = engine.step(action)
        active = ~done
        completed.copy_(torch.where(active, torch.full_like(completed, decision), completed))
        first_damage.copy_(
            torch.where(
                (first_damage < 0) & (engine.player_hits_taken > 0),
                torch.full_like(first_damage, decision),
                first_damage,
            )
        )
        displacement = torch.sqrt(
            (engine.enemy_x[:, 0] - initial_x) ** 2 + (engine.enemy_y[:, 0] - initial_y) ** 2
        )
        current_distance = torch.sqrt(
            (engine.enemy_x[:, 0] - engine.x) ** 2 + (engine.enemy_y[:, 0] - engine.y) ** 2
        )
        minimum_distance.copy_(torch.minimum(minimum_distance, current_distance))
        first_motion.copy_(
            torch.where(
                (first_motion < 0) & (displacement > 0.5),
                torch.full_like(first_motion, decision),
                first_motion,
            )
        )
        done |= terminated | truncated
    displacement = torch.sqrt(
        (engine.enemy_x[:, 0] - initial_x) ** 2 + (engine.enemy_y[:, 0] - initial_y) ** 2
    )
    return [
        {
            "damage_taken": float(engine.player_damage_taken[lane]),
            "decisions": int(completed[lane]),
            "died": bool(engine.player_dead[lane]),
            "final_displacement": float(displacement[lane]),
            "first_damage_decision": (
                None if int(first_damage[lane]) < 0 else int(first_damage[lane])
            ),
            "first_motion_decision": (
                None if int(first_motion[lane]) < 0 else int(first_motion[lane])
            ),
            "game_seed": int(game_seeds[lane]),
            "health": float(engine.health[lane]),
            "hits_taken": float(engine.player_hits_taken[lane]),
            "minimum_monster_player_distance": float(minimum_distance[lane]),
            "player_displacement": float(
                torch.sqrt(
                    (engine.x[lane] - initial_player_x[lane]) ** 2
                    + (engine.y[lane] - initial_player_y[lane]) ** 2
                )
            ),
        }
        for lane in range(num_envs)
    ]


def _optional_mean(records: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if record[name] is not None]
    return None if not values else statistics.fmean(values)


def _normal_summary(values: Sequence[float]) -> dict[str, Any]:
    """Summarize an isolated outcome distribution without pairing RNG streams."""

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
    return {
        "damage_taken": [float(record["damage_taken"]) for record in records],
        "final_displacement": [float(record["final_displacement"]) for record in records],
        "first_damage_decision_when_observed": [
            float(record["first_damage_decision"])
            for record in records
            if record["first_damage_decision"] is not None
        ],
        "first_damage_observed": [
            float(record["first_damage_decision"] is not None) for record in records
        ],
        "first_motion_decision_when_observed": [
            float(record["first_motion_decision"])
            for record in records
            if record["first_motion_decision"] is not None
        ],
        "first_motion_observed": [
            float(record["first_motion_decision"] is not None) for record in records
        ],
        "hits_taken": [float(record["hits_taken"]) for record in records],
        "minimum_monster_player_distance": [
            float(record["minimum_monster_player_distance"]) for record in records
        ],
        "player_displacement": [float(record["player_displacement"]) for record in records],
    }


def _distribution_comparison(
    reference: Sequence[Mapping[str, Any]],
    gradoom: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report unpaired provider deltas; trajectories intentionally need not align."""

    reference_values = _outcome_values(reference)
    gradoom_values = _outcome_values(gradoom)
    comparison: dict[str, Any] = {}
    for name in reference_values:
        reference_summary = _normal_summary(reference_values[name])
        gradoom_summary = _normal_summary(gradoom_values[name])
        reference_mean = reference_summary["mean"]
        gradoom_mean = gradoom_summary["mean"]
        reference_error = reference_summary["standard_error"]
        gradoom_error = gradoom_summary["standard_error"]
        if reference_mean is None or gradoom_mean is None:
            delta = None
            delta_error = None
            interval = None
        else:
            delta = gradoom_mean - reference_mean
            if reference_error is None or gradoom_error is None:
                delta_error = None
                interval = None
            else:
                delta_error = math.hypot(reference_error, gradoom_error)
                interval = [delta - 1.96 * delta_error, delta + 1.96 * delta_error]
        comparison[name] = {
            "gradoom": gradoom_summary,
            "gradoom_minus_reference": delta,
            "gradoom_minus_reference_normal_95_ci": interval,
            "gradoom_minus_reference_standard_error": delta_error,
            "reference": reference_summary,
        }
    return comparison


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(records),
        "damage_taken_mean": statistics.fmean(float(record["damage_taken"]) for record in records),
        "died_rate": statistics.fmean(float(record["died"]) for record in records),
        "final_displacement_mean": statistics.fmean(
            float(record["final_displacement"]) for record in records
        ),
        "first_damage_decision_mean_when_observed": _optional_mean(
            records, "first_damage_decision"
        ),
        "first_damage_observed_rate": statistics.fmean(
            float(record["first_damage_decision"] is not None) for record in records
        ),
        "first_motion_decision_mean_when_observed": _optional_mean(
            records, "first_motion_decision"
        ),
        "first_motion_observed_rate": statistics.fmean(
            float(record["first_motion_decision"] is not None) for record in records
        ),
        "health_mean": statistics.fmean(float(record["health"]) for record in records),
        "hits_taken_mean": statistics.fmean(float(record["hits_taken"]) for record in records),
        "minimum_monster_player_distance_mean": statistics.fmean(
            float(record["minimum_monster_player_distance"]) for record in records
        ),
        "player_displacement_mean": statistics.fmean(
            float(record["player_displacement"]) for record in records
        ),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.episodes <= 0 or args.decisions <= 0 or args.frame_skip <= 0 or args.workers <= 0:
        raise ValueError("episodes, decisions, frame-skip, and workers must be positive")
    # Stop before the ACS loop's first spawn check at tic 106.  One initial
    # materialization action has already consumed frame_skip tics.
    if (args.decisions + 1) * args.frame_skip >= 106:
        raise ValueError("comparison must end before the ACS spawn loop starts at tic 106")
    config = args.config.expanduser().resolve()
    scenario = args.scenario.expanduser().resolve()
    iwad = args.iwad.expanduser().resolve()
    game_seeds = [_game_seed(args.seed + lane) for lane in range(args.episodes)]
    results: list[dict[str, Any]] = []
    for enemy_type, monster_name in enumerate(MONSTER_NAMES):
        if monster_name not in args.classes:
            continue
        for program in args.programs:
            with ThreadPoolExecutor(max_workers=min(args.workers, args.episodes)) as executor:
                reference = list(
                    executor.map(
                        lambda game_seed, monster_name=monster_name, program=program: (
                            _run_vizdoom_episode(
                                config=config,
                                iwad=iwad,
                                game_seed=game_seed,
                                monster_name=monster_name,
                                program=program,
                                decisions=args.decisions,
                                frame_skip=args.frame_skip,
                            )
                        ),
                        game_seeds,
                    )
                )
            gradoom = _run_gradoom(
                scenario_path=scenario,
                iwad=iwad,
                reference=reference,
                enemy_type=enemy_type,
                program=program,
                decisions=args.decisions,
                frame_skip=args.frame_skip,
            )
            results.append(
                {
                    "class": monster_name,
                    "distribution_comparison": _distribution_comparison(reference, gradoom),
                    "program": program,
                    "gradoom": _summary(gradoom),
                    "reference": _summary(reference),
                }
            )
    result = {
        "decisions": args.decisions,
        "episodes": args.episodes,
        "frame_skip": args.frame_skip,
        "results": results,
        "schema": "gradoom.summoned-monster-outcomes.v5",
        "seed": args.seed,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
