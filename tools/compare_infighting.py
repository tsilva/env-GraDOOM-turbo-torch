"""Compare an aligned two-monster infighting setup with ViZDoom.

A hitscan monster is placed directly behind a weaker blocker, both facing the
same stationary player.  The blocker should intercept the rear monster's
shots, retaliate, and produce the same kill and player-damage distributions in
both providers.  The run ends before deathmatch.acs begins spawning actors.
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

from gradoom.actions import DEATHMATCH_BUTTONS
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
MONSTER_RADII = (20.0, 20.0, 16.0, 20.0, 30.0, 24.0)
ANGLE_SETUP_TICS = 14
VARIABLE_NAMES = (
    "KILLCOUNT",
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
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--decisions", type=int, default=44)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--front-class", choices=MONSTER_NAMES, default="Zombieman")
    parser.add_argument("--rear-class", choices=MONSTER_NAMES, default="ShotgunGuy")
    parser.add_argument("--front-distance", type=float, default=100.0)
    parser.add_argument("--rear-distance", type=float, default=200.0)
    return parser


def _game_seed(provider_seed: int) -> int:
    generator = np.random.default_rng(provider_seed)
    return int(generator.integers(0, UINT32_MASK + 1, dtype=np.uint32))


def _queue_spawn_at(
    game: Any,
    *,
    monster_name: str,
    radius: float,
    center_x: float,
    center_y: float,
    direction_x: float,
    direction_y: float,
    distance: float,
) -> None:
    # The setup turns the player west so its first shot wakes the actors
    # without hitting them. GZDoom's summon command places these actors
    # 2*radius+16 units west of that temporary player location. Warp east of
    # the requested actor center so both actors still land on the eastward
    # player-to-monster sight line.
    summon_distance = 2.0 * radius + 16.0
    player_distance = distance + summon_distance
    x = round(center_x + player_distance * direction_x)
    y = round(center_y + player_distance * direction_y)
    game.send_game_command(f"warp {x} {y}")
    game.send_game_command(f"summon {monster_name}")


def _objects(state: Any, names: Sequence[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for actor in () if state is None else state.objects:
        if actor.name in names:
            if actor.name in result:
                raise RuntimeError(f"expected one {actor.name} object")
            result[actor.name] = {
                "angle": float(actor.angle),
                "x": float(actor.position_x),
                "y": float(actor.position_y),
                "z": float(actor.position_z),
            }
    return result


def _run_vizdoom_episode(
    *,
    config: Path,
    iwad: Path,
    game_seed: int,
    front_class: str,
    rear_class: str,
    front_distance: float,
    rear_distance: float,
    decisions: int,
    frame_skip: int,
) -> dict[str, Any]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("infighting comparison requires vizdoom") from exc

    game = vzd.DoomGame()
    config_directory = tempfile.TemporaryDirectory(prefix="gradoom-vizdoom-infight-")
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
        buttons = tuple(value.name for value in game.get_available_buttons())
        noop = [0.0] * len(buttons)
        # The deathmatch teleport leaves seven reaction tics. Once it expires,
        # make one exact keyboard turn by temporarily setting all four turn
        # speeds to the current 16-bit angle. This puts every episode on the
        # same central east-facing line without modifying the map.
        game.make_action(noop, 10)
        angle = float(game.get_game_variable(vzd.GameVariable.ANGLE))
        angle_units = round(angle / 360.0 * 65536.0)
        game.send_game_command(
            f"turnspeeds {angle_units} {angle_units} {angle_units} {angle_units}"
        )
        game.make_action(noop, 1)
        turn = noop.copy()
        turn[buttons.index("TURN_RIGHT")] = 1.0
        game.make_action(turn, 1)
        game.send_game_command("turnspeeds 32768 32768 32768 32768")
        game.make_action(noop, 1)
        game.make_action(turn, 1)
        direction_x = 1.0
        direction_y = 0.0
        center_x = 512.0
        center_y = 512.0
        front_type = MONSTER_NAMES.index(front_class)
        rear_type = MONSTER_NAMES.index(rear_class)
        _queue_spawn_at(
            game,
            monster_name=front_class,
            radius=MONSTER_RADII[front_type],
            center_x=center_x,
            center_y=center_y,
            direction_x=direction_x,
            direction_y=direction_y,
            distance=front_distance,
        )
        _queue_spawn_at(
            game,
            monster_name=rear_class,
            radius=MONSTER_RADII[rear_type],
            center_x=center_x,
            center_y=center_y,
            direction_x=direction_x,
            direction_y=direction_y,
            distance=rear_distance,
        )
        game.send_game_command(f"warp {round(center_x)} {round(center_y)}")
        game.make_action(noop, frame_skip)
        state = game.get_state()
        monsters = _objects(state, (front_class, rear_class))
        if set(monsters) != {front_class, rear_class}:
            raise RuntimeError(f"summoned actors did not materialize: {sorted(monsters)}")
        values = {
            name.casefold(): float(game.get_game_variable(variable))
            for name, variable in zip(VARIABLE_NAMES, variables, strict=True)
        }
        initial_kills = values["killcount"]
        initial_hits = values["hits_taken"]
        initial_damage = values["damage_taken"]
        attack = noop.copy()
        attack[buttons.index("ATTACK")] = 1.0
        first_kill: int | None = None
        executed = 0
        for decision in range(1, decisions + 1):
            if game.is_episode_finished() or game.is_player_dead():
                break
            game.make_action(attack if decision == 1 else noop, frame_skip)
            executed = decision
            kills = float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)) - initial_kills
            if first_kill is None and kills > 0:
                first_kill = decision
        return {
            "damage_taken": float(game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN))
            - initial_damage,
            "decisions": executed,
            "died": bool(game.is_player_dead()),
            "first_infighting_decision": None,
            "first_kill_decision": first_kill,
            "game_seed": game_seed,
            "health": float(game.get_game_variable(vzd.GameVariable.HEALTH)),
            "hits_taken": float(game.get_game_variable(vzd.GameVariable.HITS_TAKEN)) - initial_hits,
            "initial": {
                "episode_time": int(game.get_episode_time()) - executed * frame_skip,
                "monsters": monsters,
                "player": values,
            },
            "kills": float(game.get_game_variable(vzd.GameVariable.KILLCOUNT)) - initial_kills,
        }
    finally:
        game.close()
        config_directory.cleanup()


def _align_players(engine: TorchDeathmatchEngine, records: Sequence[Mapping[str, Any]]) -> None:
    players = [record["initial"]["player"] for record in records]
    x = torch.tensor([player["position_x"] for player in players], device=engine.device)
    y = torch.tensor([player["position_y"] for player in players], device=engine.device)
    z = torch.tensor([player["position_z"] for player in players], device=engine.device)
    camera_z = torch.tensor(
        [player["camera_position_z"] for player in players], device=engine.device
    )
    angle_degrees = torch.tensor([player["angle"] for player in players], device=engine.device)
    engine._x_fixed.copy_(torch.round(x * FIXED_UNIT).to(torch.int64))
    engine._y_fixed.copy_(torch.round(y * FIXED_UNIT).to(torch.int64))
    engine.x.copy_(engine._x_fixed.to(torch.float32) / FIXED_UNIT)
    engine.y.copy_(engine._y_fixed.to(torch.float32) / FIXED_UNIT)
    engine.z.copy_(z)
    engine.view_z.copy_(camera_z)
    engine.view_height.copy_(camera_z - z)
    angle_bam = torch.bitwise_and(
        torch.round(angle_degrees / 360.0 * (1 << 32)).to(torch.int64), UINT32_MASK
    )
    engine._angle_bam.copy_(angle_bam)
    engine.angle.copy_(angle_bam.to(torch.float32) * BAM_TO_RADIANS)
    engine.health.copy_(
        torch.tensor([player["health"] for player in players], device=engine.device)
    )
    engine.armor.copy_(torch.tensor([player["armor"] for player in players], device=engine.device))
    # The reference setup consumed fourteen real tics before the aligned
    # state, so the spawn teleport lock and pistol raise have both completed.
    engine.reaction_time.zero_()
    engine.weapon_raise_cooldown.zero_()
    engine.weapon_ready_tics.fill_(6)
    sector = engine._sector_at(engine.x, engine.y)
    engine.player_floor_z.copy_(engine.map.sector_heights[sector, 0])
    engine.previous_player_floor_z.copy_(engine.player_floor_z)
    engine.player_ceiling_z.copy_(engine.map.sector_heights[sector, 1])


def _initialize_monster(
    engine: TorchDeathmatchEngine,
    records: Sequence[Mapping[str, Any]],
    *,
    enemy_type: int,
    monster_name: str,
    slot_index: int,
) -> None:
    monsters = [record["initial"]["monsters"][monster_name] for record in records]
    x = torch.tensor([monster["x"] for monster in monsters], device=engine.device)
    y = torch.tensor([monster["y"] for monster in monsters], device=engine.device)
    z = torch.tensor([monster["z"] for monster in monsters], device=engine.device)
    angle = torch.tensor(
        [monster["angle"] * math.pi / 180.0 for monster in monsters], device=engine.device
    )
    spawn = torch.ones(engine.num_envs, device=engine.device, dtype=torch.bool)
    slot = torch.full((engine.num_envs,), slot_index, device=engine.device, dtype=torch.int64)
    engine._initialize_enemy_spawn_cuda(enemy_type, spawn, slot, x, y, angle)
    rows = torch.arange(engine.num_envs, device=engine.device)
    sector = engine._sector_at(x, y)
    floor = engine.map.sector_heights[sector, 0]
    ceiling = engine.map.sector_heights[sector, 1]
    z_fixed = torch.round(z * FIXED_UNIT).to(torch.int64)
    engine.enemy_z[rows, slot] = z
    engine._enemy_z_fixed[rows, slot] = z_fixed
    engine._enemy_floor_z_fixed[rows, slot] = torch.round(floor * FIXED_UNIT).to(torch.int64)
    engine._enemy_ceiling_z_fixed[rows, slot] = torch.round(ceiling * FIXED_UNIT).to(torch.int64)
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
    front_type: int,
    rear_type: int,
    decisions: int,
    frame_skip: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda")
    num_envs = len(reference)
    engine = TorchDeathmatchEngine(
        compile_deathmatch_scenario(scenario_path, iwad),
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
    _initialize_monster(
        engine,
        reference,
        enemy_type=front_type,
        monster_name=MONSTER_NAMES[front_type],
        slot_index=0,
    )
    _initialize_monster(
        engine,
        reference,
        enemy_type=rear_type,
        monster_name=MONSTER_NAMES[rear_type],
        slot_index=1,
    )
    engine.episode_time.copy_(
        torch.tensor(
            [record["initial"]["episode_time"] for record in reference],
            device=device,
            dtype=torch.int32,
        )
    )
    engine.next_spawn_check.fill_(1 << 30)
    first_infighting = torch.full((num_envs,), -1, device=device, dtype=torch.int32)
    first_kill = torch.full((num_envs,), -1, device=device, dtype=torch.int32)
    completed = torch.zeros(num_envs, device=device, dtype=torch.int32)
    done = torch.zeros(num_envs, device=device, dtype=torch.bool)
    noop = torch.zeros((num_envs, len(DEATHMATCH_BUTTONS)), device=device, dtype=torch.bool)
    attack = noop.clone()
    attack[:, DEATHMATCH_BUTTONS.index("ATTACK")] = True
    for decision in range(1, decisions + 1):
        _frames, _rewards, terminated, truncated = engine.step(attack if decision == 1 else noop)
        active = ~done
        completed.copy_(torch.where(active, torch.full_like(completed, decision), completed))
        has_monster_target = torch.any(engine.enemy_target_slot >= 0, dim=1)
        first_infighting.copy_(
            torch.where(
                (first_infighting < 0) & has_monster_target,
                torch.full_like(first_infighting, decision),
                first_infighting,
            )
        )
        first_kill.copy_(
            torch.where(
                (first_kill < 0) & (engine.killcount > 0),
                torch.full_like(first_kill, decision),
                first_kill,
            )
        )
        done |= terminated | truncated
    return [
        {
            "damage_taken": float(engine.player_damage_taken[lane]),
            "decisions": int(completed[lane]),
            "died": bool(engine.player_dead[lane]),
            "first_infighting_decision": (
                None if int(first_infighting[lane]) < 0 else int(first_infighting[lane])
            ),
            "first_kill_decision": (None if int(first_kill[lane]) < 0 else int(first_kill[lane])),
            "game_seed": int(game_seeds[lane]),
            "health": float(engine.health[lane]),
            "hits_taken": float(engine.player_hits_taken[lane]),
            "kills": float(engine.killcount[lane]),
        }
        for lane in range(num_envs)
    ]


def _optional_mean(records: Sequence[Mapping[str, Any]], name: str) -> float | None:
    values = [float(record[name]) for record in records if record[name] is not None]
    return None if not values else statistics.fmean(values)


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "damage_taken_mean": statistics.fmean(float(record["damage_taken"]) for record in records),
        "died_rate": statistics.fmean(float(record["died"]) for record in records),
        "episodes": len(records),
        "first_infighting_decision_mean_when_observed": _optional_mean(
            records, "first_infighting_decision"
        ),
        "first_kill_decision_mean_when_observed": _optional_mean(records, "first_kill_decision"),
        "health_mean": statistics.fmean(float(record["health"]) for record in records),
        "hits_taken_mean": statistics.fmean(float(record["hits_taken"]) for record in records),
        "kill_observed_rate": statistics.fmean(float(record["kills"] > 0) for record in records),
        "kills_mean": statistics.fmean(float(record["kills"]) for record in records),
    }


def main() -> int:
    args = _parser().parse_args()
    if args.episodes <= 0 or args.decisions <= 0 or args.frame_skip <= 0 or args.workers <= 0:
        raise ValueError("episodes, decisions, frame-skip, and workers must be positive")
    if ANGLE_SETUP_TICS + (args.decisions + 1) * args.frame_skip >= 106:
        raise ValueError("comparison must end before the ACS spawn loop starts at tic 106")
    if args.front_class == args.rear_class:
        raise ValueError("front and rear classes must differ for object-pose alignment")
    if args.front_distance >= args.rear_distance:
        raise ValueError("front-distance must be less than rear-distance")
    config = args.config.expanduser().resolve()
    scenario = args.scenario.expanduser().resolve()
    iwad = args.iwad.expanduser().resolve()
    game_seeds = [_game_seed(args.seed + lane) for lane in range(args.episodes)]
    with ThreadPoolExecutor(max_workers=min(args.workers, args.episodes)) as executor:
        reference = list(
            executor.map(
                lambda game_seed: _run_vizdoom_episode(
                    config=config,
                    iwad=iwad,
                    game_seed=game_seed,
                    front_class=args.front_class,
                    rear_class=args.rear_class,
                    front_distance=args.front_distance,
                    rear_distance=args.rear_distance,
                    decisions=args.decisions,
                    frame_skip=args.frame_skip,
                ),
                game_seeds,
            )
        )
    gradoom = _run_gradoom(
        scenario_path=scenario,
        iwad=iwad,
        reference=reference,
        front_type=MONSTER_NAMES.index(args.front_class),
        rear_type=MONSTER_NAMES.index(args.rear_class),
        decisions=args.decisions,
        frame_skip=args.frame_skip,
    )
    print(
        json.dumps(
            {
                "decisions": args.decisions,
                "episodes": args.episodes,
                "frame_skip": args.frame_skip,
                "front_class": args.front_class,
                "gradoom": _summary(gradoom),
                "rear_class": args.rear_class,
                "reference": _summary(reference),
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
