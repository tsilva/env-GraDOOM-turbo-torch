"""Compare deterministic env-Doom-turbo-torch transitions with an aligned ViZDoom oracle.

The certified scenario randomizes its initial pose through a stochastic stream
that env-Doom-turbo-torch is not required to reproduce. This diagnostic copies ViZDoom's
initial pose into env-Doom-turbo-torch, then compares deterministic player-facing state up
to (but not including) the first ACS monster spawn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import torch

from env_doom_turbo_torch.actions import DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS
from env_doom_turbo_torch.engine import TorchDeathmatchEngine
from env_doom_turbo_torch.scenario import CompiledScenario, compile_deathmatch_scenario

VARIABLES = (
    "KILLCOUNT",
    "HEALTH",
    "ARMOR",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "CAMERA_POSITION_Z",
    "ANGLE",
    "VELOCITY_X",
    "VELOCITY_Y",
    "VELOCITY_Z",
    "WEAPON1",
    "WEAPON2",
    "WEAPON3",
    "WEAPON4",
    "WEAPON5",
    "WEAPON6",
    "AMMO1",
    "AMMO2",
    "AMMO3",
    "AMMO4",
    "AMMO5",
    "AMMO6",
)
PROGRAMS = (
    "noop",
    "forward",
    "backward",
    "run-forward",
    "strafe-left",
    "strafe-right",
    "turn-left",
    "turn-right",
    "spiral",
    "forward-fire",
    "weapon-next-fire",
    "weapon-switch-fire",
)
_FIXED_UNIT = 1 << 16
_BAM_TO_RADIANS = 2.0 * math.pi / float(1 << 32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_index(program: str, step: int) -> int:
    if program == "noop":
        return 0
    if program == "forward":
        return 2
    if program == "backward":
        return 3
    if program == "run-forward":
        return 8
    if program == "strafe-left":
        return 4
    if program == "strafe-right":
        return 5
    if program == "turn-left":
        return 6
    if program == "turn-right":
        return 7
    if program == "forward-fire":
        return 9
    if program == "weapon-next-fire":
        return 0 if step == 0 else 15 if step == 1 else 1
    if program == "weapon-switch-fire":
        return 0 if step == 0 else 16 if step == 1 else 1
    return 13 if (step // 20) % 2 == 0 else 14


def _action_matrix(available: tuple[str, ...]) -> tuple[list[float], ...]:
    if available != DEATHMATCH_BUTTONS:
        raise RuntimeError(f"reference buttons differ: {available!r}")
    indices = {name: index for index, name in enumerate(available)}
    rows: list[list[float]] = []
    for labels in DEATHMATCH_ACTIONS:
        row = [0.0] * len(available)
        for label in labels:
            row[indices[label]] = 1.0
        rows.append(row)
    return tuple(rows)


def _align_pose(engine: TorchDeathmatchEngine, values: dict[str, float]) -> None:
    x_fixed = round(values["POSITION_X"] * _FIXED_UNIT)
    y_fixed = round(values["POSITION_Y"] * _FIXED_UNIT)
    angle_bam = round(values["ANGLE"] / 360.0 * (1 << 32)) & ((1 << 32) - 1)
    engine._x_fixed[0] = x_fixed
    engine._y_fixed[0] = y_fixed
    engine.x[0] = x_fixed / _FIXED_UNIT
    engine.y[0] = y_fixed / _FIXED_UNIT
    engine.z[0] = values["POSITION_Z"]
    engine.view_z[0] = values["CAMERA_POSITION_Z"]
    engine.view_height[0] = values["CAMERA_POSITION_Z"] - values["POSITION_Z"]
    engine._angle_bam[0] = angle_bam
    engine.angle[0] = angle_bam * _BAM_TO_RADIANS
    sector = engine._sector_at(engine.x, engine.y)
    engine.player_floor_z[0] = engine.map.sector_heights[sector, 0]
    engine.previous_player_floor_z.copy_(engine.player_floor_z)
    engine.player_ceiling_z[0] = engine.map.sector_heights[sector, 1]


def _align_give_all(engine: TorchDeathmatchEngine) -> None:
    """Mirror the state produced when ViZDoom processes ``give all``."""

    engine.armor.fill_(200)
    engine.armor_save_fraction.fill_(0.5)
    engine.chainsaw_owned.fill_(True)
    engine.shotgun_owned.fill_(True)
    engine.super_shotgun_owned.fill_(True)
    engine.weapons[0] = torch.tensor(
        (2, 1, 2, 1, 1, 1),
        device=engine.device,
        dtype=torch.float32,
    )
    engine.ammo[0] = torch.tensor(
        (0, 400, 100, 400, 100, 600),
        device=engine.device,
        dtype=torch.float32,
    )


def _engine_values(engine: TorchDeathmatchEngine) -> dict[str, float]:
    values = {
        "KILLCOUNT": float(engine.killcount[0]),
        "HEALTH": float(engine.health[0]),
        "ARMOR": float(engine.armor[0]),
        "SELECTED_WEAPON": float(engine.selected_weapon[0]),
        "POSITION_X": float(engine.x[0]),
        "POSITION_Y": float(engine.y[0]),
        "POSITION_Z": float(engine.z[0]),
        "CAMERA_POSITION_Z": float(engine.view_z[0]),
        "ANGLE": math.degrees(float(engine.angle[0])) % 360.0,
        "VELOCITY_X": float(engine.momentum_x[0]),
        "VELOCITY_Y": float(engine.momentum_y[0]),
        "VELOCITY_Z": float(engine.velocity_z[0]),
    }
    selected_ammo = int(engine.selected_weapon[0]) - 1
    values["SELECTED_WEAPON_AMMO"] = float(engine.ammo[0, selected_ammo])
    for index in range(6):
        values[f"WEAPON{index + 1}"] = float(engine.weapons[0, index])
        values[f"AMMO{index + 1}"] = float(engine.ammo[0, index])
    return values


def _difference(name: str, reference: float, actual: float) -> float:
    difference = abs(reference - actual)
    if name == "ANGLE":
        difference = min(difference, 360.0 - difference)
    return difference


def _run_case(
    *,
    config: Path,
    scenario: CompiledScenario,
    iwad: Path,
    seed: int,
    steps: int,
    frame_skip: int,
    program: str,
    tolerance: float,
    device: torch.device,
) -> dict[str, Any]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("compare_behavior.py requires the reference vizdoom package") from exc

    game = vzd.DoomGame()
    config_directory = tempfile.TemporaryDirectory(prefix="env_doom_turbo_torch-vizdoom-parity-")
    game.load_config(str(config))
    game.set_doom_config_path(str(Path(config_directory.name) / "engine.ini"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_audio_buffer_enabled(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_doom_game_path(str(iwad))
    variable_types = tuple(getattr(vzd.GameVariable, name) for name in VARIABLES)
    for variable in variable_types:
        if variable not in game.get_available_game_variables():
            game.add_available_game_variable(variable)
    game.set_seed(seed)
    game.init()
    try:
        game.new_episode()
        if program in {"weapon-next-fire", "weapon-switch-fire"}:
            game.send_game_command("give all")
        available = tuple(value.name for value in game.get_available_buttons())
        actions = _action_matrix(available)
        engine = TorchDeathmatchEngine(
            scenario,
            1,
            device=device,
            frame_skip=frame_skip,
            debug_checks=False,
        )
        engine.reset(
            torch.ones(1, dtype=torch.bool, device=device),
            torch.tensor([seed], device=device),
        )
        blank = torch.zeros((1, 84, 84), dtype=torch.uint8, device=device)

        def render_blank() -> torch.Tensor:
            return blank

        engine.render_frame = render_blank
        reference = {
            name: float(game.get_game_variable(variable))
            for name, variable in zip(VARIABLES, variable_types, strict=True)
        }
        _align_pose(engine, reference)
        previous_engine_reward = 0.0
        previous_reference_reward = 0.0

        for step in range(steps + 1):
            reference = {
                name: float(game.get_game_variable(variable))
                for name, variable in zip(VARIABLES, variable_types, strict=True)
            }
            actual = _engine_values(engine)
            for name in VARIABLES:
                difference = _difference(name, reference[name], actual[name])
                if difference > tolerance:
                    return {
                        "actual": actual[name],
                        "difference": reference[name] - actual[name],
                        "matched_transitions": step,
                        "program": program,
                        "reference": reference[name],
                        "seed": seed,
                        "status": "diverged",
                        "variable": name,
                    }
            metadata = (
                ("EPISODE_TIME", float(game.get_episode_time()), float(engine.episode_time[0])),
                ("PLAYER_DEAD", float(game.is_player_dead()), float(engine.player_dead[0])),
                ("REWARD", previous_reference_reward, previous_engine_reward),
            )
            for name, reference_value, actual_value in metadata:
                if abs(reference_value - actual_value) > tolerance:
                    return {
                        "actual": actual_value,
                        "difference": reference_value - actual_value,
                        "matched_transitions": step,
                        "program": program,
                        "reference": reference_value,
                        "seed": seed,
                        "status": "diverged",
                        "variable": name,
                    }
            if step == steps:
                break
            action_index = _action_index(program, step)
            previous_reference_reward = float(game.make_action(actions[action_index], frame_skip))
            _frames, reward, _terminated, _truncated = engine.step(
                torch.tensor(actions[action_index], dtype=torch.bool, device=device)
            )
            previous_engine_reward = float(reward[0])
            if program in {"weapon-next-fire", "weapon-switch-fire"} and step == 0:
                _align_give_all(engine)
        return {
            "matched_transitions": steps + 1,
            "program": program,
            "seed": seed,
            "status": "matched",
        }
    finally:
        game.close()
        config_directory.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--iwad", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1337))
    parser.add_argument("--programs", choices=PROGRAMS, nargs="+", default=PROGRAMS)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be positive")
    if args.steps * args.frame_skip >= 106:
        parser.error(
            "comparison must stop before the first stochastic ACS monster spawn at episode time 106"
        )
    config = args.config.expanduser().resolve()
    scenario_path = args.scenario.expanduser().resolve()
    iwad = args.iwad.expanduser().resolve()
    scenario = compile_deathmatch_scenario(scenario_path, iwad)
    records = [
        _run_case(
            config=config,
            scenario=scenario,
            iwad=iwad,
            seed=seed,
            steps=args.steps,
            frame_skip=args.frame_skip,
            program=program,
            tolerance=args.tolerance,
            device=torch.device(args.device),
        )
        for seed in args.seeds
        for program in args.programs
    ]
    result = {
        "config_sha256": _sha256(config),
        "iwad_sha256": _sha256(iwad),
        "records": records,
        "scenario_sha256": _sha256(scenario_path),
        "schema": "env_doom_turbo_torch.behavior-parity.aligned-prefix.v1",
    }
    print(json.dumps(result, sort_keys=True))
    return int(any(record["status"] != "matched" for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
