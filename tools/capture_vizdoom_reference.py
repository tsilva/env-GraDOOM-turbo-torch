"""Capture a deterministic ViZDoom oracle trace as JSON Lines.

Run this with the reference ViZDoom environment available on PYTHONPATH. The
tool is intentionally outside the env-Doom-turbo-torch runtime dependency graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from env_doom_turbo_torch.actions import DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS

VARIABLES = (
    "KILLCOUNT",
    "HEALTH",
    "ARMOR",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
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


def _action_matrix(available: tuple[str, ...]) -> np.ndarray:
    indices = {name: index for index, name in enumerate(available)}
    matrix = np.zeros((len(DEATHMATCH_ACTIONS), len(available)), dtype=np.float64)
    for action_index, labels in enumerate(DEATHMATCH_ACTIONS):
        for label in labels:
            matrix[action_index, indices[label]] = 1
    return matrix


def _objects(state: Any) -> dict[str, Any]:
    objects = state.objects if state is not None else ()
    counts = Counter(item.name for item in objects)
    dynamic_names = {
        "DoomPlayer",
        "Zombieman",
        "ShotgunGuy",
        "MarineChainsawVzd",
        "ChaingunGuy",
        "Demon",
        "HellKnight",
        "TeleportFog",
    }
    dynamic = [
        {
            "angle": round(float(item.angle), 6),
            "id": int(item.id),
            "name": item.name,
            "x": round(float(item.position_x), 6),
            "y": round(float(item.position_y), 6),
            "z": round(float(item.position_z), 6),
        }
        for item in objects
        if item.name in dynamic_names
    ]
    return {"counts": dict(sorted(counts.items())), "dynamic": dynamic}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--iwad", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--doom-skill", type=int, default=1)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--program",
        choices=(
            "noop",
            "forward",
            "strafe-right",
            "turn-left",
            "spiral",
            "forward-fire",
            "weapon-switch-fire",
            "weapon3-switch-fire",
            "summon-kill",
            "summon-forward",
            "summon-noop",
            "script-kill",
        ),
        default="spiral",
    )
    parser.add_argument("--summon-class", default="Zombieman")
    args = parser.parse_args()

    import vizdoom as vzd

    game = vzd.DoomGame()
    config_directory = tempfile.TemporaryDirectory(prefix="env_doom_turbo_torch-vizdoom-trace-")
    game.load_config(str(Path(args.config).expanduser().resolve()))
    game.set_doom_config_path(str(Path(config_directory.name) / "engine.ini"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_audio_buffer_enabled(False)
    game.set_doom_game_path(str(Path(args.iwad).expanduser().resolve()))
    game.set_doom_skill(args.doom_skill)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_objects_info_enabled(True)
    for name in VARIABLES:
        variable = getattr(vzd.GameVariable, name)
        if variable not in game.get_available_game_variables():
            game.add_available_game_variable(variable)
    game.set_seed(args.seed)
    game.init()
    try:
        game.new_episode()
        if args.program in {"summon-kill", "summon-forward", "summon-noop"}:
            game.send_game_command(f"summon {args.summon_class}")
        elif args.program in {"weapon-switch-fire", "weapon3-switch-fire"}:
            game.send_game_command("give all")
        available = tuple(value.name for value in game.get_available_buttons())
        if available != DEATHMATCH_BUTTONS:
            raise RuntimeError(f"reference buttons differ: {available!r}")
        actions = _action_matrix(available)
        weapon3_action = np.zeros(len(available), dtype=np.float64)
        weapon3_action[available.index("SELECT_WEAPON3")] = 1
        variable_types = tuple(getattr(vzd.GameVariable, name) for name in VARIABLES)
        previous_reward = 0.0
        print(
            json.dumps(
                {
                    "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                    "frame_skip": args.frame_skip,
                    "iwad_sha256": hashlib.sha256(Path(args.iwad).read_bytes()).hexdigest(),
                    "program": args.program,
                    "schema": "env_doom_turbo_torch.vizdoom-reference-trace.v1",
                    "seed": args.seed,
                    "type": "header",
                },
                sort_keys=True,
            )
        )
        for step in range(args.steps + 1):
            state = game.get_state()
            screen = None if state is None else np.asarray(state.screen_buffer)
            values = {
                name.casefold(): float(game.get_game_variable(variable))
                for name, variable in zip(VARIABLES, variable_types, strict=True)
            }
            record: dict[str, Any] = {
                "dead": bool(game.is_player_dead()),
                "episode_finished": bool(game.is_episode_finished()),
                "episode_time": int(game.get_episode_time()),
                "previous_action_reward": previous_reward,
                "step": step,
                "type": "transition",
                "variables": values,
            }
            if not args.compact:
                record["objects"] = _objects(state)
                record["screen_sha256"] = (
                    None if screen is None else hashlib.sha256(screen.tobytes()).hexdigest()
                )
            print(json.dumps(record, sort_keys=True))
            if step == args.steps or game.is_episode_finished() or game.is_player_dead():
                break
            if args.program in {"noop", "script-kill", "summon-noop"}:
                if args.program == "script-kill" and step == 350:
                    game.send_game_command("kill monsters")
                action_index = 0
            elif args.program == "forward":
                action_index = 2
            elif args.program == "strafe-right":
                action_index = 5
            elif args.program == "turn-left":
                action_index = 6
            elif args.program == "forward-fire":
                action_index = 9
            elif args.program == "summon-kill":
                if step == 0:
                    game.send_game_command("kill monsters")
                action_index = 0
            elif args.program == "summon-forward":
                action_index = 2
            elif args.program == "weapon-switch-fire":
                action_index = 0 if step == 0 else 16 if step == 1 else 1
            elif args.program == "weapon3-switch-fire":
                action_index = 1
            else:
                action_index = 13 if (step // 20) % 2 == 0 else 14
            action = (
                weapon3_action.tolist()
                if args.program == "weapon3-switch-fire" and step == 1
                else actions[action_index].tolist()
            )
            previous_reward = float(game.make_action(action, args.frame_skip))
    finally:
        game.close()
        config_directory.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
