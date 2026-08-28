from __future__ import annotations

import argparse
import copy
import json

COMMON_BEHAVIORS = {
    "constructor": {
        "accepted": True,
        "parameters": [
            "game",
            "state",
            "scenario",
            "info",
            "use_restricted_actions",
            "record",
            "players",
            "inttype",
            "obs_type",
            "render_mode",
            "num_envs",
            "num_threads",
            "rom_path",
            "transport",
            "obs_copy",
            "obs_resize",
            "obs_crop",
            "obs_crop_mode",
            "obs_crop_fill",
            "obs_grayscale",
            "obs_resize_algorithm",
            "obs_layout",
            "frame_skip",
            "frame_stack",
            "maxpool_last_two",
            "noop_reset_max",
            "use_fire_reset",
            "sticky_action_prob",
            "reward_clip",
            "info_filter",
            "info_frame_stack_keys",
            "state_catalog",
        ],
    },
    "action_meanings": ["noop", "attack", "move_forward"],
    "observation_shapes": {"reset": [2, 4, 84, 84], "step": [2, 4, 84, 84]},
    "signal_shapes": {
        "reset": {"health": [2], "player_killcount": [2]},
        "step": {"health": [2], "player_killcount": [2]},
    },
    "rewards": {"shape": [2], "dtype": "float32", "sample": [0.0, 1.0]},
    "reset": {"returns_observation_and_signals": True},
    "step": {"returns_five_tuple": True},
    "masked_reset": {"supported": True, "selected_lane_only": True},
    "termination": {"reported_separately": True, "requires_reset": True},
    "truncation": {"reported_separately": True, "requires_reset": True},
    "episode": {"step_before_reset_rejected": True, "autoreset": False},
    "player_killcount": {"present": True, "player_kill_delta": 1},
    "player_killcount.enemy_on_enemy_exclusion": {"enemy_on_enemy_delta": 0},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("gradoom", "env-vizdoom-turbo"))
    parser.add_argument("--mismatch")
    args = parser.parse_args()
    behaviors = copy.deepcopy(COMMON_BEHAVIORS)
    if args.mismatch == "rewards":
        behaviors["rewards"]["sample"] = [9.0, 9.0]
    contract = {
        "schema_version": 1,
        "provider": args.provider,
        "revision": f"fixture-{args.provider}-revision",
        "behaviors": behaviors,
    }
    if args.provider == "gradoom":
        contract["tensor_device"] = {
            "declared_device": "cpu",
            "reset_mask_input": {"transport": "torch", "device": "cpu"},
            "step_action_input": {"transport": "torch", "device": "cpu"},
            "reset_outputs": {"transport": "torch", "device": "cpu"},
            "step_outputs": {"transport": "torch", "device": "cpu"},
        }
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
