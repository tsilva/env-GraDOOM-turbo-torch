#!/usr/bin/env python3
"""CPU-only fixture for the standalone GraDOOM trainer process contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seeds-file", type=Path)
    parser.add_argument("--evaluation-seed", type=int, default=123)
    parser.add_argument("--evaluation-stochastic", action=argparse.BooleanOptionalAction)
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--fixture-outcomes", required=True)
    parser.add_argument("--fixture-fail-training-seed", type=int)
    parser.add_argument("--fixture-fail-evaluation-step", type=int)
    parser.add_argument("--fixture-omit-player-killcount", action="store_true")
    parser.add_argument("--fixture-training-step-offset", type=int, default=0)
    parser.add_argument("--fixture-hardlink-checkpoint-to", type=Path)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit(path: Path, *records: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> int:
    args, _unknown = _parser().parse_known_args()
    outcomes = json.loads(args.fixture_outcomes)
    if args.evaluate_checkpoint is None:
        if args.fixture_fail_training_seed == args.seed:
            return 17
        assert args.checkpoint is not None
        assert args.timesteps is not None
        actual_step = args.timesteps + args.fixture_training_step_offset
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(
            json.dumps(
                {
                    "format": "standalone-gradoom-ppo-v1",
                    "seed": args.seed,
                    "step": actual_step,
                    "resumed": args.resume is not None,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if (
            args.fixture_hardlink_checkpoint_to is not None
            and not args.fixture_hardlink_checkpoint_to.exists()
        ):
            os.link(args.checkpoint, args.fixture_hardlink_checkpoint_to)
        records: list[dict[str, object]] = [
            {
                "type": "config",
                "contract": "standalone-gradoom-deathmatch-ppo-v2",
                "operation": "train",
                "requested_timesteps": args.timesteps,
                "execution_timesteps": actual_step,
                "initialization": {
                    "mode": "random",
                    "checkpoint": None,
                    "checkpoint_sha256": None,
                },
                "state_initialization": {
                    "policy_state": "resumed" if args.resume is not None else "fresh_random",
                    "optimizer_state": "resumed" if args.resume is not None else "fresh",
                },
            }
        ]
        if args.resume is not None:
            records.append(
                {
                    "type": "event",
                    "event": "resumed",
                    "checkpoint": str(args.resume),
                }
            )
        records.append(
            {
                "type": "summary",
                "status": "completed",
                "train/global_step": actual_step,
                "requested_timesteps": args.timesteps,
                "execution_timesteps": actual_step,
                "checkpoint": str(args.checkpoint),
                "training_transitions_per_second": 1000.0,
            }
        )
        _emit(args.metrics_jsonl, *records)
        return 0

    checkpoint = json.loads(args.evaluate_checkpoint.read_text(encoding="utf-8"))
    step = int(checkpoint["step"])
    if args.fixture_fail_evaluation_step == step:
        return 19
    assert args.evaluation_seeds_file is not None
    episode_seeds = json.loads(args.evaluation_seeds_file.read_text(encoding="utf-8"))
    assert args.evaluation_episodes == len(episode_seeds)
    player_quality, compatibility_quality = outcomes.get(
        f"{checkpoint['seed']}:{step}",
        outcomes.get(str(step), [0.0, 0.0]),
    )
    episodes = []
    for index, game_seed in enumerate(episode_seeds):
        episode = {
            "index": index,
            "game_seed": game_seed,
            "killcount": float(compatibility_quality),
            "length": 10,
            "terminated": True,
            "truncated": False,
        }
        if args.fixture_omit_player_killcount:
            episode["kills"] = float(player_quality)
        else:
            episode["player_killcount"] = float(player_quality)
        episodes.append(episode)
    _emit(
        args.metrics_jsonl,
        {
            "type": "config",
            "contract": "standalone-gradoom-deathmatch-ppo-v2",
            "operation": "evaluate",
            "evaluation": {
                "episodes": args.evaluation_episodes,
                "seed": args.evaluation_seed,
                "stochastic_actions": args.evaluation_stochastic,
                "kills_signal": "player_killcount",
                "vizdoom_compatibility_kills_signal": "killcount",
            },
        },
        {
            "type": "evaluation",
            "status": "completed",
            "checkpoint": str(args.evaluate_checkpoint),
            "checkpoint_sha256": _sha256(args.evaluate_checkpoint),
            "checkpoint_step": step,
            "deterministic_actions": not args.evaluation_stochastic,
            "evaluation/episode/count": len(episodes),
            "evaluation/kills/signal": "player_killcount",
            "episodes": episodes,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
