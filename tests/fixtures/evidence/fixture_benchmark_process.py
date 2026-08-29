#!/usr/bin/env python3
"""CPU-only fixture for the standalone GraDOOM trainer process contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _apply_startup_delay() -> None:
    try:
        index = sys.argv.index("--fixture-startup-delay-seconds")
    except ValueError:
        return
    time.sleep(float(sys.argv[index + 1]))


_apply_startup_delay()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--reusable-time-budget-seconds", type=float)
    parser.add_argument("--reusable-time-deadline-monotonic", type=float)
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
    parser.add_argument("--fixture-diagnostic-quality", type=float, default=0.0)
    parser.add_argument("--fixture-diagnostic-transitions", type=int, default=1000)
    parser.add_argument("--fixture-diagnostic-elapsed-seconds", type=float, default=1.0)
    parser.add_argument("--fixture-startup-delay-seconds", type=float, default=0.0)
    parser.add_argument("--fixture-episode-length", type=int, default=10)
    parser.add_argument(
        "--fixture-terminal-mode",
        choices=("terminated", "truncated", "neither", "both"),
        default="terminated",
    )
    parser.add_argument("--fixture-mutate-checkpoint-after-evaluation", action="store_true")
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
        diagnostic = args.reusable_time_budget_seconds is not None
        before_deadline = (
            args.reusable_time_deadline_monotonic is None
            or time.monotonic() < args.reusable_time_deadline_monotonic
        )
        actual_step = (
            args.fixture_diagnostic_transitions
            if diagnostic and before_deadline
            else 0
            if diagnostic
            else args.timesteps + args.fixture_training_step_offset
        )
        if diagnostic and args.reusable_time_deadline_monotonic is not None:
            time.sleep(max(0.0, args.reusable_time_deadline_monotonic - time.monotonic()))
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(
            json.dumps(
                {
                    "format": "standalone-gradoom-ppo-v1",
                    "seed": args.seed,
                    "step": actual_step,
                    "resumed": args.resume is not None,
                    "fixed_time": diagnostic,
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
                "reusable_time_budget_seconds": args.reusable_time_budget_seconds,
                "reusable_time_deadline_monotonic": args.reusable_time_deadline_monotonic,
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
                "training_transitions": actual_step,
                "frame_skip": 2,
                "reusable_time_budget_seconds": args.reusable_time_budget_seconds,
                "reusable_time_elapsed_seconds": (
                    args.fixture_diagnostic_elapsed_seconds if diagnostic else None
                ),
                "stop_reason": "reusable_time_budget" if diagnostic else "timestep_budget",
                "reusable_time_deadline_monotonic": args.reusable_time_deadline_monotonic,
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
    if checkpoint.get("fixed_time"):
        player_quality, compatibility_quality = args.fixture_diagnostic_quality, 0.0
    else:
        player_quality, compatibility_quality = outcomes.get(
            f"{checkpoint['seed']}:{step}",
            outcomes.get(str(step), [0.0, 0.0]),
        )
    episodes = []
    for index, game_seed in enumerate(episode_seeds):
        episode = {
            "index": index,
            "game_seed": game_seed,
            "compatibility_killcount": float(compatibility_quality),
            "length": args.fixture_episode_length,
            "terminated": args.fixture_terminal_mode in {"terminated", "both"},
            "truncated": args.fixture_terminal_mode in {"truncated", "both"},
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
                "compatibility_killcount_signal": "killcount",
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
    if args.fixture_mutate_checkpoint_after_evaluation:
        with args.evaluate_checkpoint.open("ab") as stream:
            stream.write(b"mutated-after-evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
