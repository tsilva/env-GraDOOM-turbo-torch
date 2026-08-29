#!/usr/bin/env python3
"""CPU-only fixture for the standalone GraDOOM trainer process contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import time
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
    parser.add_argument("--fixture-fail-training-once-marker", type=Path)
    parser.add_argument("--fixture-fail-after-resume-once-marker", type=Path)
    parser.add_argument("--fixture-fail-evaluation-step", type=int)
    parser.add_argument("--fixture-omit-player-killcount", action="store_true")
    parser.add_argument("--fixture-training-step-offset", type=int, default=0)
    parser.add_argument("--fixture-hardlink-checkpoint-to", type=Path)
    parser.add_argument("--fixture-mutate-bootstrap", type=Path)
    parser.add_argument("--fixture-mutate-trainer-code", type=Path)
    parser.add_argument("--fixture-interrupt-once-at-step", type=int)
    parser.add_argument("--fixture-hard-crash-once-at-step", type=int)
    parser.add_argument("--fixture-hold-after-recovery-checkpoint-marker", type=Path)
    parser.add_argument("--fixture-recovery-child-exited-marker", type=Path)
    parser.add_argument("--evidence-run-identity")
    parser.add_argument("--evidence-attempt-identity")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit(path: Path, *records: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _mutate_bootstrap(path: Path | None) -> None:
    if path is None:
        return
    path.chmod(0o600)
    with path.open("ab") as stream:
        stream.write(b"mutated during benchmark\n")


def main() -> int:
    args, _unknown = _parser().parse_known_args()

    def stop_after_checkpoint(_signum: int, _frame: object) -> None:
        if args.fixture_recovery_child_exited_marker is not None:
            args.fixture_recovery_child_exited_marker.write_text(
                "child-exited-after-forwarded-signal\n", encoding="utf-8"
            )
        raise SystemExit(130)

    signal.signal(signal.SIGINT, stop_after_checkpoint)
    signal.signal(signal.SIGTERM, stop_after_checkpoint)
    outcomes = json.loads(args.fixture_outcomes)
    if args.evaluate_checkpoint is None:
        if (
            args.resume is not None
            and args.fixture_fail_after_resume_once_marker is not None
            and not args.fixture_fail_after_resume_once_marker.exists()
        ):
            args.fixture_fail_after_resume_once_marker.write_text("failed\n", encoding="utf-8")
            return 23
        if (
            args.fixture_fail_training_once_marker is not None
            and not args.fixture_fail_training_once_marker.exists()
        ):
            args.fixture_fail_training_once_marker.write_text("failed\n", encoding="utf-8")
            return 17
        if args.fixture_fail_training_seed == args.seed:
            return 17
        assert args.checkpoint is not None
        assert args.timesteps is not None
        resumed_checkpoint = None
        if args.resume is not None:
            resumed_checkpoint = json.loads(args.resume.read_text(encoding="utf-8"))
            assert resumed_checkpoint["evidence_run_identity"] == args.evidence_run_identity
            assert resumed_checkpoint["evidence_attempt_identity"] == args.evidence_attempt_identity
        cooperative_interruption = args.fixture_interrupt_once_at_step == args.timesteps
        hard_crash = args.fixture_hard_crash_once_at_step == args.timesteps
        should_interrupt = (cooperative_interruption or hard_crash) and not (
            resumed_checkpoint or {}
        ).get("interrupted", False)
        actual_step = (
            max(1, args.timesteps // 2)
            if should_interrupt
            else args.timesteps + args.fixture_training_step_offset
        )
        execution_timesteps = args.timesteps if should_interrupt else actual_step
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        args.checkpoint.write_text(
            json.dumps(
                {
                    "format": "standalone-gradoom-ppo-v1",
                    "seed": args.seed,
                    "step": actual_step,
                    "resumed": args.resume is not None,
                    "interrupted": should_interrupt,
                    "policy_state": "fixture-policy-state",
                    "optimizer_state": "fixture-optimizer-state",
                    "rng_state": "fixture-rng-state",
                    "progress": {
                        "global_step": actual_step,
                        "rollouts": actual_step,
                        "environment_state": {"format": "fixture-live-snapshot-v1", "lanes": 1},
                        "observations": [[actual_step]],
                        "context": [[actual_step]],
                        "episode_starts": [False],
                        "dones": [False],
                        "episode_returns": [float(actual_step)],
                        "episode_lengths": [actual_step],
                        "episode_index": [0],
                    },
                    "evidence_run_identity": args.evidence_run_identity,
                    "evidence_attempt_identity": args.evidence_attempt_identity,
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
                "execution_timesteps": execution_timesteps,
                "initialization": {
                    "mode": "random",
                    "checkpoint": None,
                    "checkpoint_sha256": None,
                },
                "state_initialization": {
                    "policy_state": "resumed" if args.resume is not None else "fresh_random",
                    "optimizer_state": "resumed" if args.resume is not None else "fresh",
                },
                "evidence_binding": {
                    "run_identity": args.evidence_run_identity,
                    "attempt_identity": args.evidence_attempt_identity,
                },
            }
        ]
        if args.resume is not None:
            records.append(
                {
                    "type": "event",
                    "event": "resumed",
                    "checkpoint": str(args.resume),
                    "train/global_step": int(resumed_checkpoint["step"]),
                    "restored_state": {
                        "policy": "policy_state" in resumed_checkpoint,
                        "optimizer": "optimizer_state" in resumed_checkpoint,
                        "rng": "rng_state" in resumed_checkpoint,
                        "progress": all(
                            key in resumed_checkpoint.get("progress", {})
                            for key in (
                                "global_step",
                                "rollouts",
                                "environment_state",
                                "observations",
                                "context",
                                "episode_starts",
                                "dones",
                                "episode_returns",
                                "episode_lengths",
                                "episode_index",
                            )
                        ),
                    },
                    "evidence_binding": {
                        "run_identity": args.evidence_run_identity,
                        "attempt_identity": args.evidence_attempt_identity,
                    },
                }
            )
        records.append(
            {
                "type": "summary",
                "status": "interrupted" if should_interrupt else "completed",
                "train/global_step": actual_step,
                "requested_timesteps": args.timesteps,
                "execution_timesteps": execution_timesteps,
                "checkpoint": str(args.checkpoint),
                "training_transitions_per_second": 1000.0,
            }
        )
        _emit(args.metrics_jsonl, *records)
        _mutate_bootstrap(args.fixture_mutate_bootstrap)
        if args.fixture_mutate_trainer_code is not None:
            with args.fixture_mutate_trainer_code.open("a", encoding="utf-8") as stream:
                stream.write("\n# mutated during cohort\n")
        if should_interrupt and args.fixture_hold_after_recovery_checkpoint_marker is not None:
            args.fixture_hold_after_recovery_checkpoint_marker.write_text(
                "checkpoint-ready\n", encoding="utf-8"
            )
            time.sleep(0.75)
            if args.fixture_recovery_child_exited_marker is not None:
                args.fixture_recovery_child_exited_marker.write_text(
                    "child-exited\n", encoding="utf-8"
                )
        if should_interrupt and hard_crash:
            os._exit(17)
        return 130 if should_interrupt else 0

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
            "compatibility_killcount": float(compatibility_quality),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
