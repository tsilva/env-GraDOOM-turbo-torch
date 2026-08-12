"""Audit fixed-evaluation learnability evidence and measured training throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

EVALUATION_PROTOCOL = "standalone-gradoom-deathmatch-checkpoint-eval-v3-balanced-seed-grid"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", action="append", required=True, type=Path)
    parser.add_argument("--training-metrics", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--minimum-mean-kills", type=float, default=10.0)
    parser.add_argument("--median-mean-kills", type=float, default=10.0)
    parser.add_argument("--best-mean-kills", type=float, default=15.0)
    return parser


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: JSONL record must be an object")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _single(records: Sequence[Mapping[str, Any]], kind: str, path: Path) -> Mapping[str, Any]:
    matches = [record for record in records if record.get("type") == kind]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one {kind!r} record, found {len(matches)}")
    return matches[0]


def _evaluation(path: Path) -> dict[str, Any]:
    record = _single(_records(path), "evaluation", path)
    checkpoint = Path(str(record["checkpoint"]))
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != record["checkpoint_sha256"]:
        raise ValueError(f"{path}: checkpoint SHA-256 does not match")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError(f"{checkpoint}: checkpoint payload must be a mapping")
    parameter_groups = checkpoint_payload["optimizer_state_dict"]["param_groups"]
    actual_learning_rates = sorted({float(group["lr"]) for group in parameter_groups})
    config = record["checkpoint_config"]
    recipe = config["effective_recipe"]
    initialization = config["initialization"]
    checks = {
        "episodes_exactly_100": record["evaluation/episode/count"] == 100,
        "initialization_random": initialization["mode"] == "random"
        and initialization["checkpoint"] is None,
        "no_privileged_inputs": recipe["privileged_imitation_coef"] == 0.0,
        "protocol_fixed_balanced_grid": record["protocol"] == EVALUATION_PROTOCOL,
        "reward_native": recipe["reward_shape"] == "native-v1",
        "stochastic_actions": record["deterministic_actions"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"{path}: failed evaluation checks: {checks}")
    return {
        "actual_optimizer_learning_rates": actual_learning_rates,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(record["checkpoint_step"]),
        "evaluation_process_seconds": float(record["process_elapsed_seconds"]),
        "kills_min": float(record["evaluation/kills/min"]),
        "kills_max": float(record["evaluation/kills/max"]),
        "kills_mean": float(record["evaluation/kills/mean"]),
        "kills_median": float(record["evaluation/kills/median"]),
        "metrics": str(path),
        "protocol": str(record["protocol"]),
        "reported_learning_rate": float(recipe["learning_rate"]),
        "training_seed": int(recipe["seed"]),
        "validation": checks,
    }


def _training_metrics(path: Path) -> dict[str, Any]:
    records = _records(path)
    config = _single(records, "config", path)
    events = [record for record in records if record.get("type") == "event"]
    rollouts = [record for record in records if record.get("type") == "rollout"]
    if not rollouts:
        raise ValueError(f"{path}: no rollout records")
    summaries = [record for record in records if record.get("type") == "summary"]
    loop_seconds = sum(
        float(record["train/throughput/rollout/seconds"])
        + float(record["train/throughput/update/seconds"])
        for record in rollouts
    )
    steady_rates = [float(record["train/throughput/loop/rate"]) for record in rollouts[1:]]
    recipe = config["effective_recipe"]
    initialization = config["initialization"]
    resumed_events = [record for record in events if record.get("event") == "resumed"]
    if len(resumed_events) > 1:
        raise ValueError(f"{path}: expected at most one resumed event")
    external_initialization_events = [
        record
        for record in events
        if record.get("event") in {"initialized", "policy_initialized"}
    ]
    checks = {
        "initialization_declared_random": initialization["mode"] == "random"
        and initialization["checkpoint"] is None,
        "no_external_initialization_event": not external_initialization_events,
        "no_privileged_inputs": recipe["privileged_imitation_coef"] == 0.0,
        "reward_native": recipe["reward_shape"] == "native-v1",
        "standalone": config["standalone"] is True,
    }
    if not all(checks.values()):
        raise ValueError(f"{path}: failed training checks: {checks}")
    resumed_event = resumed_events[0] if resumed_events else None
    produced_checkpoints = [
        str(Path(str(record["checkpoint"])).resolve())
        for record in events
        if record.get("event") == "checkpoint_saved"
    ]
    if summaries:
        produced_checkpoints.append(
            str(Path(str(summaries[-1]["checkpoint"])).resolve())
        )
    return {
        "end_step": int(rollouts[-1]["train/global_step"]),
        "measured_loop_seconds": loop_seconds,
        "metrics": str(path),
        "process_seconds": (
            float(summaries[-1]["process_elapsed_seconds"]) if summaries else None
        ),
        "reported_learning_rate": float(recipe["learning_rate"]),
        "rollout_shape": [int(recipe["num_envs"]), int(recipe["n_steps"])],
        "seed": int(recipe["seed"]),
        "start_step": (
            int(resumed_event["train/global_step"]) if resumed_event is not None else 0
        ),
        "resumed_from_checkpoint": (
            str(Path(str(resumed_event["checkpoint"])).resolve())
            if resumed_event is not None
            else None
        ),
        "produced_checkpoints": sorted(set(produced_checkpoints)),
        "steady_transitions_per_second": statistics.median(steady_rates),
        "summary_present": bool(summaries),
        "validation": checks,
    }


def _lineage(
    evaluation: Mapping[str, Any], training: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    producers: dict[str, Mapping[str, Any]] = {}
    for segment in training:
        for checkpoint in segment["produced_checkpoints"]:
            if checkpoint in producers:
                raise ValueError(f"multiple training logs produced {checkpoint}")
            producers[checkpoint] = segment

    checkpoint = str(Path(str(evaluation["checkpoint"])).resolve())
    segments: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    while True:
        if checkpoint in seen:
            raise ValueError(f"cycle in checkpoint lineage at {checkpoint}")
        seen.add(checkpoint)
        try:
            segment = producers[checkpoint]
        except KeyError as error:
            raise ValueError(f"no training log produces {checkpoint}") from error
        segments.append(segment)
        parent = segment["resumed_from_checkpoint"]
        if parent is None:
            break
        checkpoint = str(parent)

    segments.reverse()
    seed = int(evaluation["training_seed"])
    if any(int(segment["seed"]) != seed for segment in segments):
        raise ValueError(f"seed {seed}: checkpoint lineage crosses training seeds")
    process_seconds_complete = all(segment["process_seconds"] is not None for segment in segments)
    observed_training_seconds = sum(
        float(segment["process_seconds"])
        if segment["process_seconds"] is not None
        else float(segment["measured_loop_seconds"])
        for segment in segments
    )
    return {
        "checkpoint": str(evaluation["checkpoint"]),
        "evaluation_process_seconds": float(evaluation["evaluation_process_seconds"]),
        "observed_training_seconds": observed_training_seconds,
        "observed_training_plus_evaluation_seconds": observed_training_seconds
        + float(evaluation["evaluation_process_seconds"]),
        "process_seconds_complete_for_all_segments": process_seconds_complete,
        "random_initialization_root": segments[0]["resumed_from_checkpoint"] is None,
        "segments": [segment["metrics"] for segment in segments],
        "training_seed": seed,
    }


def main() -> int:
    args = _parser().parse_args()
    evaluations = [_evaluation(path.resolve()) for path in args.evaluation]
    training = [_training_metrics(path.resolve()) for path in args.training_metrics]
    lineages = [_lineage(evaluation, training) for evaluation in evaluations]
    means = [record["kills_mean"] for record in evaluations]
    seeds = {record["training_seed"] for record in evaluations}
    acceptance = {
        "at_least_minimum_distinct_training_seeds": len(seeds) >= args.minimum_seeds,
        "every_seed_mean_at_or_above_minimum_threshold": all(
            mean >= args.minimum_mean_kills for mean in means
        ),
        "at_least_one_seed_mean_at_or_above_best_threshold": max(means)
        >= args.best_mean_kills,
        "median_seed_mean_at_or_above_threshold": statistics.median(means)
        >= args.median_mean_kills,
        "all_checkpoints_trace_to_random_training_roots": all(
            lineage["random_initialization_root"] for lineage in lineages
        ),
    }
    output = {
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "evaluation_count": len(evaluations),
        "evaluations": sorted(evaluations, key=lambda record: record["training_seed"]),
        "best_seed_mean_kills": max(means),
        "checkpoint_lineages": sorted(lineages, key=lambda record: record["training_seed"]),
        "median_seed_mean_kills": statistics.median(means),
        "thresholds": {
            "best_seed_mean_kills": args.best_mean_kills,
            "median_seed_mean_kills": args.median_mean_kills,
            "minimum_seed_mean_kills": args.minimum_mean_kills,
            "minimum_distinct_training_seeds": args.minimum_seeds,
        },
        "training_metrics": training,
        "training_steady_throughput_median": (
            statistics.median(record["steady_transitions_per_second"] for record in training)
            if training
            else None
        ),
    }
    serialized = json.dumps(output, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    return 0 if output["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
