from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from typing import Any

from .benchmark import (
    _EVALUATION_EPISODES,
    _TRAINER_CONTRACT,
    _UINT32_MAX,
    _fsync_file,
    _read_jsonl,
    _required_mapping,
    _run_process,
    _seed_list,
    _validate_evaluation_records,
    _validate_runtime_assets,
    _validate_trainer,
    _write_seed_file,
)
from .report import (
    EvidenceError,
    _canonical_sha256,
    _load_manifest,
    _parse_json_document,
    _resolve_evidence_path,
    _sha256_bytes,
    _validate_code_provenance,
    _validate_declared_inputs,
    _validate_schema_version,
    _validate_sha256,
)
from .wad_profile import validate_wad_profile

_WORKFLOW = "fixed_time_training_diagnostic"
_MAX_TRAINING_STEPS = (1 << 63) - 1
_TIMING_RULES = {
    "clock": "monotonic_wall_clock",
    "start": "before_attempt_setup_and_public_subprocess_start",
    "stop": "after_trainer_exit_and_durable_training_evidence_write",
    "device_synchronization": "before_and_after_measured_gpu_work",
    "includes": [
        "evaluation_seed_manifest_write",
        "interpreter_and_module_import_startup",
        "recurring_initialization",
        "per_process_or_uncached_compilation",
        "warmup",
        "training",
        "durable_checkpoint_write",
        "durable_training_metrics_write",
    ],
    "excludes": ["final_held_out_evaluation"],
}


def _positive_finite(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise EvidenceError(f"{field} must be finite and positive")
    return float(value)


def _load_matching_benchmark(
    value: object,
    *,
    manifest_directory: Path,
) -> tuple[dict[str, Any], Path, str]:
    declaration = _required_mapping(value, "diagnostic.matching_benchmark_report")
    raw_path = declaration.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise EvidenceError(
            "diagnostic.matching_benchmark_report.path must be a non-whitespace path"
        )
    expected_sha256 = _validate_sha256(
        declaration.get("sha256"),
        "diagnostic.matching_benchmark_report.sha256",
    )
    path = _resolve_evidence_path(Path(raw_path), base_directory=manifest_directory)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"cannot read matching benchmark report: {path}") from error
    actual_sha256 = _sha256_bytes(payload)
    if actual_sha256 != expected_sha256:
        raise EvidenceError(
            "diagnostic.matching_benchmark_report SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    report = _parse_json_document(payload, document="matching benchmark report")
    if not isinstance(report, dict):
        raise EvidenceError("matching benchmark report must be a JSON object")
    _validate_schema_version(report.get("schema_version"), document="matching benchmark report")
    if report.get("workflow") not in {
        "development_training_benchmark",
        "primary_training_benchmark",
    }:
        raise EvidenceError("matching benchmark report is not a training benchmark")
    if not isinstance(report.get("run_identity"), str):
        raise EvidenceError("matching benchmark report run_identity is missing")
    evidence_index = _required_mapping(
        report.get("evidence_index"),
        "matching benchmark report evidence_index",
    )
    entries = evidence_index.get("entries")
    if not isinstance(entries, list):
        raise EvidenceError("matching benchmark report evidence_index.entries must be an array")
    if evidence_index.get("sha256") != _canonical_sha256(entries, document="report"):
        raise EvidenceError("matching benchmark report evidence index SHA-256 mismatch")
    declared_inputs = report.get("declared_inputs")
    protocol = report.get("benchmark_protocol")
    if not isinstance(declared_inputs, list) or not isinstance(protocol, dict):
        raise EvidenceError("matching benchmark report identity-bearing fields are missing")
    normalized_inputs: list[dict[str, Any]] = []
    for index, item in enumerate(declared_inputs):
        if not isinstance(item, dict):
            raise EvidenceError(f"matching benchmark report declared_inputs[{index}] is invalid")
        name = item.get("name")
        digest = item.get("sha256")
        if not isinstance(name, str):
            raise EvidenceError(
                f"matching benchmark report declared_inputs[{index}].name is invalid"
            )
        _validate_sha256(
            digest,
            f"matching benchmark report declared_inputs[{index}].sha256",
        )
        normalized_inputs.append({"name": name, "sha256": digest})
    expected_identity = _canonical_sha256(
        {
            "schema_version": report["schema_version"],
            "workflow": report["workflow"],
            "evidence_level": report.get("evidence_level"),
            "fixture": report.get("fixture"),
            "code_provenance": report.get("code_provenance"),
            "declared_inputs": sorted(normalized_inputs, key=lambda item: item["name"]),
            "benchmark_protocol": protocol,
        },
        document="matching benchmark report",
    )
    if report["run_identity"] != expected_identity:
        raise EvidenceError("matching benchmark report run_identity does not match its protocol")
    return report, path, actual_sha256


def _validate_diagnostic(
    manifest: dict[str, Any],
    *,
    manifest_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    _validate_schema_version(manifest.get("schema_version"), document="manifest")
    if manifest.get("workflow") != _WORKFLOW:
        raise EvidenceError(f"this command path requires workflow {_WORKFLOW}")
    if manifest.get("evidence_level") not in {"development", "formal"}:
        raise EvidenceError("fixed-time diagnostic requires development or formal evidence")
    if type(manifest.get("fixture")) is not bool:
        raise EvidenceError("fixture is required and must be a boolean")
    code_provenance = _validate_code_provenance(manifest.get("code_provenance"))
    diagnostic = _required_mapping(manifest.get("diagnostic"), "diagnostic")
    budget = _positive_finite(
        diagnostic.get("reusable_time_budget_seconds"),
        "diagnostic.reusable_time_budget_seconds",
    )
    training_seeds = _seed_list(
        diagnostic.get("training_seeds"),
        "diagnostic.training_seeds",
    )
    evaluation_seeds = _seed_list(
        diagnostic.get("evaluation_episode_seeds"),
        "diagnostic.evaluation_episode_seeds",
        exact_count=_EVALUATION_EPISODES,
    )
    evaluation_action_seed = diagnostic.get("evaluation_action_seed")
    if type(evaluation_action_seed) is not int or not 0 <= evaluation_action_seed <= _UINT32_MAX:
        raise EvidenceError(
            f"diagnostic.evaluation_action_seed must be an integer in [0, {_UINT32_MAX}]"
        )
    recipe = _validate_trainer(diagnostic.get("recipe"))
    timing_rules = diagnostic.get("timing_rules")
    if timing_rules != _TIMING_RULES:
        raise EvidenceError(
            "diagnostic.timing_rules must equal the fixed-time reusable-run timing protocol"
        )
    artifacts_directory = diagnostic.get("artifacts_directory")
    if not isinstance(artifacts_directory, str) or not artifacts_directory.strip():
        raise EvidenceError("diagnostic.artifacts_directory must be a non-whitespace path")
    benchmark, benchmark_path, benchmark_sha256 = _load_matching_benchmark(
        diagnostic.get("matching_benchmark_report"),
        manifest_directory=manifest_directory,
    )
    if benchmark.get("evidence_level") != manifest["evidence_level"]:
        raise EvidenceError("fixed-time diagnostic evidence level does not match benchmark")
    if benchmark.get("fixture") is not manifest["fixture"]:
        raise EvidenceError("fixed-time diagnostic fixture status does not match benchmark")
    if benchmark.get("code_provenance") != code_provenance:
        raise EvidenceError("fixed-time diagnostic code provenance does not match benchmark")
    benchmark_protocol = _required_mapping(
        benchmark.get("benchmark_protocol"),
        "matching benchmark report benchmark_protocol",
    )
    comparisons = {
        "training seeds": (training_seeds, benchmark_protocol.get("training_seeds")),
        "evaluation episode seeds": (
            evaluation_seeds,
            benchmark_protocol.get("evaluation_episode_seeds"),
        ),
        "evaluation action seed": (
            evaluation_action_seed,
            benchmark_protocol.get("evaluation_action_seed"),
        ),
        "recipe": (recipe, benchmark_protocol.get("trainer")),
    }
    for name, (actual, expected) in comparisons.items():
        if actual != expected:
            raise EvidenceError(f"fixed-time diagnostic {name} do not match benchmark")
    return (
        {
            "code_provenance": code_provenance,
            "reusable_time_budget_seconds": budget,
            "training_seeds": training_seeds,
            "evaluation_episode_seeds": evaluation_seeds,
            "evaluation_action_seed": evaluation_action_seed,
            "recipe": recipe,
            "timing_rules": timing_rules,
            "artifacts_directory": artifacts_directory,
        },
        benchmark,
        benchmark_path,
        benchmark_sha256,
    )


def _training_failure(
    *,
    seed: int,
    process: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "phase": "training",
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _validate_training(
    records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    budget: float,
    deadline: float,
    manifest_directory: Path,
    wad_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any], int, int]:
    configs = [record for record in records if record.get("type") == "config"]
    summaries = [record for record in records if record.get("type") == "summary"]
    if len(configs) != 1 or len(summaries) != 1:
        raise EvidenceError("fixed-time training must emit exactly one config and summary")
    config = configs[0]
    summary = summaries[0]
    if config.get("contract") != _TRAINER_CONTRACT or config.get("operation") != "train":
        raise EvidenceError("fixed-time diagnostic did not use the standalone trainer")
    if config.get("initialization") != {
        "mode": "random",
        "checkpoint": None,
        "checkpoint_sha256": None,
    }:
        raise EvidenceError("fixed-time training must use fresh random policy state")
    if config.get("state_initialization") != {
        "policy_state": "fresh_random",
        "optimizer_state": "fresh",
    }:
        raise EvidenceError("fixed-time training must use fresh policy and optimizer state")
    if float(config.get("reusable_time_budget_seconds", -1.0)) != budget:
        raise EvidenceError("training config did not bind the reusable-time budget")
    if float(config.get("reusable_time_deadline_monotonic", -1.0)) != deadline:
        raise EvidenceError("training config did not bind the outer monotonic deadline")
    if summary.get("status") != "completed" or summary.get("stop_reason") != (
        "reusable_time_budget"
    ):
        raise EvidenceError("training did not complete at the reusable-time budget")
    if float(summary.get("reusable_time_budget_seconds", -1.0)) != budget:
        raise EvidenceError("training summary did not bind the reusable-time budget")
    if float(summary.get("reusable_time_deadline_monotonic", -1.0)) != deadline:
        raise EvidenceError("training summary did not bind the outer monotonic deadline")
    transitions = summary.get("training_transitions")
    frame_skip = summary.get("frame_skip")
    if type(transitions) is not int or transitions < 0:
        raise EvidenceError("training summary training_transitions must be non-negative")
    if type(frame_skip) is not int or frame_skip <= 0:
        raise EvidenceError("training summary frame_skip must be a positive integer")
    recorded_checkpoint = summary.get("checkpoint")
    if not isinstance(recorded_checkpoint, str) or (
        _resolve_evidence_path(Path(recorded_checkpoint), base_directory=manifest_directory)
        != checkpoint
    ):
        raise EvidenceError("fixed-time training reported a different checkpoint path")
    _validate_runtime_assets(summary, phase="training", wad_profile=wad_profile)
    return summary, transitions, frame_skip


def _run_diagnostic_attempt(
    *,
    seed: int,
    protocol: dict[str, Any],
    run_directory: Path,
    manifest_directory: Path,
    evidence_entries: list[dict[str, str]],
    wad_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    attempt_started = time.monotonic()
    reusable_time_deadline = attempt_started + protocol["reusable_time_budget_seconds"]
    attempt_directory = run_directory / f"seed-{seed}"
    attempt_directory.mkdir()
    seed_file = attempt_directory / "evaluation-seeds.json"
    seed_sha256 = _write_seed_file(seed_file, protocol["evaluation_episode_seeds"])
    evidence_entries.append({"name": f"seed-{seed}-evaluation-seeds", "sha256": seed_sha256})
    checkpoint = attempt_directory / "final-checkpoint.pt"
    training_metrics = attempt_directory / "training.jsonl"
    base_command = [*protocol["recipe"]["command"], *protocol["recipe"]["arguments"]]
    training_command = [
        *base_command,
        "--seed",
        str(seed),
        "--timesteps",
        str(_MAX_TRAINING_STEPS),
        "--reusable-time-budget-seconds",
        str(protocol["reusable_time_budget_seconds"]),
        "--reusable-time-deadline-monotonic",
        str(reusable_time_deadline),
        "--checkpoint",
        str(checkpoint),
        "--metrics-jsonl",
        str(training_metrics),
    ]
    training_process = _run_process(training_command, cwd=manifest_directory)
    if training_process.returncode != 0:
        failure = _training_failure(seed=seed, process=training_process)
        return {
            "seed": seed,
            "status": "failed",
            "final_mean_player_killcount": None,
            "episodes": [],
            "throughput": None,
            "failures": [failure],
            "process_elapsed_seconds": time.monotonic() - attempt_started,
        }
    try:
        training_records = _read_jsonl(training_metrics, phase="fixed-time training")
        training, transitions, frame_skip = _validate_training(
            training_records,
            checkpoint=checkpoint,
            budget=protocol["reusable_time_budget_seconds"],
            deadline=reusable_time_deadline,
            manifest_directory=manifest_directory,
            wad_profile=wad_profile,
        )
        checkpoint_sha256 = _fsync_file(checkpoint, field="fixed-time checkpoint")
        training_metrics_sha256 = _fsync_file(
            training_metrics,
            field="fixed-time training metrics",
        )
        reusable_time_elapsed = time.monotonic() - attempt_started
        if reusable_time_elapsed < protocol["reusable_time_budget_seconds"]:
            raise EvidenceError("training stopped before the outer reusable-time budget")
    except (EvidenceError, TypeError, ValueError) as error:
        failure = {"seed": seed, "phase": "training_evidence", "message": str(error)}
        return {
            "seed": seed,
            "status": "failed",
            "final_mean_player_killcount": None,
            "episodes": [],
            "throughput": None,
            "failures": [failure],
            "process_elapsed_seconds": time.monotonic() - attempt_started,
        }
    evidence_entries.extend(
        (
            {"name": f"seed-{seed}-checkpoint", "sha256": checkpoint_sha256},
            {"name": f"seed-{seed}-training-metrics", "sha256": training_metrics_sha256},
        )
    )
    evaluation_metrics = attempt_directory / "evaluation.jsonl"
    evaluation_command = [
        *base_command,
        "--evaluate-checkpoint",
        str(checkpoint),
        "--evaluation-episodes",
        str(_EVALUATION_EPISODES),
        "--evaluation-seeds-file",
        str(seed_file),
        "--evaluation-seed",
        str(protocol["evaluation_action_seed"]),
        "--evaluation-stochastic",
        "--metrics-jsonl",
        str(evaluation_metrics),
    ]
    evaluation_process = _run_process(evaluation_command, cwd=manifest_directory)
    if evaluation_process.returncode != 0:
        failure = {
            "seed": seed,
            "phase": "evaluation",
            "returncode": evaluation_process.returncode,
            "stdout": evaluation_process.stdout,
            "stderr": evaluation_process.stderr,
        }
        return {
            "seed": seed,
            "status": "failed",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "final_mean_player_killcount": None,
            "episodes": [],
            "throughput": None,
            "failures": [failure],
            "process_elapsed_seconds": time.monotonic() - attempt_started,
        }
    try:
        evaluation_records = _read_jsonl(evaluation_metrics, phase="fixed-time evaluation")
        evaluation, episodes, mean_player, mean_compatibility = _validate_evaluation_records(
            evaluation_records,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            episode_seeds=protocol["evaluation_episode_seeds"],
            evaluation_action_seed=protocol["evaluation_action_seed"],
            manifest_directory=manifest_directory,
            wad_profile=wad_profile,
        )
        evaluation_metrics_sha256 = _fsync_file(
            evaluation_metrics,
            field="fixed-time evaluation metrics",
        )
    except EvidenceError as error:
        failure = {"seed": seed, "phase": "evaluation_evidence", "message": str(error)}
        return {
            "seed": seed,
            "status": "failed",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "final_mean_player_killcount": None,
            "episodes": [],
            "throughput": None,
            "failures": [failure],
            "process_elapsed_seconds": time.monotonic() - attempt_started,
        }
    evidence_entries.append(
        {"name": f"seed-{seed}-evaluation-metrics", "sha256": evaluation_metrics_sha256}
    )
    simulated_tics = transitions * frame_skip
    return {
        "seed": seed,
        "status": "completed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "final_mean_player_killcount": mean_player,
        "mean_killcount": mean_compatibility,
        "training": training,
        "evaluation": evaluation,
        "episodes": episodes,
        "throughput": {
            "simulated_tics_per_second": simulated_tics / reusable_time_elapsed,
            "transitions_per_second": transitions / reusable_time_elapsed,
            "timer": {
                "elapsed_seconds": reusable_time_elapsed,
                "source": "public_evidence_subprocess",
                "rules": protocol["timing_rules"],
            },
            "workload": {
                "frame_skip": frame_skip,
                "simulated_tics": simulated_tics,
                "transitions": transitions,
            },
        },
        "failures": [],
        "process_elapsed_seconds": time.monotonic() - attempt_started,
    }


def _generated_artifact_specs(
    run_directory: Path,
    seeds: list[int],
) -> list[tuple[int, str, Path]]:
    return [
        (seed, f"seed-{seed}-{kind}", run_directory / f"seed-{seed}" / filename)
        for seed in seeds
        for kind, filename in (
            ("evaluation-seeds", "evaluation-seeds.json"),
            ("checkpoint", "final-checkpoint.pt"),
            ("training-metrics", "training.jsonl"),
            ("evaluation-metrics", "evaluation.jsonl"),
        )
    ]


def _validate_unique_evidence_entries(entries: list[dict[str, str]]) -> None:
    names: set[str] = set()
    for index, entry in enumerate(entries):
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise EvidenceError(f"evidence entry {index} has an invalid name")
        if name in names:
            raise EvidenceError(f"evidence index contains duplicate name {name!r}")
        names.add(name)
        _validate_sha256(entry.get("sha256"), f"evidence entry {name!r}.sha256")


def _reconcile_generated_artifacts(
    *,
    attempts: list[dict[str, Any]],
    specs: list[tuple[int, str, Path]],
    evidence_entries: list[dict[str, str]],
) -> list[dict[str, Any]]:
    _validate_unique_evidence_entries(evidence_entries)
    entries = {entry["name"]: entry["sha256"] for entry in evidence_entries}
    attempts_by_seed = {attempt["seed"]: attempt for attempt in attempts}
    artifacts: list[dict[str, Any]] = []
    for seed, name, path in specs:
        if not path.is_file():
            if attempts_by_seed[seed]["status"] == "completed":
                failure = {
                    "seed": seed,
                    "phase": "artifact_evidence",
                    "message": f"generated artifact {name} is missing",
                }
                attempts_by_seed[seed]["failures"].append(failure)
                attempts_by_seed[seed]["status"] = "evidence_failed"
            continue
        actual_sha256 = _sha256_bytes(path.read_bytes())
        indexed_sha256 = entries.get(name)
        matched = actual_sha256 == indexed_sha256
        artifacts.append(
            {
                "name": name,
                "path": str(path),
                "sha256": actual_sha256,
                "evidence_entry_sha256": indexed_sha256,
                "status": "matched" if matched else "mismatched",
            }
        )
        if not matched:
            failure = {
                "seed": seed,
                "phase": "artifact_evidence",
                "message": (
                    f"generated artifact {name} SHA-256 changed or has no unique evidence entry"
                ),
            }
            attempts_by_seed[seed]["failures"].append(failure)
            attempts_by_seed[seed]["status"] = "evidence_failed"
    return artifacts


def build_fixed_time_diagnostic_report(manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_payload = _load_manifest(manifest_path)
    validated, benchmark, benchmark_path, benchmark_sha256 = _validate_diagnostic(
        manifest,
        manifest_directory=manifest_path.parent,
    )
    declared_inputs = _validate_declared_inputs(
        manifest.get("declared_inputs"),
        base_directory=manifest_path.parent,
    )
    generated_specs = _generated_artifact_specs(Path(), validated["training_seeds"])
    reserved_names = {
        "manifest",
        "matching-benchmark-report",
        *(name for _seed, name, _path in generated_specs),
    }
    for declared_input in declared_inputs:
        if declared_input["name"] in reserved_names:
            raise EvidenceError(f"declared input name {declared_input['name']!r} is reserved")
    evidence_entries = [
        {"name": "manifest", "sha256": _sha256_bytes(manifest_payload)},
        {"name": "matching-benchmark-report", "sha256": benchmark_sha256},
    ]
    for declared_input in declared_inputs:
        input_path = _resolve_evidence_path(
            Path(declared_input["path"]),
            base_directory=manifest_path.parent,
        )
        try:
            actual_sha256 = _sha256_bytes(input_path.read_bytes())
        except OSError as error:
            raise EvidenceError(f"cannot read declared input {declared_input['name']!r}") from error
        if actual_sha256 != declared_input["sha256"]:
            raise EvidenceError(f"declared input {declared_input['name']!r} SHA-256 mismatch")
        evidence_entries.append({"name": declared_input["name"], "sha256": actual_sha256})
    wad_profile = None
    if "wad_profile" in manifest:
        wad_profile, wad_entries = validate_wad_profile(
            manifest["wad_profile"],
            base_directory=manifest_path.parent,
        )
        evidence_entries.extend(wad_entries)
        if wad_profile["status"] != "matched":
            raise EvidenceError("fixed-time diagnostic WAD profile did not match")
    elif not manifest["fixture"]:
        raise EvidenceError("non-fixture fixed-time diagnostic requires wad_profile")
    benchmark_binding = benchmark["benchmark_protocol"].get("wad_profile_binding_sha256")
    diagnostic_binding = None if wad_profile is None else wad_profile["binding_sha256"]
    if diagnostic_binding != benchmark_binding:
        raise EvidenceError("fixed-time diagnostic WAD profile does not match benchmark")
    protocol = {
        "reusable_time_budget_seconds": validated["reusable_time_budget_seconds"],
        "training_seeds": validated["training_seeds"],
        "evaluation_episode_seeds": validated["evaluation_episode_seeds"],
        "evaluation_action_seed": validated["evaluation_action_seed"],
        "evaluation": {
            "episodes": _EVALUATION_EPISODES,
            "signal": "player_killcount",
            "stochastic_actions": True,
        },
        "recipe": validated["recipe"],
        "recipe_sha256": _canonical_sha256(validated["recipe"], document="manifest"),
        "timing_rules": validated["timing_rules"],
        "wad_profile_binding_sha256": diagnostic_binding,
    }
    identity_payload = {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": manifest["evidence_level"],
        "fixture": manifest["fixture"],
        "code_provenance": validated["code_provenance"],
        "matching_benchmark_report_sha256": benchmark_sha256,
        "declared_inputs": sorted(
            ({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
            key=lambda item: item["name"],
        ),
        "diagnostic_protocol": protocol,
    }
    run_identity = _canonical_sha256(identity_payload, document="manifest")
    artifacts_root = _resolve_evidence_path(
        Path(validated["artifacts_directory"]),
        base_directory=manifest_path.parent,
    )
    run_directory = artifacts_root / run_identity
    generated_specs = _generated_artifact_specs(
        run_directory,
        validated["training_seeds"],
    )
    _validate_unique_evidence_entries(evidence_entries)
    try:
        run_directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise EvidenceError(
            f"diagnostic artifact directory already exists; refusing to overwrite: {run_directory}"
        ) from error
    attempts = [
        _run_diagnostic_attempt(
            seed=seed,
            protocol=protocol,
            run_directory=run_directory,
            manifest_directory=manifest_path.parent,
            evidence_entries=evidence_entries,
            wad_profile=wad_profile,
        )
        for seed in validated["training_seeds"]
    ]
    generated_artifacts = _reconcile_generated_artifacts(
        attempts=attempts,
        specs=generated_specs,
        evidence_entries=evidence_entries,
    )
    failures = [failure for attempt in attempts for failure in attempt["failures"]]
    completed = all(attempt["status"] == "completed" for attempt in attempts)
    benchmark_claim_eligible = benchmark.get("claim_eligible") is True
    return {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": manifest["evidence_level"],
        "fixture": manifest["fixture"],
        "authoritative": False,
        "status": "completed" if completed else "failed",
        "claim_eligible": False,
        "claim_reasons": [
            {
                "code": "diagnostic_evidence",
                "message": "Fixed-time evidence is diagnostic and cannot alter benchmark passage.",
            }
        ],
        "run_identity": run_identity,
        "code_provenance": validated["code_provenance"],
        "declared_inputs": [
            *declared_inputs,
            {
                "name": "matching-benchmark-report",
                "path": str(benchmark_path),
                "sha256": benchmark_sha256,
            },
        ],
        "diagnostic_protocol": protocol,
        "wad_profile": wad_profile,
        "diagnostics": {
            "fixed_time": {
                "status": "completed" if completed else "failed",
                "affects_passage": False,
                "matching_benchmark": {
                    "matched": True,
                    "path": str(benchmark_path),
                    "sha256": benchmark_sha256,
                    "run_identity": benchmark["run_identity"],
                    "passage": {"status": benchmark.get("status"), "unchanged": True},
                },
                "attempts": attempts,
            }
        },
        "failures": failures,
        "public_performance_evidence": {
            "complete": completed and benchmark_claim_eligible and not manifest["fixture"],
            "reason": (
                "complete"
                if completed and benchmark_claim_eligible and not manifest["fixture"]
                else "matching_fixed_time_diagnostic_failed"
                if not completed
                else "matching_benchmark_is_not_claim_eligible"
            ),
        },
        "generated_artifacts": generated_artifacts,
        "evidence_index": {
            "algorithm": "sha256",
            "entries": evidence_entries,
            "sha256": _canonical_sha256(evidence_entries, document="report"),
        },
    }
