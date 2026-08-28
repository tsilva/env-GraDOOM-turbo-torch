from __future__ import annotations

import json
import math
import os
import stat
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

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

_WORKFLOW = "development_training_benchmark"
_TRAINER_CONTRACT = "standalone-gradoom-deathmatch-ppo-v2"
_DEFAULT_TRAINING_SEED = 123
_UINT32_MAX = (1 << 32) - 1
_QUALITY_THRESHOLD = 30.0
_EVALUATION_EPISODES = 100
_CONTROLLED_ARGUMENTS = {
    "--checkpoint",
    "--checkpoint-every-rollouts",
    "--config-only",
    "--evaluate-checkpoint",
    "--evaluation-episodes",
    "--evaluation-seed",
    "--evaluation-seeds-file",
    "--evaluation-stochastic",
    "--evidence-attempt-identity",
    "--evidence-run-identity",
    "--initialize-from",
    "--metrics-jsonl",
    "--no-evaluation-stochastic",
    "--resume",
    "--seed",
    "--timesteps",
}

_RESTORABLE_STATE = {
    "policy": True,
    "optimizer": True,
    "rng": True,
    "progress": True,
}


def _attempt_journal_payload(attempt: dict[str, Any], *, run_identity: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_identity": run_identity,
        "attempt_identity": attempt.get("attempt_identity"),
        "seed": attempt.get("seed"),
        "status": attempt.get("status"),
        "reusable_elapsed_seconds": attempt.get("reusable_elapsed_seconds"),
        "cold_start": attempt.get("cold_start"),
        "checkpoint": attempt.get("checkpoint"),
        "checkpoint_sha256": attempt.get("checkpoint_sha256"),
        "outcomes_sha256": _canonical_sha256(
            attempt.get("outcomes"),
            document="benchmark attempt outcomes",
        ),
        "failures_sha256": _canonical_sha256(
            attempt.get("failures"),
            document="benchmark attempt failures",
        ),
        "recovery_sha256": _canonical_sha256(
            {
                "recovery": attempt.get("recovery"),
                "recovery_history": attempt.get("recovery_history"),
                "recovery_journal": attempt.get("recovery_journal"),
            },
            document="benchmark attempt recovery",
        ),
    }


def _required_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} is required and must be an object")
    return value


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise EvidenceError(f"{field} must be a positive integer")
    return value


def _seed_list(value: object, field: str, *, exact_count: int | None = None) -> list[int]:
    if not isinstance(value, list):
        raise EvidenceError(f"{field} must be an array")
    if exact_count is not None and len(value) != exact_count:
        raise EvidenceError(f"{field} must contain exactly {exact_count} seeds")
    if not value:
        raise EvidenceError(f"{field} must contain at least one seed")
    seeds: list[int] = []
    for index, seed in enumerate(value):
        if type(seed) is not int or not 0 <= seed <= _UINT32_MAX:
            raise EvidenceError(f"{field}[{index}] must be an integer in [0, {_UINT32_MAX}]")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise EvidenceError(f"{field} must be unique")
    return seeds


def _string_array(value: object, field: str, *, non_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (non_empty and not value):
        qualifier = "a non-empty array" if non_empty else "an array"
        raise EvidenceError(f"{field} must be {qualifier} of strings")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise EvidenceError(f"{field}[{index}] must be a non-empty string")
    return value


def _validate_trainer(value: object) -> dict[str, list[str]]:
    trainer = _required_mapping(value, "benchmark.trainer")
    command = _string_array(trainer.get("command"), "benchmark.trainer.command", non_empty=True)
    arguments = _string_array(trainer.get("arguments", []), "benchmark.trainer.arguments")
    for argument in arguments:
        controlled = next(
            (
                option
                for option in _CONTROLLED_ARGUMENTS
                if argument == option or argument.startswith(f"{option}=")
            ),
            None,
        )
        if controlled is not None:
            raise EvidenceError(
                f"benchmark.trainer.arguments must not control {controlled!r}; "
                "the evidence command owns cold-start, timing, checkpoint, and evaluation flags"
            )
    return {"command": command, "arguments": arguments}


def _validate_certificate(value: object) -> dict[str, Any]:
    if value is None:
        return {
            "available": False,
            "reason": "No current parity certificate was declared.",
        }
    certificate = _required_mapping(value, "benchmark.parity_certificate")
    if type(certificate.get("available")) is not bool:
        raise EvidenceError("benchmark.parity_certificate.available must be a boolean")
    if not certificate["available"]:
        reason = certificate.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise EvidenceError(
                "benchmark.parity_certificate.reason must explain an unavailable certificate"
            )
    return certificate


_BOOTSTRAP_STATE_FIELDS = {
    "learned",
    "optimizer",
    "rollout",
    "seed_specific",
    "candidate_specific",
}


def _validate_bootstrap_artifacts(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvidenceError("benchmark.bootstrap_artifacts must be an array")
    artifacts: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_artifact in enumerate(value):
        field = f"benchmark.bootstrap_artifacts[{index}]"
        artifact = _required_mapping(raw_artifact, field)
        expected_fields = {
            "name",
            "path",
            "sha256",
            "creation_elapsed_seconds",
            "creation_protocol",
            "reuse_conditions",
            "persistent",
            "run_independent",
            "reused_unchanged",
            "contains_state",
        }
        undeclared_fields = sorted(set(artifact) - expected_fields)
        if undeclared_fields:
            formatted = ", ".join(repr(name) for name in undeclared_fields)
            raise EvidenceError(f"{field} has undeclared fields: {formatted}")
        name = artifact.get("name")
        if not isinstance(name, str) or not name.strip():
            raise EvidenceError(f"{field}.name must be a non-whitespace string")
        if name in names:
            raise EvidenceError(f"{field}.name {name!r} is duplicated")
        names.add(name)
        path = artifact.get("path")
        if not isinstance(path, str) or not path.strip():
            raise EvidenceError(f"{field}.path must be a non-whitespace path")
        sha256 = _validate_sha256(artifact.get("sha256"), f"{field}.sha256")
        creation_elapsed = artifact.get("creation_elapsed_seconds")
        if (
            type(creation_elapsed) not in (int, float)
            or not math.isfinite(float(creation_elapsed))
            or float(creation_elapsed) < 0.0
        ):
            raise EvidenceError(f"{field}.creation_elapsed_seconds must be finite and non-negative")
        creation_protocol = artifact.get("creation_protocol")
        if not isinstance(creation_protocol, str) or not creation_protocol.strip():
            raise EvidenceError(f"{field}.creation_protocol must be a non-whitespace string")
        reuse_conditions = _string_array(
            artifact.get("reuse_conditions"),
            f"{field}.reuse_conditions",
            non_empty=True,
        )
        for required_true in ("persistent", "run_independent", "reused_unchanged"):
            if artifact.get(required_true) is not True:
                raise EvidenceError(f"{field}.{required_true} must be true for excluded work")
        contains_state = _required_mapping(
            artifact.get("contains_state"), f"{field}.contains_state"
        )
        missing_state = sorted(_BOOTSTRAP_STATE_FIELDS - set(contains_state))
        undeclared_state = sorted(set(contains_state) - _BOOTSTRAP_STATE_FIELDS)
        if missing_state or undeclared_state:
            details = []
            if missing_state:
                details.append(f"missing {', '.join(missing_state)}")
            if undeclared_state:
                details.append(f"unknown {', '.join(undeclared_state)}")
            raise EvidenceError(
                f"{field}.contains_state must disclose every state class: {'; '.join(details)}"
            )
        forbidden = sorted(key for key, present in contains_state.items() if present is not False)
        if forbidden:
            raise EvidenceError(
                f"{field} cannot exclude an artifact containing state: {', '.join(forbidden)}"
            )
        artifacts.append(
            {
                "name": name,
                "path": path,
                "sha256": sha256,
                "creation_elapsed_seconds": float(creation_elapsed),
                "creation_protocol": creation_protocol,
                "reuse_conditions": reuse_conditions,
                "persistent": True,
                "run_independent": True,
                "reused_unchanged": True,
                "contains_state": {key: False for key in sorted(_BOOTSTRAP_STATE_FIELDS)},
            }
        )
    return artifacts


def _validate_bootstrap_files(
    declarations: list[dict[str, Any]],
    *,
    base_directory: Path,
    artifacts_root: Path,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for declaration in declarations:
        path = _resolve_evidence_path(Path(declaration["path"]), base_directory=base_directory)
        if path == artifacts_root or path.is_relative_to(artifacts_root):
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} must persist outside "
                "benchmark artifacts"
            )
        try:
            metadata = path.lstat()
            payload = path.read_bytes()
        except OSError as error:
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} is missing or unreadable: {path}"
            ) from error
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} must be a regular file"
            )
        if metadata.st_nlink != 1 or metadata.st_mode & 0o222:
            raise EvidenceError(f"bootstrap artifact {declaration['name']!r} is mutable")
        actual_sha256 = _sha256_bytes(payload)
        if actual_sha256 != declaration["sha256"]:
            raise EvidenceError(
                f"bootstrap artifact {declaration['name']!r} SHA-256 mismatch: "
                f"expected {declaration['sha256']}, got {actual_sha256}"
            )
        validated.append(
            {
                **declaration,
                "path": str(path),
                "validated_before_cohort": True,
                "reverified_unchanged_after_cohort": False,
            }
        )
    return validated


def _reverify_bootstrap_files(artifacts: list[dict[str, Any]]) -> None:
    for artifact in artifacts:
        path = Path(artifact["path"])
        try:
            metadata = path.lstat()
            actual_sha256 = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise EvidenceError(
                f"bootstrap artifact {artifact['name']!r} changed during the cohort"
            ) from error
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o222
            or actual_sha256 != artifact["sha256"]
        ):
            raise EvidenceError(
                f"bootstrap artifact {artifact['name']!r} changed during the cohort"
            )
        artifact["reverified_unchanged_after_cohort"] = True


def _load_benchmark_continuation(
    path: Path,
    *,
    run_identity: str,
    protocol: dict[str, Any],
    code_provenance: dict[str, Any],
    declared_inputs: list[dict[str, Any]],
    wad_profile: dict[str, Any] | None,
    initial_evidence_entries: list[dict[str, str]],
    manifest_directory: Path,
) -> dict[str, Any]:
    try:
        report = _parse_json_document(path.read_bytes(), document="benchmark continuation report")
    except OSError as error:
        raise EvidenceError(f"cannot read benchmark continuation report: {path}") from error
    if not isinstance(report, dict):
        raise EvidenceError("benchmark continuation report must be a JSON object")
    exact_fields = {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": "development",
        "run_identity": run_identity,
        "code_provenance": code_provenance,
        "declared_inputs": declared_inputs,
        "benchmark_protocol": protocol,
        "wad_profile": wad_profile,
    }
    for field, expected in exact_fields.items():
        if report.get(field) != expected:
            raise EvidenceError(f"cannot continue benchmark with unlike {field.replace('_', ' ')}")
    if report.get("claim_eligible") is not False or report.get("authoritative") is not False:
        raise EvidenceError("benchmark continuation report changed development evidence status")
    evidence_index = _required_mapping(
        report.get("evidence_index"),
        "benchmark continuation report evidence_index",
    )
    entries = evidence_index.get("entries")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise EvidenceError("benchmark continuation report evidence_index.entries must be an array")
    if evidence_index.get("sha256") != _canonical_sha256(
        entries, document="benchmark continuation"
    ):
        raise EvidenceError("benchmark continuation report evidence index SHA-256 mismatch")
    entries_by_name = {entry.get("name"): entry for entry in entries}
    if len(entries_by_name) != len(entries):
        raise EvidenceError("benchmark continuation report evidence index has duplicate names")
    for current in initial_evidence_entries:
        if entries_by_name.get(current["name"]) != current:
            raise EvidenceError(
                f"benchmark continuation evidence {current['name']!r} does not match current inputs"
            )
    generated = report.get("generated_artifacts")
    if not isinstance(generated, list):
        raise EvidenceError("benchmark continuation report generated_artifacts must be an array")
    paths_by_name = {
        item.get("name"): item.get("path")
        for item in generated
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("path"), str)
    }
    initial_names = {entry["name"] for entry in initial_evidence_entries}
    for name, entry in entries_by_name.items():
        if name in initial_names:
            continue
        raw_path = paths_by_name.get(name)
        if raw_path is None:
            raise EvidenceError(f"benchmark continuation evidence {name!r} has no artifact path")
        artifact_path = _resolve_evidence_path(
            Path(raw_path),
            base_directory=manifest_directory,
        )
        if _fsync_file(
            artifact_path, field=f"benchmark continuation evidence {name!r}"
        ) != entry.get("sha256"):
            raise EvidenceError(f"benchmark continuation evidence {name!r} SHA-256 mismatch")
    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        raise EvidenceError("benchmark continuation report attempts must be an array")
    expected_seeds = protocol["training_seeds"]
    if [attempt.get("seed") for attempt in attempts if isinstance(attempt, dict)] != expected_seeds:
        raise EvidenceError("benchmark continuation cannot replace or reorder training seeds")
    for attempt in attempts:
        if not isinstance(attempt, dict):
            raise EvidenceError("benchmark continuation attempt must be an object")
        expected_attempt_identity = _canonical_sha256(
            {"run_identity": run_identity, "seed": attempt["seed"]},
            document="benchmark attempt",
        )
        if attempt.get("attempt_identity") != expected_attempt_identity:
            raise EvidenceError("benchmark continuation attempt identity mismatch")
        if attempt.get("status") not in {
            "succeeded",
            "exhausted",
            "crashed",
            "evaluation_failed",
            "evidence_failed",
            "interrupted",
        }:
            raise EvidenceError("benchmark continuation attempt has invalid status")
        attempt_journal = attempt.get("attempt_journal")
        if not isinstance(attempt_journal, dict):
            raise EvidenceError("benchmark continuation attempt has no attempt journal")
        attempt_journal_path = _resolve_evidence_path(
            Path(attempt_journal.get("path", "")),
            base_directory=manifest_directory,
        )
        try:
            attempt_journal_bytes = attempt_journal_path.read_bytes()
            stored_attempt = _parse_json_document(
                attempt_journal_bytes,
                document="benchmark attempt journal",
            )
        except OSError as error:
            raise EvidenceError("benchmark attempt journal is missing") from error
        if stored_attempt != _attempt_journal_payload(
            attempt, run_identity=run_identity
        ) or attempt_journal.get("sha256") != _sha256_bytes(attempt_journal_bytes):
            raise EvidenceError("attempt journal does not match completed unit")
        if attempt["status"] == "interrupted":
            recovery = attempt.get("recovery")
            if not isinstance(recovery, dict):
                raise EvidenceError("interrupted benchmark attempt has no recovery checkpoint")
            if (
                recovery.get("run_identity") != run_identity
                or recovery.get("attempt_identity") != expected_attempt_identity
                or recovery.get("restorable_state") != _RESTORABLE_STATE
                or recovery.get("accumulated_reusable_elapsed_seconds")
                != attempt.get("reusable_elapsed_seconds")
            ):
                raise EvidenceError(
                    "interrupted benchmark recovery identity or state is incomplete"
                )
            journal = attempt.get("recovery_journal")
            if not isinstance(journal, dict):
                raise EvidenceError("interrupted benchmark attempt has no recovery journal")
            journal_path = _resolve_evidence_path(
                Path(journal.get("path", "")),
                base_directory=manifest_directory,
            )
            try:
                journal_payload = _parse_json_document(
                    journal_path.read_bytes(),
                    document="recovery journal",
                )
            except OSError as error:
                raise EvidenceError("recovery journal is missing") from error
            expected_journal = {
                "schema_version": 1,
                "run_identity": run_identity,
                "attempt_identity": expected_attempt_identity,
                "seed": attempt["seed"],
                "checkpoint": recovery["checkpoint"],
                "checkpoint_sha256": recovery["checkpoint_sha256"],
                "progress_step": recovery["progress_step"],
                "restorable_state": _RESTORABLE_STATE,
                "accumulated_reusable_elapsed_seconds": attempt["reusable_elapsed_seconds"],
            }
            if journal_payload != expected_journal or journal.get("sha256") != _sha256_bytes(
                journal_path.read_bytes()
            ):
                raise EvidenceError("recovery journal does not match interrupted attempt")
    return report


def _validate_benchmark(manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_schema_version(manifest.get("schema_version"), document="manifest")
    if manifest.get("workflow") != _WORKFLOW:
        raise EvidenceError(f"this command path requires workflow {_WORKFLOW}")
    if manifest.get("evidence_level") != "development":
        raise EvidenceError("development training benchmark requires development evidence")
    if type(manifest.get("fixture")) is not bool:
        raise EvidenceError("fixture is required and must be a boolean")
    code_provenance = _validate_code_provenance(manifest.get("code_provenance"))
    benchmark = _required_mapping(manifest.get("benchmark"), "benchmark")
    training_seeds = _seed_list(
        benchmark.get("training_seeds", [_DEFAULT_TRAINING_SEED]),
        "benchmark.training_seeds",
    )
    failure_budget = _positive_integer(
        benchmark.get("failure_budget_steps"),
        "benchmark.failure_budget_steps",
    )
    checkpoint_steps = benchmark.get("checkpoint_steps")
    if not isinstance(checkpoint_steps, list) or not checkpoint_steps:
        raise EvidenceError("benchmark.checkpoint_steps must be a non-empty array")
    validated_steps = [
        _positive_integer(step, f"benchmark.checkpoint_steps[{index}]")
        for index, step in enumerate(checkpoint_steps)
    ]
    if validated_steps != sorted(set(validated_steps)):
        raise EvidenceError("benchmark.checkpoint_steps must be unique and strictly increasing")
    if validated_steps[-1] != failure_budget:
        raise EvidenceError("benchmark.checkpoint_steps must end at benchmark.failure_budget_steps")
    evaluation_seeds = _seed_list(
        benchmark.get("evaluation_episode_seeds"),
        "benchmark.evaluation_episode_seeds",
        exact_count=_EVALUATION_EPISODES,
    )
    evaluation_action_seed = benchmark.get("evaluation_action_seed", _DEFAULT_TRAINING_SEED)
    if type(evaluation_action_seed) is not int or not 0 <= evaluation_action_seed <= _UINT32_MAX:
        raise EvidenceError(
            f"benchmark.evaluation_action_seed must be an integer in [0, {_UINT32_MAX}]"
        )
    trainer = _validate_trainer(benchmark.get("trainer"))
    artifacts_directory = benchmark.get("artifacts_directory")
    if not isinstance(artifacts_directory, str) or not artifacts_directory.strip():
        raise EvidenceError("benchmark.artifacts_directory must be a non-whitespace path")
    certificate = _validate_certificate(benchmark.get("parity_certificate"))
    bootstrap_artifacts = _validate_bootstrap_artifacts(benchmark.get("bootstrap_artifacts"))
    return {
        "code_provenance": code_provenance,
        "training_seeds": training_seeds,
        "failure_budget_steps": failure_budget,
        "checkpoint_steps": validated_steps,
        "evaluation_episode_seeds": evaluation_seeds,
        "evaluation_action_seed": evaluation_action_seed,
        "trainer": trainer,
        "artifacts_directory": artifacts_directory,
        "parity_certificate": certificate,
        "bootstrap_artifacts": bootstrap_artifacts,
    }


def _read_jsonl(path: Path, *, phase: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise EvidenceError(f"{phase} process did not write metrics JSONL: {path}") from error
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        parsed = _parse_json_document(line, document=f"{phase} metrics line {index}")
        if not isinstance(parsed, dict):
            raise EvidenceError(f"{phase} metrics line {index} must be a JSON object")
        records.append(parsed)
    if not records:
        raise EvidenceError(f"{phase} process wrote no metrics records")
    return records


def _record(records: list[dict[str, Any]], record_type: str, *, phase: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("type") == record_type]
    if len(matches) != 1:
        raise EvidenceError(
            f"{phase} process must emit exactly one {record_type!r} record, got {len(matches)}"
        )
    return matches[0]


def _resolved_record_path(value: object, *, field: str, base_directory: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{field} must be a non-whitespace path")
    return _resolve_evidence_path(Path(value), base_directory=base_directory)


def _validate_training_records(
    records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    requested_step: int,
    previous_checkpoint: dict[str, Any] | None,
    manifest_directory: Path,
    wad_profile: dict[str, Any] | None,
    run_identity: str,
    attempt_identity: str,
    interrupted: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    config = _record(records, "config", phase="training")
    if config.get("contract") != _TRAINER_CONTRACT or config.get("operation") != "train":
        raise EvidenceError("training process did not use the standalone GPU-resident trainer")
    expected_binding = {
        "run_identity": run_identity,
        "attempt_identity": attempt_identity,
    }
    if config.get("evidence_binding") != expected_binding:
        raise EvidenceError("training process did not bind the exact run and attempt identity")
    initialization = config.get("initialization")
    if not isinstance(initialization, dict) or initialization.get("mode") != "random":
        raise EvidenceError("training process must declare random policy initialization")
    if initialization.get("checkpoint") is not None:
        raise EvidenceError("training process initialized policy from learned state")
    state_initialization = config.get("state_initialization")
    expected_state_initialization = (
        {"policy_state": "fresh_random", "optimizer_state": "fresh"}
        if previous_checkpoint is None
        else {"policy_state": "resumed", "optimizer_state": "resumed"}
    )
    if state_initialization != expected_state_initialization:
        raise EvidenceError(
            "training process did not declare the required fresh or continuous policy and "
            "optimizer state"
        )
    resumed = [record for record in records if record.get("event") == "resumed"]
    if previous_checkpoint is None and resumed:
        raise EvidenceError("cold-start training process unexpectedly resumed learned state")
    if previous_checkpoint is not None:
        if len(resumed) != 1:
            raise EvidenceError("continued training process must emit exactly one resumed event")
        resumed_path = _resolved_record_path(
            resumed[0].get("checkpoint"),
            field="training resumed checkpoint",
            base_directory=manifest_directory,
        )
        if resumed_path != previous_checkpoint["path"]:
            raise EvidenceError("continued training process resumed the wrong checkpoint")
        if resumed[0].get("train/global_step") != previous_checkpoint["progress_step"]:
            raise EvidenceError("continued training process resumed the wrong progress step")
        if resumed[0].get("restored_state") != _RESTORABLE_STATE:
            raise EvidenceError(
                "continued training process did not restore policy, optimizer, RNG, and progress"
            )
        if resumed[0].get("evidence_binding") != expected_binding:
            raise EvidenceError("continued training process resumed unlike run or attempt identity")
    summary = _record(records, "summary", phase="training")
    expected_status = "interrupted" if interrupted else "completed"
    if summary.get("status") != expected_status:
        raise EvidenceError(f"training process reported status {summary.get('status')!r}")
    step = summary.get("train/global_step")
    if (
        type(step) is not int
        or (interrupted and not 0 < step < requested_step)
        or (not interrupted and step != requested_step)
    ):
        boundary = (
            "a recoverable intermediate step" if interrupted else "the predeclared checkpoint step"
        )
        raise EvidenceError(f"training process stopped outside {boundary}")
    if config.get("requested_timesteps") != requested_step:
        raise EvidenceError("training config did not bind the predeclared checkpoint step")
    if config.get("execution_timesteps") != requested_step:
        raise EvidenceError("training config would execute outside the predeclared checkpoint step")
    if summary.get("requested_timesteps") != requested_step:
        raise EvidenceError("training summary did not bind the predeclared checkpoint step")
    if summary.get("execution_timesteps") != requested_step:
        raise EvidenceError("training summary executed outside the predeclared checkpoint step")
    recorded_checkpoint = _resolved_record_path(
        summary.get("checkpoint"),
        field="training summary checkpoint",
        base_directory=manifest_directory,
    )
    if recorded_checkpoint != checkpoint:
        raise EvidenceError("training process reported a different checkpoint path")
    _validate_runtime_assets(summary, phase="training", wad_profile=wad_profile)
    return summary, (None if previous_checkpoint is None else resumed[0])


def _validate_runtime_assets(
    record: dict[str, Any],
    *,
    phase: str,
    wad_profile: dict[str, Any] | None,
) -> None:
    if wad_profile is None:
        return
    binding = wad_profile["binding_identity"]
    providers = binding["providers"]
    gradoom_provider = next(provider for provider in providers if provider["id"] == "gradoom")
    if record.get("iwad_sha256") != gradoom_provider["iwad_sha256"]:
        raise EvidenceError(f"{phase} process used an IWAD outside the declared WAD profile")
    if record.get("scenario_sha256") != gradoom_provider["pwad_sha256"]:
        raise EvidenceError(f"{phase} process used a PWAD outside the declared WAD profile")


def _validate_evaluation_records(
    records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    episode_seeds: list[int],
    evaluation_action_seed: int,
    manifest_directory: Path,
    wad_profile: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], float, float | None]:
    config = _record(records, "config", phase="evaluation")
    if config.get("contract") != _TRAINER_CONTRACT or config.get("operation") != "evaluate":
        raise EvidenceError("evaluation process did not use the standalone trainer evaluation path")
    evaluation_config = config.get("evaluation")
    if not isinstance(evaluation_config, dict):
        raise EvidenceError("evaluation config is missing")
    if evaluation_config.get("episodes") != _EVALUATION_EPISODES:
        raise EvidenceError("evaluation process did not declare exactly 100 episodes")
    if evaluation_config.get("stochastic_actions") is not True:
        raise EvidenceError("evaluation process did not declare stochastic actions")
    if evaluation_config.get("seed") != evaluation_action_seed:
        raise EvidenceError("evaluation process did not use the predeclared stochastic action seed")
    if evaluation_config.get("kills_signal") != "player_killcount":
        raise EvidenceError("evaluation process did not declare player_killcount quality")
    evaluation = _record(records, "evaluation", phase="evaluation")
    if evaluation.get("status") != "completed":
        raise EvidenceError(f"evaluation process reported status {evaluation.get('status')!r}")
    if evaluation.get("checkpoint_sha256") != checkpoint_sha256:
        raise EvidenceError("evaluation checkpoint SHA-256 does not match the durable checkpoint")
    recorded_checkpoint = _resolved_record_path(
        evaluation.get("checkpoint"),
        field="evaluation checkpoint",
        base_directory=manifest_directory,
    )
    if recorded_checkpoint != checkpoint:
        raise EvidenceError("evaluation process reported a different checkpoint path")
    _validate_runtime_assets(evaluation, phase="evaluation", wad_profile=wad_profile)
    if evaluation.get("deterministic_actions") is not False:
        raise EvidenceError("evaluation process did not execute stochastic policy actions")
    if evaluation.get("evaluation/episode/count") != _EVALUATION_EPISODES:
        raise EvidenceError("evaluation process did not complete exactly 100 episodes")
    if evaluation.get("evaluation/kills/signal") != "player_killcount":
        raise EvidenceError("evaluation result did not use player_killcount quality")
    raw_episodes = evaluation.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != _EVALUATION_EPISODES:
        raise EvidenceError("evaluation episodes must contain exactly 100 outcomes")
    episodes: list[dict[str, Any]] = []
    player_killcounts: list[float] = []
    compatibility_killcounts: list[float] = []
    for index, raw_episode in enumerate(raw_episodes):
        if not isinstance(raw_episode, dict):
            raise EvidenceError(f"evaluation episodes[{index}] must be an object")
        if raw_episode.get("index") != index:
            raise EvidenceError("evaluation episode indices do not match their declared order")
        if raw_episode.get("game_seed") != episode_seeds[index]:
            raise EvidenceError("evaluation episode seeds do not match the predeclared seed grid")
        player_value = raw_episode.get("player_killcount")
        if type(player_value) not in (int, float) or not math.isfinite(float(player_value)):
            raise EvidenceError(f"evaluation episodes[{index}].player_killcount must be finite")
        player_killcounts.append(float(player_value))
        compatibility_value = raw_episode.get(
            "compatibility_killcount",
            raw_episode.get("killcount", raw_episode.get("vizdoom_killcount")),
        )
        if compatibility_value is not None:
            if type(compatibility_value) not in (int, float) or not math.isfinite(
                float(compatibility_value)
            ):
                raise EvidenceError(
                    f"evaluation episodes[{index}].compatibility_killcount must be finite"
                )
            compatibility_killcounts.append(float(compatibility_value))
        episodes.append(raw_episode)
    mean_player = statistics.fmean(player_killcounts)
    mean_compatibility = (
        statistics.fmean(compatibility_killcounts)
        if len(compatibility_killcounts) == _EVALUATION_EPISODES
        else None
    )
    return evaluation, episodes, mean_player, mean_compatibility


def _run_process(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"cannot execute benchmark process {command[0]!r}: {error}",
        )


def _fsync_file(path: Path, *, field: str) -> str:
    try:
        with path.open("rb") as stream:
            payload = stream.read()
            os.fsync(stream.fileno())
    except OSError as error:
        raise EvidenceError(f"{field} is not a durable readable file: {path}") from error
    try:
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as error:
        raise EvidenceError(f"{field} directory cannot be made durable: {path.parent}") from error
    return _sha256_bytes(payload)


def _write_seed_file(path: Path, seeds: list[int]) -> str:
    path.write_text(json.dumps(seeds, separators=(",", ":")) + "\n", encoding="utf-8")
    return _fsync_file(path, field="evaluation seed file")


def _failure(
    *,
    seed: int,
    phase: str,
    checkpoint_step: int,
    process: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "phase": phase,
        "checkpoint_step": checkpoint_step,
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _run_attempt(
    *,
    seed: int,
    protocol: dict[str, Any],
    run_identity: str,
    run_directory: Path,
    manifest_directory: Path,
    evidence_entries: list[dict[str, str]],
    wad_profile: dict[str, Any] | None,
    existing_attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    attempt_directory = run_directory / f"seed-{seed}"
    attempt_identity = _canonical_sha256(
        {"run_identity": run_identity, "seed": seed},
        document="benchmark attempt",
    )
    if existing_attempt is None:
        attempt_directory.mkdir()
    elif existing_attempt.get("attempt_identity") != attempt_identity:
        raise EvidenceError(f"seed {seed} continuation has unlike attempt identity")
    elif existing_attempt.get("status") != "interrupted":
        return existing_attempt
    seed_file = attempt_directory / "evaluation-seeds.json"
    if seed_file.exists():
        seed_file_sha256 = _fsync_file(seed_file, field="evaluation seed file")
        expected_seed_sha256 = _sha256_bytes(
            (
                json.dumps(protocol["evaluation_episode_seeds"], separators=(",", ":")) + "\n"
            ).encode()
        )
        if seed_file_sha256 != expected_seed_sha256:
            raise EvidenceError(f"seed {seed} continuation changed its evaluation seed grid")
    else:
        seed_file_sha256 = _write_seed_file(seed_file, protocol["evaluation_episode_seeds"])
        evidence_entries.append(
            {"name": f"seed-{seed}-evaluation-seeds", "sha256": seed_file_sha256}
        )
    base_command = [
        *protocol["trainer"]["command"],
        *protocol["trainer"]["arguments"],
    ]
    outcomes: list[dict[str, Any]] = list((existing_attempt or {}).get("outcomes", []))
    failures: list[dict[str, Any]] = list((existing_attempt or {}).get("failures", []))
    recovery_history: list[dict[str, Any]] = list(
        (existing_attempt or {}).get("recovery_history", [])
    )
    prior_elapsed = float((existing_attempt or {}).get("reusable_elapsed_seconds", 0.0))
    recovery_journal: dict[str, Any] | None = (existing_attempt or {}).get("recovery_journal")
    existing_recovery = (existing_attempt or {}).get("recovery")
    previous_checkpoint: dict[str, Any] | None = None
    if isinstance(existing_recovery, dict):
        recovery_path = _resolve_evidence_path(
            Path(existing_recovery["checkpoint"]),
            base_directory=manifest_directory,
        )
        if _fsync_file(recovery_path, field="recovery checkpoint") != existing_recovery.get(
            "checkpoint_sha256"
        ):
            raise EvidenceError(f"seed {seed} recovery checkpoint SHA-256 mismatch")
        previous_checkpoint = {
            "path": recovery_path,
            "progress_step": existing_recovery["progress_step"],
            "kind": "recovery",
            "checkpoint_sha256": existing_recovery["checkpoint_sha256"],
            "prior_reusable_elapsed_seconds": prior_elapsed,
        }
    elif outcomes:
        previous_checkpoint = {
            "path": _resolve_evidence_path(
                Path(outcomes[-1]["checkpoint"]),
                base_directory=manifest_directory,
            ),
            "progress_step": outcomes[-1]["checkpoint_step"],
            "kind": "checkpoint",
            "checkpoint_sha256": outcomes[-1]["checkpoint_sha256"],
        }
    final_checkpoint = None if not outcomes else Path(outcomes[-1]["checkpoint"])
    final_checkpoint_sha256 = None if not outcomes else outcomes[-1]["checkpoint_sha256"]
    recovery: dict[str, Any] | None = None
    generated_artifacts: list[dict[str, str]] = []
    status = "exhausted"
    for checkpoint_step in protocol["checkpoint_steps"]:
        if any(outcome["checkpoint_step"] == checkpoint_step for outcome in outcomes):
            continue
        generation = 0
        checkpoint = attempt_directory / f"checkpoint-step-{checkpoint_step}.pt"
        while checkpoint.exists():
            generation += 1
            checkpoint = (
                attempt_directory / f"checkpoint-step-{checkpoint_step}-recovery-{generation}.pt"
            )
        suffix = "" if generation == 0 else f"-recovery-{generation}"
        training_metrics = attempt_directory / f"training-step-{checkpoint_step}{suffix}.jsonl"
        training_command = [
            *base_command,
            "--evidence-run-identity",
            run_identity,
            "--evidence-attempt-identity",
            attempt_identity,
            "--seed",
            str(seed),
            "--timesteps",
            str(checkpoint_step),
            "--checkpoint",
            str(checkpoint),
            "--metrics-jsonl",
            str(training_metrics),
        ]
        if previous_checkpoint is not None:
            training_command.extend(("--resume", str(previous_checkpoint["path"])))
        training_process = _run_process(training_command, cwd=manifest_directory)
        if training_process.returncode not in {0, 130}:
            failures.append(
                _failure(
                    seed=seed,
                    phase="training",
                    checkpoint_step=checkpoint_step,
                    process=training_process,
                )
            )
            status = "crashed"
            break
        was_interrupted = training_process.returncode == 130
        try:
            training_records = _read_jsonl(training_metrics, phase="training")
            training_summary, resumed_record = _validate_training_records(
                training_records,
                checkpoint=checkpoint,
                requested_step=checkpoint_step,
                previous_checkpoint=previous_checkpoint,
                manifest_directory=manifest_directory,
                wad_profile=wad_profile,
                run_identity=run_identity,
                attempt_identity=attempt_identity,
                interrupted=was_interrupted,
            )
            checkpoint_sha256 = _fsync_file(checkpoint, field="training checkpoint")
            training_metrics_sha256 = _fsync_file(
                training_metrics,
                field="training metrics",
            )
        except EvidenceError as error:
            failures.append(
                {
                    "seed": seed,
                    "phase": "training_evidence",
                    "checkpoint_step": checkpoint_step,
                    "message": str(error),
                }
            )
            status = "evidence_failed"
            break
        evidence_entries.extend(
            (
                {
                    "name": f"seed-{seed}-step-{checkpoint_step}-checkpoint",
                    "sha256": checkpoint_sha256,
                },
                {
                    "name": f"seed-{seed}-step-{checkpoint_step}-training-metrics",
                    "sha256": training_metrics_sha256,
                },
            )
        )
        if suffix:
            evidence_entries[-2]["name"] = f"seed-{seed}-step-{checkpoint_step}{suffix}-checkpoint"
            evidence_entries[-1]["name"] = (
                f"seed-{seed}-step-{checkpoint_step}{suffix}-training-metrics"
            )
        generated_artifacts.extend(
            (
                {"name": evidence_entries[-2]["name"], "path": str(checkpoint)},
                {"name": evidence_entries[-1]["name"], "path": str(training_metrics)},
            )
        )
        if previous_checkpoint is not None and previous_checkpoint.get("kind") == "recovery":
            assert resumed_record is not None
            recovery_history.append(
                {
                    "checkpoint": str(previous_checkpoint["path"]),
                    "checkpoint_sha256": previous_checkpoint["checkpoint_sha256"],
                    "progress_step": previous_checkpoint["progress_step"],
                    "restored_state": resumed_record["restored_state"],
                    "prior_reusable_elapsed_seconds": previous_checkpoint[
                        "prior_reusable_elapsed_seconds"
                    ],
                }
            )
        if was_interrupted:
            final_checkpoint = checkpoint
            final_checkpoint_sha256 = checkpoint_sha256
            recovery = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "progress_step": training_summary["train/global_step"],
                "restorable_state": dict(_RESTORABLE_STATE),
                "run_identity": run_identity,
                "attempt_identity": attempt_identity,
                "accumulated_reusable_elapsed_seconds": 0.0,
            }
            status = "interrupted"
            break
        evaluation_metrics = attempt_directory / f"evaluation-step-{checkpoint_step}.jsonl"
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
            failures.append(
                _failure(
                    seed=seed,
                    phase="evaluation",
                    checkpoint_step=checkpoint_step,
                    process=evaluation_process,
                )
            )
            status = "evaluation_failed"
            final_checkpoint = checkpoint
            final_checkpoint_sha256 = checkpoint_sha256
            break
        try:
            evaluation_records = _read_jsonl(evaluation_metrics, phase="evaluation")
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
                field="evaluation metrics",
            )
        except EvidenceError as error:
            failures.append(
                {
                    "seed": seed,
                    "phase": "evaluation_evidence",
                    "checkpoint_step": checkpoint_step,
                    "message": str(error),
                }
            )
            status = "evidence_failed"
            final_checkpoint = checkpoint
            final_checkpoint_sha256 = checkpoint_sha256
            break
        evidence_entries.append(
            {
                "name": f"seed-{seed}-step-{checkpoint_step}-evaluation-metrics",
                "sha256": evaluation_metrics_sha256,
            }
        )
        generated_artifacts.append(
            {"name": evidence_entries[-1]["name"], "path": str(evaluation_metrics)}
        )
        passed = mean_player >= _QUALITY_THRESHOLD
        outcomes.append(
            {
                "checkpoint_step": checkpoint_step,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "mean_player_killcount": mean_player,
                "mean_killcount": mean_compatibility,
                "passed": passed,
                "training": training_summary,
                "evaluation": evaluation,
                "episodes": episodes,
            }
        )
        previous_checkpoint = {
            "path": checkpoint,
            "progress_step": checkpoint_step,
            "kind": "checkpoint",
            "checkpoint_sha256": checkpoint_sha256,
        }
        final_checkpoint = checkpoint
        final_checkpoint_sha256 = checkpoint_sha256
        if passed:
            status = "succeeded"
            break
    elapsed = prior_elapsed + time.perf_counter() - started
    if recovery is not None:
        recovery["accumulated_reusable_elapsed_seconds"] = elapsed
        recovery_journal_path = attempt_directory / (
            f"recovery-step-{recovery['progress_step']}-{len(recovery_history)}.json"
        )
        recovery_journal_payload = {
            "schema_version": 1,
            "run_identity": run_identity,
            "attempt_identity": attempt_identity,
            "seed": seed,
            "checkpoint": recovery["checkpoint"],
            "checkpoint_sha256": recovery["checkpoint_sha256"],
            "progress_step": recovery["progress_step"],
            "restorable_state": _RESTORABLE_STATE,
            "accumulated_reusable_elapsed_seconds": elapsed,
        }
        recovery_journal_path.write_text(
            json.dumps(recovery_journal_payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        recovery_journal_sha256 = _fsync_file(
            recovery_journal_path,
            field="recovery journal",
        )
        recovery_journal = {
            "path": str(recovery_journal_path),
            "sha256": recovery_journal_sha256,
        }
        recovery_evidence_name = (
            f"seed-{seed}-recovery-step-{recovery['progress_step']}-{len(recovery_history)}"
        )
        evidence_entries.append({"name": recovery_evidence_name, "sha256": recovery_journal_sha256})
        generated_artifacts.append(
            {"name": recovery_evidence_name, "path": str(recovery_journal_path)}
        )
    attempt = {
        "seed": seed,
        "attempt_identity": attempt_identity,
        "cold_start": {
            "policy_state": "fresh_random",
            "optimizer_state": "fresh",
            "learned_initialization": False,
        },
        "status": status,
        "reusable_elapsed_seconds": elapsed,
        "checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        "checkpoint_sha256": final_checkpoint_sha256,
        "outcomes": outcomes,
        "failures": failures,
        "recovery": recovery,
        "recovery_history": recovery_history,
        "recovery_journal": recovery_journal,
        "generated_artifacts": generated_artifacts,
    }
    attempt_journal_generation = len(recovery_history)
    attempt_journal_path = attempt_directory / (f"attempt-state-{attempt_journal_generation}.json")
    attempt_journal_path.write_text(
        json.dumps(
            _attempt_journal_payload(attempt, run_identity=run_identity),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    attempt_journal_sha256 = _fsync_file(
        attempt_journal_path,
        field="benchmark attempt journal",
    )
    attempt_journal_name = f"seed-{seed}-attempt-state-{attempt_journal_generation}"
    evidence_entries.append({"name": attempt_journal_name, "sha256": attempt_journal_sha256})
    generated_artifacts.append({"name": attempt_journal_name, "path": str(attempt_journal_path)})
    attempt["attempt_journal"] = {
        "path": str(attempt_journal_path),
        "sha256": attempt_journal_sha256,
    }
    return attempt


def build_development_benchmark_report(
    manifest_path: Path,
    *,
    merge_path: Path | None = None,
) -> dict[str, Any]:
    manifest, manifest_payload = _load_manifest(manifest_path)
    validated = _validate_benchmark(manifest)
    declared_inputs = _validate_declared_inputs(
        manifest.get("declared_inputs"),
        base_directory=manifest_path.parent,
    )
    evidence_entries = [{"name": "manifest", "sha256": _sha256_bytes(manifest_payload)}]
    artifacts_root = _resolve_evidence_path(
        Path(validated["artifacts_directory"]),
        base_directory=manifest_path.parent,
    )
    bootstrap_exclusions = _validate_bootstrap_files(
        validated["bootstrap_artifacts"],
        base_directory=manifest_path.parent,
        artifacts_root=artifacts_root,
    )
    evidence_entries.extend(
        {"name": f"bootstrap-{item['name']}", "sha256": item["sha256"]}
        for item in bootstrap_exclusions
    )
    wad_profile = None
    if "wad_profile" in manifest:
        wad_profile, wad_entries = validate_wad_profile(
            manifest["wad_profile"],
            base_directory=manifest_path.parent,
        )
        evidence_entries.extend(wad_entries)
        if wad_profile["status"] != "matched":
            raise EvidenceError("development benchmark WAD profile did not match")
    elif not manifest["fixture"]:
        raise EvidenceError("non-fixture development benchmark requires wad_profile")
    reserved_evidence_names = {entry["name"] for entry in evidence_entries}
    reserved_evidence_names.update(
        f"seed-{seed}-evaluation-seeds" for seed in validated["training_seeds"]
    )
    reserved_evidence_names.update(
        name
        for seed in validated["training_seeds"]
        for step in validated["checkpoint_steps"]
        for name in (
            f"seed-{seed}-step-{step}-checkpoint",
            f"seed-{seed}-step-{step}-training-metrics",
            f"seed-{seed}-step-{step}-evaluation-metrics",
        )
    )
    for declared_input in declared_inputs:
        if declared_input["name"] in reserved_evidence_names:
            raise EvidenceError(
                f"declared input name {declared_input['name']!r} is reserved by the benchmark"
            )
        input_path = _resolve_evidence_path(
            Path(declared_input["path"]),
            base_directory=manifest_path.parent,
        )
        try:
            actual_sha256 = _sha256_bytes(input_path.read_bytes())
        except OSError as error:
            raise EvidenceError(
                f"cannot read declared input {declared_input['name']!r}: {error}"
            ) from error
        if actual_sha256 != declared_input["sha256"]:
            raise EvidenceError(
                f"declared input {declared_input['name']!r} SHA-256 mismatch: "
                f"expected {declared_input['sha256']}, got {actual_sha256}"
            )
        evidence_entries.append({"name": declared_input["name"], "sha256": actual_sha256})
    protocol = {
        "training_seeds": validated["training_seeds"],
        "failure_budget_steps": validated["failure_budget_steps"],
        "checkpoint_steps": validated["checkpoint_steps"],
        "evaluation_episode_seeds": validated["evaluation_episode_seeds"],
        "evaluation_action_seed": validated["evaluation_action_seed"],
        "quality_gate": {
            "episodes": _EVALUATION_EPISODES,
            "mean_at_least": _QUALITY_THRESHOLD,
            "signal": "player_killcount",
            "stochastic_actions": True,
        },
        "cold_start": {
            "policy_state": "fresh_random",
            "optimizer_state": "fresh",
            "learned_initialization_allowed": False,
        },
        "timer_includes": [
            "recurring_initialization",
            "per_process_or_uncached_compilation",
            "graph_capture",
            "warmup",
            "training",
            "checkpoint_evaluation",
            "durable_checkpoint_write",
        ],
        "timer_boundaries": {
            "start": "immediately_before_recurring_attempt_setup",
            "resume": "add_prior_hashed_recovery_elapsed_before_recurring_recovery_work",
            "stop": "after_durable_checkpoint_or_terminal_attempt_state",
        },
        "trainer": validated["trainer"],
        "parity_certificate": validated["parity_certificate"],
        "wad_profile_binding_sha256": (
            None if wad_profile is None else wad_profile["binding_sha256"]
        ),
        "bootstrap_artifacts": [
            {
                key: value
                for key, value in artifact.items()
                if key not in {"validated_before_cohort", "reverified_unchanged_after_cohort"}
            }
            for artifact in bootstrap_exclusions
        ],
    }
    protocol["continuation_identity"] = {
        "schema_sha256": _canonical_sha256(
            {
                "schema_version": 1,
                "workflow": _WORKFLOW,
                "evidence_level": "development",
                "trainer_contract": _TRAINER_CONTRACT,
            },
            document="benchmark schema identity",
        ),
        "recipe_sha256": _canonical_sha256(
            protocol["trainer"],
            document="benchmark recipe identity",
        ),
        "asset_sha256": _canonical_sha256(
            {
                "wad_profile_binding_sha256": protocol["wad_profile_binding_sha256"],
                "declared_inputs": sorted(
                    ({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
                    key=lambda item: item["name"],
                ),
                "bootstrap_artifacts": [
                    {"name": item["name"], "sha256": item["sha256"]}
                    for item in protocol["bootstrap_artifacts"]
                ],
            },
            document="benchmark asset identity",
        ),
        "seed_sha256": _canonical_sha256(
            {
                "training_seeds": protocol["training_seeds"],
                "evaluation_episode_seeds": protocol["evaluation_episode_seeds"],
                "evaluation_action_seed": protocol["evaluation_action_seed"],
            },
            document="benchmark seed identity",
        ),
        "timer_sha256": _canonical_sha256(
            {
                "includes": protocol["timer_includes"],
                "boundaries": protocol["timer_boundaries"],
            },
            document="benchmark timer identity",
        ),
    }
    identity_payload = {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": "development",
        "fixture": manifest["fixture"],
        "code_provenance": validated["code_provenance"],
        "declared_inputs": sorted(
            ({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
            key=lambda item: item["name"],
        ),
        "benchmark_protocol": protocol,
    }
    run_identity = _canonical_sha256(identity_payload, document="manifest")
    continuation = None
    if merge_path is not None:
        continuation = _load_benchmark_continuation(
            merge_path,
            run_identity=run_identity,
            protocol=protocol,
            code_provenance=validated["code_provenance"],
            declared_inputs=declared_inputs,
            wad_profile=wad_profile,
            initial_evidence_entries=evidence_entries,
            manifest_directory=manifest_path.parent,
        )
        evidence_entries = [dict(entry) for entry in continuation["evidence_index"]["entries"]]
    run_directory = artifacts_root / run_identity
    if continuation is None:
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise EvidenceError(
                "benchmark artifact directory already exists; refusing to overwrite: "
                f"{run_directory}"
            ) from error
    elif not run_directory.is_dir():
        raise EvidenceError("benchmark continuation artifact directory is missing")
    existing_by_seed = {
        attempt["seed"]: attempt for attempt in (continuation or {}).get("attempts", [])
    }
    attempts = []
    actual_generated_artifacts: list[dict[str, str]] = []
    for seed in validated["training_seeds"]:
        attempt = _run_attempt(
            seed=seed,
            protocol=protocol,
            run_identity=run_identity,
            run_directory=run_directory,
            manifest_directory=manifest_path.parent,
            evidence_entries=evidence_entries,
            wad_profile=wad_profile,
            existing_attempt=existing_by_seed.get(seed),
        )
        actual_generated_artifacts.extend(attempt.pop("generated_artifacts", []))
        attempts.append(attempt)
    _reverify_bootstrap_files(bootstrap_exclusions)
    generated_artifacts: list[dict[str, str]] = list(
        (continuation or {}).get("generated_artifacts", [])
    )
    for seed in validated["training_seeds"]:
        attempt_directory = run_directory / f"seed-{seed}"
        generated_artifacts.append(
            {
                "name": f"seed-{seed}-evaluation-seeds",
                "path": str(attempt_directory / "evaluation-seeds.json"),
            }
        )
        for step in validated["checkpoint_steps"]:
            for kind, path in (
                ("checkpoint", attempt_directory / f"checkpoint-step-{step}.pt"),
                ("training-metrics", attempt_directory / f"training-step-{step}.jsonl"),
                ("evaluation-metrics", attempt_directory / f"evaluation-step-{step}.jsonl"),
            ):
                generated_artifacts.append(
                    {
                        "name": f"seed-{seed}-step-{step}-{kind}",
                        "path": str(path),
                    }
                )
    generated_artifacts.extend(actual_generated_artifacts)
    generated_artifacts = list(
        {
            (artifact["name"], artifact["path"]): artifact for artifact in generated_artifacts
        }.values()
    )
    failures = [failure for attempt in attempts for failure in attempt["failures"]]
    evidence_names = [entry["name"] for entry in evidence_entries]
    if len(evidence_names) != len(set(evidence_names)):
        raise EvidenceError("benchmark evidence index contains duplicate entry names")
    all_succeeded = all(attempt["status"] == "succeeded" for attempt in attempts)
    claim_reasons: list[dict[str, Any]] = [
        {
            "code": "development_evidence",
            "message": "Development evidence is non-authoritative and cannot support claims.",
        }
    ]
    if manifest["fixture"]:
        claim_reasons.append(
            {
                "code": "fixture_evidence",
                "message": "Fixture evidence cannot support public claims.",
            }
        )
    certificate = validated["parity_certificate"]
    if not certificate["available"]:
        claim_reasons.append(
            {
                "code": "missing_current_parity_certificate",
                "message": certificate["reason"],
            }
        )
    return {
        "schema_version": 1,
        "workflow": _WORKFLOW,
        "evidence_level": "development",
        "fixture": manifest["fixture"],
        "authoritative": False,
        "status": "passed" if all_succeeded else "failed",
        "claim_eligible": False,
        "claim_reasons": claim_reasons,
        "run_identity": run_identity,
        "code_provenance": validated["code_provenance"],
        "declared_inputs": declared_inputs,
        "benchmark_protocol": protocol,
        "bootstrap_exclusions": bootstrap_exclusions,
        "wad_profile": wad_profile,
        "attempts": attempts,
        "failures": failures,
        "generated_artifacts": generated_artifacts,
        "evidence_index": {
            "algorithm": "sha256",
            "entries": evidence_entries,
            "sha256": _canonical_sha256(evidence_entries, document="manifest"),
        },
    }
