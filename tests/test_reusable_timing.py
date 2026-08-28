from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_PROCESS = Path(__file__).parent / "fixtures" / "evidence" / "fixture_benchmark_process.py"
EVALUATION_SEEDS = list(range(20_000, 20_100))


def _run_evidence(*args: str) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None
    return subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _manifest(
    tmp_path: Path,
    *,
    outcomes: dict[str, list[float]],
    bootstrap_artifacts: list[dict[str, object]] | None = None,
    extra_arguments: list[str] | None = None,
) -> Path:
    benchmark: dict[str, object] = {
        "training_seeds": [123],
        "failure_budget_steps": 10,
        "checkpoint_steps": [10],
        "evaluation_episode_seeds": EVALUATION_SEEDS,
        "trainer": {
            "command": [sys.executable, str(FIXTURE_PROCESS)],
            "arguments": [
                "--fixture-outcomes",
                json.dumps(outcomes, sort_keys=True),
                *(extra_arguments or []),
            ],
        },
        "artifacts_directory": "benchmark-artifacts",
        "parity_certificate": {
            "available": False,
            "reason": "No current parity certificate exists for the fixture profile.",
        },
    }
    if bootstrap_artifacts is not None:
        benchmark["bootstrap_artifacts"] = bootstrap_artifacts
    manifest = {
        "schema_version": 1,
        "workflow": "development_training_benchmark",
        "evidence_level": "development",
        "fixture": True,
        "code_provenance": {
            "repository": "tsilva/env-GraDOOM-turbo-torch",
            "revision": "fixture-revision",
            "dirty": False,
        },
        "declared_inputs": [],
        "benchmark": benchmark,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _bootstrap_artifact(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    artifact = tmp_path / "compiled-kernel.bin"
    artifact.write_bytes(b"run-independent compiled kernel fixture\n")
    artifact.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    declaration: dict[str, object] = {
        "name": "compiled-kernel",
        "path": artifact.name,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "creation_elapsed_seconds": 12.5,
        "creation_protocol": "fixture-compiler-v1 --target generic",
        "reuse_conditions": [
            "exact compiler and target identity",
            "read-only bytes reused without transformation",
        ],
        "persistent": True,
        "run_independent": True,
        "reused_unchanged": True,
        "contains_state": {
            "learned": False,
            "optimizer": False,
            "rollout": False,
            "seed_specific": False,
            "candidate_specific": False,
        },
    }
    return artifact, declaration


def test_public_command_accepts_only_disclosed_immutable_bootstrap_exclusion(
    tmp_path: Path,
) -> None:
    artifact, declaration = _bootstrap_artifact(tmp_path)
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["benchmark_protocol"]["timer_includes"] == [
        "recurring_initialization",
        "per_process_or_uncached_compilation",
        "graph_capture",
        "warmup",
        "training",
        "checkpoint_evaluation",
        "durable_checkpoint_write",
    ]
    assert report["bootstrap_exclusions"] == [
        {
            **declaration,
            "path": str(artifact.resolve()),
            "validated_before_cohort": True,
            "reverified_unchanged_after_cohort": True,
        }
    ]
    assert {(entry["name"], entry["sha256"]) for entry in report["evidence_index"]["entries"]} >= {
        ("bootstrap-compiled-kernel", declaration["sha256"])
    }
    continuation_identity = report["benchmark_protocol"]["continuation_identity"]
    assert set(continuation_identity) == {
        "asset_sha256",
        "recipe_sha256",
        "schema_sha256",
        "seed_sha256",
        "timer_sha256",
    }
    assert all(
        isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in continuation_identity.values()
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda artifact, declaration: artifact.unlink(), "missing or unreadable"),
        (
            lambda artifact, declaration: artifact.chmod(stat.S_IRUSR | stat.S_IWUSR),
            "is mutable",
        ),
        (
            lambda artifact, declaration: declaration.update(sha256="0" * 64),
            "SHA-256 mismatch",
        ),
        (
            lambda artifact, declaration: declaration["contains_state"].update(learned=True),
            "cannot exclude an artifact containing state: learned",
        ),
        (
            lambda artifact, declaration: declaration.pop("contains_state"),
            "contains_state is required",
        ),
    ],
)
def test_public_command_rejects_untrustworthy_bootstrap_exclusions(
    tmp_path: Path,
    mutation: object,
    error: str,
) -> None:
    artifact, declaration = _bootstrap_artifact(tmp_path)
    mutation(artifact, declaration)
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 2
    assert error in result.stderr
    assert not output.exists()


def test_public_command_rejects_bootstrap_artifact_changed_during_cohort(
    tmp_path: Path,
) -> None:
    artifact, declaration = _bootstrap_artifact(tmp_path)
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
        extra_arguments=["--fixture-mutate-bootstrap", str(artifact)],
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 2
    assert "bootstrap artifact 'compiled-kernel' changed during the cohort" in result.stderr
    assert not output.exists()


def test_public_command_output_cannot_overwrite_excluded_bootstrap_artifact(
    tmp_path: Path,
) -> None:
    artifact, declaration = _bootstrap_artifact(tmp_path)
    original = artifact.read_bytes()
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(artifact))

    assert result.returncode == 2
    assert "output path aliases bootstrap artifact 'compiled-kernel'" in result.stderr
    assert artifact.read_bytes() == original


def test_public_command_recovers_one_continuous_interrupted_attempt(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
    )
    interrupted_output = tmp_path / "interrupted-report.json"

    interrupted_result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(interrupted_output),
    )

    assert interrupted_result.returncode == 0, interrupted_result.stderr
    interrupted_report = json.loads(interrupted_output.read_text(encoding="utf-8"))
    interrupted = interrupted_report["attempts"][0]
    assert interrupted["status"] == "interrupted"
    assert interrupted["outcomes"] == []
    assert interrupted["recovery"]["progress_step"] == 5
    assert interrupted["recovery"]["restorable_state"] == {
        "policy": True,
        "optimizer": True,
        "rng": True,
        "progress": True,
    }
    assert interrupted["recovery"]["run_identity"] == interrupted_report["run_identity"]
    assert interrupted["recovery"]["attempt_identity"] == interrupted["attempt_identity"]
    assert (
        interrupted["recovery"]["accumulated_reusable_elapsed_seconds"]
        == (interrupted["reusable_elapsed_seconds"])
    )

    recovered_output = tmp_path / "recovered-report.json"
    recovered_result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(recovered_output),
        "--merge",
        str(interrupted_output),
    )

    assert recovered_result.returncode == 0, recovered_result.stderr
    recovered_report = json.loads(recovered_output.read_text(encoding="utf-8"))
    recovered = recovered_report["attempts"][0]
    assert recovered_report["run_identity"] == interrupted_report["run_identity"]
    assert recovered["attempt_identity"] == interrupted["attempt_identity"]
    assert recovered["status"] == "succeeded"
    assert recovered["cold_start"] == interrupted["cold_start"]
    assert recovered["reusable_elapsed_seconds"] >= interrupted["reusable_elapsed_seconds"]
    assert recovered["recovery"] is None
    assert recovered["recovery_history"] == [
        {
            "checkpoint": interrupted["recovery"]["checkpoint"],
            "checkpoint_sha256": interrupted["recovery"]["checkpoint_sha256"],
            "progress_step": 5,
            "restored_state": {
                "policy": True,
                "optimizer": True,
                "rng": True,
                "progress": True,
            },
            "prior_reusable_elapsed_seconds": interrupted["reusable_elapsed_seconds"],
        }
    ]
    assert [outcome["checkpoint_step"] for outcome in recovered["outcomes"]] == [10]


def test_public_command_does_not_replace_a_failed_seed_during_continuation(
    tmp_path: Path,
) -> None:
    fail_once_marker = tmp_path / "trainer-failed-once"
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-fail-training-once-marker", str(fail_once_marker)],
    )
    failed_output = tmp_path / "failed-report.json"

    failed_result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(failed_output),
    )

    assert failed_result.returncode == 0, failed_result.stderr
    failed_report = json.loads(failed_output.read_text(encoding="utf-8"))
    assert failed_report["attempts"][0]["status"] == "crashed"
    assert fail_once_marker.is_file()

    continued_output = tmp_path / "continued-report.json"
    continued_result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(continued_output),
        "--merge",
        str(failed_output),
    )

    assert continued_result.returncode == 0, continued_result.stderr
    continued_report = json.loads(continued_output.read_text(encoding="utf-8"))
    assert continued_report["attempts"] == failed_report["attempts"]
    assert continued_report["status"] == "failed"


@pytest.mark.parametrize(
    "mismatch",
    ["recipe", "asset", "seed", "timer", "schema"],
)
def test_public_command_rejects_mismatched_continuation_identity(
    tmp_path: Path,
    mismatch: str,
) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
    )
    interrupted_output = tmp_path / "interrupted-report.json"
    initial = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(interrupted_output),
    )
    assert initial.returncode == 0, initial.stderr

    merge_path = interrupted_output
    if mismatch in {"recipe", "seed"}:
        changed_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if mismatch == "recipe":
            changed_manifest["benchmark"]["trainer"]["arguments"].extend(
                ["--fixture-training-step-offset", "1"]
            )
        else:
            changed_manifest["benchmark"]["training_seeds"] = [124]
        manifest.write_text(json.dumps(changed_manifest), encoding="utf-8")
    else:
        changed_report = json.loads(interrupted_output.read_text(encoding="utf-8"))
        if mismatch == "asset":
            changed_report["benchmark_protocol"]["wad_profile_binding_sha256"] = "0" * 64
        elif mismatch == "timer":
            changed_report["benchmark_protocol"]["timer_includes"].remove("graph_capture")
        else:
            changed_report["schema_version"] = 2
        merge_path = tmp_path / f"{mismatch}-report.json"
        merge_path.write_text(json.dumps(changed_report), encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued-report.json"),
        "--merge",
        str(merge_path),
    )

    assert result.returncode == 2
    assert "cannot continue benchmark with unlike" in result.stderr


def test_public_command_rejects_tampered_accumulated_recovery_elapsed(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
    )
    interrupted_output = tmp_path / "interrupted-report.json"
    initial = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(interrupted_output),
    )
    assert initial.returncode == 0, initial.stderr
    changed_report = json.loads(interrupted_output.read_text(encoding="utf-8"))
    changed_report["attempts"][0]["reusable_elapsed_seconds"] = 0.0
    changed_report["attempts"][0]["recovery"]["accumulated_reusable_elapsed_seconds"] = 0.0
    tampered = tmp_path / "tampered-report.json"
    tampered.write_text(json.dumps(changed_report), encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued-report.json"),
        "--merge",
        str(tampered),
    )

    assert result.returncode == 2
    assert "journal does not match" in result.stderr


def test_public_command_rejects_tampered_completed_unit_elapsed(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, outcomes={"10": [30.0, 0.0]})
    completed_output = tmp_path / "completed-report.json"
    initial = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(completed_output),
    )
    assert initial.returncode == 0, initial.stderr
    changed_report = json.loads(completed_output.read_text(encoding="utf-8"))
    changed_report["attempts"][0]["reusable_elapsed_seconds"] = 0.0
    tampered = tmp_path / "tampered-report.json"
    tampered.write_text(json.dumps(changed_report), encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued-report.json"),
        "--merge",
        str(tampered),
    )

    assert result.returncode == 2
    assert "attempt journal does not match completed unit" in result.stderr
