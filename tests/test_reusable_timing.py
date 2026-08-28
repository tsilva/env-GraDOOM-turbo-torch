from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gradoom.evidence.benchmark import build_development_benchmark_report

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
    trainer_script: Path = FIXTURE_PROCESS,
) -> Path:
    benchmark: dict[str, object] = {
        "training_seeds": [123],
        "failure_budget_steps": 10,
        "checkpoint_steps": [10],
        "evaluation_episode_seeds": EVALUATION_SEEDS,
        "trainer": {
            "command": [sys.executable, str(trainer_script)],
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


def _bootstrap_artifact(
    tmp_path: Path,
    *,
    payload: bytes = b"run-independent compiled kernel fixture\n",
) -> tuple[Path, dict[str, object]]:
    artifact = tmp_path / "compiled-kernel.bin"
    artifact.write_bytes(payload)
    artifact.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    creation_protocol = "fixture-compiler-v1 --target generic"
    reuse_conditions = [
        "exact compiler and target identity",
        "read-only bytes reused without transformation",
    ]
    receipt = tmp_path / "compiled-kernel.creation.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_sha256": artifact_sha256,
                "creation_elapsed_seconds": 12.5,
                "creation_protocol": creation_protocol,
                "reuse_conditions": reuse_conditions,
                "reproduction": {
                    "varied_inputs": [
                        "candidate_identity",
                        "run_identity",
                        "training_seed",
                    ],
                    "independent_builds": [
                        {
                            "context": "verification-a",
                            "artifact_sha256": artifact_sha256,
                            "elapsed_seconds": 12.5,
                        },
                        {
                            "context": "verification-b",
                            "artifact_sha256": artifact_sha256,
                            "elapsed_seconds": 12.75,
                        },
                    ],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    receipt.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    declaration: dict[str, object] = {
        "name": "compiled-kernel",
        "path": artifact.name,
        "sha256": artifact_sha256,
        "creation_elapsed_seconds": 12.5,
        "creation_protocol": creation_protocol,
        "creation_receipt": {
            "path": receipt.name,
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        },
        "reuse_conditions": reuse_conditions,
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


@pytest.mark.parametrize(
    "hidden_state",
    [
        {"policy_state_dict": {"weight": [1.0]}},
        {"optimizer_state_dict": {"momentum": [0.5]}},
        {"rollout_state": {"observations": [1, 2, 3]}},
        {"training_seed": 123},
        {"candidate_identity": "recipe-7"},
    ],
)
def test_public_command_rejects_state_bearing_bootstrap_bytes_declared_clean(
    tmp_path: Path,
    hidden_state: dict[str, object],
) -> None:
    _artifact, declaration = _bootstrap_artifact(
        tmp_path,
        payload=json.dumps(hidden_state, sort_keys=True).encode("utf-8"),
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
    )

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "contains prohibited benchmark state" in result.stderr


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
        "command_parsing",
        "manifest_and_configuration_validation",
        "identity_and_input_hashing",
        "artifact_directory_setup",
        "continuation_and_recovery_verification",
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
            "creation_receipt": {
                **declaration["creation_receipt"],
                "path": str((tmp_path / "compiled-kernel.creation.json").resolve()),
            },
            "eligibility_evidence": {
                "contract": "deterministic-run-independence-v1",
                "receipt_sha256": declaration["creation_receipt"]["sha256"],
                "independent_builds": 2,
                "varied_inputs": [
                    "candidate_identity",
                    "run_identity",
                    "training_seed",
                ],
                "content_scan": "prohibited-benchmark-state-markers-v1",
            },
            "validated_before_cohort": True,
            "reverified_unchanged_after_cohort": True,
        }
    ]
    assert {(entry["name"], entry["sha256"]) for entry in report["evidence_index"]["entries"]} >= {
        ("bootstrap-compiled-kernel", declaration["sha256"]),
        (
            "bootstrap-compiled-kernel-creation-receipt",
            declaration["creation_receipt"]["sha256"],
        ),
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


def test_reusable_timer_starts_before_recurring_manifest_and_identity_setup(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, outcomes={"10": [30.0, 0.0]})

    report = build_development_benchmark_report(
        manifest,
        invocation_started=10.0,
        clock=lambda: 42.5,
    )

    assert report["attempts"][0]["reusable_elapsed_seconds"] == 32.5
    assert report["benchmark_protocol"]["timer_boundaries"]["start"] == (
        "before_command_parsing_manifest_validation_identity_hashing_artifact_setup_"
        "and_continuation_verification"
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
            lambda artifact, declaration: declaration.update(creation_elapsed_seconds=99.0),
            "does not corroborate creation elapsed seconds",
        ),
        (
            lambda artifact, declaration: declaration.update(
                creation_protocol="different compiler"
            ),
            "does not corroborate creation protocol",
        ),
        (
            lambda artifact, declaration: declaration.update(
                reuse_conditions=["different conditions"]
            ),
            "does not corroborate reuse conditions",
        ),
        (
            lambda artifact, declaration: (artifact.parent / "compiled-kernel.creation.json").chmod(
                stat.S_IRUSR | stat.S_IWUSR
            ),
            "creation receipt is mutable",
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


def test_public_command_recovers_after_actual_child_crash(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-hard-crash-once-at-step", "10"],
    )
    crashed_output = tmp_path / "crashed-report.json"

    crashed = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(crashed_output),
    )

    assert crashed.returncode == 0, crashed.stderr
    crashed_report = json.loads(crashed_output.read_text(encoding="utf-8"))
    interrupted = crashed_report["attempts"][0]
    assert interrupted["status"] == "interrupted"
    assert interrupted["recovery"]["progress_step"] == 5

    recovered_output = tmp_path / "recovered-report.json"
    recovered = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(recovered_output),
        "--merge",
        str(crashed_output),
    )

    assert recovered.returncode == 0, recovered.stderr
    recovered_report = json.loads(recovered_output.read_text(encoding="utf-8"))
    assert recovered_report["attempts"][0]["status"] == "succeeded"
    assert recovered_report["attempts"][0]["attempt_identity"] == interrupted["attempt_identity"]
    assert (
        recovered_report["attempts"][0]["reusable_elapsed_seconds"]
        >= interrupted["reusable_elapsed_seconds"]
    )


def test_public_command_recovers_after_parent_process_interruption(
    tmp_path: Path,
) -> None:
    checkpoint_ready = tmp_path / "checkpoint-ready"
    child_exited = tmp_path / "child-exited"
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=[
            "--fixture-hard-crash-once-at-step",
            "10",
            "--fixture-hold-after-recovery-checkpoint-marker",
            str(checkpoint_ready),
            "--fixture-recovery-child-exited-marker",
            str(child_exited),
        ],
    )
    command = shutil.which("gradoom-evidence")
    assert command is not None
    lost_output = tmp_path / "lost-report.json"
    parent = subprocess.Popen(
        [command, "--manifest", str(manifest), "--output", str(lost_output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 10.0
    while not checkpoint_ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert checkpoint_ready.exists()
    parent.terminate()
    assert parent.wait(timeout=5.0) != 0
    deadline = time.monotonic() + 5.0
    while not child_exited.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_exited.exists()
    assert not lost_output.exists()

    recovered_output = tmp_path / "recovered-report.json"
    recovered = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(recovered_output),
    )

    assert recovered.returncode == 0, recovered.stderr
    report = json.loads(recovered_output.read_text(encoding="utf-8"))
    attempt = report["attempts"][0]
    assert attempt["status"] == "succeeded"
    assert len(attempt["recovery_history"]) == 1
    prior_elapsed = attempt["recovery_history"][0]["prior_reusable_elapsed_seconds"]
    assert prior_elapsed > 0.0
    assert attempt["reusable_elapsed_seconds"] >= prior_elapsed


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


def test_public_command_rejects_same_path_trainer_script_replacement(
    tmp_path: Path,
) -> None:
    trainer_script = tmp_path / "trainer.py"
    shutil.copyfile(FIXTURE_PROCESS, trainer_script)
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
        trainer_script=trainer_script,
    )
    interrupted_output = tmp_path / "interrupted-report.json"
    initial = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(interrupted_output),
    )
    assert initial.returncode == 0, initial.stderr
    trainer_script.write_text(
        trainer_script.read_text(encoding="utf-8") + "\n# replaced in place\n",
        encoding="utf-8",
    )

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued-report.json"),
        "--merge",
        str(interrupted_output),
    )

    assert result.returncode == 2
    assert "cannot continue benchmark with unlike run identity" in result.stderr


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
