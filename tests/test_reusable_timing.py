from __future__ import annotations

import base64
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gradoom.evidence import benchmark as benchmark_module
from gradoom.evidence import cli as cli_module
from gradoom.evidence.benchmark import (
    _attempt_journal_payload,
    build_development_benchmark_report,
)
from gradoom.evidence.report import _canonical_sha256

FIXTURE_PROCESS = Path(__file__).parent / "fixtures" / "evidence" / "fixture_benchmark_process.py"
EVALUATION_SEEDS = list(range(20_000, 20_100))
ANCHOR_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)


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
    trainer_command: list[str] | None = None,
    trainer_code_root: Path | None = None,
    declared_inputs: list[dict[str, object]] | None = None,
    elapsed_time_anchor: bool = True,
    training_seeds: list[int] | None = None,
) -> Path:
    effective_training_seeds = training_seeds or [123]
    benchmark: dict[str, object] = {
        "training_seeds": effective_training_seeds,
        "failure_budget_steps": 10,
        "checkpoint_steps": [10],
        "evaluation_episode_seeds": EVALUATION_SEEDS,
        "trainer": {
            "command": trainer_command or [sys.executable, str(trainer_script)],
            "code_root": str(trainer_code_root or trainer_script.parent),
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
    if elapsed_time_anchor:
        benchmark["elapsed_time_anchors"] = []
        for seed in effective_training_seeds:
            anchor_payload = {
                "schema_version": 1,
                "authority": "gradoom-fixture-independent-anchor-v1",
                "seed": seed,
                "started_unix_ns": time.time_ns(),
            }
            anchor_bytes = json.dumps(
                anchor_payload, sort_keys=True, separators=(",", ":")
            ).encode()
            benchmark["elapsed_time_anchors"].append(
                {
                    "payload": anchor_payload,
                    "public_key": base64.b64encode(
                        ANCHOR_PRIVATE_KEY.public_key().public_bytes(
                            serialization.Encoding.Raw,
                            serialization.PublicFormat.Raw,
                        )
                    ).decode(),
                    "signature": base64.b64encode(ANCHOR_PRIVATE_KEY.sign(anchor_bytes)).decode(),
                }
            )
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
        "declared_inputs": declared_inputs or [],
        "benchmark": benchmark,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _bootstrap_artifact(
    tmp_path: Path,
    *,
    payload: bytes | None = None,
    compiler_target: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    immutable_input = tmp_path / "compiler-target.json"
    immutable_input.write_text(
        json.dumps(compiler_target or {"target": "generic"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    immutable_input_sha256 = hashlib.sha256(immutable_input.read_bytes()).hexdigest()
    binding = {
        "contract": "gradoom-declarative-bootstrap-v1",
        "immutable_inputs": [
            {"name": "compiler-target", "sha256": immutable_input_sha256},
        ],
        "protocol": "canonical-declared-input-binding-v1",
    }
    artifact = tmp_path / "compiled-kernel.json"
    artifact.write_bytes(
        payload
        if payload is not None
        else (json.dumps(binding, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )
    artifact.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    artifact_stat = artifact.stat()
    artifact_identity = {
        "resolved_path": str(artifact.resolve()),
        "device": artifact_stat.st_dev,
        "inode": artifact_stat.st_ino,
    }
    reuse_conditions = [
        "exact compiler and target identity",
        "read-only bytes reused without transformation",
    ]
    declaration: dict[str, object] = {
        "name": "compiled-kernel",
        "path": artifact.name,
        "sha256": artifact_sha256,
        "creation_elapsed_seconds": 12.5,
        "contract": binding["contract"],
        "immutable_inputs": binding["immutable_inputs"],
        "creation_protocol": binding["protocol"],
        "reuse_conditions": reuse_conditions,
        "persistent": True,
        "run_independent": True,
        "reused_unchanged": True,
    }
    attestation_payload = {
        "schema_version": 1,
        "authority": "gradoom-fixture-independent-anchor-v1",
        "artifact_name": declaration["name"],
        "artifact_sha256": declaration["sha256"],
        "creation_elapsed_seconds": declaration["creation_elapsed_seconds"],
        "creation_protocol": declaration["creation_protocol"],
        "immutable_inputs": declaration["immutable_inputs"],
        "reuse_conditions": declaration["reuse_conditions"],
        "artifact_identity": artifact_identity,
        "creation_event": {
            "event_id": "fixture-creation-event",
            "artifact_sha256": declaration["sha256"],
        },
        "prior_reuse_event": {
            "event_id": "fixture-prior-reuse-event",
            "artifact_sha256": declaration["sha256"],
            "artifact_identity": artifact_identity,
        },
    }
    declaration["eligibility_attestation"] = {
        "payload": attestation_payload,
        "public_key": base64.b64encode(
            ANCHOR_PRIVATE_KEY.public_key().public_bytes(
                serialization.Encoding.Raw,
                serialization.PublicFormat.Raw,
            )
        ).decode(),
        "signature": base64.b64encode(
            ANCHOR_PRIVATE_KEY.sign(
                json.dumps(attestation_payload, sort_keys=True, separators=(",", ":")).encode()
            )
        ).decode(),
    }
    return artifact, declaration


def _bootstrap_declared_inputs(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "compiler-target.json"
    return [
        {
            "name": "compiler-target",
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]


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
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
    )

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "does not match the canonical declarative bootstrap contract" in result.stderr


def test_public_command_rejects_forged_self_authored_bootstrap_receipt(tmp_path: Path) -> None:
    _artifact, declaration = _bootstrap_artifact(tmp_path)
    declaration["creation_receipt"] = {
        "path": "forged.json",
        "sha256": "0" * 64,
    }
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "out.json"))

    assert result.returncode == 2
    assert "creation_receipt" in result.stderr


def test_public_command_rejects_signed_seed_specific_bootstrap_input(tmp_path: Path) -> None:
    _artifact, declaration = _bootstrap_artifact(
        tmp_path, compiler_target={"target": "generic", "training_seed": 123}
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "compiler-target input has unsupported fields" in result.stderr


def test_public_command_rejects_bootstrap_without_prior_reuse_event(tmp_path: Path) -> None:
    _artifact, declaration = _bootstrap_artifact(tmp_path)
    attestation = declaration["eligibility_attestation"]
    assert isinstance(attestation, dict)
    payload = attestation["payload"]
    assert isinstance(payload, dict)
    payload.pop("prior_reuse_event")
    attestation["signature"] = base64.b64encode(
        ANCHOR_PRIVATE_KEY.sign(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    ).decode()
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "prior unchanged reuse event" in result.stderr


def test_public_command_rejects_bootstrap_signature_replayed_at_new_location(
    tmp_path: Path,
) -> None:
    artifact, declaration = _bootstrap_artifact(tmp_path)
    copied = tmp_path / "copied-kernel.json"
    shutil.copyfile(artifact, copied)
    copied.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    declaration["path"] = copied.name
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "externally attested artifact object identity" in result.stderr


def test_public_command_rejects_deleted_and_recreated_bootstrap_object(tmp_path: Path) -> None:
    artifact, declaration = _bootstrap_artifact(tmp_path)
    payload = artifact.read_bytes()
    artifact.unlink()
    (tmp_path / "inode-occupier").write_text("occupy old object\n", encoding="utf-8")
    artifact.write_bytes(payload)
    artifact.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "externally attested artifact object identity" in result.stderr


def test_public_command_accepts_only_disclosed_immutable_bootstrap_exclusion(
    tmp_path: Path,
) -> None:
    artifact, declaration = _bootstrap_artifact(tmp_path)
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        bootstrap_artifacts=[declaration],
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
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
        "terminal_evidence_verification",
        "report_validation_serialization_replacement_and_fsync",
        "durable_authority_elapsed_seal",
    ]
    assert report["bootstrap_exclusions"] == [
        {
            **declaration,
            "path": str(artifact.resolve()),
            "eligibility_evidence": {
                "contract": "gradoom-declarative-bootstrap-v1",
                "derivation": "canonical-declared-input-binding-v1",
                "opaque_payload_allowed": False,
            },
            "validated_before_cohort": True,
            "reverified_unchanged_after_cohort": True,
        }
    ]
    assert {(entry["name"], entry["sha256"]) for entry in report["evidence_index"]["entries"]} >= {
        ("bootstrap-compiled-kernel", declaration["sha256"]),
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


def test_reusable_setup_time_is_included_for_every_active_seed(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        training_seeds=[123, 124],
    )

    report = build_development_benchmark_report(
        manifest,
        invocation_started=10.0,
        clock=lambda: 42.5,
    )

    assert [attempt["reusable_elapsed_seconds"] for attempt in report["attempts"]] == [32.5, 32.5]


def test_terminal_elapsed_includes_durable_journal_and_authority_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path, outcomes={"10": [30.0, 0.0]})
    now = [10.0]
    original_write = benchmark_module._write_durable_json
    original_sign = benchmark_module._sign_generation_attestation

    def delayed_write(*args: object, **kwargs: object) -> str:
        result = original_write(*args, **kwargs)
        if kwargs.get("field") == "benchmark attempt journal":
            now[0] += 40.0
        return result

    def delayed_sign(*args: object, **kwargs: object) -> dict[str, object]:
        now[0] += 60.0
        return original_sign(*args, **kwargs)

    monkeypatch.setattr(benchmark_module, "_write_durable_json", delayed_write)
    monkeypatch.setattr(benchmark_module, "_sign_generation_attestation", delayed_sign)

    report = build_development_benchmark_report(
        manifest,
        invocation_started=10.0,
        clock=lambda: now[0],
    )

    assert report["attempts"][0]["reusable_elapsed_seconds"] >= 100.0


def test_terminal_elapsed_seal_covers_final_verification_and_report_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path, outcomes={"10": [30.0, 0.0]})
    output = tmp_path / "report.json"
    now = [10.0]
    original_verify = benchmark_module._reverify_trainer_files
    original_write = cli_module._write_report

    def delayed_verify(trainer: dict[str, object]) -> None:
        original_verify(trainer)
        now[0] += 20.0

    def delayed_write(path: Path, report: dict[str, object]) -> None:
        original_write(path, report)
        now[0] += 30.0

    monkeypatch.setattr(benchmark_module, "_reverify_trainer_files", delayed_verify)
    monkeypatch.setattr(cli_module, "_write_report", delayed_write)
    monkeypatch.setattr(cli_module.time, "perf_counter", lambda: now[0])
    monkeypatch.setitem(
        benchmark_module.build_development_benchmark_report.__kwdefaults__,
        "clock",
        lambda: now[0],
    )

    result = cli_module.main(["--manifest", str(manifest), "--output", str(output)])

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["attempts"][0]["reusable_elapsed_seconds"] >= now[0] - 10.0
    assert (
        report["attempts"][0]["attempt_journal"]["authority_attestation"]["payload"][
            "reusable_elapsed_seconds"
        ]
        == report["attempts"][0]["reusable_elapsed_seconds"]
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
            "eligibility_attestation has no valid creation event",
        ),
        (
            lambda artifact, declaration: declaration.update(
                creation_protocol="different compiler"
            ),
            "canonical declarative bootstrap protocol",
        ),
        (
            lambda artifact, declaration: declaration.update(contract="opaque-v1"),
            "must use gradoom-declarative-bootstrap-v1",
        ),
        (
            lambda artifact, declaration: declaration.update(immutable_inputs=[]),
            "immutable_inputs must contain at least one",
        ),
        (
            lambda artifact, declaration: declaration.update(creation_receipt={}),
            "creation_receipt",
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
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
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
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
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
        declared_inputs=_bootstrap_declared_inputs(tmp_path),
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


def test_public_command_does_not_advertise_recovery_without_signed_elapsed_anchor(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
        elapsed_time_anchor=False,
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 2
    assert "elapsed_time_anchors are required" in result.stderr
    assert not output.exists()


def test_public_command_rejects_invalid_elapsed_anchor_signature(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, outcomes={"10": [30.0, 0.0]})
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["benchmark"]["elapsed_time_anchors"][0]["signature"] = base64.b64encode(
        bytes(64)
    ).decode()
    manifest.write_text(json.dumps(document), encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "signature is invalid" in result.stderr


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


def test_public_command_forwards_and_recovers_parent_signal_during_evaluation(
    tmp_path: Path,
) -> None:
    evaluation_started = tmp_path / "evaluation-started"
    child_exited = tmp_path / "evaluation-child-exited"
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=[
            "--fixture-hold-evaluation-once-marker",
            str(evaluation_started),
            "--fixture-evaluation-child-exited-marker",
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
    while not evaluation_started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert evaluation_started.exists()
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
    attempt = json.loads(recovered_output.read_text(encoding="utf-8"))["attempts"][0]
    assert attempt["status"] == "succeeded"
    assert attempt["recovery_history"][-1]["resumed_phase"] == "evaluation"
    assert attempt["recovery_history"][-1]["prior_reusable_elapsed_seconds"] > 0.0


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


def test_failed_recovery_appends_attempt_journal_without_overwriting_prior(
    tmp_path: Path,
) -> None:
    fail_marker = tmp_path / "resume-failed"
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=[
            "--fixture-interrupt-once-at-step",
            "10",
            "--fixture-fail-after-resume-once-marker",
            str(fail_marker),
        ],
    )
    interrupted_output = tmp_path / "interrupted.json"
    initial = _run_evidence("--manifest", str(manifest), "--output", str(interrupted_output))
    assert initial.returncode == 0, initial.stderr
    initial_report = json.loads(interrupted_output.read_text(encoding="utf-8"))
    original_journal = Path(initial_report["attempts"][0]["attempt_journal"]["path"])
    original_bytes = original_journal.read_bytes()

    failed_output = tmp_path / "failed.json"
    failed = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(failed_output),
        "--merge",
        str(interrupted_output),
    )

    assert failed.returncode == 0, failed.stderr
    failed_report = json.loads(failed_output.read_text(encoding="utf-8"))
    assert failed_report["attempts"][0]["status"] == "crashed"
    assert original_journal.read_bytes() == original_bytes
    assert Path(failed_report["attempts"][0]["attempt_journal"]["path"]).name == (
        "attempt-state-1.json"
    )


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
    if mismatch == "seed":
        assert "elapsed_time_anchors" in result.stderr
    else:
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


def test_public_command_rejects_shell_eval_trainer_indirection(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_command=[sys.executable, "-c", "exec(open('hidden.py').read())"],
        trainer_code_root=tmp_path,
    )

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "shell, -c, eval, and opaque trainer indirection are forbidden" in result.stderr


def test_public_command_rejects_same_path_hidden_trainer_mutation(tmp_path: Path) -> None:
    code_root = tmp_path / "trainer-code"
    code_root.mkdir()
    hidden = code_root / "hidden_trainer.py"
    shutil.copyfile(FIXTURE_PROCESS, hidden)
    launcher = code_root / "launcher.py"
    launcher.write_text(
        "from hidden_trainer import main\nraise SystemExit(main())\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
        trainer_script=launcher,
        trainer_code_root=code_root,
    )
    interrupted = tmp_path / "interrupted.json"
    initial = _run_evidence("--manifest", str(manifest), "--output", str(interrupted))
    assert initial.returncode == 0, initial.stderr
    hidden.write_text(
        hidden.read_text(encoding="utf-8") + "\n# hidden mutation\n",
        encoding="utf-8",
    )

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued.json"),
        "--merge",
        str(interrupted),
    )

    assert result.returncode == 2
    assert "cannot continue benchmark with unlike run identity" in result.stderr


def test_public_command_rejects_trainer_code_changed_during_cohort(tmp_path: Path) -> None:
    code_root = tmp_path / "trainer-code"
    code_root.mkdir()
    trainer = code_root / "trainer.py"
    shutil.copyfile(FIXTURE_PROCESS, trainer)
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-mutate-trainer-code", str(trainer)],
        trainer_script=trainer,
        trainer_code_root=code_root,
    )

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "bound trainer file changed during the cohort" in result.stderr


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
    assert "authority attestation" in result.stderr


def test_public_command_rejects_consistently_rewritten_local_elapsed_evidence(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
    )
    interrupted = tmp_path / "interrupted.json"
    initial = _run_evidence("--manifest", str(manifest), "--output", str(interrupted))
    assert initial.returncode == 0, initial.stderr
    report = json.loads(interrupted.read_text(encoding="utf-8"))
    attempt = report["attempts"][0]
    attempt["reusable_elapsed_seconds"] = 0.0
    attempt["recovery"]["accumulated_reusable_elapsed_seconds"] = 0.0
    recovery_path = Path(attempt["recovery_journal"]["path"])
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recovery["accumulated_reusable_elapsed_seconds"] = 0.0
    recovery_path.write_text(
        json.dumps(recovery, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    recovery_sha = hashlib.sha256(recovery_path.read_bytes()).hexdigest()
    attempt["recovery_journal"]["sha256"] = recovery_sha
    journal_path = Path(attempt["attempt_journal"]["path"])
    journal_path.write_text(
        json.dumps(
            _attempt_journal_payload(attempt, run_identity=report["run_identity"]),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    journal_sha = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    attempt["attempt_journal"]["sha256"] = journal_sha
    path_hashes = {str(recovery_path): recovery_sha, str(journal_path): journal_sha}
    names_by_path = {item["path"]: item["name"] for item in report["generated_artifacts"]}
    for path, digest in path_hashes.items():
        name = names_by_path[path]
        next(entry for entry in report["evidence_index"]["entries"] if entry["name"] == name)[
            "sha256"
        ] = digest
    report["evidence_index"]["sha256"] = _canonical_sha256(
        report["evidence_index"]["entries"], document="forged evidence index"
    )
    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps(report), encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued.json"),
        "--merge",
        str(forged),
    )

    assert result.returncode == 2
    assert "attempt journal generation is invalid" in result.stderr


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
    assert "authority attestation" in result.stderr


def test_public_command_rejects_consistently_rewritten_completed_elapsed(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, outcomes={"10": [30.0, 0.0]})
    completed = tmp_path / "completed.json"
    initial = _run_evidence("--manifest", str(manifest), "--output", str(completed))
    assert initial.returncode == 0, initial.stderr
    report = json.loads(completed.read_text(encoding="utf-8"))
    attempt = report["attempts"][0]
    attempt["reusable_elapsed_seconds"] = 0.0
    journal_path = Path(attempt["attempt_journal"]["path"])
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["reusable_elapsed_seconds"] = 0.0
    journal_path.write_text(
        json.dumps(journal, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    journal_sha = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    attempt["attempt_journal"]["sha256"] = journal_sha
    artifact_name = next(
        item["name"] for item in report["generated_artifacts"] if item["path"] == str(journal_path)
    )
    next(item for item in report["evidence_index"]["entries"] if item["name"] == artifact_name)[
        "sha256"
    ] = journal_sha
    report["evidence_index"]["sha256"] = _canonical_sha256(
        report["evidence_index"]["entries"], document="forged completed index"
    )
    forged = tmp_path / "forged-completed.json"
    forged.write_text(json.dumps(report), encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued.json"),
        "--merge",
        str(forged),
    )

    assert result.returncode == 2
    assert "authority attestation" in result.stderr


def test_public_command_rejects_stale_interrupted_report_after_terminal_generation(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "resume-failed"
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        extra_arguments=[
            "--fixture-interrupt-once-at-step",
            "10",
            "--fixture-fail-after-resume-once-marker",
            str(marker),
        ],
    )
    interrupted = tmp_path / "interrupted.json"
    first = _run_evidence("--manifest", str(manifest), "--output", str(interrupted))
    assert first.returncode == 0, first.stderr
    terminal = tmp_path / "terminal.json"
    second = _run_evidence(
        "--manifest", str(manifest), "--output", str(terminal), "--merge", str(interrupted)
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(terminal.read_text(encoding="utf-8"))["attempts"][0]["status"] == "crashed"

    replay = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "replayed.json"),
        "--merge",
        str(interrupted),
    )

    assert replay.returncode == 2
    assert "stale attempt journal generation" in replay.stderr


def test_trainer_relative_executable_is_resolved_from_manifest_directory(
    tmp_path: Path,
) -> None:
    trainer = tmp_path / "trainer.py"
    shutil.copyfile(FIXTURE_PROCESS, trainer)
    launcher = tmp_path / "trainer"
    launcher.symlink_to(Path(sys.executable))
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_script=Path("trainer.py"),
        trainer_command=["./trainer", "trainer.py"],
        trainer_code_root=tmp_path,
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 0, result.stderr


def test_public_command_binds_from_package_import_submodule(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    package = code_root / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    worker = package / "worker.py"
    shutil.copyfile(FIXTURE_PROCESS, worker)
    launcher = code_root / "launcher.py"
    launcher.write_text(
        "from pkg import worker\nraise SystemExit(worker.main())\n", encoding="utf-8"
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_script=launcher,
        trainer_code_root=code_root,
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
    )
    interrupted = tmp_path / "interrupted.json"
    first = _run_evidence("--manifest", str(manifest), "--output", str(interrupted))
    assert first.returncode == 0, first.stderr
    worker.write_text(worker.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued.json"),
        "--merge",
        str(interrupted),
    )

    assert result.returncode == 2
    assert "unlike run identity" in result.stderr


def test_public_command_binds_executed_parent_package_initializers(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    package = code_root / "pkg"
    package.mkdir(parents=True)
    initializer = package / "__init__.py"
    initializer.write_text("PACKAGE_VALUE = 1\n", encoding="utf-8")
    worker = package / "worker.py"
    shutil.copyfile(FIXTURE_PROCESS, worker)
    launcher = code_root / "launcher.py"
    launcher.write_text(
        "import pkg.worker as worker\nraise SystemExit(worker.main())\n", encoding="utf-8"
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_script=launcher,
        trainer_code_root=code_root,
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
    )
    interrupted = tmp_path / "interrupted.json"
    first = _run_evidence("--manifest", str(manifest), "--output", str(interrupted))
    assert first.returncode == 0, first.stderr
    initializer.write_text("PACKAGE_VALUE = 2\n", encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "next.json"),
        "--merge",
        str(interrupted),
    )

    assert result.returncode == 2
    assert "unlike run identity" in result.stderr


def test_public_command_binds_relative_from_import_submodule(tmp_path: Path) -> None:
    code_root = tmp_path / "code"
    package = code_root / "pkg"
    helpers = package / "helpers"
    helpers.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (helpers / "__init__.py").write_text("", encoding="utf-8")
    worker = helpers / "worker.py"
    shutil.copyfile(FIXTURE_PROCESS, worker)
    launcher = package / "launcher.py"
    launcher.write_text(
        "from .helpers import worker\nraise SystemExit(worker.main())\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_script=launcher,
        trainer_code_root=code_root,
        extra_arguments=["--fixture-interrupt-once-at-step", "10"],
    )
    interrupted = tmp_path / "interrupted.json"
    first = _run_evidence("--manifest", str(manifest), "--output", str(interrupted))
    assert first.returncode == 0, first.stderr
    worker.write_text(worker.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "continued.json"),
        "--merge",
        str(interrupted),
    )

    assert result.returncode == 2
    assert "unlike run identity" in result.stderr


def test_public_command_rejects_os_exec_trainer_indirection(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import os\nos.execv('/tmp/hidden-trainer', ['/tmp/hidden-trainer'])\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_script=launcher,
        trainer_code_root=tmp_path,
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "opaque trainer indirection" in result.stderr


def test_public_command_rejects_nonliteral_dynamic_exec_indirection(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import os\nname = 'exec' + 'v'\nrunners = [os]\n"
        "getattr(runners[0], name)('/tmp/hidden', ['/tmp/hidden'])\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_script=launcher,
        trainer_code_root=tmp_path,
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "opaque trainer indirection" in result.stderr


@pytest.mark.parametrize(
    "launcher_source",
    [
        "import os\nos.__dict__['exec' + 'v']('/tmp/hidden', ['/tmp/hidden'])\n",
        "import os\nvars(os)['exec' + 'v']('/tmp/hidden', ['/tmp/hidden'])\n",
        ("import os\ngetattr(os, '__dict__')['exec' + 'v']('/tmp/hidden', ['/tmp/hidden'])\n"),
        (
            "import os\nrunners = {'selected': os}\n"
            "runners['selected'].__dict__['exec' + 'v']('/tmp/hidden', ['/tmp/hidden'])\n"
        ),
        ("import os\nglobals()['os'].__dict__['exec' + 'v']('/tmp/hidden', ['/tmp/hidden'])\n"),
    ],
)
def test_public_command_rejects_computed_module_namespace_exec_indirection(
    tmp_path: Path, launcher_source: str
) -> None:
    launcher = tmp_path / "launcher.py"
    launcher.write_text(launcher_source, encoding="utf-8")
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [30.0, 0.0]},
        trainer_script=launcher,
        trainer_code_root=tmp_path,
    )

    result = _run_evidence("--manifest", str(manifest), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "opaque trainer indirection" in result.stderr


def test_documented_train_python_closure_allows_importlib_resources() -> None:
    repository = Path(__file__).resolve().parents[1]

    closure = benchmark_module._python_source_closure(repository / "train.py", repository)

    assert repository / "src/gradoom/scenario.py" in closure
