from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from gradoom.evidence.benchmark import _validate_elapsed_time_anchors
from gradoom.evidence.report import EvidenceError
from gradoom.evidence.time_authority import ReusableTimeAuthority


def _authority(
    state: Path,
    operation: str,
    payload: dict[str, object] | None = None,
    *,
    environment: dict[str, str] | None = None,
):
    command = shutil.which("gradoom-time-authority")
    assert command is not None
    return subprocess.run(
        [command, "--state-directory", str(state), operation],
        input=None if payload is None else json.dumps(payload),
        check=False,
        capture_output=True,
        text=True,
        env=None if environment is None else {**os.environ, **environment},
    )


def _artifact_request(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    artifact = tmp_path / "compiled-kernel.json"
    return artifact, {
        "artifact_name": "compiled-kernel",
        "artifact_path": str(artifact),
        "creation_protocol": "canonical-declared-input-binding-v1",
        "immutable_inputs": [{"name": "compiler-target", "sha256": "1" * 64}],
        "reuse_conditions": [
            "exact compiler and target identity",
            "read-only bytes reused without transformation",
        ],
    }


def test_repository_authority_requires_chronological_unchanged_reuse(tmp_path: Path) -> None:
    state = tmp_path / "authority"
    initialized = _authority(state, "init")
    assert initialized.returncode == 0, initialized.stderr
    artifact, request = _artifact_request(tmp_path)
    created = _authority(state, "create-bootstrap", request)
    assert created.returncode == 0, created.stderr
    declaration = json.loads(created.stdout)

    premature = _authority(state, "attest-bootstrap-reuse", declaration)
    assert premature.returncode == 2
    assert "distinct prior reuse" in premature.stderr

    reused = _authority(state, "record-bootstrap-reuse", declaration)
    assert reused.returncode == 0, reused.stderr
    attested = _authority(state, "attest-bootstrap-reuse", declaration)
    assert attested.returncode == 0, attested.stderr
    attestation = json.loads(attested.stdout)
    creation = attestation["payload"]["creation_event"]
    prior_reuse = attestation["payload"]["prior_reuse_event"]
    assert creation["sequence"] < prior_reuse["sequence"]
    assert creation["event_id"] != prior_reuse["event_id"]
    assert attestation["payload"]["artifact_identity"]["resolved_path"] == str(artifact.resolve())


def test_repository_authority_rejects_ledger_replay_and_state_reset(tmp_path: Path) -> None:
    state = tmp_path / "authority"
    assert _authority(state, "init").returncode == 0
    _artifact, request = _artifact_request(tmp_path)
    created = _authority(state, "create-bootstrap", request)
    assert created.returncode == 0, created.stderr
    declaration = json.loads(created.stdout)
    old_ledger = (state / "ledger.json").read_bytes()
    assert _authority(state, "record-bootstrap-reuse", declaration).returncode == 0
    attested = _authority(state, "attest-bootstrap-reuse", declaration)
    assert attested.returncode == 0, attested.stderr
    attestation = json.loads(attested.stdout)

    (state / "ledger.json").write_bytes(old_ledger)
    replay = _authority(state, "attest-bootstrap-reuse", declaration)
    assert replay.returncode == 2
    assert "rollback" in replay.stderr

    shutil.rmtree(state)
    refused_reset = _authority(state, "init")
    assert refused_reset.returncode == 2
    assert "witness directory is not empty" in refused_reset.stderr
    shutil.rmtree(ReusableTimeAuthority.default_witness_directory(state))
    assert _authority(state, "init").returncode == 0
    reset = _authority(state, "verify-bootstrap-reuse", attestation)
    assert reset.returncode == 2
    assert "authority identity" in reset.stderr


def test_repository_authority_rejects_bootstrap_object_replacement(tmp_path: Path) -> None:
    state = tmp_path / "authority"
    assert _authority(state, "init").returncode == 0
    artifact, request = _artifact_request(tmp_path)
    created = _authority(state, "create-bootstrap", request)
    assert created.returncode == 0, created.stderr
    declaration = json.loads(created.stdout)
    artifact.unlink()
    artifact.write_text('{"contract":"replacement"}\n', encoding="utf-8")
    artifact.chmod(0o444)

    reused = _authority(state, "record-bootstrap-reuse", declaration)

    assert reused.returncode == 2
    assert "object identity" in reused.stderr


def test_repository_authority_rejects_replayed_journal_head(tmp_path: Path) -> None:
    state = tmp_path / "authority"
    assert _authority(state, "init").returncode == 0
    anchor_result = _authority(state, "start-attempt", {"seed": 123})
    assert anchor_result.returncode == 0, anchor_result.stderr
    anchor = json.loads(anchor_result.stdout)
    first_head = {
        "schema_version": 1,
        "authority": anchor["payload"]["authority"],
        "seed": 123,
        "started_unix_ns": anchor["payload"]["started_unix_ns"],
        "generation": 0,
        "previous_journal_sha256": None,
        "journal_sha256": "2" * 64,
        "status": "interrupted",
        "prior_reusable_elapsed_seconds": 0.0,
        "minimum_reusable_elapsed_seconds": 1.0,
    }
    first = _authority(state, "sign-journal-head", first_head)
    assert first.returncode == 0, first.stderr
    first_attestation = json.loads(first.stdout)
    second_head = {
        **first_head,
        "generation": 1,
        "previous_journal_sha256": "2" * 64,
        "journal_sha256": "3" * 64,
        "status": "succeeded",
        "prior_reusable_elapsed_seconds": first_attestation["payload"]["reusable_elapsed_seconds"],
    }
    second = _authority(state, "sign-journal-head", second_head)
    assert second.returncode == 0, second.stderr

    replay = _authority(state, "verify-latest-journal-head", first_attestation)

    assert replay.returncode == 2
    assert "stale journal head" in replay.stderr
    recovery = _authority(state, "recover-latest-journal-head", first_attestation)
    assert recovery.returncode == 2
    assert "different generation" in recovery.stderr


def test_repository_authority_seals_seed_local_elapsed_not_precommand_anchor_age(
    tmp_path: Path,
) -> None:
    state = tmp_path / "authority"
    assert _authority(state, "init").returncode == 0
    anchor_result = _authority(state, "start-attempt", {"seed": 124})
    assert anchor_result.returncode == 0, anchor_result.stderr
    anchor = json.loads(anchor_result.stdout)
    # This gap represents another seed's work after all pre-command anchors were issued.
    time.sleep(0.3)

    sealed = _authority(
        state,
        "sign-journal-head",
        {
            "schema_version": 1,
            "authority": anchor["payload"]["authority"],
            "seed": 124,
            "started_unix_ns": anchor["payload"]["started_unix_ns"],
            "generation": 0,
            "previous_journal_sha256": None,
            "journal_sha256": "4" * 64,
            "status": "succeeded",
            "prior_reusable_elapsed_seconds": 0.0,
            "minimum_reusable_elapsed_seconds": 0.01,
        },
    )

    assert sealed.returncode == 0, sealed.stderr
    assert json.loads(sealed.stdout)["payload"]["reusable_elapsed_seconds"] == 0.01


def test_repository_authority_rejects_coherent_state_directory_rollback(
    tmp_path: Path,
) -> None:
    state = tmp_path / "authority"
    snapshot = tmp_path / "authority-snapshot"
    assert _authority(state, "init").returncode == 0
    anchor_result = _authority(state, "start-attempt", {"seed": 123})
    assert anchor_result.returncode == 0, anchor_result.stderr
    anchor = json.loads(anchor_result.stdout)
    first_request = {
        "schema_version": 1,
        "authority": anchor["payload"]["authority"],
        "seed": 123,
        "started_unix_ns": anchor["payload"]["started_unix_ns"],
        "generation": 0,
        "previous_journal_sha256": None,
        "journal_sha256": "2" * 64,
        "status": "interrupted",
        "prior_reusable_elapsed_seconds": 0.0,
        "minimum_reusable_elapsed_seconds": 1.0,
    }
    first = _authority(state, "sign-journal-head", first_request)
    assert first.returncode == 0, first.stderr
    first_attestation = json.loads(first.stdout)
    shutil.copytree(state, snapshot)
    second_request = {
        **first_request,
        "generation": 1,
        "previous_journal_sha256": "2" * 64,
        "journal_sha256": "3" * 64,
        "status": "succeeded",
        "prior_reusable_elapsed_seconds": first_attestation["payload"]["reusable_elapsed_seconds"],
        "minimum_reusable_elapsed_seconds": 10.0,
    }
    second = _authority(state, "sign-journal-head", second_request)
    assert second.returncode == 0, second.stderr
    shutil.rmtree(state)
    shutil.copytree(snapshot, state)

    replay = _authority(state, "verify-latest-journal-head", first_attestation)

    assert replay.returncode == 2
    assert "rollback" in replay.stderr


def test_repository_authority_recovers_durable_report_after_same_head_reseal(
    tmp_path: Path,
) -> None:
    state = tmp_path / "authority"
    assert _authority(state, "init").returncode == 0
    anchor_result = _authority(state, "start-attempt", {"seed": 123})
    assert anchor_result.returncode == 0, anchor_result.stderr
    anchor = json.loads(anchor_result.stdout)
    request = {
        "schema_version": 1,
        "authority": anchor["payload"]["authority"],
        "seed": 123,
        "started_unix_ns": anchor["payload"]["started_unix_ns"],
        "generation": 0,
        "previous_journal_sha256": None,
        "journal_sha256": "2" * 64,
        "status": "succeeded",
        "prior_reusable_elapsed_seconds": 0.0,
        "minimum_reusable_elapsed_seconds": 1.0,
    }
    first = _authority(state, "sign-journal-head", request)
    assert first.returncode == 0, first.stderr
    durable_report_attestation = json.loads(first.stdout)
    resealed = _authority(
        state,
        "sign-journal-head",
        {
            **request,
            "prior_reusable_elapsed_seconds": durable_report_attestation["payload"][
                "reusable_elapsed_seconds"
            ],
            "minimum_reusable_elapsed_seconds": 10.0,
        },
    )
    assert resealed.returncode == 0, resealed.stderr
    latest_attestation = json.loads(resealed.stdout)

    recovered = _authority(
        state,
        "recover-latest-journal-head",
        durable_report_attestation,
    )

    assert recovered.returncode == 0, recovered.stderr
    assert json.loads(recovered.stdout) == latest_attestation


@pytest.mark.parametrize("durable_step", ["ledger", "witness", "head"])
def test_repository_authority_recovers_interrupted_state_transition(
    tmp_path: Path, durable_step: str
) -> None:
    state = tmp_path / "authority"
    assert _authority(state, "init").returncode == 0

    interrupted = _authority(
        state,
        "start-attempt",
        {"seed": 123},
        environment={"GRADOOM_TIME_AUTHORITY_TEST_INTERRUPT_AFTER": durable_step},
    )

    assert interrupted.returncode == 91
    reopened = _authority(state, "identity")
    assert reopened.returncode == 0, reopened.stderr
    next_attempt = _authority(state, "start-attempt", {"seed": 456})
    assert next_attempt.returncode == 0, next_attempt.stderr
    assert json.loads(next_attempt.stdout)["payload"]["seed"] == 456


def test_formal_benchmark_binds_repository_authority_state_not_arbitrary_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "authority"
    assert _authority(state, "init").returncode == 0
    anchor_result = _authority(state, "start-attempt", {"seed": 123})
    assert anchor_result.returncode == 0, anchor_result.stderr
    anchor = json.loads(anchor_result.stdout)
    monkeypatch.setenv("GRADOOM_REUSABLE_TIME_AUTHORITY_STATE", str(state))
    witness = ReusableTimeAuthority.default_witness_directory(state)
    monkeypatch.setenv("GRADOOM_REUSABLE_TIME_AUTHORITY_WITNESS", str(witness))
    monkeypatch.setenv("GRADOOM_EVIDENCE_AUTHORITY", "/tmp/operator-selected-signer")

    assert _validate_elapsed_time_anchors([anchor], training_seeds=[123], fixture=False) == [anchor]

    shutil.rmtree(state)
    shutil.rmtree(witness)
    assert _authority(state, "init").returncode == 0
    with pytest.raises(EvidenceError, match="not rooted in the pinned public authority"):
        _validate_elapsed_time_anchors([anchor], training_seeds=[123], fixture=False)
