from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

FIXTURE_PROCESS = Path(__file__).parent / "fixtures" / "evidence" / "fixture_benchmark_process.py"
EVALUATION_SEEDS = list(range(10_000, 10_100))
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
    training_seeds: list[int] | None = None,
    checkpoint_steps: list[int] | None = None,
    extra_arguments: list[str] | None = None,
) -> Path:
    command_arguments = ["--fixture-outcomes", json.dumps(outcomes, sort_keys=True)]
    command_arguments.extend(extra_arguments or [])
    benchmark: dict[str, object] = {
        "failure_budget_steps": (checkpoint_steps or [10, 20, 30])[-1],
        "checkpoint_steps": checkpoint_steps or [10, 20, 30],
        "evaluation_episode_seeds": EVALUATION_SEEDS,
        "trainer": {
            "command": [sys.executable, str(FIXTURE_PROCESS)],
            "arguments": command_arguments,
        },
        "artifacts_directory": "benchmark-artifacts",
        "parity_certificate": {
            "available": False,
            "reason": "No current parity certificate exists for the fixture profile.",
        },
    }
    if training_seeds is not None:
        benchmark["training_seeds"] = training_seeds
    effective_seeds = training_seeds or [123]
    anchors = []
    for seed in effective_seeds:
        payload = {
            "schema_version": 1,
            "authority": "gradoom-fixture-independent-anchor-v1",
            "seed": seed,
            "started_unix_ns": time.time_ns(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        anchors.append(
            {
                "payload": payload,
                "public_key": base64.b64encode(
                    ANCHOR_PRIVATE_KEY.public_key().public_bytes(
                        serialization.Encoding.Raw,
                        serialization.PublicFormat.Raw,
                    )
                ).decode(),
                "signature": base64.b64encode(ANCHOR_PRIVATE_KEY.sign(encoded)).decode(),
            }
        )
    benchmark["elapsed_time_anchors"] = anchors
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


def test_development_benchmark_defaults_to_one_cold_seed_and_stops_at_first_pass(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        outcomes={"10": [29.99, 999.0], "20": [30.0, 0.0], "30": [45.0, 45.0]},
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workflow"] == "development_training_benchmark"
    assert report["evidence_level"] == "development"
    assert report["authoritative"] is False
    assert report["claim_eligible"] is False
    assert report["status"] == "passed"
    assert report["benchmark_protocol"]["training_seeds"] == [123]
    assert report["benchmark_protocol"]["failure_budget_steps"] == 30
    assert report["benchmark_protocol"]["quality_gate"] == {
        "episodes": 100,
        "mean_at_least": 30.0,
        "signal": "player_killcount",
        "stochastic_actions": True,
    }
    assert report["benchmark_protocol"]["evaluation_episode_seeds"] == EVALUATION_SEEDS
    assert report["benchmark_protocol"]["evaluation_action_seed"] == 123
    assert report["benchmark_protocol"]["time_authority"] == {
        "authority": "gradoom-fixture-independent-anchor-v1",
        "public_key": "MfMyLUkj02xBwQm9sAmRkxh77ZmUIJbkkmokx379DS8=",
        "monotonic_witness": None,
        "claim_eligible": False,
    }
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
    assert report["attempts"][0]["status"] == "succeeded"
    assert [candidate["checkpoint_step"] for candidate in report["attempts"][0]["outcomes"]] == [
        10,
        20,
    ]
    first, passing = report["attempts"][0]["outcomes"]
    assert first["mean_player_killcount"] == 29.99
    assert first["mean_killcount"] == 999.0
    assert first["passed"] is False
    assert passing["mean_player_killcount"] == 30.0
    assert passing["mean_killcount"] == 0.0
    assert passing["passed"] is True
    assert len(passing["episodes"]) == 100
    assert [episode["game_seed"] for episode in passing["episodes"]] == EVALUATION_SEEDS
    checkpoint = Path(report["attempts"][0]["checkpoint"])
    assert checkpoint.is_file()
    assert (
        report["attempts"][0]["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert report["attempts"][0]["reusable_elapsed_seconds"] >= 0.0
    assert {reason["code"] for reason in report["claim_reasons"]} == {
        "development_evidence",
        "fixture_evidence",
        "missing_current_parity_certificate",
    }


def test_development_benchmark_accepts_multiple_unique_seeds_and_retains_failures(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        training_seeds=[7, 8, 9],
        checkpoint_steps=[10],
        outcomes={"7:10": [31.0, 0.0], "8:10": [3.0, 90.0]},
        extra_arguments=["--fixture-fail-training-seed", "9"],
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert [attempt["seed"] for attempt in report["attempts"]] == [7, 8, 9]
    assert [attempt["status"] for attempt in report["attempts"]] == [
        "succeeded",
        "exhausted",
        "crashed",
    ]
    assert report["attempts"][1]["outcomes"][0]["mean_killcount"] == 90.0
    assert report["attempts"][1]["outcomes"][0]["passed"] is False
    assert report["attempts"][2]["failures"][0]["phase"] == "training"
    assert report["failures"] == report["attempts"][2]["failures"]
    assert report["claim_eligible"] is False
    assert report["authoritative"] is False


def test_development_benchmark_retains_process_launch_failures(tmp_path: Path) -> None:
    manifest_path = _manifest(
        tmp_path,
        checkpoint_steps=[10],
        outcomes={"10": [30.0, 30.0]},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark"]["trainer"]["command"] = [str(tmp_path / "missing-trainer")]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 2
    assert "executed-code closure cannot be proven" in result.stderr
    assert not output.exists()


def test_development_benchmark_rejects_legacy_kills_without_player_killcount(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        checkpoint_steps=[10],
        outcomes={"10": [31.0, 999.0]},
        extra_arguments=["--fixture-omit-player-killcount"],
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    attempt = report["attempts"][0]
    assert report["status"] == "failed"
    assert attempt["status"] == "evidence_failed"
    assert attempt["outcomes"] == []
    assert "episodes[0].player_killcount must be finite" in attempt["failures"][0]["message"]


def test_development_benchmark_rejects_training_past_predeclared_step(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        checkpoint_steps=[10],
        outcomes={"10": [31.0, 0.0]},
        extra_arguments=["--fixture-training-step-offset", "4086"],
    )
    output = tmp_path / "report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    attempt = report["attempts"][0]
    assert report["status"] == "failed"
    assert attempt["status"] == "evidence_failed"
    assert attempt["outcomes"] == []
    assert "outside the predeclared checkpoint step" in attempt["failures"][0]["message"]


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_report_output_cannot_alias_generated_benchmark_artifacts(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    alias_output = tmp_path / f"{alias_kind}-report.json"
    extra_arguments = (
        ["--fixture-hardlink-checkpoint-to", str(alias_output)] if alias_kind == "hardlink" else []
    )
    manifest_path = _manifest(
        tmp_path,
        checkpoint_steps=[10],
        outcomes={"10": [31.0, 0.0]},
        extra_arguments=extra_arguments,
    )
    seed_report = tmp_path / "seed-report.json"
    seeded = _run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(seed_report),
    )
    assert seeded.returncode == 0, seeded.stderr
    run_identity = json.loads(seed_report.read_text(encoding="utf-8"))["run_identity"]
    alias_output.unlink(missing_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark"]["artifacts_directory"] = "protected-artifacts"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    attempt_directory = tmp_path / "protected-artifacts" / run_identity / "seed-123"
    checkpoint = attempt_directory / "checkpoint-step-10.pt"
    if alias_kind == "direct":
        output = checkpoint
    elif alias_kind == "symlink":
        alias_output.symlink_to(checkpoint)
        output = alias_output
    else:
        output = alias_output

    result = _run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert "output path aliases generated benchmark artifact" in result.stderr
    assert checkpoint.is_file()
    evaluation_metrics = attempt_directory / "evaluation-step-10.jsonl"
    evaluation = json.loads(evaluation_metrics.read_text(encoding="utf-8").splitlines()[-1])
    assert evaluation["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["format"] == (
        "standalone-gradoom-ppo-v1"
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda benchmark: benchmark.update(training_seeds=[4, 4]), "must be unique"),
        (
            lambda benchmark: benchmark.update(evaluation_episode_seeds=EVALUATION_SEEDS[:-1]),
            "must contain exactly 100",
        ),
        (
            lambda benchmark: benchmark["trainer"]["arguments"].extend(
                ["--initialize-from", "learned.pt"]
            ),
            "must not control '--initialize-from'",
        ),
    ],
)
def test_development_benchmark_rejects_invalid_cold_start_or_evaluation_contract(
    tmp_path: Path,
    mutation: object,
    error: str,
) -> None:
    manifest_path = _manifest(tmp_path, outcomes={"10": [30.0, 30.0]})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest["benchmark"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert error in result.stderr
    assert not (tmp_path / "report.json").exists()
