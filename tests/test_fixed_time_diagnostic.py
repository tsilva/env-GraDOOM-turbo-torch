from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURE_PROCESS = Path(__file__).parent / "fixtures" / "evidence" / "fixture_benchmark_process.py"
EVALUATION_SEEDS = list(range(10_000, 10_100))
TIMING_RULES = {
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


def _run_evidence(*args: str) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None
    return subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _trainer(outcomes: dict[str, list[float]], *extra: str) -> dict[str, object]:
    return {
        "command": [sys.executable, str(FIXTURE_PROCESS)],
        "arguments": [
            "--fixture-outcomes",
            json.dumps(outcomes, sort_keys=True),
            *extra,
        ],
    }


def _benchmark_report(
    tmp_path: Path,
    trainer: dict[str, object],
    *,
    training_seeds: list[int] | None = None,
) -> tuple[Path, dict[str, object]]:
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
        "benchmark": {
            "training_seeds": training_seeds or [123],
            "failure_budget_steps": 10,
            "checkpoint_steps": [10],
            "evaluation_episode_seeds": EVALUATION_SEEDS,
            "trainer": trainer,
            "artifacts_directory": "benchmark-artifacts",
            "parity_certificate": {
                "available": False,
                "reason": "No current parity certificate exists for the fixture profile.",
            },
        },
    }
    manifest_path = tmp_path / "benchmark-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report_path = tmp_path / "benchmark-report.json"
    result = _run_evidence("--manifest", str(manifest_path), "--output", str(report_path))
    assert result.returncode == 0, result.stderr
    return report_path, json.loads(report_path.read_text(encoding="utf-8"))


def _diagnostic_manifest(
    tmp_path: Path,
    *,
    benchmark_report: Path,
    trainer: dict[str, object],
    training_seeds: list[int] | None = None,
    reusable_time_budget_seconds: float = 0.5,
) -> Path:
    manifest = {
        "schema_version": 1,
        "workflow": "fixed_time_training_diagnostic",
        "evidence_level": "development",
        "fixture": True,
        "code_provenance": {
            "repository": "tsilva/env-GraDOOM-turbo-torch",
            "revision": "fixture-revision",
            "dirty": False,
        },
        "declared_inputs": [],
        "diagnostic": {
            "reusable_time_budget_seconds": reusable_time_budget_seconds,
            "training_seeds": training_seeds or [123],
            "evaluation_episode_seeds": EVALUATION_SEEDS,
            "evaluation_action_seed": 123,
            "recipe": trainer,
            "timing_rules": TIMING_RULES,
            "artifacts_directory": "diagnostic-artifacts",
            "matching_benchmark_report": {
                "path": str(benchmark_report),
                "sha256": hashlib.sha256(benchmark_report.read_bytes()).hexdigest(),
            },
        },
    }
    path = tmp_path / "diagnostic-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_matching_fixed_time_diagnostic_reports_quality_and_throughput_without_passage(
    tmp_path: Path,
) -> None:
    trainer = _trainer(
        {"10": [31.0, 0.0]},
        "--fixture-diagnostic-quality",
        "27.5",
        "--fixture-diagnostic-transitions",
        "6000",
        "--fixture-diagnostic-elapsed-seconds",
        "12.0",
    )
    training_seeds = [123, 456]
    benchmark_path, benchmark = _benchmark_report(
        tmp_path,
        trainer,
        training_seeds=training_seeds,
    )
    manifest = _diagnostic_manifest(
        tmp_path,
        benchmark_report=benchmark_path,
        trainer=trainer,
        training_seeds=training_seeds,
    )
    output = tmp_path / "diagnostic-report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workflow"] == "fixed_time_training_diagnostic"
    assert report["status"] == "completed"
    assert report["claim_eligible"] is False
    assert report["diagnostic_protocol"]["reusable_time_budget_seconds"] == 0.5
    assert report["diagnostic_protocol"]["training_seeds"] == training_seeds
    assert report["diagnostic_protocol"]["evaluation_episode_seeds"] == EVALUATION_SEEDS
    assert report["diagnostic_protocol"]["timing_rules"] == TIMING_RULES
    fixed_time = report["diagnostics"]["fixed_time"]
    assert fixed_time["status"] == "completed"
    assert fixed_time["affects_passage"] is False
    assert fixed_time["matching_benchmark"]["matched"] is True
    assert fixed_time["matching_benchmark"]["run_identity"] == benchmark["run_identity"]
    assert fixed_time["matching_benchmark"]["passage"] == {
        "status": "passed",
        "unchanged": True,
    }
    assert [attempt["seed"] for attempt in fixed_time["attempts"]] == training_seeds
    assert all(
        attempt["throughput"]["timer"]["elapsed_seconds"] >= 0.5
        for attempt in fixed_time["attempts"]
    )
    attempt = fixed_time["attempts"][0]
    assert attempt["status"] == "completed"
    assert attempt["final_mean_player_killcount"] == 27.5
    assert len(attempt["episodes"]) == 100
    assert [episode["game_seed"] for episode in attempt["episodes"]] == EVALUATION_SEEDS
    throughput = attempt["throughput"]
    elapsed = throughput["timer"]["elapsed_seconds"]
    assert throughput["timer"] == {
        "elapsed_seconds": elapsed,
        "source": "public_evidence_subprocess",
        "rules": TIMING_RULES,
    }
    assert throughput["workload"] == {
        "frame_skip": 2,
        "simulated_tics": 12000,
        "transitions": 6000,
    }
    assert throughput["transitions_per_second"] == 6000 / elapsed
    assert throughput["simulated_tics_per_second"] == 12000 / elapsed
    evidence_entries = {
        entry["name"]: entry["sha256"] for entry in report["evidence_index"]["entries"]
    }
    assert len(evidence_entries) == len(report["evidence_index"]["entries"])
    assert len(report["generated_artifacts"]) == 4 * len(training_seeds)
    assert all(
        artifact["status"] == "matched"
        and artifact["sha256"]
        == artifact["evidence_entry_sha256"]
        == evidence_entries[artifact["name"]]
        for artifact in report["generated_artifacts"]
    )
    assert report["public_performance_evidence"] == {
        "complete": False,
        "reason": "matching_benchmark_is_not_claim_eligible",
    }
    assert benchmark["status"] == "passed"


@pytest.mark.parametrize("mismatch", ["training_seeds", "recipe"])
def test_fixed_time_diagnostic_rejects_evidence_that_does_not_match_benchmark(
    tmp_path: Path,
    mismatch: str,
) -> None:
    trainer = _trainer({"10": [31.0, 0.0]})
    benchmark_path, _benchmark = _benchmark_report(tmp_path, trainer)
    manifest_path = _diagnostic_manifest(
        tmp_path,
        benchmark_report=benchmark_path,
        trainer=trainer,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mismatch == "training_seeds":
        manifest["diagnostic"]["training_seeds"] = [999]
    else:
        manifest["diagnostic"]["recipe"]["arguments"].extend(
            ["--fixture-diagnostic-quality", "1.0"]
        )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "diagnostic-report.json"

    result = _run_evidence("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 2
    assert f"fixed-time diagnostic {mismatch.replace('_', ' ')} do not match benchmark" in (
        result.stderr
    )
    assert not output.exists()
    assert not (tmp_path / "diagnostic-artifacts").exists()


def test_fixed_time_diagnostic_retains_failed_evaluation_without_changing_passage(
    tmp_path: Path,
) -> None:
    trainer = _trainer(
        {"10": [31.0, 0.0]},
        "--fixture-diagnostic-transitions",
        "6000",
        "--fixture-diagnostic-elapsed-seconds",
        "12.0",
        "--fixture-fail-evaluation-step",
        "6000",
    )
    benchmark_path, _benchmark = _benchmark_report(tmp_path, trainer)
    manifest = _diagnostic_manifest(
        tmp_path,
        benchmark_report=benchmark_path,
        trainer=trainer,
    )
    output = tmp_path / "diagnostic-report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    fixed_time = report["diagnostics"]["fixed_time"]
    assert report["status"] == "failed"
    assert fixed_time["status"] == "failed"
    assert fixed_time["affects_passage"] is False
    assert fixed_time["matching_benchmark"]["passage"] == {
        "status": "passed",
        "unchanged": True,
    }
    assert fixed_time["attempts"][0]["status"] == "failed"
    assert fixed_time["attempts"][0]["failures"][0]["phase"] == "evaluation"
    assert report["public_performance_evidence"] == {
        "complete": False,
        "reason": "matching_fixed_time_diagnostic_failed",
    }


@pytest.mark.parametrize(
    ("fixture_arguments", "message"),
    [
        (("--fixture-episode-length", "0"), "length must be a positive integer"),
        (("--fixture-terminal-mode", "neither"), "must declare terminal completion"),
        (("--fixture-terminal-mode", "both"), "cannot be both terminated and truncated"),
    ],
)
def test_fixed_time_diagnostic_rejects_incomplete_episode_records_before_quality(
    tmp_path: Path,
    fixture_arguments: tuple[str, ...],
    message: str,
) -> None:
    trainer = _trainer(
        {"10": [31.0, 0.0]},
        "--fixture-diagnostic-quality",
        "999.0",
        *fixture_arguments,
    )
    benchmark_path, _benchmark = _benchmark_report(tmp_path, trainer)
    manifest = _diagnostic_manifest(
        tmp_path,
        benchmark_report=benchmark_path,
        trainer=trainer,
    )
    output = tmp_path / "diagnostic-report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    attempt = report["diagnostics"]["fixed_time"]["attempts"][0]
    assert report["status"] == "failed"
    assert attempt["status"] == "evidence_failed"
    assert attempt["final_mean_player_killcount"] is None
    assert attempt["episodes"] == []
    assert attempt["failures"][0]["phase"] == "evaluation_evidence"
    assert message in attempt["failures"][0]["message"]


def test_fixed_time_diagnostic_rejects_declared_input_name_collision_before_execution(
    tmp_path: Path,
) -> None:
    trainer = _trainer({"10": [31.0, 0.0]})
    benchmark_path, _benchmark = _benchmark_report(tmp_path, trainer)
    manifest_path = _diagnostic_manifest(
        tmp_path,
        benchmark_report=benchmark_path,
        trainer=trainer,
    )
    collision = tmp_path / "collision.bin"
    collision.write_bytes(b"declared collision")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"] = [
        {
            "name": "seed-123-checkpoint",
            "path": str(collision),
            "sha256": hashlib.sha256(collision.read_bytes()).hexdigest(),
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "diagnostic-report.json"

    result = _run_evidence("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 2
    assert "declared input name 'seed-123-checkpoint' is reserved" in result.stderr
    assert not output.exists()
    assert not (tmp_path / "diagnostic-artifacts").exists()


def test_fixed_time_diagnostic_retains_generated_artifact_digest_mismatch(
    tmp_path: Path,
) -> None:
    trainer = _trainer(
        {"10": [31.0, 0.0]},
        "--fixture-mutate-checkpoint-after-evaluation",
    )
    benchmark_path, _benchmark = _benchmark_report(tmp_path, trainer)
    manifest = _diagnostic_manifest(
        tmp_path,
        benchmark_report=benchmark_path,
        trainer=trainer,
    )
    output = tmp_path / "diagnostic-report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    attempt = report["diagnostics"]["fixed_time"]["attempts"][0]
    assert report["status"] == "failed"
    assert attempt["status"] == "evidence_failed"
    assert attempt["failures"][-1]["phase"] == "artifact_evidence"
    assert "seed-123-checkpoint SHA-256 changed" in attempt["failures"][-1]["message"]
    names = [entry["name"] for entry in report["evidence_index"]["entries"]]
    assert len(names) == len(set(names))
    checkpoint_artifact = next(
        artifact
        for artifact in report["generated_artifacts"]
        if artifact["name"] == "seed-123-checkpoint"
    )
    assert checkpoint_artifact["status"] == "mismatched"
    assert checkpoint_artifact["sha256"] != next(
        entry["sha256"]
        for entry in report["evidence_index"]["entries"]
        if entry["name"] == "seed-123-checkpoint"
    )
    assert checkpoint_artifact["evidence_entry_sha256"] != checkpoint_artifact["sha256"]


def test_fixed_time_budget_and_rates_include_subprocess_startup(
    tmp_path: Path,
) -> None:
    trainer = _trainer(
        {"10": [31.0, 0.0]},
        "--fixture-startup-delay-seconds",
        "0.1",
        "--fixture-diagnostic-transitions",
        "6000",
        "--fixture-diagnostic-elapsed-seconds",
        "0.001",
    )
    benchmark_path, _benchmark = _benchmark_report(tmp_path, trainer)
    manifest = _diagnostic_manifest(
        tmp_path,
        benchmark_report=benchmark_path,
        trainer=trainer,
        reusable_time_budget_seconds=0.05,
    )
    output = tmp_path / "diagnostic-report.json"

    result = _run_evidence("--manifest", str(manifest), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    attempt = report["diagnostics"]["fixed_time"]["attempts"][0]
    throughput = attempt["throughput"]
    assert attempt["status"] == "completed"
    assert throughput["workload"]["transitions"] == 0
    assert throughput["workload"]["simulated_tics"] == 0
    assert throughput["timer"]["source"] == "public_evidence_subprocess"
    assert throughput["timer"]["elapsed_seconds"] >= 0.1
    assert throughput["transitions_per_second"] == 0.0
    assert throughput["simulated_tics_per_second"] == 0.0
