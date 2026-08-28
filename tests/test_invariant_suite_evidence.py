from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "evidence"


def run_evidence(*args: str) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None
    return subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def write_manifest(tmp_path: Path, *, mismatch: str | None = None) -> Path:
    source = FIXTURES / "invariant-readiness-manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    runner = FIXTURES / "invariant-provider.py"
    manifest["declared_inputs"][0]["path"] = str(runner)
    manifest["declared_inputs"][0]["sha256"] = hashlib.sha256(runner.read_bytes()).hexdigest()
    for provider in manifest["invariant_suite"]["providers"]:
        provider["command"][1] = str(runner)
    if mismatch is not None:
        manifest["invariant_suite"]["providers"][1]["command"].extend(["--mismatch", mismatch])
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def test_public_readiness_runs_the_versioned_invariant_suite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path)),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    suite = report["invariant_suite"]
    assert suite["version"] == "1.0.0"
    assert suite["status"] == "passed"
    assert suite["failures"] == []
    assert {check["behavior"] for check in suite["checks"]} == {
        "constructor",
        "action_meanings",
        "observation_shapes",
        "signal_shapes",
        "rewards",
        "reset",
        "step",
        "masked_reset",
        "termination",
        "truncation",
        "episode",
        "player_killcount",
        "player_killcount.enemy_on_enemy_exclusion",
        "gradoom.tensor_inputs",
        "gradoom.tensor_outputs",
        "gradoom.device",
    }
    assert all(check["status"] == "passed" for check in suite["checks"])
    assert suite["diagnostics"]["affects_verdict"] is False
    assert {item["id"] for item in suite["diagnostics"]["tools"]} == {
        "mechanics",
        "trace",
        "distribution",
        "observation",
        "rendering",
    }
    assert report["status"] == "unavailable"
    assert report["claim_eligible"] is False
    assert any(
        reason.get("prerequisite") == "real_pretrained_policy_corpus"
        for reason in report["claim_reasons"]
    )


def test_failed_public_behavior_blocks_readiness_and_is_named(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path, mismatch="rewards")),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["claim_eligible"] is False
    assert report["invariant_suite"]["status"] == "failed"
    assert report["invariant_suite"]["failures"] == [
        {
            "behavior": "rewards",
            "message": ("env-vizdoom-turbo public behavior does not match gradoom for rewards."),
            "provider": "env-vizdoom-turbo",
        }
    ]
    assert any(
        reason.get("code") == "invariant_failure" and reason.get("behavior") == "rewards"
        for reason in report["claim_reasons"]
    )


def test_unavailable_runtime_is_reported_without_executing_a_provider(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = manifest["invariant_suite"]["providers"][1]
    reference.clear()
    reference.update(
        {
            "id": "env-vizdoom-turbo",
            "available": False,
            "reason": "The pinned reference runtime is not installed.",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "unavailable"
    assert report["invariant_suite"]["status"] == "unavailable"
    assert report["invariant_suite"]["failures"] == []
    assert report["invariant_suite"]["unavailable_reasons"] == [
        {
            "code": "provider_unavailable",
            "provider": "env-vizdoom-turbo",
            "message": "The pinned reference runtime is not installed.",
        }
    ]


def test_merge_rejects_a_tampered_invariant_verdict(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    report_path = tmp_path / "report.json"
    initial = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(report_path),
    )
    assert initial.returncode == 0, initial.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["invariant_suite"]["status"] = "failed"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "merged.json"),
        "--merge",
        str(report_path),
    )

    assert result.returncode == 2
    assert "invariant_suite does not match the current invariant execution" in result.stderr


def test_non_fixture_reference_revision_mismatch_blocks_readiness(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    failed_behaviors = {failure["behavior"] for failure in report["invariant_suite"]["failures"]}
    assert failed_behaviors == {"gradoom.revision", "env-vizdoom-turbo.revision"}


def test_provider_runner_hash_is_verified_before_execution(tmp_path: Path) -> None:
    sentinel = tmp_path / "executed"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"][0]["path"] = str(runner)
    manifest["declared_inputs"][0]["sha256"] = "0" * 64
    for provider in manifest["invariant_suite"]["providers"]:
        provider["command"][1] = str(runner)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "declared input 'invariant_provider_runner' SHA-256 mismatch" in result.stderr
    assert not sentinel.exists()
