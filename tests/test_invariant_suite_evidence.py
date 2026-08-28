from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gradoom.evidence import invariant_suite

FIXTURES = Path(__file__).parent / "fixtures" / "evidence"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "gradoom" / "evidence" / "invariant_runner.py"


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
    manifest["declared_inputs"][0]["path"] = str(RUNNER)
    manifest["declared_inputs"][0]["sha256"] = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    if mismatch is not None:
        assert mismatch == "rewards"
        manifest["invariant_suite"]["fixture_case"] = "reward_mismatch"
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
    manifest["fixture"] = False
    manifest["invariant_suite"] = {
        "version": "1.0.0",
        "mode": "real",
        "runner_input": "invariant_runner",
    }
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
    reasons = report["invariant_suite"]["unavailable_reasons"]
    assert reasons
    assert all(reason["code"] for reason in reasons)
    assert report["invariant_suite"]["providers"] == []


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


def test_fixture_runner_cannot_support_a_non_fixture_claim(tmp_path: Path) -> None:
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

    assert result.returncode == 2
    assert "invariant_suite.mode must match the manifest fixture state" in result.stderr
    assert not output.exists()


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
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "declared input 'invariant_runner' SHA-256 mismatch" in result.stderr
    assert not sentinel.exists()


def test_arbitrary_provider_commands_are_rejected_even_when_their_file_is_hashed(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["invariant_suite"]["providers"] = [
        {
            "id": "gradoom",
            "available": True,
            "runner_input": "invariant_runner",
            "command": ["python", str(RUNNER)],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert "invariant_suite.providers is not supported" in result.stderr
    assert not output.exists()


def test_unrelated_hashed_executable_cannot_replace_the_trusted_runner(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("raise SystemExit(0)\n", encoding="utf-8")
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"].append(
        {
            "name": "unrelated_runner",
            "path": str(unrelated),
            "sha256": hashlib.sha256(unrelated.read_bytes()).hexdigest(),
        }
    )
    manifest["invariant_suite"]["runner_input"] = "unrelated_runner"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "must bind the repository-owned invariant runner" in result.stderr


def test_static_contract_emitter_cannot_impersonate_the_trusted_runner(tmp_path: Path) -> None:
    emitter = tmp_path / "static-emitter.py"
    emitter.write_text(
        "import json\n"
        "print(json.dumps({'protocol_version': 1, 'status': 'complete', "
        "'contracts': [], 'unavailable_reasons': []}))\n",
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"][0]["path"] = str(emitter)
    manifest["declared_inputs"][0]["sha256"] = hashlib.sha256(emitter.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "must bind the repository-owned invariant runner" in result.stderr


def test_minimal_bogus_contract_cannot_impersonate_complete_evidence(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus-contract.py"
    bogus.write_text(
        'print(\'{"schema_version":1,"provider":"gradoom","behaviors":{}}\')\n',
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"][0].update(
        {
            "path": str(bogus),
            "sha256": hashlib.sha256(bogus.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "must bind the repository-owned invariant runner" in result.stderr


def test_authenticated_runner_contract_still_rejects_a_minimal_bogus_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimal_contracts = [
        {
            "schema_version": 1,
            "provider": provider,
            "revision": "bogus",
            "behaviors": {},
            **({"tensor_device": {}} if provider == "gradoom" else {}),
        }
        for provider in ("gradoom", "env-vizdoom-turbo")
    ]
    monkeypatch.setattr(
        invariant_suite,
        "_execute_runner",
        lambda **_kwargs: {
            "protocol_version": 1,
            "challenge": "ignored-after-authentication",
            "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            "status": "complete",
            "contracts": minimal_contracts,
            "unavailable_reasons": [],
        },
    )

    with pytest.raises(
        invariant_suite.InvariantSuiteError,
        match="gradoom invariant contract behavior set is incomplete",
    ):
        invariant_suite.run_invariant_suite(
            {
                "version": "1.0.0",
                "mode": "fixture",
                "runner_input": "invariant_runner",
            },
            base_directory=ROOT,
            declared_inputs=[
                {
                    "name": "invariant_runner",
                    "path": str(RUNNER),
                    "sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                }
            ],
            fixture=True,
            gradoom_revision="fixture-revision",
        )


def test_omitted_invariant_suite_blocks_readiness(tmp_path: Path) -> None:
    manifest = json.loads((FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8"))
    manifest["declared_inputs"][0]["path"] = str(FIXTURES / "provider-contract.json")
    manifest["prerequisites"] = [
        {"id": "pinned_reference_provider", "available": True},
        {"id": "real_pretrained_policy_corpus", "available": True},
    ]
    manifest_path = tmp_path / "manifest.json"
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
    assert any(
        reason["code"] == "invariant_suite_unavailable" for reason in report["claim_reasons"]
    )


def test_invariant_pass_without_a_declared_real_corpus_is_unavailable(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prerequisites"] = [
        {"id": "pinned_reference_provider", "available": True},
    ]
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
    assert report["invariant_suite"]["status"] == "passed"
    assert report["status"] == "unavailable"
    assert report["claim_eligible"] is False
    assert any(
        reason.get("code") == "missing_required_prerequisite"
        and reason.get("prerequisite") == "real_pretrained_policy_corpus"
        for reason in report["claim_reasons"]
    )
