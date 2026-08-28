from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "evidence"


def run_evidence(*args: str) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None, "the installed project must expose gradoom-evidence"
    return subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_fixture_manifest_emits_non_claim_eligible_readiness_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "readiness-report.json"

    result = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(report_path),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["workflow"] == "parity_readiness"
    assert report["evidence_level"] == "development"
    assert report["fixture"] is True
    assert report["status"] == "unavailable"
    assert report["claim_eligible"] is False
    assert report["code_provenance"] == {
        "repository": "tsilva/env-GraDOOM-turbo-torch",
        "revision": "fixture-revision",
        "dirty": False,
    }
    assert report["declared_inputs"] == [
        {
            "name": "provider_contract",
            "path": "provider-contract.json",
            "sha256": "09634113116b633334e06263d1c236cc402107c6a98effabe5ff2dbbee0b15d7",
        }
    ]
    fixture_manifest = json.loads(
        (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
    )
    assert report["prerequisites"] == fixture_manifest["prerequisites"]
    assert report["run_identity"] == (
        "7974e35ef137d77a42dcec2766b8ae912c43078c3aaca6df6bcf3ffc1c8d9b4e"
    )
    assert {reason["code"] for reason in report["claim_reasons"]} == {
        "development_evidence",
        "fixture_evidence",
        "missing_prerequisite",
    }
    missing = {
        reason["prerequisite"]
        for reason in report["claim_reasons"]
        if reason["code"] == "missing_prerequisite"
    }
    assert missing == {
        "certified_freedoom2_wad_profile",
        "pinned_reference_provider",
        "real_pretrained_policy_corpus",
    }
    assert report["evidence_index"]["algorithm"] == "sha256"
    assert report["evidence_index"]["sha256"] == (
        "bff82949c8cfc702355ef8250aecb554b61870b48d8b81107f1eab891dc41573"
    )
    assert {entry["name"] for entry in report["evidence_index"]["entries"]} == {
        "manifest",
        "provider_contract",
    }


def test_malformed_manifest_fails_with_a_clear_error(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
                {
                    "schema_version": 1,
                    "workflow": "parity_readiness",
                    "evidence_level": "development",
                    "fixture": True,
                    "code_provenance": {
                        "repository": "tsilva/env-GraDOOM-turbo-torch",
                        "revision": "fixture-revision",
                        "dirty": False,
                    },
                "declared_inputs": [{}],
                "prerequisites": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "declared_inputs[0].name is required" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("document", ["manifest", "merge report"])
def test_invalid_utf8_json_fails_with_a_clear_error(
    tmp_path: Path,
    document: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_bytes(b"\xff")
    output = tmp_path / "report.json"
    if document == "manifest":
        arguments = ["--manifest", str(invalid_path), "--output", str(output)]
    else:
        arguments = [
            "--manifest",
            str(FIXTURES / "readiness-manifest.json"),
            "--output",
            str(output),
            "--merge",
            str(invalid_path),
        ]

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert f"{document} is not valid UTF-8" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("document", ["manifest", "merge report"])
def test_lone_surrogate_json_fails_with_a_clear_error(
    tmp_path: Path,
    document: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
        )
        payload["declared_inputs"][0]["path"] = str(
            FIXTURES / "provider-contract.json"
        )
        arguments = ["--manifest", str(invalid_path), "--output", str(output)]
    else:
        initial_report = tmp_path / "initial-report.json"
        initial = run_evidence(
            "--manifest",
            str(FIXTURES / "readiness-manifest.json"),
            "--output",
            str(initial_report),
        )
        assert initial.returncode == 0, initial.stderr
        payload = json.loads(initial_report.read_text(encoding="utf-8"))
        arguments = [
            "--manifest",
            str(FIXTURES / "readiness-manifest.json"),
            "--output",
            str(output),
            "--merge",
            str(invalid_path),
        ]
    payload["code_provenance"]["revision"] = "\ud800"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert f"{document} contains invalid Unicode" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_unknown_manifest_schema_version_fails_with_a_clear_error(tmp_path: Path) -> None:
    manifest = json.loads(
        (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
    )
    manifest["schema_version"] = 2
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "unsupported manifest schema_version 2; supported version is 1" in result.stderr


def test_changed_declared_input_hash_fails_with_a_clear_error(tmp_path: Path) -> None:
    input_path = tmp_path / "provider-contract.json"
    input_path.write_text('{"provider":"changed"}\n', encoding="utf-8")
    manifest = json.loads(
        (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
    )
    manifest["declared_inputs"][0]["path"] = input_path.name
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "declared input 'provider_contract' SHA-256 mismatch" in result.stderr


def test_merge_rejects_an_unlike_run_identity(tmp_path: Path) -> None:
    first_report = tmp_path / "first-report.json"
    first = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(first_report),
    )
    assert first.returncode == 0, first.stderr

    manifest = json.loads(
        (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
    )
    manifest["code_provenance"]["revision"] = "different-fixture-revision"
    manifest["declared_inputs"][0]["path"] = str(
        FIXTURES / "provider-contract.json"
    )
    unlike_manifest = tmp_path / "unlike-manifest.json"
    unlike_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "merged-report.json"

    result = run_evidence(
        "--manifest",
        str(unlike_manifest),
        "--output",
        str(output),
        "--merge",
        str(first_report),
    )

    assert result.returncode == 2
    assert "cannot merge unlike run identities" in result.stderr
    assert not output.exists()


def test_merge_accepts_the_same_stable_run_identity(tmp_path: Path) -> None:
    first_report = tmp_path / "first-report.json"
    first = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(first_report),
    )
    assert first.returncode == 0, first.stderr
    merged_report = tmp_path / "merged-report.json"

    merged = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(merged_report),
        "--merge",
        str(first_report),
    )

    assert merged.returncode == 0, merged.stderr
    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    merged_payload = json.loads(merged_report.read_text(encoding="utf-8"))
    assert merged_payload["run_identity"] == first_payload["run_identity"]


def test_merge_rejects_a_changed_evidence_index_hash(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    initial = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(report_path),
    )
    assert initial.returncode == 0, initial.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evidence_index"]["entries"][0]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(tmp_path / "merged-report.json"),
        "--merge",
        str(report_path),
    )

    assert result.returncode == 2
    assert "merge report evidence_index SHA-256 mismatch" in result.stderr


@pytest.mark.parametrize(
    "tampered_field",
    [
        "evidence_level",
        "fixture",
        "code_provenance",
        "declared_input_hash",
        "prerequisites",
    ],
)
def test_merge_rejects_tampered_identity_bearing_report_fields(
    tmp_path: Path,
    tampered_field: str,
) -> None:
    report_path = tmp_path / "report.json"
    initial = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(report_path),
    )
    assert initial.returncode == 0, initial.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if tampered_field == "evidence_level":
        report["evidence_level"] = "formal"
    elif tampered_field == "fixture":
        report["fixture"] = False
    elif tampered_field == "code_provenance":
        report["code_provenance"]["revision"] = "tampered-revision"
    elif tampered_field == "declared_input_hash":
        report["declared_inputs"][0]["sha256"] = "0" * 64
    else:
        report["prerequisites"][0]["id"] = "tampered_prerequisite"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "merged-report.json"

    result = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(output),
        "--merge",
        str(report_path),
    )

    assert result.returncode == 2
    assert (
        "merge report run_identity does not match its identity-bearing fields"
        in result.stderr
    )
    assert not output.exists()
