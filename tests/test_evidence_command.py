from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "evidence"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def rehash_evidence_index(report: dict[str, object]) -> None:
    evidence_index = report["evidence_index"]
    assert isinstance(evidence_index, dict)
    evidence_index["sha256"] = canonical_sha256(evidence_index["entries"])


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
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_constants_fail_with_a_clear_error(
    tmp_path: Path,
    document: str,
    constant: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
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
    payload["prerequisites"][0]["extension"] = {"retained": "INVALID_CONSTANT"}
    serialized = json.dumps(payload).replace('"INVALID_CONSTANT"', constant)
    invalid_path.write_text(serialized, encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert (
        f"{document} is not valid JSON: non-standard constant {constant}"
        in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("document", ["manifest", "merge report"])
@pytest.mark.parametrize(
    ("invalid_value", "expected_error"),
    [
        ("1e400", "numeric value is not finite"),
        ("9" * 5000, "integer value exceeds supported range"),
        ("[" * 1100 + "0" + "]" * 1100, "nesting is too deep"),
    ],
)
def test_json_resource_and_numeric_failures_are_clear_validation_errors(
    tmp_path: Path,
    document: str,
    invalid_value: str,
    expected_error: str,
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
    payload["prerequisites"][0]["extension"] = {"retained": "INVALID_VALUE"}
    serialized = json.dumps(payload).replace('"INVALID_VALUE"', invalid_value)
    invalid_path.write_text(serialized, encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert f"{document} is not valid JSON: {expected_error}" in result.stderr
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


@pytest.mark.parametrize(
    ("document", "field"),
    [
        ("manifest", "declared_inputs[0].path"),
        ("manifest", "prerequisites[0].reason"),
        ("merge report", "declared_inputs[0].path"),
        ("merge report", "prerequisites[0].reason"),
    ],
)
def test_all_schema_strings_reject_lone_surrogates(
    tmp_path: Path,
    document: str,
    field: str,
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

    if field == "declared_inputs[0].path":
        payload["declared_inputs"][0]["path"] = "\ud800"
    else:
        payload["prerequisites"][0]["reason"] = "\ud800"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert f"{document} contains invalid Unicode in {field}" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("document", ["manifest", "merge report"])
def test_declared_input_paths_reject_embedded_nulls(
    tmp_path: Path,
    document: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
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
    payload["declared_inputs"][0]["path"] = "provider\u0000-contract.json"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert (
        f"{document} contains U+0000 in declared_inputs[0].path" in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("document", "invalid_kind", "expected_field"),
    [
        ("manifest", "value", "prerequisites[0].extension.nested[0]"),
        ("manifest", "key", "prerequisites[0].extension.<key>"),
        ("merge report", "value", "prerequisites[0].extension.nested[0]"),
        ("merge report", "key", "prerequisites[0].extension.<key>"),
    ],
)
def test_recursive_extension_strings_and_keys_reject_lone_surrogates(
    tmp_path: Path,
    document: str,
    invalid_kind: str,
    expected_field: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
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
    if invalid_kind == "value":
        payload["prerequisites"][0]["extension"] = {"nested": ["\ud800"]}
    else:
        payload["prerequisites"][0]["extension"] = {"\ud800": "value"}
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert (
        f"{document} contains invalid Unicode in {expected_field}" in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("document", ["manifest", "merge report"])
def test_whitespace_only_prerequisite_reasons_fail_with_a_clear_error(
    tmp_path: Path,
    document: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
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
    payload["prerequisites"][0]["reason"] = " \t\n"
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert (
        "prerequisites[0].reason must be a human-readable non-whitespace string"
        in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("document", "invalid_path", "expected_error"),
    [
        (
            "manifest",
            "   ",
            "declared_inputs[0].path must be a non-whitespace path",
        ),
        (
            "merge report",
            "   ",
            "declared_inputs[0].path must be a non-whitespace path",
        ),
    ],
)
def test_declared_input_paths_reject_whitespace(
    tmp_path: Path,
    document: str,
    invalid_path: str,
    expected_error: str,
) -> None:
    document_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
        )
        arguments = ["--manifest", str(document_path), "--output", str(output)]
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
            str(document_path),
        ]
    payload["declared_inputs"][0]["path"] = invalid_path
    document_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("document", ["manifest", "merge report"])
def test_declared_input_paths_must_be_unique(
    tmp_path: Path,
    document: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
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
    duplicate = dict(payload["declared_inputs"][0])
    duplicate["name"] = "duplicate_provider_contract"
    payload["declared_inputs"].append(duplicate)
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert "declared_inputs[1].path is duplicated" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("document", ["manifest", "merge report"])
def test_declared_input_normalized_path_aliases_must_be_unique(
    tmp_path: Path,
    document: str,
) -> None:
    invalid_path = tmp_path / "invalid.json"
    output = tmp_path / "report.json"
    if document == "manifest":
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
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
    duplicate = dict(payload["declared_inputs"][0])
    duplicate["name"] = "duplicate_provider_contract"
    duplicate["path"] = "subdirectory/../provider-contract.json"
    payload["declared_inputs"].append(duplicate)
    invalid_path.write_text(json.dumps(payload), encoding="utf-8")

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert "declared_inputs[1].path aliases an earlier declared input" in result.stderr
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


@pytest.mark.parametrize(
    ("protected_source", "expected_error"),
    [
        ("manifest", "output path aliases the manifest"),
        (
            "declared_input",
            "output path aliases declared input 'provider_contract'",
        ),
        ("merge_report", "output path aliases the merge report"),
    ],
)
def test_output_cannot_overwrite_evidence_sources_through_a_path_alias(
    tmp_path: Path,
    protected_source: str,
    expected_error: str,
) -> None:
    alias_directory = tmp_path / "alias"
    alias_directory.mkdir()
    if protected_source == "manifest":
        source_path = tmp_path / "manifest.json"
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
        )
        payload["declared_inputs"][0]["path"] = str(
            FIXTURES / "provider-contract.json"
        )
        source_path.write_text(json.dumps(payload), encoding="utf-8")
        arguments = [
            "--manifest",
            str(source_path),
            "--output",
            str(alias_directory / ".." / source_path.name),
        ]
    elif protected_source == "declared_input":
        source_path = tmp_path / "provider-contract.json"
        source_path.write_bytes((FIXTURES / "provider-contract.json").read_bytes())
        payload = json.loads(
            (FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8")
        )
        payload["declared_inputs"][0]["path"] = str(source_path)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        arguments = [
            "--manifest",
            str(manifest_path),
            "--output",
            str(alias_directory / ".." / source_path.name),
        ]
    else:
        source_path = tmp_path / "merge-report.json"
        initial = run_evidence(
            "--manifest",
            str(FIXTURES / "readiness-manifest.json"),
            "--output",
            str(source_path),
        )
        assert initial.returncode == 0, initial.stderr
        arguments = [
            "--manifest",
            str(FIXTURES / "readiness-manifest.json"),
            "--output",
            str(alias_directory / ".." / source_path.name),
            "--merge",
            str(source_path),
        ]
    source_before = source_path.read_bytes()

    result = run_evidence(*arguments)

    assert result.returncode == 2
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert source_path.read_bytes() == source_before


def test_output_cannot_overwrite_input_relative_to_a_symlinked_manifest(
    tmp_path: Path,
) -> None:
    manifest_target_directory = tmp_path / "manifest-target"
    manifest_target_directory.mkdir()
    manifest_target = manifest_target_directory / "manifest.json"
    manifest_target.write_bytes((FIXTURES / "readiness-manifest.json").read_bytes())

    supplied_directory = tmp_path / "supplied"
    supplied_directory.mkdir()
    supplied_manifest = supplied_directory / "manifest.json"
    supplied_manifest.symlink_to(manifest_target)
    protected_input = supplied_directory / "provider-contract.json"
    protected_input.write_bytes((FIXTURES / "provider-contract.json").read_bytes())
    input_before = protected_input.read_bytes()

    result = run_evidence(
        "--manifest",
        str(supplied_manifest),
        "--output",
        str(protected_input),
    )

    assert result.returncode == 2
    assert "output path aliases declared input 'provider_contract'" in result.stderr
    assert "Traceback" not in result.stderr
    assert protected_input.read_bytes() == input_before
    assert not (manifest_target_directory / "provider-contract.json").exists()


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


@pytest.mark.parametrize(
    "missing_field",
    [
        "schema_version",
        "workflow",
        "evidence_level",
        "fixture",
        "status",
        "claim_eligible",
        "claim_reasons",
        "run_identity",
        "code_provenance",
        "declared_inputs",
        "prerequisites",
        "invariant_suite",
        "evidence_index",
    ],
)
def test_merge_rejects_missing_required_report_fields(
    tmp_path: Path,
    missing_field: str,
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
    del report[missing_field]
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
    assert f"merge report {missing_field} is required" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        (
            "status",
            "ready",
            "merge report status must be 'unavailable' for its prerequisites",
        ),
        (
            "claim_eligible",
            True,
            "merge report claim_eligible must be false for development evidence",
        ),
        (
            "claim_reasons",
            [],
            "merge report claim_reasons do not match its fixture state and "
            "prerequisites",
        ),
    ],
)
def test_merge_rejects_incoherent_report_envelope_fields(
    tmp_path: Path,
    field: str,
    value: object,
    expected_error: str,
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
    report[field] = value
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
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        (
            "workflow",
            1,
            "merge report workflow is required and must be a non-empty string",
        ),
        (
            "evidence_level",
            [],
            "merge report evidence_level is required and must be a non-empty string",
        ),
        (
            "status",
            False,
            "merge report status is required and must be a non-empty string",
        ),
        (
            "claim_eligible",
            0,
            "merge report claim_eligible must be a boolean",
        ),
        (
            "claim_reasons",
            {},
            "merge report claim_reasons must be an array",
        ),
    ],
)
def test_merge_rejects_wrong_typed_envelope_fields_before_identity_validation(
    tmp_path: Path,
    field: str,
    value: object,
    expected_error: str,
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
    report[field] = value
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
    assert expected_error in result.stderr
    assert "run_identity" not in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            "empty",
            "merge report evidence_index.entries missing required names: "
            "'manifest', 'provider_contract'",
        ),
        (
            "missing_declared_input",
            "merge report evidence_index.entries missing required names: "
            "'provider_contract'",
        ),
        (
            "extra",
            "merge report evidence_index.entries has unexpected names: 'extra'",
        ),
        (
            "wrong_declared_digest",
            "merge report evidence_index entry 'provider_contract' SHA-256 does not "
            "match declared_inputs",
        ),
        (
            "duplicate",
            "merge report evidence_index.entries[2].name 'manifest' is duplicated",
        ),
    ],
)
def test_merge_rejects_self_consistently_rehashed_mismatched_evidence_indexes(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
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
    entries = report["evidence_index"]["entries"]
    if mutation == "empty":
        entries.clear()
    elif mutation == "missing_declared_input":
        entries[:] = [entry for entry in entries if entry["name"] == "manifest"]
    elif mutation == "extra":
        entries.append({"name": "extra", "sha256": "0" * 64})
    elif mutation == "wrong_declared_digest":
        next(
            entry for entry in entries if entry["name"] == "provider_contract"
        )["sha256"] = "0" * 64
    else:
        entries.append(dict(entries[0]))
    rehash_evidence_index(report)
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
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            "non_object_entry",
            "merge report evidence_index.entries[0] must be an object",
        ),
        (
            "missing_name",
            "merge report evidence_index.entries[0].name is required and must be a "
            "non-empty string",
        ),
        (
            "wrong_name_type",
            "merge report evidence_index.entries[0].name is required and must be a "
            "non-empty string",
        ),
        (
            "missing_digest",
            "merge report evidence_index.entries[0].sha256 must be a lowercase "
            "SHA-256 digest",
        ),
        (
            "invalid_digest",
            "merge report evidence_index.entries[0].sha256 must be a lowercase "
            "SHA-256 digest",
        ),
    ],
)
def test_merge_rejects_self_consistently_rehashed_malformed_evidence_entries(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
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
    entries = report["evidence_index"]["entries"]
    if mutation == "non_object_entry":
        entries[0] = None
    elif mutation == "missing_name":
        del entries[0]["name"]
    elif mutation == "wrong_name_type":
        entries[0]["name"] = 1
    elif mutation == "missing_digest":
        del entries[0]["sha256"]
    else:
        entries[0]["sha256"] = "not-a-digest"
    rehash_evidence_index(report)
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
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_merge_rejects_undeclared_evidence_entry_members(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    initial = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(report_path),
    )
    assert initial.returncode == 0, initial.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["evidence_index"]["entries"][0]["path"] = "invented-manifest.json"
    rehash_evidence_index(report)
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
        "merge report evidence_index.entries[0] has undeclared fields: 'path'"
        in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output.exists()


def test_merge_rejects_a_self_rehashed_false_manifest_digest(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    initial = run_evidence(
        "--manifest",
        str(FIXTURES / "readiness-manifest.json"),
        "--output",
        str(report_path),
    )
    assert initial.returncode == 0, initial.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_entry = next(
        entry
        for entry in report["evidence_index"]["entries"]
        if entry["name"] == "manifest"
    )
    manifest_entry["sha256"] = "0" * 64
    rehash_evidence_index(report)
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
        "merge report evidence_index entry 'manifest' SHA-256 does not match "
        "the source manifest" in result.stderr
    )
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            "missing_algorithm",
            "merge report evidence_index.algorithm is required",
        ),
        (
            "wrong_algorithm",
            "merge report evidence_index.algorithm must be 'sha256'",
        ),
        ("missing_entries", "merge report evidence_index.entries is required"),
        ("wrong_entries_type", "merge report evidence_index.entries must be an array"),
        ("missing_hash", "merge report evidence_index.sha256 is required"),
        (
            "invalid_hash",
            "merge report evidence_index.sha256 must be a lowercase SHA-256 digest",
        ),
    ],
)
def test_merge_rejects_malformed_evidence_index_envelopes(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
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
    evidence_index = report["evidence_index"]
    if mutation == "missing_algorithm":
        del evidence_index["algorithm"]
    elif mutation == "wrong_algorithm":
        evidence_index["algorithm"] = "md5"
    elif mutation == "missing_entries":
        del evidence_index["entries"]
    elif mutation == "wrong_entries_type":
        evidence_index["entries"] = {}
    elif mutation == "missing_hash":
        del evidence_index["sha256"]
    else:
        evidence_index["sha256"] = "not-a-digest"
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
    assert expected_error in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


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
