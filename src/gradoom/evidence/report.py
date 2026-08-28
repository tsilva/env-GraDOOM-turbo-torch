from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """An evidence manifest or report violates the public contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object, *, document: str) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except UnicodeEncodeError as error:
        raise EvidenceError(
            f"{document} contains invalid Unicode at character {error.start}"
        ) from error
    return _sha256_bytes(payload)


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    try:
        manifest = json.loads(payload)
    except UnicodeDecodeError as error:
        raise EvidenceError(
            f"manifest is not valid UTF-8 at byte {error.start}"
        ) from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"manifest is not valid JSON: {error.msg}") from error
    if not isinstance(manifest, dict):
        raise EvidenceError("manifest must be a JSON object")
    return manifest, payload


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{field} is required and must be a non-empty string")
    return value


def _validate_schema_version(value: object, *, document: str) -> None:
    if type(value) is not int or value != 1:
        raise EvidenceError(
            f"unsupported {document} schema_version {value!r}; supported version is 1"
        )


def _validate_code_provenance(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("code_provenance must be an object")
    _required_string(value.get("repository"), "code_provenance.repository")
    _required_string(value.get("revision"), "code_provenance.revision")
    if type(value.get("dirty")) is not bool:
        raise EvidenceError("code_provenance.dirty is required and must be a boolean")
    return value


def _validate_declared_inputs(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceError("declared_inputs must be an array")
    inputs: list[dict[str, Any]] = []
    names: set[str] = {"manifest"}
    for index, item in enumerate(value):
        field = f"declared_inputs[{index}]"
        if not isinstance(item, dict):
            raise EvidenceError(f"{field} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise EvidenceError(f"{field}.name is required and must be a non-empty string")
        if name in names:
            raise EvidenceError(f"{field}.name {name!r} is duplicated or reserved")
        names.add(name)
        _required_string(item.get("path"), f"{field}.path")
        sha256 = item.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise EvidenceError(f"{field}.sha256 must be a lowercase SHA-256 digest")
        inputs.append(item)
    return inputs


def _validate_prerequisites(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceError("prerequisites must be an array")
    prerequisites: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(value):
        field = f"prerequisites[{index}]"
        if not isinstance(item, dict):
            raise EvidenceError(f"{field} must be an object")
        identifier = _required_string(item.get("id"), f"{field}.id")
        if identifier in identifiers:
            raise EvidenceError(f"{field}.id {identifier!r} is duplicated")
        identifiers.add(identifier)
        if type(item.get("available")) is not bool:
            raise EvidenceError(f"{field}.available is required and must be a boolean")
        if not item["available"]:
            _required_string(item.get("reason"), f"{field}.reason")
        prerequisites.append(item)
    return prerequisites


def _run_identity(
    manifest: dict[str, Any],
    code_provenance: dict[str, Any],
    declared_inputs: list[dict[str, Any]],
    prerequisites: list[dict[str, Any]],
    *,
    document: str,
) -> str:
    identity = {
        "schema_version": manifest["schema_version"],
        "workflow": manifest["workflow"],
        "evidence_level": manifest["evidence_level"],
        "fixture": manifest.get("fixture", False),
        "code_provenance": code_provenance,
        "declared_inputs": sorted(
            ({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
            key=lambda item: item["name"],
        ),
        "prerequisites": sorted(item["id"] for item in prerequisites),
    }
    return _canonical_sha256(identity, document=document)


def build_readiness_report(manifest_path: Path) -> dict[str, Any]:
    manifest, manifest_payload = _load_manifest(manifest_path)
    _validate_schema_version(manifest.get("schema_version"), document="manifest")
    if manifest.get("workflow") != "parity_readiness":
        raise EvidenceError("this command path requires workflow parity_readiness")
    if manifest.get("evidence_level") != "development":
        raise EvidenceError("parity readiness fixtures require development evidence")
    if type(manifest.get("fixture")) is not bool:
        raise EvidenceError("fixture is required and must be a boolean")

    code_provenance = _validate_code_provenance(manifest.get("code_provenance"))
    declared_inputs = _validate_declared_inputs(manifest.get("declared_inputs"))
    prerequisites = _validate_prerequisites(manifest.get("prerequisites"))

    evidence_entries = [
        {"name": "manifest", "sha256": _sha256_bytes(manifest_payload)},
    ]
    for declared_input in declared_inputs:
        name = declared_input["name"]
        input_path = Path(declared_input["path"])
        if not input_path.is_absolute():
            input_path = manifest_path.parent / input_path
        actual_sha256 = _sha256_bytes(input_path.read_bytes())
        if actual_sha256 != declared_input["sha256"]:
            raise EvidenceError(
                f"declared input {name!r} SHA-256 mismatch: "
                f"expected {declared_input['sha256']}, got {actual_sha256}"
            )
        evidence_entries.append({"name": name, "sha256": actual_sha256})

    missing = [item for item in prerequisites if not item["available"]]
    claim_reasons = [
        {
            "code": "development_evidence",
            "message": "Development evidence is non-authoritative and cannot support claims.",
        }
    ]
    if manifest["fixture"]:
        claim_reasons.append(
            {
                "code": "fixture_evidence",
                "message": "Fixture evidence cannot support public claims.",
            }
        )
    claim_reasons.extend(
        {
            "code": "missing_prerequisite",
            "prerequisite": item["id"],
            "message": item["reason"],
        }
        for item in missing
    )
    evidence_index = {
        "algorithm": "sha256",
        "entries": evidence_entries,
        "sha256": _canonical_sha256(evidence_entries, document="manifest"),
    }
    return {
        "schema_version": 1,
        "workflow": "parity_readiness",
        "evidence_level": "development",
        "fixture": manifest["fixture"],
        "status": "unavailable" if missing else "ready",
        "claim_eligible": False,
        "claim_reasons": claim_reasons,
        "run_identity": _run_identity(
            manifest,
            code_provenance,
            declared_inputs,
            prerequisites,
            document="manifest",
        ),
        "code_provenance": code_provenance,
        "declared_inputs": declared_inputs,
        "prerequisites": prerequisites,
        "evidence_index": evidence_index,
    }


def validate_merge_report(path: Path, expected_run_identity: object) -> None:
    try:
        report = json.loads(path.read_bytes())
    except UnicodeDecodeError as error:
        raise EvidenceError(
            f"merge report is not valid UTF-8 at byte {error.start}"
        ) from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"merge report is not valid JSON: {error.msg}") from error
    if not isinstance(report, dict):
        raise EvidenceError("merge report must be a JSON object")
    _validate_schema_version(report.get("schema_version"), document="report")
    _required_string(report.get("workflow"), "merge report workflow")
    _required_string(report.get("evidence_level"), "merge report evidence_level")
    if type(report.get("fixture")) is not bool:
        raise EvidenceError("merge report fixture is required and must be a boolean")
    code_provenance = _validate_code_provenance(report.get("code_provenance"))
    declared_inputs = _validate_declared_inputs(report.get("declared_inputs"))
    prerequisites = _validate_prerequisites(report.get("prerequisites"))
    stored_run_identity = _required_string(
        report.get("run_identity"), "merge report run_identity"
    )
    recomputed_run_identity = _run_identity(
        report,
        code_provenance,
        declared_inputs,
        prerequisites,
        document="merge report",
    )
    if stored_run_identity != recomputed_run_identity:
        raise EvidenceError(
            "merge report run_identity does not match its identity-bearing fields"
        )
    evidence_index = report.get("evidence_index")
    if not isinstance(evidence_index, dict):
        raise EvidenceError("merge report evidence_index must be an object")
    entries = evidence_index.get("entries")
    if evidence_index.get("algorithm") != "sha256" or not isinstance(entries, list):
        raise EvidenceError("merge report evidence_index is malformed")
    if evidence_index.get("sha256") != _canonical_sha256(
        entries, document="merge report"
    ):
        raise EvidenceError("merge report evidence_index SHA-256 mismatch")
    if recomputed_run_identity != expected_run_identity:
        raise EvidenceError(
            "cannot merge unlike run identities: "
            f"existing {recomputed_run_identity!r}, requested {expected_run_identity!r}"
        )
