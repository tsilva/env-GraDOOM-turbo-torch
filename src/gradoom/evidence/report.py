from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """An evidence manifest or report violates the public contract."""


class _NonStandardJsonConstant(ValueError):
    """A Python JSON decoder extension appeared in an evidence document."""


_READINESS_REPORT_FIELDS = (
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
    "evidence_index",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_non_standard_json_constant(constant: str) -> None:
    raise _NonStandardJsonConstant(constant)


def _parse_json_document(payload: bytes, *, document: str) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_non_standard_json_constant,
        )
    except UnicodeDecodeError as error:
        raise EvidenceError(
            f"{document} is not valid UTF-8 at byte {error.start}"
        ) from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"{document} is not valid JSON: {error.msg}") from error
    except _NonStandardJsonConstant as error:
        raise EvidenceError(
            f"{document} is not valid JSON: non-standard constant {error}"
        ) from error


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


def _validate_string_content(
    value: object,
    *,
    document: str,
    field: str = "",
) -> None:
    if isinstance(value, str):
        null_index = value.find("\0")
        if null_index >= 0:
            location = field or "<root>"
            raise EvidenceError(
                f"{document} contains U+0000 in {location} "
                f"at character {null_index}"
            )
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as error:
            location = field or "<root>"
            raise EvidenceError(
                f"{document} contains invalid Unicode in {location} "
                f"at character {error.start}"
            ) from error
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            child_field = f"{field}[{index}]" if field else f"[{index}]"
            _validate_string_content(
                item,
                document=document,
                field=child_field,
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_field = f"{field}.<key>" if field else "<key>"
            _validate_string_content(
                key,
                document=document,
                field=key_field,
            )
            child_field = f"{field}.{key}" if field else key
            _validate_string_content(
                item,
                document=document,
                field=child_field,
            )


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    manifest = _parse_json_document(payload, document="manifest")
    if not isinstance(manifest, dict):
        raise EvidenceError("manifest must be a JSON object")
    _validate_string_content(manifest, document="manifest")
    return manifest, payload


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{field} is required and must be a non-empty string")
    return value


def _validate_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceError(f"{field} must be a lowercase SHA-256 digest")
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
    paths: set[Path] = set()
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
        path = _required_string(item.get("path"), f"{field}.path")
        if not path.strip():
            raise EvidenceError(f"{field}.path must be a non-whitespace path")
        normalized_path = Path(path)
        if normalized_path in paths:
            raise EvidenceError(f"{field}.path is duplicated")
        paths.add(normalized_path)
        _validate_sha256(item.get("sha256"), f"{field}.sha256")
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
            reason = item.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise EvidenceError(
                    f"{field}.reason must be a human-readable non-whitespace string"
                )
        prerequisites.append(item)
    return prerequisites


def _readiness_claim_reasons(
    fixture: bool,
    prerequisites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = [
        {
            "code": "development_evidence",
            "message": "Development evidence is non-authoritative and cannot support claims.",
        }
    ]
    if fixture:
        reasons.append(
            {
                "code": "fixture_evidence",
                "message": "Fixture evidence cannot support public claims.",
            }
        )
    reasons.extend(
        {
            "code": "missing_prerequisite",
            "prerequisite": item["id"],
            "message": item["reason"],
        }
        for item in prerequisites
        if not item["available"]
    )
    return reasons


def _validate_readiness_envelope(
    report: dict[str, Any],
    prerequisites: list[dict[str, Any]],
) -> None:
    missing = [item for item in prerequisites if not item["available"]]
    expected_status = "unavailable" if missing else "ready"
    if report["status"] != expected_status:
        raise EvidenceError(
            f"merge report status must be {expected_status!r} for its prerequisites"
        )
    if report["claim_eligible"] is not False:
        raise EvidenceError(
            "merge report claim_eligible must be false for development evidence"
        )
    claim_reasons = report["claim_reasons"]
    if not isinstance(claim_reasons, list):
        raise EvidenceError("merge report claim_reasons must be an array")
    expected_reasons = _readiness_claim_reasons(report["fixture"], prerequisites)
    canonical_reasons = sorted(
        json.dumps(reason, sort_keys=True, separators=(",", ":"))
        for reason in claim_reasons
    )
    canonical_expected = sorted(
        json.dumps(reason, sort_keys=True, separators=(",", ":"))
        for reason in expected_reasons
    )
    if canonical_reasons != canonical_expected:
        raise EvidenceError(
            "merge report claim_reasons do not match its fixture state and "
            "prerequisites"
        )


def _validate_evidence_index(
    value: object,
    declared_inputs: list[dict[str, Any]],
) -> None:
    field = "merge report evidence_index"
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} must be an object")
    if "algorithm" not in value:
        raise EvidenceError(f"{field}.algorithm is required")
    if value["algorithm"] != "sha256":
        raise EvidenceError(f"{field}.algorithm must be 'sha256'")
    if "entries" not in value:
        raise EvidenceError(f"{field}.entries is required")
    entries = value["entries"]
    if not isinstance(entries, list):
        raise EvidenceError(f"{field}.entries must be an array")
    if "sha256" not in value:
        raise EvidenceError(f"{field}.sha256 is required")
    stored_sha256 = _validate_sha256(value["sha256"], f"{field}.sha256")

    entries_by_name: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        entry_field = f"{field}.entries[{index}]"
        if not isinstance(entry, dict):
            raise EvidenceError(f"{entry_field} must be an object")
        name = _required_string(entry.get("name"), f"{entry_field}.name")
        if name in entries_by_name:
            raise EvidenceError(f"{entry_field}.name {name!r} is duplicated")
        _validate_sha256(entry.get("sha256"), f"{entry_field}.sha256")
        entries_by_name[name] = entry

    if stored_sha256 != _canonical_sha256(entries, document="merge report"):
        raise EvidenceError("merge report evidence_index SHA-256 mismatch")

    expected_names = {"manifest", *(item["name"] for item in declared_inputs)}
    actual_names = set(entries_by_name)
    missing_names = sorted(expected_names - actual_names)
    if missing_names:
        formatted = ", ".join(repr(name) for name in missing_names)
        raise EvidenceError(f"{field}.entries missing required names: {formatted}")
    unexpected_names = sorted(actual_names - expected_names)
    if unexpected_names:
        formatted = ", ".join(repr(name) for name in unexpected_names)
        raise EvidenceError(f"{field}.entries has unexpected names: {formatted}")
    for declared_input in declared_inputs:
        entry = entries_by_name[declared_input["name"]]
        if entry["sha256"] != declared_input["sha256"]:
            raise EvidenceError(
                f"{field} entry {declared_input['name']!r} SHA-256 does not match "
                "declared_inputs"
            )


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
    claim_reasons = _readiness_claim_reasons(manifest["fixture"], prerequisites)
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
    report = _parse_json_document(path.read_bytes(), document="merge report")
    if not isinstance(report, dict):
        raise EvidenceError("merge report must be a JSON object")
    _validate_string_content(report, document="merge report")
    for field in _READINESS_REPORT_FIELDS:
        if field not in report:
            raise EvidenceError(f"merge report {field} is required")
    _validate_schema_version(report["schema_version"], document="merge report")
    _required_string(report["workflow"], "merge report workflow")
    _required_string(report["evidence_level"], "merge report evidence_level")
    if type(report["fixture"]) is not bool:
        raise EvidenceError("merge report fixture is required and must be a boolean")
    _required_string(report["status"], "merge report status")
    if type(report["claim_eligible"]) is not bool:
        raise EvidenceError("merge report claim_eligible must be a boolean")
    if not isinstance(report["claim_reasons"], list):
        raise EvidenceError("merge report claim_reasons must be an array")
    if not isinstance(report["evidence_index"], dict):
        raise EvidenceError("merge report evidence_index must be an object")
    code_provenance = _validate_code_provenance(report["code_provenance"])
    declared_inputs = _validate_declared_inputs(report["declared_inputs"])
    prerequisites = _validate_prerequisites(report["prerequisites"])
    stored_run_identity = _required_string(
        report["run_identity"], "merge report run_identity"
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
    if report["workflow"] != "parity_readiness":
        raise EvidenceError("merge report workflow must be parity_readiness")
    if report["evidence_level"] != "development":
        raise EvidenceError("merge report evidence_level must be development")
    _validate_readiness_envelope(report, prerequisites)
    _validate_evidence_index(report["evidence_index"], declared_inputs)
    if recomputed_run_identity != expected_run_identity:
        raise EvidenceError(
            "cannot merge unlike run identities: "
            f"existing {recomputed_run_identity!r}, requested {expected_run_identity!r}"
        )
