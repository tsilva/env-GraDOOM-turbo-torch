from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .invariant_suite import InvariantSuiteError, run_invariant_suite
from .wad_profile import validate_wad_profile


class EvidenceError(ValueError):
    """An evidence manifest or report violates the public contract."""


class _NonStandardJsonConstant(ValueError):
    """A Python JSON decoder extension appeared in an evidence document."""


class _InvalidJsonNumber(ValueError):
    """A JSON number cannot be represented safely by the evidence contract."""


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
    "invariant_suite",
    "evidence_index",
)

_MAX_JSON_NESTING = 256
_REQUIRED_READINESS_PREREQUISITES = ("real_pretrained_policy_corpus",)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_non_standard_json_constant(constant: str) -> None:
    raise _NonStandardJsonConstant(constant)


def _parse_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _InvalidJsonNumber("numeric value is not finite")
    return parsed


def _parse_json_integer(value: str) -> int:
    try:
        return int(value)
    except (OverflowError, ValueError) as error:
        raise _InvalidJsonNumber("integer value exceeds supported range") from error


def _validate_json_nesting(value: object, *, document: str) -> None:
    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > _MAX_JSON_NESTING:
            raise EvidenceError(f"{document} is not valid JSON: nesting is too deep")
        if isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
        elif isinstance(current, dict):
            pending.extend((item, depth + 1) for item in current.values())


def _parse_json_document(payload: bytes, *, document: str) -> Any:
    try:
        parsed = json.loads(
            payload,
            parse_constant=_reject_non_standard_json_constant,
            parse_float=_parse_json_float,
            parse_int=_parse_json_integer,
        )
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{document} is not valid UTF-8 at byte {error.start}") from error
    except json.JSONDecodeError as error:
        raise EvidenceError(f"{document} is not valid JSON: {error.msg}") from error
    except _NonStandardJsonConstant as error:
        raise EvidenceError(
            f"{document} is not valid JSON: non-standard constant {error}"
        ) from error
    except _InvalidJsonNumber as error:
        raise EvidenceError(f"{document} is not valid JSON: {error}") from error
    except RecursionError as error:
        raise EvidenceError(f"{document} is not valid JSON: nesting is too deep") from error
    except (OverflowError, ValueError) as error:
        raise EvidenceError(f"{document} is not valid JSON: numeric value is invalid") from error
    _validate_json_nesting(parsed, document=document)
    return parsed


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
    except (RecursionError, ValueError) as error:
        raise EvidenceError(f"{document} cannot be encoded as standard JSON") from error
    return _sha256_bytes(payload)


def _validate_string_content(
    value: object,
    *,
    document: str,
    field: str = "",
) -> None:
    pending = [(value, field)]
    while pending:
        current, current_field = pending.pop()
        if isinstance(current, str):
            null_index = current.find("\0")
            if null_index >= 0:
                location = current_field or "<root>"
                raise EvidenceError(
                    f"{document} contains U+0000 in {location} at character {null_index}"
                )
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as error:
                location = current_field or "<root>"
                raise EvidenceError(
                    f"{document} contains invalid Unicode in {location} at character {error.start}"
                ) from error
        elif isinstance(current, list):
            pending.extend(
                (
                    item,
                    f"{current_field}[{index}]" if current_field else f"[{index}]",
                )
                for index, item in reversed(list(enumerate(current)))
            )
        elif isinstance(current, dict):
            for key, item in reversed(list(current.items())):
                child_field = f"{current_field}.{key}" if current_field else key
                key_field = f"{current_field}.<key>" if current_field else "<key>"
                pending.append((item, child_field))
                pending.append((key, key_field))


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


def _resolve_evidence_path(path: Path, *, base_directory: Path) -> Path:
    if not path.is_absolute():
        path = base_directory / path
    return path.resolve(strict=False)


def _paths_alias(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def _validate_declared_inputs(
    value: object,
    *,
    base_directory: Path,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceError("declared_inputs must be an array")
    inputs: list[dict[str, Any]] = []
    names: set[str] = {"manifest"}
    paths: set[Path] = set()
    resolved_paths: list[Path] = []
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
        resolved_path = _resolve_evidence_path(
            normalized_path,
            base_directory=base_directory,
        )
        if any(_paths_alias(resolved_path, previous) for previous in resolved_paths):
            raise EvidenceError(f"{field}.path aliases an earlier declared input")
        resolved_paths.append(resolved_path)
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
    wad_profile: dict[str, Any] | None = None,
    invariant_suite: dict[str, Any] | None = None,
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
    if wad_profile is not None and wad_profile["status"] == "failed":
        authority = wad_profile.get("authority")
        if isinstance(authority, dict) and authority.get("status") == "failed":
            failure = wad_profile["failures"][0]
            reasons.append(
                {
                    "code": "wad_profile_authority_failure",
                    "context": failure["context"],
                    "message": failure["message"],
                }
            )
        else:
            reasons.append(
                {
                    "code": "wad_profile_mismatch",
                    "message": "The certified Freedoom2 WAD profile did not match.",
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
    declared_prerequisites = {item["id"] for item in prerequisites}
    reasons.extend(
        {
            "code": "missing_required_prerequisite",
            "prerequisite": identifier,
            "message": f"Required readiness prerequisite {identifier!r} was not declared.",
        }
        for identifier in _REQUIRED_READINESS_PREREQUISITES
        if identifier not in declared_prerequisites
    )
    if invariant_suite is not None:
        reasons.extend(
            {
                "code": "invariant_failure",
                "behavior": failure["behavior"],
                "message": failure["message"],
            }
            for failure in invariant_suite["failures"]
        )
        reasons.extend(
            {
                "code": "invariant_suite_unavailable",
                **{
                    key: value
                    for key, value in unavailable.items()
                    if key in {"provider", "message"}
                },
            }
            for unavailable in invariant_suite["unavailable_reasons"]
        )
    return reasons


def _validate_readiness_envelope(
    report: dict[str, Any],
    prerequisites: list[dict[str, Any]],
    wad_profile: dict[str, Any] | None = None,
    invariant_suite: dict[str, Any] | None = None,
) -> None:
    missing = [item for item in prerequisites if not item["available"]]
    declared_prerequisites = {item["id"] for item in prerequisites}
    omitted_required = set(_REQUIRED_READINESS_PREREQUISITES) - declared_prerequisites
    expected_status = (
        "failed"
        if (
            (wad_profile is not None and wad_profile["status"] == "failed")
            or (invariant_suite is not None and invariant_suite["status"] == "failed")
        )
        else "unavailable"
        if missing
        or omitted_required
        or (invariant_suite is not None and invariant_suite["status"] == "unavailable")
        else "ready"
    )
    if report["status"] != expected_status:
        raise EvidenceError(
            f"merge report status must be {expected_status!r} for its prerequisites"
        )
    if report["claim_eligible"] is not False:
        raise EvidenceError("merge report claim_eligible must be false for development evidence")
    claim_reasons = report["claim_reasons"]
    if not isinstance(claim_reasons, list):
        raise EvidenceError("merge report claim_reasons must be an array")
    expected_reasons = _readiness_claim_reasons(
        report["fixture"], prerequisites, wad_profile, invariant_suite
    )
    canonical_reasons = sorted(
        json.dumps(reason, sort_keys=True, separators=(",", ":")) for reason in claim_reasons
    )
    canonical_expected = sorted(
        json.dumps(reason, sort_keys=True, separators=(",", ":")) for reason in expected_reasons
    )
    if canonical_reasons != canonical_expected:
        raise EvidenceError(
            "merge report claim_reasons do not match its fixture state and prerequisites"
        )


def _validate_evidence_index(
    value: object,
    declared_inputs: list[dict[str, Any]],
    *,
    expected_manifest_sha256: str,
    expected_entries: list[dict[str, str]] | None = None,
) -> None:
    field = "merge report evidence_index"
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} must be an object")
    undeclared_fields = sorted(set(value) - {"algorithm", "entries", "sha256"})
    if undeclared_fields:
        formatted = ", ".join(repr(name) for name in undeclared_fields)
        raise EvidenceError(f"{field} has undeclared fields: {formatted}")
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
        undeclared_fields = sorted(set(entry) - {"name", "sha256"})
        if undeclared_fields:
            formatted = ", ".join(repr(name) for name in undeclared_fields)
            raise EvidenceError(f"{entry_field} has undeclared fields: {formatted}")
        name = _required_string(entry.get("name"), f"{entry_field}.name")
        if name in entries_by_name:
            raise EvidenceError(f"{entry_field}.name {name!r} is duplicated")
        _validate_sha256(entry.get("sha256"), f"{entry_field}.sha256")
        entries_by_name[name] = entry

    if stored_sha256 != _canonical_sha256(entries, document="merge report"):
        raise EvidenceError("merge report evidence_index SHA-256 mismatch")

    expected_by_name = {
        "manifest": expected_manifest_sha256,
        **{item["name"]: item["sha256"] for item in declared_inputs},
    }
    for entry in expected_entries or ():
        expected_by_name[entry["name"]] = entry["sha256"]
    expected_names = set(expected_by_name)
    actual_names = set(entries_by_name)
    missing_names = sorted(expected_names - actual_names)
    if missing_names:
        formatted = ", ".join(repr(name) for name in missing_names)
        raise EvidenceError(f"{field}.entries missing required names: {formatted}")
    unexpected_names = sorted(actual_names - expected_names)
    if unexpected_names:
        formatted = ", ".join(repr(name) for name in unexpected_names)
        raise EvidenceError(f"{field}.entries has unexpected names: {formatted}")
    for name, expected_sha256 in expected_by_name.items():
        entry = entries_by_name[name]
        if entry["sha256"] != expected_sha256:
            if name == "manifest":
                raise EvidenceError(
                    "merge report evidence_index entry 'manifest' SHA-256 does not "
                    "match the source manifest"
                )
            if name in {item["name"] for item in declared_inputs}:
                raise EvidenceError(
                    f"{field} entry {name!r} SHA-256 does not match declared_inputs"
                )
            raise EvidenceError(
                f"{field} entry {name!r} SHA-256 does not match current evidence sources"
            )


def _run_identity(
    manifest: dict[str, Any],
    code_provenance: dict[str, Any],
    declared_inputs: list[dict[str, Any]],
    prerequisites: list[dict[str, Any]],
    *,
    document: str,
    wad_profile: dict[str, Any] | None = None,
    invariant_suite: dict[str, Any] | None = None,
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
    if wad_profile is not None:
        binding_identity = wad_profile["binding_identity"]
        identity["wad_profile"] = (
            binding_identity
            if binding_identity is not None
            else {"authority": wad_profile["authority"]}
        )
    if invariant_suite is not None and invariant_suite["configured"]:
        identity["invariant_suite"] = {
            "version": invariant_suite["version"],
            "providers": invariant_suite["providers"],
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
    declared_inputs = _validate_declared_inputs(
        manifest.get("declared_inputs"),
        base_directory=manifest_path.parent,
    )
    prerequisites = _validate_prerequisites(manifest.get("prerequisites"))

    wad_profile = None
    wad_evidence_entries: list[dict[str, str]] = []
    if "wad_profile" in manifest:
        wad_profile, wad_evidence_entries = validate_wad_profile(
            manifest["wad_profile"],
            base_directory=manifest_path.parent,
        )
        profile_prerequisite = next(
            (item for item in prerequisites if item["id"] == "certified_freedoom2_wad_profile"),
            None,
        )
        if profile_prerequisite is None:
            raise EvidenceError("wad_profile requires prerequisite certified_freedoom2_wad_profile")
        if wad_profile["status"] == "matched":
            profile_prerequisite["available"] = True
            profile_prerequisite.pop("reason", None)
        else:
            profile_prerequisite["available"] = False
            profile_prerequisite["reason"] = "The certified Freedoom2 WAD profile did not match."
    elif any(
        item["id"] == "certified_freedoom2_wad_profile" and item["available"]
        for item in prerequisites
    ):
        raise EvidenceError("available certified_freedoom2_wad_profile requires wad_profile")

    evidence_entries = [
        {"name": "manifest", "sha256": _sha256_bytes(manifest_payload)},
    ]
    for declared_input in declared_inputs:
        name = declared_input["name"]
        input_path = _resolve_evidence_path(
            Path(declared_input["path"]),
            base_directory=manifest_path.parent,
        )
        actual_sha256 = _sha256_bytes(input_path.read_bytes())
        if actual_sha256 != declared_input["sha256"]:
            raise EvidenceError(
                f"declared input {name!r} SHA-256 mismatch: "
                f"expected {declared_input['sha256']}, got {actual_sha256}"
            )
        evidence_entries.append({"name": name, "sha256": actual_sha256})
    declared_evidence_names = {entry["name"] for entry in evidence_entries}
    profile_evidence_names = {entry["name"] for entry in wad_evidence_entries}
    colliding_names = sorted(declared_evidence_names & profile_evidence_names)
    if colliding_names:
        formatted = ", ".join(repr(name) for name in colliding_names)
        raise EvidenceError(f"declared_inputs use reserved WAD profile evidence names: {formatted}")
    evidence_entries.extend(wad_evidence_entries)
    try:
        invariant_suite = run_invariant_suite(
            manifest.get("invariant_suite"),
            base_directory=manifest_path.parent,
            declared_inputs=declared_inputs,
            fixture=manifest["fixture"],
            gradoom_revision=code_provenance["revision"],
            wad_profile=wad_profile,
        )
    except InvariantSuiteError as error:
        raise EvidenceError(str(error)) from error

    missing = [item for item in prerequisites if not item["available"]]
    declared_prerequisites = {item["id"] for item in prerequisites}
    omitted_required = set(_REQUIRED_READINESS_PREREQUISITES) - declared_prerequisites
    claim_reasons = _readiness_claim_reasons(
        manifest["fixture"], prerequisites, wad_profile, invariant_suite
    )
    evidence_index = {
        "algorithm": "sha256",
        "entries": evidence_entries,
        "sha256": _canonical_sha256(evidence_entries, document="manifest"),
    }
    report = {
        "schema_version": 1,
        "workflow": "parity_readiness",
        "evidence_level": "development",
        "fixture": manifest["fixture"],
        "status": (
            "failed"
            if (
                (wad_profile is not None and wad_profile["status"] == "failed")
                or invariant_suite["status"] == "failed"
            )
            else "unavailable"
            if missing or omitted_required or invariant_suite["status"] == "unavailable"
            else "ready"
        ),
        "claim_eligible": False,
        "claim_reasons": claim_reasons,
        "run_identity": _run_identity(
            manifest,
            code_provenance,
            declared_inputs,
            prerequisites,
            document="manifest",
            wad_profile=wad_profile,
            invariant_suite=invariant_suite,
        ),
        "code_provenance": code_provenance,
        "declared_inputs": declared_inputs,
        "prerequisites": prerequisites,
        "invariant_suite": invariant_suite,
        "evidence_index": evidence_index,
    }
    if wad_profile is not None:
        report["wad_profile"] = wad_profile
    return report


def validate_merge_report(
    path: Path,
    expected_run_identity: object,
    *,
    expected_manifest_sha256: str,
    manifest_directory: Path,
    expected_wad_profile: dict[str, Any] | None = None,
    expected_invariant_suite: dict[str, Any] | None = None,
    expected_evidence_entries: list[dict[str, str]] | None = None,
) -> None:
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
    declared_inputs = _validate_declared_inputs(
        report["declared_inputs"],
        base_directory=manifest_directory,
    )
    prerequisites = _validate_prerequisites(report["prerequisites"])
    wad_profile = report.get("wad_profile")
    if wad_profile != expected_wad_profile:
        raise EvidenceError(
            "merge report wad_profile does not match the current profile validation"
        )
    invariant_suite = report["invariant_suite"]
    if invariant_suite != expected_invariant_suite:
        raise EvidenceError(
            "merge report invariant_suite does not match the current invariant execution"
        )
    stored_run_identity = _required_string(report["run_identity"], "merge report run_identity")
    recomputed_run_identity = _run_identity(
        report,
        code_provenance,
        declared_inputs,
        prerequisites,
        document="merge report",
        wad_profile=wad_profile,
        invariant_suite=invariant_suite,
    )
    if stored_run_identity != recomputed_run_identity:
        raise EvidenceError("merge report run_identity does not match its identity-bearing fields")
    if recomputed_run_identity != expected_run_identity:
        raise EvidenceError(
            "cannot merge unlike run identities: "
            f"existing {recomputed_run_identity!r}, requested {expected_run_identity!r}"
        )
    if report["workflow"] != "parity_readiness":
        raise EvidenceError("merge report workflow must be parity_readiness")
    if report["evidence_level"] != "development":
        raise EvidenceError("merge report evidence_level must be development")
    _validate_readiness_envelope(report, prerequisites, wad_profile, invariant_suite)
    _validate_evidence_index(
        report["evidence_index"],
        declared_inputs,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_entries=expected_evidence_entries,
    )
