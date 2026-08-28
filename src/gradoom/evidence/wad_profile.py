from __future__ import annotations

import copy
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

PROFILE_ID = "freedoom2-deathmatch-v1"
PROVIDER_IDS = ("gradoom", "env-vizdoom-turbo")
PROFILE_CANONICAL_SHA256 = "a3953ddfd4de7c8a99f51fed58dfbdc7002f6bf1c561ebbd25819aedf6e0cde7"
_EXPECTED_PROFILE = {
    "schema_version": 1,
    "profile_id": PROFILE_ID,
    "assets": {
        "iwad": {
            "name": "Freedoom2",
            "sha256": "a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b",
        },
        "pwad": {
            "name": "ViZDoom deathmatch",
            "sha256": "1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d",
        },
    },
    "configuration": {
        "map": "MAP01",
        "skill": 1,
        "scenario": {
            "game": "VizdoomDeathmatch-v1",
            "mode": "PLAYER",
            "screen_resolution": [320, 240],
            "episode_start_time": 1,
            "player_death_termination": True,
            "episode_timeout_as_truncation": True,
            "render_hud": False,
            "render_screen_flashes": False,
        },
        "action_mode": {
            "kind": "custom_discrete",
            "count": 17,
            "table_sha256": ("0bd9dd28d67a88ef6bc54734f53d55bc4af597e672665a7f20d4b204098036af"),
        },
        "frame_skip": 2,
        "episode_horizon_tics": 4200,
        "observation": {
            "crop_or_mask": {
                "kind": "mask",
                "edges": [0, 32, 0, 0],
                "fill": 0,
            },
            "resize": {"shape": [84, 84], "algorithm": "area"},
            "grayscale": {
                "enabled": True,
                "conversion": "env-vizdoom-turbo-rgb-area-gray8-v1",
            },
            "layout": "chw",
            "frame_stack": 4,
        },
    },
}


class _ProfileAuthorityError(ValueError):
    def __init__(
        self,
        failure_code: str,
        message: str,
        *,
        actual_canonical_sha256: str | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.actual_canonical_sha256 = actual_canonical_sha256


class _DuplicateProfileKey(ValueError):
    pass


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateProfileKey(key)
        result[key] = value
    return result


def _reject_profile_constant(value: str) -> None:
    raise ValueError(f"non-standard constant {value}")


def _strict_value_mismatch(expected: object, actual: object, *, field: str) -> str | None:
    if type(expected) is not type(actual):
        return field
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        if set(expected) != set(actual):
            return field
        for key in sorted(expected):
            mismatch = _strict_value_mismatch(
                expected[key],
                actual[key],
                field=f"{field}.{key}",
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        assert isinstance(actual, list)
        if len(expected) != len(actual):
            return field
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            mismatch = _strict_value_mismatch(
                expected_item,
                actual_item,
                field=f"{field}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    return None if expected == actual else field


def _load_profile() -> tuple[dict[str, Any], bytes]:
    try:
        resource = files("gradoom.evidence").joinpath("profiles/freedoom2-deathmatch-v1.json")
        payload = resource.read_bytes()
    except OSError as error:
        raise _ProfileAuthorityError(
            "profile_resource_missing",
            f"Bundled certified WAD profile resource is unavailable: {error}.",
        ) from error
    try:
        profile = json.loads(
            payload,
            object_pairs_hook=_profile_object,
            parse_constant=_reject_profile_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateProfileKey, ValueError) as error:
        raise _ProfileAuthorityError(
            "profile_resource_invalid_json",
            f"Bundled certified WAD profile resource is invalid JSON: {error}.",
        ) from error
    except (RecursionError, OverflowError) as error:
        raise _ProfileAuthorityError(
            "profile_resource_invalid_json",
            "Bundled certified WAD profile resource has invalid JSON structure.",
        ) from error
    try:
        actual_canonical_sha256 = _canonical_sha256(profile)
        mismatch = _strict_value_mismatch(_EXPECTED_PROFILE, profile, field="profile")
    except (RecursionError, UnicodeEncodeError, ValueError) as error:
        raise _ProfileAuthorityError(
            "profile_resource_invalid_json",
            "Bundled certified WAD profile resource cannot be represented as canonical JSON.",
        ) from error
    if mismatch is not None or actual_canonical_sha256 != PROFILE_CANONICAL_SHA256:
        context = mismatch or "profile"
        raise _ProfileAuthorityError(
            "profile_resource_semantic_mismatch",
            "Bundled certified WAD profile resource does not match the independently "
            f"pinned authority at {context}.",
            actual_canonical_sha256=actual_canonical_sha256,
        )
    assert isinstance(profile, dict)
    return profile, payload


def _authority_failure_report(error: _ProfileAuthorityError) -> dict[str, Any]:
    authority = {
        "status": "failed",
        "failure_code": error.failure_code,
        "expected_canonical_sha256": PROFILE_CANONICAL_SHA256,
        "actual_canonical_sha256": error.actual_canonical_sha256,
    }
    return {
        "status": "failed",
        "authority": authority,
        "profile": None,
        "profile_identity": None,
        "providers": [],
        "binding_identity": None,
        "binding_sha256": None,
        "failures": [
            {
                "code": "profile_authority_failure",
                "context": error.failure_code,
                "message": str(error),
            }
        ],
    }


def _configuration_failures(
    expected: object,
    actual: object,
    *,
    provider: str,
    field: str = "configuration",
) -> list[dict[str, Any]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        failures: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{field}.{key}"
            if key not in actual:
                failures.append(
                    {
                        "code": "configuration_mismatch",
                        "provider": provider,
                        "field": child,
                        "expected": expected[key],
                        "actual": None,
                        "message": f"{provider} is missing required setting {child}.",
                    }
                )
            elif key not in expected:
                failures.append(
                    {
                        "code": "configuration_mismatch",
                        "provider": provider,
                        "field": child,
                        "expected": None,
                        "actual": actual[key],
                        "message": f"{provider} declares unsupported setting {child}.",
                    }
                )
            else:
                failures.extend(
                    _configuration_failures(
                        expected[key],
                        actual[key],
                        provider=provider,
                        field=child,
                    )
                )
        return failures
    if type(expected) is not type(actual) or expected != actual:
        return [
            {
                "code": "configuration_mismatch",
                "provider": provider,
                "field": field,
                "expected": expected,
                "actual": actual,
                "message": f"{provider} setting {field} does not match the certified profile.",
            }
        ]
    return []


def validate_wad_profile(
    declaration: object,
    *,
    base_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        profile, profile_payload = _load_profile()
    except _ProfileAuthorityError as error:
        return _authority_failure_report(error), []
    expected_iwad = profile["assets"]["iwad"]["sha256"]
    expected_pwad = profile["assets"]["pwad"]["sha256"]
    expected_configuration = profile["configuration"]
    failures: list[dict[str, Any]] = []
    evidence_entries = [
        {
            "name": "wad_profile_manifest",
            "sha256": hashlib.sha256(profile_payload).hexdigest(),
        }
    ]

    if not isinstance(declaration, dict):
        declaration = {}
        failures.append(
            {
                "code": "profile_declaration_missing",
                "field": "wad_profile",
                "expected": PROFILE_ID,
                "actual": None,
                "message": "The certified Freedoom2 WAD profile declaration is missing.",
            }
        )
    declared_profile_id = declaration.get("profile_id")
    if declared_profile_id != PROFILE_ID:
        failures.append(
            {
                "code": "profile_id_mismatch",
                "field": "wad_profile.profile_id",
                "expected": PROFILE_ID,
                "actual": declared_profile_id,
                "message": "The requested WAD profile is not the first certified profile.",
            }
        )

    raw_providers = declaration.get("providers")
    if not isinstance(raw_providers, list):
        raw_providers = []
        failures.append(
            {
                "code": "provider_binding_mismatch",
                "field": "wad_profile.providers",
                "expected": list(PROVIDER_IDS),
                "actual": None,
                "message": "WAD profile providers must be an array.",
            }
        )
    providers_by_id: dict[str, dict[str, Any]] = {}
    for provider_index, raw_provider in enumerate(raw_providers):
        if not isinstance(raw_provider, dict):
            failures.append(
                {
                    "code": "provider_binding_mismatch",
                    "field": f"wad_profile.providers[{provider_index}]",
                    "expected": "provider object",
                    "actual": raw_provider,
                    "message": "Each WAD profile provider binding must be an object.",
                }
            )
            continue
        provider_id = raw_provider.get("id")
        if provider_id not in PROVIDER_IDS or provider_id in providers_by_id:
            failures.append(
                {
                    "code": "provider_binding_mismatch",
                    "field": f"wad_profile.providers[{provider_index}].id",
                    "expected": list(PROVIDER_IDS),
                    "actual": provider_id,
                    "message": "WAD profile provider IDs must bind each approved provider once.",
                }
            )
            continue
        providers_by_id[provider_id] = raw_provider

    provider_reports: list[dict[str, Any]] = []
    binding_providers: list[dict[str, Any]] = []
    for provider_id in PROVIDER_IDS:
        provider = providers_by_id.get(provider_id, {})
        if not provider:
            failures.append(
                {
                    "code": "provider_binding_mismatch",
                    "provider": provider_id,
                    "field": "wad_profile.providers",
                    "expected": provider_id,
                    "actual": None,
                    "message": f"The {provider_id} WAD binding is missing.",
                }
            )
        asset_reports: dict[str, dict[str, Any]] = {}
        for asset_name, expected_sha256 in (
            ("iwad", expected_iwad),
            ("pwad", expected_pwad),
        ):
            path_field = f"{asset_name}_path"
            raw_path = provider.get(path_field)
            actual_sha256: str | None = None
            resolved_path: Path | None = None
            if isinstance(raw_path, str) and raw_path.strip():
                candidate = Path(raw_path)
                resolved_path = (
                    candidate if candidate.is_absolute() else base_directory / candidate
                ).resolve(strict=False)
                try:
                    actual_sha256 = _sha256_file(resolved_path)
                except OSError as error:
                    failures.append(
                        {
                            "code": "asset_unavailable",
                            "provider": provider_id,
                            "field": path_field,
                            "expected": expected_sha256,
                            "actual": None,
                            "message": (
                                f"{provider_id} {asset_name.upper()} is unavailable: {error}."
                            ),
                        }
                    )
            else:
                failures.append(
                    {
                        "code": "asset_unavailable",
                        "provider": provider_id,
                        "field": path_field,
                        "expected": expected_sha256,
                        "actual": None,
                        "message": f"{provider_id} {asset_name.upper()} path is missing.",
                    }
                )
            if actual_sha256 is not None:
                evidence_entries.append(
                    {
                        "name": f"wad_profile.{provider_id}.{asset_name}",
                        "sha256": actual_sha256,
                    }
                )
                if actual_sha256 != expected_sha256:
                    failures.append(
                        {
                            "code": "asset_hash_mismatch",
                            "provider": provider_id,
                            "field": asset_name,
                            "expected": expected_sha256,
                            "actual": actual_sha256,
                            "message": (
                                f"{provider_id} {asset_name.upper()} does not match the "
                                "certified asset hash."
                            ),
                        }
                    )
            asset_reports[asset_name] = {
                "path": raw_path,
                "sha256": actual_sha256,
            }

        configuration = provider.get("configuration")
        failures.extend(
            _configuration_failures(
                expected_configuration,
                configuration,
                provider=provider_id,
            )
        )
        provider_reports.append(
            {
                "id": provider_id,
                **asset_reports,
                "configuration": configuration,
            }
        )
        binding_providers.append(
            {
                "id": provider_id,
                "iwad_sha256": asset_reports["iwad"]["sha256"],
                "pwad_sha256": asset_reports["pwad"]["sha256"],
                "configuration": configuration,
            }
        )

    for asset_name in ("iwad", "pwad"):
        hashes = [provider[asset_name]["sha256"] for provider in provider_reports]
        if any(value is not None for value in hashes) and len(set(hashes)) != 1:
            failures.append(
                {
                    "code": "provider_asset_mismatch",
                    "field": asset_name,
                    "expected": "byte-identical provider files",
                    "actual": hashes,
                    "message": f"Providers would receive different {asset_name.upper()} bytes.",
                }
            )

    binding_identity = {
        "profile": profile,
        "providers": binding_providers,
    }
    report = {
        "status": "failed" if failures else "matched",
        "authority": {
            "status": "verified",
            "failure_code": None,
            "expected_canonical_sha256": PROFILE_CANONICAL_SHA256,
            "actual_canonical_sha256": PROFILE_CANONICAL_SHA256,
        },
        "profile": copy.deepcopy(profile),
        "profile_identity": _canonical_sha256(profile),
        "providers": provider_reports,
        "binding_identity": binding_identity,
        "binding_sha256": _canonical_sha256(binding_identity),
        "failures": failures,
    }
    return report, evidence_entries
