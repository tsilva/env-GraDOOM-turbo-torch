from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from .reference_provider import REFERENCE_REVISION

INVARIANT_SUITE_VERSION = "1.0.0"

_PROVIDER_IDS = ("gradoom", "env-vizdoom-turbo")
_COMMON_BEHAVIORS = (
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
)
_DIAGNOSTIC_IDS = ("mechanics", "trace", "distribution", "observation", "rendering")


class InvariantSuiteError(ValueError):
    """The invariant-suite declaration is not a valid public evidence request."""


def _canonical_payload(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvariantSuiteError(f"{field} is required and must be a non-empty string")
    if "\0" in value:
        raise InvariantSuiteError(f"{field} must not contain U+0000")
    return value


def _parse_provider_contract(payload: str, *, provider: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard constant {value}")

    try:
        contract = json.loads(payload, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise InvariantSuiteError(
            f"{provider} invariant provider returned invalid JSON: {error}"
        ) from error
    if not isinstance(contract, dict):
        raise InvariantSuiteError(f"{provider} invariant provider contract must be an object")
    pending = [(contract, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > 256:
            raise InvariantSuiteError(f"{provider} invariant provider contract nesting is too deep")
        if isinstance(value, str):
            if "\0" in value:
                raise InvariantSuiteError(f"{provider} invariant provider contract contains U+0000")
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as error:
                raise InvariantSuiteError(
                    f"{provider} invariant provider contract contains invalid Unicode"
                ) from error
        elif isinstance(value, float) and not math.isfinite(value):
            raise InvariantSuiteError(
                f"{provider} invariant provider contract contains a non-finite number"
            )
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
        elif isinstance(value, dict):
            pending.extend((key, depth + 1) for key in value)
            pending.extend((item, depth + 1) for item in value.values())
    if contract.get("schema_version") != 1:
        raise InvariantSuiteError(
            f"{provider} invariant provider contract requires schema_version 1"
        )
    if contract.get("provider") != provider:
        raise InvariantSuiteError(
            f"{provider} invariant provider contract identifies {contract.get('provider')!r}"
        )
    _required_string(contract.get("revision"), f"{provider} contract revision")
    if not isinstance(contract.get("behaviors"), dict):
        raise InvariantSuiteError(f"{provider} contract behaviors must be an object")
    return contract


def _failed_provider_execution(provider: str, message: str) -> dict[str, str]:
    return {
        "behavior": f"{provider}.provider_execution",
        "provider": provider,
        "message": message,
    }


def _normalized_command(provider: dict[str, Any]) -> list[str]:
    provider_id = provider["id"]
    command = provider.get("command")
    if not isinstance(command, list) or not command:
        raise InvariantSuiteError(
            f"invariant_suite provider {provider_id!r} command must be a non-empty array"
        )
    return [
        _required_string(item, f"invariant_suite provider {provider_id!r} command[{index}]")
        for index, item in enumerate(command)
    ]


def _run_provider(
    provider: dict[str, Any],
    *,
    base_directory: Path,
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    provider_id = provider["id"]
    normalized_command = _normalized_command(provider)
    try:
        completed = subprocess.run(
            normalized_command,
            cwd=base_directory,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        return None, _failed_provider_execution(
            provider_id,
            f"{provider_id} public invariant provider could not execute: {error}",
        )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        return None, _failed_provider_execution(
            provider_id,
            f"{provider_id} public invariant provider failed: {detail}",
        )
    try:
        return _parse_provider_contract(completed.stdout, provider=provider_id), None
    except InvariantSuiteError as error:
        return None, _failed_provider_execution(provider_id, str(error))


def _shape(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(type(dimension) is int and dimension > 0 for dimension in value)
    )


def _valid_common_behavior(behavior: str, value: object) -> bool:
    if behavior == "constructor":
        return (
            isinstance(value, dict)
            and value.get("accepted") is True
            and isinstance(value.get("parameters"), list)
            and bool(value["parameters"])
            and all(isinstance(name, str) and name for name in value["parameters"])
        )
    if behavior == "action_meanings":
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(meaning, str) and meaning for meaning in value)
        )
    if behavior == "observation_shapes":
        return isinstance(value, dict) and _shape(value.get("reset")) and _shape(value.get("step"))
    if behavior == "signal_shapes":
        return isinstance(value, dict) and all(
            isinstance(value.get(operation), dict)
            and _shape(value[operation].get("player_killcount"))
            for operation in ("reset", "step")
        )
    if behavior == "rewards":
        return (
            isinstance(value, dict)
            and _shape(value.get("shape"))
            and value.get("dtype") in {"float32", "float64"}
            and isinstance(value.get("sample"), list)
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
                for item in value["sample"]
            )
        )
    expected_flags = {
        "reset": {"returns_observation_and_signals": True},
        "step": {"returns_five_tuple": True},
        "masked_reset": {"supported": True, "selected_lane_only": True},
        "termination": {"reported_separately": True, "requires_reset": True},
        "truncation": {"reported_separately": True, "requires_reset": True},
        "episode": {"step_before_reset_rejected": True, "autoreset": False},
        "player_killcount": {"present": True, "player_kill_delta": 1},
        "player_killcount.enemy_on_enemy_exclusion": {"enemy_on_enemy_delta": 0},
    }
    return isinstance(value, dict) and all(
        value.get(field) == expected for field, expected in expected_flags[behavior].items()
    )


def _common_checks(contracts: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    gradoom_behaviors = contracts["gradoom"]["behaviors"]
    reference_behaviors = contracts["env-vizdoom-turbo"]["behaviors"]
    for behavior in _COMMON_BEHAVIORS:
        gradoom_value = gradoom_behaviors.get(behavior)
        reference_value = reference_behaviors.get(behavior)
        if not _valid_common_behavior(behavior, gradoom_value):
            checks.append(
                {
                    "behavior": behavior,
                    "status": "failed",
                    "provider": "gradoom",
                    "message": f"gradoom public behavior is invalid for {behavior}.",
                }
            )
        elif not _valid_common_behavior(behavior, reference_value):
            checks.append(
                {
                    "behavior": behavior,
                    "status": "failed",
                    "provider": "env-vizdoom-turbo",
                    "message": (f"env-vizdoom-turbo public behavior is invalid for {behavior}."),
                }
            )
        elif gradoom_value != reference_value:
            checks.append(
                {
                    "behavior": behavior,
                    "status": "failed",
                    "provider": "env-vizdoom-turbo",
                    "message": (
                        f"env-vizdoom-turbo public behavior does not match gradoom for {behavior}."
                    ),
                }
            )
        else:
            checks.append({"behavior": behavior, "status": "passed"})
    return checks


def _gradoom_device_checks(contract: dict[str, Any]) -> list[dict[str, str]]:
    value = contract.get("tensor_device")
    declared_device = value.get("declared_device") if isinstance(value, dict) else None

    def descriptor_matches(name: str) -> bool:
        descriptor = value.get(name) if isinstance(value, dict) else None
        return (
            isinstance(descriptor, dict)
            and descriptor.get("transport") == "torch"
            and descriptor.get("device") == declared_device
        )

    checks: list[dict[str, str]] = []
    predicates = {
        "gradoom.tensor_inputs": descriptor_matches("reset_mask_input")
        and descriptor_matches("step_action_input"),
        "gradoom.tensor_outputs": descriptor_matches("reset_outputs")
        and descriptor_matches("step_outputs"),
        "gradoom.device": isinstance(declared_device, str) and bool(declared_device),
    }
    for behavior, passed in predicates.items():
        checks.append(
            {"behavior": behavior, "status": "passed"}
            if passed
            else {
                "behavior": behavior,
                "status": "failed",
                "provider": "gradoom",
                "message": f"gradoom public behavior is invalid for {behavior}.",
            }
        )
    return checks


def _diagnostics(value: object) -> dict[str, Any]:
    declared = value if isinstance(value, list) else []
    by_id = {
        item.get("id"): item
        for item in declared
        if isinstance(item, dict) and item.get("id") in _DIAGNOSTIC_IDS
    }
    return {
        "affects_verdict": False,
        "tools": [
            by_id.get(identifier, {"id": identifier, "status": "not_run"})
            for identifier in _DIAGNOSTIC_IDS
        ],
    }


def run_invariant_suite(
    declaration: object,
    *,
    base_directory: Path,
    declared_input_names: set[str],
    fixture: bool,
    gradoom_revision: str,
) -> dict[str, Any]:
    if declaration is None:
        return {
            "version": INVARIANT_SUITE_VERSION,
            "configured": False,
            "status": "unavailable",
            "checks": [],
            "failures": [],
            "unavailable_reasons": [
                {
                    "code": "invariant_suite_not_configured",
                    "message": "No invariant-suite provider execution was declared.",
                }
            ],
            "providers": [],
            "diagnostics": _diagnostics(None),
        }
    if not isinstance(declaration, dict):
        raise InvariantSuiteError("invariant_suite must be an object")
    version = declaration.get("version")
    if version != INVARIANT_SUITE_VERSION:
        raise InvariantSuiteError(
            f"unsupported invariant_suite.version {version!r}; supported version is "
            f"{INVARIANT_SUITE_VERSION!r}"
        )
    providers = declaration.get("providers")
    if not isinstance(providers, list):
        raise InvariantSuiteError("invariant_suite.providers must be an array")
    providers_by_id: dict[str, dict[str, Any]] = {}
    for index, provider in enumerate(providers):
        field = f"invariant_suite.providers[{index}]"
        if not isinstance(provider, dict):
            raise InvariantSuiteError(f"{field} must be an object")
        provider_id = provider.get("id")
        if provider_id not in _PROVIDER_IDS or provider_id in providers_by_id:
            raise InvariantSuiteError(
                f"{field}.id must bind each of {list(_PROVIDER_IDS)!r} exactly once"
            )
        if type(provider.get("available")) is not bool:
            raise InvariantSuiteError(f"{field}.available must be a boolean")
        if provider["available"]:
            runner_input = _required_string(provider.get("runner_input"), f"{field}.runner_input")
            if runner_input not in declared_input_names:
                raise InvariantSuiteError(
                    f"{field}.runner_input {runner_input!r} is not a declared input"
                )
            _normalized_command(provider)
        else:
            _required_string(provider.get("reason"), f"{field}.reason")
        providers_by_id[provider_id] = provider
    if set(providers_by_id) != set(_PROVIDER_IDS):
        raise InvariantSuiteError(
            f"invariant_suite.providers must bind each of {list(_PROVIDER_IDS)!r} exactly once"
        )

    unavailable = [
        {
            "code": "provider_unavailable",
            "provider": provider_id,
            "message": providers_by_id[provider_id]["reason"],
        }
        for provider_id in _PROVIDER_IDS
        if not providers_by_id[provider_id]["available"]
    ]
    provider_reports = [
        {
            "id": provider_id,
            "status": ("pending" if providers_by_id[provider_id]["available"] else "unavailable"),
            **(
                {
                    "runner_input": providers_by_id[provider_id]["runner_input"],
                    "command": providers_by_id[provider_id]["command"],
                }
                if providers_by_id[provider_id]["available"]
                else {"reason": providers_by_id[provider_id]["reason"]}
            ),
        }
        for provider_id in _PROVIDER_IDS
    ]
    diagnostics = _diagnostics(declaration.get("diagnostics"))
    if unavailable:
        return {
            "version": INVARIANT_SUITE_VERSION,
            "configured": True,
            "status": "unavailable",
            "checks": [],
            "failures": [],
            "unavailable_reasons": unavailable,
            "providers": provider_reports,
            "diagnostics": diagnostics,
        }

    contracts: dict[str, dict[str, Any]] = {}
    execution_failures: list[dict[str, str]] = []
    for index, provider_id in enumerate(_PROVIDER_IDS):
        contract, failure = _run_provider(
            providers_by_id[provider_id],
            base_directory=base_directory,
        )
        if failure is not None:
            execution_failures.append(failure)
            provider_reports[index] = {
                "id": provider_id,
                "status": "failed",
                "runner_input": providers_by_id[provider_id]["runner_input"],
                "command": providers_by_id[provider_id]["command"],
            }
            continue
        assert contract is not None
        contracts[provider_id] = contract
        provider_reports[index] = {
            "id": provider_id,
            "status": "executed",
            "runner_input": providers_by_id[provider_id]["runner_input"],
            "command": providers_by_id[provider_id]["command"],
            "revision": contract["revision"],
            "contract_sha256": _canonical_sha256(contract),
        }
        expected_revision = gradoom_revision if provider_id == "gradoom" else REFERENCE_REVISION
        if not fixture and contract["revision"] != expected_revision:
            execution_failures.append(
                {
                    "behavior": f"{provider_id}.revision",
                    "provider": provider_id,
                    "message": (
                        f"{provider_id} revision mismatch: expected {expected_revision}, "
                        f"found {contract['revision']}."
                    ),
                }
            )
    if execution_failures:
        return {
            "version": INVARIANT_SUITE_VERSION,
            "configured": True,
            "status": "failed",
            "checks": [],
            "failures": execution_failures,
            "unavailable_reasons": [],
            "providers": provider_reports,
            "diagnostics": diagnostics,
        }

    checks = [*_common_checks(contracts), *_gradoom_device_checks(contracts["gradoom"])]
    failures = [
        {key: value for key, value in check.items() if key != "status"}
        for check in checks
        if check["status"] == "failed"
    ]
    return {
        "version": INVARIANT_SUITE_VERSION,
        "configured": True,
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
        "unavailable_reasons": [],
        "providers": provider_reports,
        "diagnostics": diagnostics,
    }


__all__ = [
    "INVARIANT_SUITE_VERSION",
    "InvariantSuiteError",
    "run_invariant_suite",
]
