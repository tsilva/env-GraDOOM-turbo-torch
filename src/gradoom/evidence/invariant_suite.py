from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

from gradoom.actions import DEATHMATCH_ACTION_MEANINGS

from . import invariant_runner
from .invariant_contract import INVARIANT_SUITE_VERSION
from .reference_provider import REFERENCE_REVISION

_PROVIDER_IDS = ("gradoom", "env-vizdoom-turbo")
_SIGNALS = ("health", "killcount", "player_killcount", "episode_return")
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
_CONSTRUCTOR_PARAMETERS = (
    "game",
    "state",
    "scenario",
    "info",
    "use_restricted_actions",
    "record",
    "players",
    "inttype",
    "obs_type",
    "render_mode",
    "num_envs",
    "num_threads",
    "rom_path",
    "transport",
    "obs_copy",
    "obs_resize",
    "obs_crop",
    "obs_crop_mode",
    "obs_crop_fill",
    "obs_grayscale",
    "obs_resize_algorithm",
    "obs_layout",
    "frame_skip",
    "frame_stack",
    "maxpool_last_two",
    "noop_reset_max",
    "use_fire_reset",
    "sticky_action_prob",
    "reward_clip",
    "info_filter",
    "info_frame_stack_keys",
    "state_catalog",
)
_CONSTRUCTOR_DEFAULTS = (
    "required",
    None,
    None,
    None,
    "default",
    False,
    1,
    "stable",
    "image",
    None,
    1,
    None,
    None,
    "default",
    "safe_view",
    [84, 84],
    None,
    "remove",
    0,
    True,
    "area",
    "chw",
    4,
    4,
    False,
    0,
    False,
    0.0,
    False,
    "all",
    None,
    None,
)
_CONSTRUCTOR_KINDS = (
    *("POSITIONAL_OR_KEYWORD",) * 10,
    *("KEYWORD_ONLY",) * 22,
)


class InvariantSuiteError(ValueError):
    """The invariant-suite declaration is not a valid public evidence request."""


def _canonical_payload(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_payload(value)).hexdigest()


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvariantSuiteError(f"{field} is required and must be a non-empty string")
    if "\0" in value:
        raise InvariantSuiteError(f"{field} must not contain U+0000")
    return value


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


def _unconfigured_suite() -> dict[str, Any]:
    return {
        "version": INVARIANT_SUITE_VERSION,
        "configured": False,
        "status": "unavailable",
        "checks": [],
        "failures": [],
        "unavailable_reasons": [
            {
                "code": "invariant_suite_not_configured",
                "message": "No invariant-suite execution was declared.",
            }
        ],
        "providers": [],
        "diagnostics": _diagnostics(None),
    }


def _declared_runner(
    name: str,
    *,
    declared_inputs: list[dict[str, Any]],
    base_directory: Path,
) -> tuple[Path, str]:
    matches = [item for item in declared_inputs if item["name"] == name]
    if len(matches) != 1:
        raise InvariantSuiteError(f"invariant_suite.runner_input {name!r} is not a declared input")
    item = matches[0]
    path = Path(item["path"])
    resolved = (base_directory / path if not path.is_absolute() else path).resolve()
    trusted = Path(invariant_runner.__file__).resolve()
    if resolved != trusted:
        raise InvariantSuiteError(
            "invariant_suite.runner_input must bind the repository-owned invariant runner"
        )
    actual = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise InvariantSuiteError(
            f"invariant runner SHA-256 mismatch: expected {item['sha256']}, got {actual}"
        )
    return resolved, actual


def _parse_runner_response(payload: str, *, challenge: str, runner_sha256: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard constant {value}")

    try:
        response = json.loads(payload, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise InvariantSuiteError(
            f"authenticated invariant runner returned invalid JSON: {error}"
        ) from error
    if not isinstance(response, dict):
        raise InvariantSuiteError("authenticated invariant runner response must be an object")
    expected_fields = {
        "protocol_version",
        "challenge",
        "runner_sha256",
        "status",
        "contracts",
        "unavailable_reasons",
    }
    if set(response) != expected_fields:
        raise InvariantSuiteError("authenticated invariant runner response has invalid fields")
    if response["protocol_version"] != invariant_runner.RUNNER_PROTOCOL_VERSION:
        raise InvariantSuiteError("authenticated invariant runner protocol mismatch")
    if response["challenge"] != challenge:
        raise InvariantSuiteError("authenticated invariant runner challenge mismatch")
    if response["runner_sha256"] != runner_sha256:
        raise InvariantSuiteError("authenticated invariant runner source hash mismatch")
    if response["status"] not in {"complete", "unavailable"}:
        raise InvariantSuiteError("authenticated invariant runner status is invalid")
    if not isinstance(response["contracts"], list) or not isinstance(
        response["unavailable_reasons"], list
    ):
        raise InvariantSuiteError("authenticated invariant runner payload is invalid")
    if response["status"] == "complete" and (
        response["unavailable_reasons"] or len(response["contracts"]) != 2
    ):
        raise InvariantSuiteError("complete invariant execution requires exactly two contracts")
    if response["status"] == "unavailable" and (
        response["contracts"] or not response["unavailable_reasons"]
    ):
        raise InvariantSuiteError(
            "unavailable invariant execution requires reasons and no contracts"
        )
    return response


def _execute_runner(
    *,
    runner_sha256: str,
    mode: str,
    fixture_case: str,
    gradoom_revision: str,
    real_configuration: object,
) -> dict[str, Any]:
    challenge = secrets.token_hex(32)
    request = {
        "protocol_version": invariant_runner.RUNNER_PROTOCOL_VERSION,
        "challenge": challenge,
        "mode": mode,
        "fixture_case": fixture_case,
        "gradoom_revision": gradoom_revision,
        "real_configuration": real_configuration,
    }
    timeout_seconds = 60
    if mode == "real" and isinstance(real_configuration, dict):
        timeout_seconds = int(real_configuration["timeout_seconds"])
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "gradoom.evidence.invariant_runner"],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            input=json.dumps(request, allow_nan=False),
        )
    except subprocess.TimeoutExpired:
        return {
            "protocol_version": invariant_runner.RUNNER_PROTOCOL_VERSION,
            "challenge": challenge,
            "runner_sha256": runner_sha256,
            "status": "unavailable",
            "contracts": [],
            "unavailable_reasons": [
                {
                    "code": "invariant_runner_timeout",
                    "message": (
                        "Authenticated invariant execution exhausted its predeclared "
                        f"{timeout_seconds}-second timeout."
                    ),
                }
            ],
        }
    except (OSError, UnicodeError) as error:
        raise InvariantSuiteError(
            f"authenticated invariant runner could not execute: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise InvariantSuiteError(f"authenticated invariant runner failed: {detail}")
    return _parse_runner_response(
        completed.stdout,
        challenge=challenge,
        runner_sha256=runner_sha256,
    )


def _resolved_declared_input(
    name: object,
    *,
    field: str,
    declared_inputs: list[dict[str, Any]],
    base_directory: Path,
) -> Path:
    input_name = _required_string(name, field)
    matches = [item for item in declared_inputs if item["name"] == input_name]
    if len(matches) != 1:
        raise InvariantSuiteError(f"{field} {input_name!r} is not a declared input")
    path = Path(matches[0]["path"])
    return (base_directory / path if not path.is_absolute() else path).resolve()


def _validated_semantic_probes(value: object) -> dict[str, Any]:
    expected = {
        "termination",
        "truncation",
        "player_killcount",
        "player_killcount.enemy_on_enemy_exclusion",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise InvariantSuiteError(
            "invariant_suite.real_configuration.semantic_probes must declare every "
            "required lifecycle and kill probe"
        )
    result: dict[str, Any] = {}
    for behavior in sorted(expected):
        probe = value[behavior]
        field = f"invariant_suite.real_configuration.semantic_probes.{behavior}"
        if not isinstance(probe, dict) or set(probe) != {"seeds", "actions", "max_steps"}:
            raise InvariantSuiteError(f"{field} has invalid fields")
        seeds = probe["seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != 2
            or any(type(seed) is not int or not 0 <= seed <= 0xFFFFFFFF for seed in seeds)
        ):
            raise InvariantSuiteError(f"{field}.seeds must contain two uint32 values")
        actions = probe["actions"]
        if (
            not isinstance(actions, list)
            or not actions
            or any(
                not isinstance(row, list)
                or len(row) != 2
                or any(
                    type(action) is not int or not 0 <= action < len(DEATHMATCH_ACTION_MEANINGS)
                    for action in row
                )
                for row in actions
            )
        ):
            raise InvariantSuiteError(
                f"{field}.actions must contain two-lane rows of pinned action indices"
            )
        max_steps = probe["max_steps"]
        if type(max_steps) is not int or not 1 <= max_steps <= 100_000:
            raise InvariantSuiteError(f"{field}.max_steps must be in [1, 100000]")
        result[behavior] = {"seeds": seeds, "actions": actions, "max_steps": max_steps}
    return result


def _prepare_real_configuration(
    value: object,
    *,
    declared_inputs: list[dict[str, Any]],
    base_directory: Path,
    wad_profile: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "device",
        "timeout_seconds",
        "reference_scenario_config_input",
        "semantic_probes",
    }:
        raise InvariantSuiteError(
            "invariant_suite.real_configuration must declare device, timeout_seconds, "
            "reference_scenario_config_input, and semantic_probes"
        )
    device = _required_string(value["device"], "invariant_suite.real_configuration.device")
    if (
        device not in {"cpu", "cuda"}
        and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device, flags=re.ASCII) is None
    ):
        raise InvariantSuiteError(
            "invariant_suite.real_configuration.device must be 'cpu', 'cuda', or 'cuda:N'"
        )
    probes = _validated_semantic_probes(value["semantic_probes"])
    timeout_seconds = value["timeout_seconds"]
    minimum_timeout = max(
        120,
        math.ceil(sum(probe["max_steps"] for probe in probes.values()) * 0.05),
    )
    if (
        type(timeout_seconds) is not int
        or timeout_seconds < minimum_timeout
        or timeout_seconds > 86_400
    ):
        raise InvariantSuiteError(
            "invariant_suite.real_configuration.timeout_seconds must be an integer in "
            f"[{minimum_timeout}, 86400] for the declared probe budget"
        )
    reference_config = _resolved_declared_input(
        value["reference_scenario_config_input"],
        field="invariant_suite.real_configuration.reference_scenario_config_input",
        declared_inputs=declared_inputs,
        base_directory=base_directory,
    )
    if not reference_config.is_file():
        raise InvariantSuiteError("reference scenario configuration is unavailable")
    raw_providers = wad_profile.get("providers")
    if not isinstance(raw_providers, list):
        raise InvariantSuiteError("matched wad_profile is missing provider bindings")
    providers: dict[str, dict[str, Any]] = {}
    for provider in raw_providers:
        if not isinstance(provider, dict) or provider.get("id") not in _PROVIDER_IDS:
            raise InvariantSuiteError("matched wad_profile has invalid provider bindings")
        provider_id = provider["id"]
        binding: dict[str, Any] = {"configuration": provider.get("configuration")}
        for asset in ("iwad", "pwad"):
            report = provider.get(asset)
            if not isinstance(report, dict):
                raise InvariantSuiteError("matched wad_profile has incomplete provider assets")
            raw_path = report.get("path")
            if not isinstance(raw_path, str):
                raise InvariantSuiteError("matched wad_profile has incomplete provider paths")
            path = Path(raw_path)
            binding[f"{asset}_path"] = str(
                (base_directory / path if not path.is_absolute() else path).resolve()
            )
            binding[f"{asset}_sha256"] = report.get("sha256")
        providers[provider_id] = binding
    if set(providers) != set(_PROVIDER_IDS):
        raise InvariantSuiteError("matched wad_profile must bind both invariant providers")
    reference_pwad = Path(providers["env-vizdoom-turbo"]["pwad_path"])
    consumed_pwad = reference_config.with_name("deathmatch.wad")
    try:
        same_reference_pwad = consumed_pwad.samefile(reference_pwad)
    except OSError:
        same_reference_pwad = False
    if not same_reference_pwad:
        raise InvariantSuiteError(
            "reference scenario configuration must consume the validated reference PWAD"
        )
    profile_configuration = providers["env-vizdoom-turbo"]["configuration"]
    if not isinstance(profile_configuration, dict):
        raise InvariantSuiteError("matched wad_profile has invalid reference configuration")
    try:
        invariant_runner._validate_reference_scenario_config(
            reference_config,
            profile_configuration,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise InvariantSuiteError(
            "reference scenario configuration is not the exact validated scenario configuration"
        ) from error
    return {
        "device": device,
        "timeout_seconds": timeout_seconds,
        "reference_scenario_config_path": str(reference_config),
        "semantic_probes": probes,
        "wad_binding_sha256": wad_profile.get("binding_sha256"),
        "providers": providers,
    }


def _descriptor(value: object, *, shape: list[int], dtype: str) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"transport", "shape", "dtype", "device"}
        and value["transport"] in {"torch", "numpy"}
        and value["shape"] == shape
        and value["dtype"] == dtype
        and isinstance(value["device"], str)
        and bool(value["device"])
    )


def _constructor_valid(value: object) -> bool:
    return isinstance(value, dict) and value == {
        "accepted": True,
        "parameters": list(_CONSTRUCTOR_PARAMETERS),
        "defaults": list(_CONSTRUCTOR_DEFAULTS),
        "kinds": list(_CONSTRUCTOR_KINDS),
    }


def _common_value(behavior: str, value: object) -> object:
    if behavior == "player_killcount":
        assert isinstance(value, dict)
        return {
            "present": value["present"],
            "player_kill_observed": value["player_kill_delta"] > 0,
            "attribution": value["attribution"],
        }
    if behavior == "player_killcount.enemy_on_enemy_exclusion":
        assert isinstance(value, dict)
        return {
            "enemy_on_enemy_delta": value["enemy_on_enemy_delta"],
            "compatibility_kill_observed": value["compatibility_kill_delta"] > 0,
            "attribution": value["attribution"],
        }
    if behavior in {"observation_shapes", "signal_shapes", "rewards"}:
        assert isinstance(value, dict)

        def strip_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
            return {"shape": descriptor["shape"], "dtype": descriptor["dtype"]}

        if behavior == "observation_shapes":
            return {
                operation: strip_descriptor(value[operation]) for operation in ("reset", "step")
            }
        if behavior == "signal_shapes":
            return {
                operation: {
                    signal: strip_descriptor(value[operation][signal]) for signal in _SIGNALS
                }
                for operation in ("reset", "step")
            }
        return {
            "shape": value["shape"],
            "dtype": value["dtype"],
            "sample": value["sample"],
        }
    return value


def _valid_common_behavior(behavior: str, value: object) -> bool:
    if behavior == "constructor":
        return _constructor_valid(value)
    if behavior == "action_meanings":
        return value == list(DEATHMATCH_ACTION_MEANINGS)
    if behavior == "observation_shapes":
        return (
            isinstance(value, dict)
            and set(value) == {"reset", "step"}
            and all(
                _descriptor(value[operation], shape=[2, 4, 84, 84], dtype="uint8")
                for operation in ("reset", "step")
            )
        )
    if behavior == "signal_shapes":
        return (
            isinstance(value, dict)
            and set(value) == {"reset", "step"}
            and all(
                isinstance(value[operation], dict)
                and set(value[operation]) == set(_SIGNALS)
                and all(
                    _descriptor(value[operation][name], shape=[2], dtype="float64")
                    for name in _SIGNALS
                )
                for operation in ("reset", "step")
            )
        )
    if behavior == "rewards":
        return (
            isinstance(value, dict)
            and set(value) == {"transport", "shape", "dtype", "device", "sample"}
            and _descriptor(
                {key: value[key] for key in ("transport", "shape", "dtype", "device")},
                shape=[2],
                dtype="float32",
            )
            and isinstance(value["sample"], list)
            and len(value["sample"]) == 2
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(item)
                for item in value["sample"]
            )
        )
    if behavior == "player_killcount":
        return (
            isinstance(value, dict)
            and set(value) == {"present", "player_kill_delta", "attribution"}
            and value["present"] is True
            and isinstance(value["player_kill_delta"], (int, float))
            and not isinstance(value["player_kill_delta"], bool)
            and math.isfinite(value["player_kill_delta"])
            and value["player_kill_delta"] > 0
            and value["attribution"] == {"attacker": "player", "target": "enemy"}
        )
    if behavior == "player_killcount.enemy_on_enemy_exclusion":
        return (
            isinstance(value, dict)
            and set(value) == {"enemy_on_enemy_delta", "compatibility_kill_delta", "attribution"}
            and value["enemy_on_enemy_delta"] == 0
            and isinstance(value["compatibility_kill_delta"], (int, float))
            and not isinstance(value["compatibility_kill_delta"], bool)
            and math.isfinite(value["compatibility_kill_delta"])
            and value["compatibility_kill_delta"] > 0
            and value["attribution"] == {"attacker": "enemy", "target": "enemy"}
        )
    expected = {
        "reset": {"returns_observation_and_signals": True},
        "step": {"returns_five_tuple": True},
        "masked_reset": {
            "supported": True,
            "selected_lane_state_and_signals_reset": True,
            "unselected_lane_state_and_signals_unchanged": True,
            "selected_lane_continues_from_reset_state": True,
            "unselected_lane_continues_without_reset": True,
        },
        "termination": {
            "reported_separately": True,
            "requires_reset": True,
            "terminal_lane_reset": True,
            "stepping_resumed": True,
        },
        "truncation": {
            "reported_separately": True,
            "requires_reset": True,
            "terminal_lane_reset": True,
            "stepping_resumed": True,
        },
        "episode": {"step_before_reset_rejected": True, "autoreset": False},
    }
    return value == expected[behavior]


def _invalid_behavior_message(provider: str, behavior: str, value: object) -> str:
    if isinstance(value, dict) and isinstance(value.get("probe_error"), str):
        return f"{provider} public {behavior} probe failed: {value['probe_error']}."
    return f"{provider} public behavior is invalid for {behavior}."


def _parse_contracts(values: list[object]) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or set(value) not in (
            {"schema_version", "provider", "revision", "behaviors"},
            {"schema_version", "provider", "revision", "behaviors", "tensor_device"},
        ):
            raise InvariantSuiteError("invariant runner returned a malformed provider contract")
        provider = value.get("provider")
        if provider not in _PROVIDER_IDS or provider in contracts:
            raise InvariantSuiteError("invariant runner did not bind each provider exactly once")
        if value.get("schema_version") != 1 or not isinstance(value.get("revision"), str):
            raise InvariantSuiteError(f"{provider} invariant contract metadata is invalid")
        behaviors = value.get("behaviors")
        if not isinstance(behaviors, dict) or set(behaviors) != set(_COMMON_BEHAVIORS):
            raise InvariantSuiteError(f"{provider} invariant contract behavior set is incomplete")
        if provider == "gradoom" and "tensor_device" not in value:
            raise InvariantSuiteError("gradoom invariant contract requires tensor_device")
        if provider != "gradoom" and "tensor_device" in value:
            raise InvariantSuiteError(
                "reference invariant contract cannot declare GraDOOM device data"
            )
        contracts[provider] = value
    if set(contracts) != set(_PROVIDER_IDS):
        raise InvariantSuiteError("invariant runner did not bind each provider exactly once")
    return contracts


def _common_checks(contracts: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    first = contracts["gradoom"]["behaviors"]
    second = contracts["env-vizdoom-turbo"]["behaviors"]
    for behavior in _COMMON_BEHAVIORS:
        first_value = first[behavior]
        second_value = second[behavior]
        if not _valid_common_behavior(behavior, first_value):
            checks.append(
                {
                    "behavior": behavior,
                    "status": "failed",
                    "provider": "gradoom",
                    "message": _invalid_behavior_message("gradoom", behavior, first_value),
                }
            )
        elif not _valid_common_behavior(behavior, second_value):
            checks.append(
                {
                    "behavior": behavior,
                    "status": "failed",
                    "provider": "env-vizdoom-turbo",
                    "message": _invalid_behavior_message(
                        "env-vizdoom-turbo", behavior, second_value
                    ),
                }
            )
        elif _common_value(behavior, first_value) != _common_value(behavior, second_value):
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
    value = contract["tensor_device"]
    declared = value.get("declared_device") if isinstance(value, dict) else None
    expected_top = {
        "declared_device",
        "reset_mask_input",
        "step_action_input",
        "reset_outputs",
        "step_outputs",
    }
    structure = (
        isinstance(value, dict)
        and set(value) == expected_top
        and isinstance(declared, str)
        and bool(declared)
    )

    expected_device = declared
    if structure and declared == "cuda":
        input_devices = {
            item.get("device")
            for item in (value["reset_mask_input"], value["step_action_input"])
            if isinstance(item, dict)
        }
        if len(input_devices) == 1:
            concrete_device = next(iter(input_devices))
            if isinstance(concrete_device, str) and re.fullmatch(
                r"cuda:(0|[1-9]\d*)", concrete_device
            ):
                expected_device = concrete_device

    def torch_descriptor(item: object, *, shape: list[int], dtype: str) -> bool:
        return (
            _descriptor(item, shape=shape, dtype=dtype)
            and item["transport"] == "torch"
            and item["device"] == expected_device
        )  # type: ignore[index]

    inputs = (
        structure
        and torch_descriptor(value["reset_mask_input"], shape=[2], dtype="bool")
        and torch_descriptor(value["step_action_input"], shape=[2], dtype="int64")
    )
    reset = value.get("reset_outputs") if structure else None
    step = value.get("step_outputs") if structure else None
    outputs = (
        isinstance(reset, dict)
        and set(reset) == {"observation", "signals"}
        and torch_descriptor(reset["observation"], shape=[2, 4, 84, 84], dtype="uint8")
        and isinstance(reset["signals"], dict)
        and set(reset["signals"]) == set(_SIGNALS)
        and all(
            torch_descriptor(reset["signals"][name], shape=[2], dtype="float64")
            for name in _SIGNALS
        )
        and isinstance(step, dict)
        and set(step) == {"observation", "reward", "terminated", "truncated", "signals"}
        and torch_descriptor(step["observation"], shape=[2, 4, 84, 84], dtype="uint8")
        and torch_descriptor(step["reward"], shape=[2], dtype="float32")
        and torch_descriptor(step["terminated"], shape=[2], dtype="bool")
        and torch_descriptor(step["truncated"], shape=[2], dtype="bool")
        and isinstance(step["signals"], dict)
        and set(step["signals"]) == set(_SIGNALS)
        and all(
            torch_descriptor(step["signals"][name], shape=[2], dtype="float64") for name in _SIGNALS
        )
    )
    predicates = {
        "gradoom.tensor_inputs": bool(inputs),
        "gradoom.tensor_outputs": bool(outputs),
        "gradoom.device": bool(inputs and outputs),
    }
    return [
        {"behavior": behavior, "status": "passed"}
        if passed
        else {
            "behavior": behavior,
            "status": "failed",
            "provider": "gradoom",
            "message": f"gradoom public behavior is invalid for {behavior}.",
        }
        for behavior, passed in predicates.items()
    ]


def run_invariant_suite(
    declaration: object,
    *,
    base_directory: Path,
    declared_inputs: list[dict[str, Any]],
    fixture: bool,
    gradoom_revision: str,
    wad_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if declaration is None:
        return _unconfigured_suite()
    if not isinstance(declaration, dict):
        raise InvariantSuiteError("invariant_suite must be an object")
    if "providers" in declaration:
        raise InvariantSuiteError(
            "invariant_suite.providers is not supported; execution is owned by the "
            "authenticated invariant runner"
        )
    allowed = {
        "version",
        "mode",
        "runner_input",
        "fixture_case",
        "real_configuration",
        "diagnostics",
    }
    extra = sorted(set(declaration) - allowed)
    if extra:
        raise InvariantSuiteError(f"invariant_suite has undeclared fields: {extra!r}")
    if declaration.get("version") != INVARIANT_SUITE_VERSION:
        raise InvariantSuiteError(
            f"unsupported invariant_suite.version {declaration.get('version')!r}; "
            f"supported version is {INVARIANT_SUITE_VERSION!r}"
        )
    mode = declaration.get("mode")
    if mode not in {"fixture", "real"}:
        raise InvariantSuiteError("invariant_suite.mode must be 'fixture' or 'real'")
    if (mode == "fixture") is not fixture:
        raise InvariantSuiteError("invariant_suite.mode must match the manifest fixture state")
    if mode == "real" and wad_profile is None:
        raise InvariantSuiteError("real invariant execution requires a matched wad_profile")
    runner_input = _required_string(declaration.get("runner_input"), "invariant_suite.runner_input")
    _runner_path, runner_sha256 = _declared_runner(
        runner_input,
        declared_inputs=declared_inputs,
        base_directory=base_directory,
    )
    fixture_case = declaration.get("fixture_case", "pass")
    if mode == "fixture" and fixture_case not in {
        "pass",
        "reward_mismatch",
        "missing_player_killcount",
        "missing_termination",
        "ignored_masked_reset",
        "leaky_masked_reset",
        "forged_masked_reset",
        "broken_terminal_reset",
        "counter_only_kills",
    }:
        raise InvariantSuiteError("invariant_suite.fixture_case is invalid")
    if mode == "real" and "fixture_case" in declaration:
        raise InvariantSuiteError("invariant_suite.fixture_case is fixture-only")
    diagnostics = _diagnostics(declaration.get("diagnostics"))
    if mode == "real" and wad_profile is not None and wad_profile.get("status") != "matched":
        return {
            "version": INVARIANT_SUITE_VERSION,
            "configured": True,
            "status": "unavailable",
            "checks": [],
            "failures": [],
            "unavailable_reasons": [
                {
                    "code": "wad_profile_not_matched",
                    "message": "Real invariant execution requires the validated WAD profile.",
                }
            ],
            "providers": [],
            "diagnostics": diagnostics,
        }
    real_configuration = declaration.get("real_configuration")
    if mode == "real":
        assert wad_profile is not None
        real_configuration = _prepare_real_configuration(
            real_configuration,
            declared_inputs=declared_inputs,
            base_directory=base_directory,
            wad_profile=wad_profile,
        )
    elif "real_configuration" in declaration:
        raise InvariantSuiteError("invariant_suite.real_configuration is real-only")
    response = _execute_runner(
        runner_sha256=runner_sha256,
        mode=mode,
        fixture_case=str(fixture_case),
        gradoom_revision=gradoom_revision,
        real_configuration=real_configuration,
    )
    if response["status"] == "unavailable":
        reasons = response["unavailable_reasons"]
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("code"), str)
            and isinstance(item.get("message"), str)
            for item in reasons
        ):
            raise InvariantSuiteError(
                "authenticated invariant runner unavailable reasons are invalid"
            )
        contract_failures = [
            {
                "behavior": item["behavior"],
                "provider": item["provider"],
                "message": item["message"],
            }
            for item in reasons
            if item.get("code") == "provider_contract_failure"
            and isinstance(item.get("behavior"), str)
            and isinstance(item.get("provider"), str)
        ]
        if contract_failures:
            return {
                "version": INVARIANT_SUITE_VERSION,
                "configured": True,
                "status": "failed",
                "checks": [],
                "failures": contract_failures,
                "unavailable_reasons": [],
                "providers": [],
                "diagnostics": diagnostics,
            }
        return {
            "version": INVARIANT_SUITE_VERSION,
            "configured": True,
            "status": "unavailable",
            "checks": [],
            "failures": [],
            "unavailable_reasons": reasons,
            "providers": [],
            "diagnostics": diagnostics,
        }
    contracts = _parse_contracts(response["contracts"])
    revision_failures = []
    expected_revisions = {"gradoom": gradoom_revision, "env-vizdoom-turbo": REFERENCE_REVISION}
    if mode == "real":
        for provider, expected in expected_revisions.items():
            if contracts[provider]["revision"] != expected:
                revision_failures.append(
                    {
                        "behavior": f"{provider}.revision",
                        "provider": provider,
                        "message": (
                            f"{provider} revision mismatch: expected {expected}, "
                            f"found {contracts[provider]['revision']}."
                        ),
                    }
                )
    provider_reports = [
        {
            "id": provider,
            "status": "executed",
            "runner_input": runner_input,
            "runner_sha256": runner_sha256,
            "mode": mode,
            "revision": contracts[provider]["revision"],
            "contract_sha256": _canonical_sha256(contracts[provider]),
        }
        for provider in _PROVIDER_IDS
    ]
    if revision_failures:
        return {
            "version": INVARIANT_SUITE_VERSION,
            "configured": True,
            "status": "failed",
            "checks": [],
            "failures": revision_failures,
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


__all__ = ["INVARIANT_SUITE_VERSION", "InvariantSuiteError", "run_invariant_suite"]
