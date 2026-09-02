from __future__ import annotations

import contextlib
import ctypes
import fcntl
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .checkpoint_policy import CHECKPOINT_FORMAT
from .policy_execution import POLICY_PREPROCESSING_SHA256, policy_execution_identity
from .report import (
    EvidenceError,
    _canonical_sha256,
    _load_manifest,
    _parse_json_document,
    _paths_alias,
    _required_string,
    _resolve_evidence_path,
    _sha256_bytes,
    _validate_code_provenance,
    _validate_declared_inputs,
    _validate_schema_version,
    _validate_sha256,
    _validate_string_content,
)

POLICY_RUNNER_PROTOCOL_VERSION = 2
PROVIDER_IDS = ("gradoom", "env-vizdoom-turbo")
_TERMINATION_STATES = frozenset({"terminated", "truncated"})
_OUTCOME_FIELDS = frozenset(
    {
        "seed",
        "player_killcount",
        "termination_state",
        "episode_length",
        "execution_failure",
    }
)
_RETAINED_OUTCOME_FIELDS = frozenset(
    {"unit_identity", "provider_id", "policy_id", "seed_index", *_OUTCOME_FIELDS}
)
_MODEL_RUNTIME_CONTRACT_FIELDS = frozenset(
    {"contract_version", "artifact_format", "runtime", "architecture"}
)
_POLICY_FIELDS = frozenset(
    {
        "id",
        "training_provider",
        "artifact_path",
        "artifact_sha256",
        "model_runtime_contract",
        "stochastic_actions",
        "adapted",
        "provider_specific_modifications",
    }
)
_CORPUS_FIELDS = frozenset(
    {
        "schema_version",
        "corpus_version",
        "sealed",
        "shared_preprocessing_identity",
        "policies",
    }
)
_SEED_MANIFEST_FIELDS = frozenset({"schema_version", "seed_set_id", "seeds"})
_PROVIDER_FIELDS = frozenset({"id", "revision"})
_POLICY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "workflow",
        "evidence_level",
        "fixture",
        "code_provenance",
        "declared_inputs",
        "policy_evaluation",
    }
)
_CODE_PROVENANCE_FIELDS = frozenset({"repository", "revision", "dirty"})
_DECLARED_INPUT_FIELDS = frozenset({"name", "path", "sha256"})
_POLICY_EVALUATION_FIELDS = frozenset(
    {"protocol_version", "corpus_input", "seed_manifest_input", "runner_input", "providers"}
)
_POLICY_EVALUATION_OPTIONAL_FIELDS = frozenset({"timeout_seconds", "fixture_failure_seed"})
_SUPPORTED_POLICY_ARCHITECTURES = frozenset(
    {
        "nature",
        "nature-pyramid",
        "nature-waist",
        "nature-flat",
        "nature-thin",
        "nature-half",
        "nature-quarter",
    }
)
_POLICY_REPORT_FIELDS = frozenset(
    {
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
        "policy_evaluation",
        "evidence_index",
    }
)
_EVIDENCE_INDEX_FIELDS = frozenset({"algorithm", "entries", "sha256"})
_RUNNER_CAPTURE_LIMIT = 8 * 1024 * 1024
_FAILURE_MESSAGE_LIMIT = 4096
_FAILURE_CODE_LIMIT = 128
_FIELD_DIAGNOSTIC_LIMIT = 128
_FIELD_DIAGNOSTIC_COUNT = 8
_MAX_OUTCOME_NUMBER = 2**63 - 1
_MAX_TIMEOUT_SECONDS = 2**31 - 1
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008


def _sealed_memfd(payload: bytes, *, name: str, stack: contextlib.ExitStack) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    memfd_create = getattr(libc, "memfd_create", None)
    if memfd_create is None:
        raise EvidenceError("sealed policy execution requires Linux memfd sealing support")
    memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    memfd_create.restype = ctypes.c_int
    descriptor = memfd_create(name.encode(), _MFD_ALLOW_SEALING)
    if descriptor < 0:
        error_number = ctypes.get_errno()
        raise EvidenceError(
            "sealed policy execution could not create an anonymous file: "
            f"{os.strerror(error_number)}"
        )
    stack.callback(os.close, descriptor)
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]
    os.lseek(descriptor, 0, os.SEEK_SET)
    seals = _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
    try:
        fcntl.fcntl(descriptor, _F_ADD_SEALS, seals)
    except OSError as error:
        raise EvidenceError("sealed policy execution could not seal an anonymous file") from error
    return descriptor


def _limit_runner_output_files() -> None:
    _soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
    limit = (
        _RUNNER_CAPTURE_LIMIT
        if hard_limit == resource.RLIM_INFINITY
        else min(_RUNNER_CAPTURE_LIMIT, hard_limit)
    )
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))


def _bounded_text(value: str, *, limit: int, suffix: str) -> str:
    payload = value.encode(errors="backslashreplace").replace(b"\0", b"\\x00")
    if len(payload) <= limit:
        return payload.decode()
    suffix_payload = suffix.encode()
    prefix = payload[: limit - len(suffix_payload)].decode(errors="ignore")
    return prefix + suffix


def _bounded_failure_message(payload: bytes) -> str:
    message = payload[: _FAILURE_MESSAGE_LIMIT + 1].decode(errors="replace").strip()
    return _bounded_text(
        message,
        limit=_FAILURE_MESSAGE_LIMIT,
        suffix="\n[runner stderr truncated]",
    )


def _validate_sources_unchanged(
    *,
    declared_inputs: list[dict[str, Any]],
    verified: dict[str, tuple[Path, bytes]],
    policies: list[dict[str, Any]],
) -> None:
    for declared_input in declared_inputs:
        try:
            current_payload = verified[declared_input["name"]][0].read_bytes()
        except OSError as error:
            raise EvidenceError(
                f"declared input {declared_input['name']!r} changed during policy evaluation: "
                f"{error}"
            ) from error
        if _sha256_bytes(current_payload) != declared_input["sha256"]:
            raise EvidenceError(
                f"declared input {declared_input['name']!r} changed during policy evaluation"
            )
    for policy in policies:
        _payload, current_sha256 = _artifact_payload(
            Path(policy["resolved_artifact_path"]), policy_id=policy["id"]
        )
        if current_sha256 != policy["artifact_sha256"]:
            raise EvidenceError(
                f"policy {policy['id']!r} artifact changed after the corpus was sealed"
            )


def _read_verified_input(
    declared_input: dict[str, Any], *, base_directory: Path
) -> tuple[Path, bytes]:
    path = _resolve_evidence_path(Path(declared_input["path"]), base_directory=base_directory)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceError(
            f"declared input {declared_input['name']!r} is unavailable: {error}"
        ) from error
    actual = _sha256_bytes(payload)
    if actual != declared_input["sha256"]:
        raise EvidenceError(
            f"declared input {declared_input['name']!r} SHA-256 mismatch: "
            f"expected {declared_input['sha256']}, got {actual}"
        )
    return path, payload


def _declared_input(
    name: object, *, inputs_by_name: dict[str, dict[str, Any]], field: str
) -> dict[str, Any]:
    input_name = _required_string(name, field)
    try:
        return inputs_by_name[input_name]
    except KeyError as error:
        raise EvidenceError(f"{field} names undeclared input {input_name!r}") from error


def _artifact_payload(path: Path, *, policy_id: str) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise EvidenceError(f"policy {policy_id!r} artifact is unavailable: {error}") from error
    return payload, _sha256_bytes(payload)


def _validate_contract(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} must be an object")
    _exact_fields(value, _MODEL_RUNTIME_CONTRACT_FIELDS, document=field)
    if (
        type(value.get("contract_version")) is not int
        or value.get("contract_version") != 1
        or value.get("artifact_format") != CHECKPOINT_FORMAT
        or value.get("runtime") != "torch"
        or type(value.get("architecture")) is not str
        or value.get("architecture") not in _SUPPORTED_POLICY_ARCHITECTURES
    ):
        raise EvidenceError(f"{field} uses an unsupported model/runtime contract")
    return {name: value[name] for name in _MODEL_RUNTIME_CONTRACT_FIELDS}


def _load_corpus(
    path: Path, payload: bytes
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
    corpus = _parse_json_document(payload, document="policy corpus manifest")
    if not isinstance(corpus, dict):
        raise EvidenceError("policy corpus manifest must be a JSON object")
    _exact_fields(corpus, _CORPUS_FIELDS, document="policy corpus manifest")
    _validate_string_content(corpus, document="policy corpus manifest")
    _validate_schema_version(corpus.get("schema_version"), document="policy corpus manifest")
    _required_string(corpus.get("corpus_version"), "policy corpus manifest corpus_version")
    if corpus.get("sealed") is not True:
        raise EvidenceError("policy corpus manifest must be sealed before evaluation")
    preprocessing = _validate_sha256(
        corpus.get("shared_preprocessing_identity"),
        "policy corpus manifest shared_preprocessing_identity",
    )
    if preprocessing != POLICY_PREPROCESSING_SHA256:
        raise EvidenceError(
            "policy corpus manifest uses an unsupported shared preprocessing identity"
        )
    raw_policies = corpus.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise EvidenceError("policy corpus manifest policies must be a non-empty array")

    policies: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    artifact_paths: list[Path] = []
    artifact_sha256s: set[str] = set()
    origins: set[str] = set()
    artifact_payloads: dict[str, bytes] = {}
    for index, raw_policy in enumerate(raw_policies):
        field = f"policy corpus manifest policies[{index}]"
        if not isinstance(raw_policy, dict):
            raise EvidenceError(f"{field} must be an object")
        _exact_fields(raw_policy, _POLICY_FIELDS, document=field)
        policy_id = _required_string(raw_policy.get("id"), f"{field}.id")
        if policy_id in identifiers:
            raise EvidenceError(f"{field}.id {policy_id!r} is duplicated")
        identifiers.add(policy_id)
        training_provider = raw_policy.get("training_provider")
        if training_provider not in PROVIDER_IDS:
            raise EvidenceError(f"{field}.training_provider must name an approved provider")
        origins.add(training_provider)
        artifact_path_string = _required_string(
            raw_policy.get("artifact_path"), f"{field}.artifact_path"
        )
        if not artifact_path_string.strip():
            raise EvidenceError(f"{field}.artifact_path must be a non-whitespace path")
        artifact_path = _resolve_evidence_path(
            Path(artifact_path_string), base_directory=path.parent
        )
        if any(_paths_alias(artifact_path, previous) for previous in artifact_paths):
            raise EvidenceError(f"{field}.artifact_path is duplicated")
        artifact_paths.append(artifact_path)
        expected_sha256 = _validate_sha256(
            raw_policy.get("artifact_sha256"), f"{field}.artifact_sha256"
        )
        artifact_payload, actual_sha256 = _artifact_payload(artifact_path, policy_id=policy_id)
        if actual_sha256 != expected_sha256:
            raise EvidenceError(
                f"policy {policy_id!r} artifact SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        if actual_sha256 in artifact_sha256s:
            raise EvidenceError(f"{field}.artifact_sha256 is duplicated")
        artifact_sha256s.add(actual_sha256)
        artifact_payloads[policy_id] = artifact_payload
        contract = _validate_contract(
            raw_policy.get("model_runtime_contract"),
            field=f"{field}.model_runtime_contract",
        )
        if raw_policy.get("stochastic_actions") is not True:
            raise EvidenceError(f"policy {policy_id!r} must require stochastic actions")
        if raw_policy.get("adapted") is not False:
            raise EvidenceError(
                f"policy {policy_id!r} is adapted or does not declare adapted=false"
            )
        if raw_policy.get("provider_specific_modifications") != []:
            raise EvidenceError(
                f"policy {policy_id!r} must not declare provider-specific modifications"
            )
        policies.append(
            {
                "id": policy_id,
                "training_provider": training_provider,
                "artifact_path": artifact_path_string,
                "resolved_artifact_path": str(artifact_path),
                "artifact_sha256": expected_sha256,
                "model_runtime_contract": contract,
                "stochastic_actions": True,
                "adapted": False,
                "provider_specific_modifications": [],
                "execution_identity": policy_execution_identity(
                    artifact_sha256=expected_sha256,
                    model_runtime_contract=contract,
                    stochastic_actions=True,
                ),
            }
        )
    if origins != set(PROVIDER_IDS):
        raise EvidenceError(
            "policy corpus manifest must contain a policy from each training provider"
        )
    normalized = {
        "schema_version": 1,
        "corpus_version": corpus["corpus_version"],
        "sealed": True,
        "shared_preprocessing_identity": preprocessing,
        "policies": policies,
    }
    return normalized, policies, artifact_payloads


def _load_seeds(payload: bytes) -> dict[str, Any]:
    document = _parse_json_document(payload, document="episode seed manifest")
    if not isinstance(document, dict):
        raise EvidenceError("episode seed manifest must be a JSON object")
    _exact_fields(document, _SEED_MANIFEST_FIELDS, document="episode seed manifest")
    _validate_string_content(document, document="episode seed manifest")
    _validate_schema_version(document.get("schema_version"), document="episode seed manifest")
    seed_set_id = _required_string(document.get("seed_set_id"), "episode seed manifest seed_set_id")
    seeds = document.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 256:
        raise EvidenceError("episode seed manifest must declare exactly 256 seeds")
    if any(type(seed) is not int or seed < 0 or seed > 2**63 - 1 for seed in seeds):
        raise EvidenceError("episode seed manifest seeds must be non-negative 64-bit integers")
    if len(set(seeds)) != len(seeds):
        raise EvidenceError("episode seed manifest seeds must be unique")
    return {"schema_version": 1, "seed_set_id": seed_set_id, "seeds": seeds}


def _providers(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(PROVIDER_IDS):
        raise EvidenceError("policy_evaluation.providers must bind each approved provider once")
    by_id: dict[str, dict[str, str]] = {}
    for index, provider in enumerate(value):
        field = f"policy_evaluation.providers[{index}]"
        if not isinstance(provider, dict) or provider.get("id") not in PROVIDER_IDS:
            raise EvidenceError(f"{field}.id must name an approved provider")
        _exact_fields(provider, _PROVIDER_FIELDS, document=field)
        provider_id = provider["id"]
        if provider_id in by_id:
            raise EvidenceError(f"{field}.id {provider_id!r} is duplicated")
        by_id[provider_id] = {
            "id": provider_id,
            "revision": _required_string(provider.get("revision"), f"{field}.revision"),
        }
    return [by_id[provider_id] for provider_id in PROVIDER_IDS]


def _unit_identity(
    evaluation_identity: str,
    *,
    provider_id: str,
    policy_id: str,
    seed_index: int,
    seed: int,
) -> str:
    return _canonical_sha256(
        {
            "evaluation_identity": evaluation_identity,
            "provider_id": provider_id,
            "policy_id": policy_id,
            "seed_index": seed_index,
            "seed": seed,
        },
        document="policy evaluation unit",
    )


def _failure_outcome(seed: int, *, code: str, message: str) -> dict[str, Any]:
    bounded_message = _bounded_text(
        message,
        limit=_FAILURE_MESSAGE_LIMIT,
        suffix="\n[failure message truncated]",
    )
    return {
        "seed": seed,
        "player_killcount": None,
        "termination_state": None,
        "episode_length": None,
        "execution_failure": {"code": code, "message": bounded_message},
    }


def _same_json_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return set(actual) == set(expected) and all(
            _same_json_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return len(actual) == len(expected) and all(
            _same_json_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _field_diagnostic(fields: list[str]) -> str:
    displayed = fields[:_FIELD_DIAGNOSTIC_COUNT]
    rendered = ", ".join(
        repr(
            field
            if len(field) <= _FIELD_DIAGNOSTIC_LIMIT
            else field[:_FIELD_DIAGNOSTIC_LIMIT] + "…"
        )
        for field in displayed
    )
    if len(fields) > len(displayed):
        rendered += f", and {len(fields) - len(displayed)} more"
    return rendered


def _exact_fields(
    value: dict[str, Any],
    expected: frozenset[str],
    *,
    document: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected - optional)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {_field_diagnostic(missing)}")
        if extra:
            details.append(f"undeclared {_field_diagnostic(extra)}")
        raise EvidenceError(f"{document} has invalid fields: {'; '.join(details)}")


def _canonical_outcome(
    value: object,
    *,
    expected_seed: int,
    document: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{document} must be an object")
    _exact_fields(value, _OUTCOME_FIELDS, document=document)
    seed = value["seed"]
    if type(seed) is not int or seed != expected_seed:
        raise EvidenceError(f"{document}.seed does not match the requested seed")
    failure = value["execution_failure"]
    if failure is not None:
        if not isinstance(failure, dict):
            raise EvidenceError(f"{document}.execution_failure must be an object or null")
        _exact_fields(failure, frozenset({"code", "message"}), document=f"{document}.failure")
        code = _required_string(failure["code"], f"{document}.failure.code")
        message = _required_string(failure["message"], f"{document}.failure.message")
        if not code.strip() or not message.strip():
            raise EvidenceError(f"{document}.execution_failure fields must be non-whitespace")
        if len(code.encode()) > _FAILURE_CODE_LIMIT:
            raise EvidenceError(f"{document}.failure.code exceeds the supported length")
        if any(
            value[field] is not None
            for field in ("player_killcount", "termination_state", "episode_length")
        ):
            raise EvidenceError(f"{document} failure metrics must be null")
        return _failure_outcome(seed, code=code, message=message)

    killcount = value["player_killcount"]
    length = value["episode_length"]
    termination = value["termination_state"]
    if (
        type(killcount) not in {int, float}
        or not math.isfinite(killcount)
        or killcount < 0
        or killcount > _MAX_OUTCOME_NUMBER
    ):
        raise EvidenceError(
            f"{document}.player_killcount must be finite, non-negative, and within "
            "the supported 64-bit range"
        )
    if type(length) is not int or length < 0:
        raise EvidenceError(f"{document}.episode_length must be a non-negative integer")
    if not isinstance(termination, str) or termination not in _TERMINATION_STATES:
        raise EvidenceError(f"{document}.termination_state must be 'terminated' or 'truncated'")
    return {
        "seed": seed,
        "player_killcount": killcount,
        "termination_state": termination,
        "episode_length": length,
        "execution_failure": None,
    }


def _normalize_runner_outcomes(
    payload: bytes, *, requested_seeds: list[int]
) -> dict[int, dict[str, Any]]:
    try:
        response = _parse_json_document(payload, document="policy runner response")
        if not isinstance(response, dict):
            raise EvidenceError("policy runner response must be an object")
        _validate_string_content(response, document="policy runner response")
        _exact_fields(
            response,
            frozenset({"protocol_version", "outcomes"}),
            document="policy runner response",
        )
        if (
            type(response.get("protocol_version")) is not int
            or response.get("protocol_version") != POLICY_RUNNER_PROTOCOL_VERSION
        ):
            raise EvidenceError("policy runner response uses an unsupported protocol")
        raw_outcomes = response.get("outcomes")
        if not isinstance(raw_outcomes, list):
            raise EvidenceError("policy runner response outcomes must be an array")
        requested = set(requested_seeds)
        normalized: dict[int, dict[str, Any]] = {}
        for index, raw in enumerate(raw_outcomes):
            if not isinstance(raw, dict):
                raise EvidenceError(f"policy runner response outcomes[{index}] must be an object")
            seed = raw.get("seed")
            if type(seed) is not int or seed not in requested or seed in normalized:
                raise EvidenceError(
                    "policy runner response contains an unexpected or duplicate seed"
                )
            normalized[seed] = _canonical_outcome(
                raw,
                expected_seed=seed,
                document=f"policy runner response outcomes[{index}]",
            )
        for seed in requested_seeds:
            normalized.setdefault(
                seed,
                _failure_outcome(
                    seed,
                    code="missing_runner_outcome",
                    message="The policy runner omitted this required seed outcome.",
                ),
            )
        return normalized
    except EvidenceError as error:
        message = _bounded_text(
            str(error),
            limit=_FAILURE_MESSAGE_LIMIT,
            suffix="\n[failure message truncated]",
        )
        return {
            seed: _failure_outcome(
                seed,
                code="invalid_runner_response",
                message=message,
            )
            for seed in requested_seeds
        }


def _execute_batch(
    runner_descriptor: int,
    artifact_descriptor: int,
    *,
    provider: dict[str, str],
    policy: dict[str, Any],
    seeds: list[int],
    fixture_failure_seed: int | None,
    timeout_seconds: float,
) -> dict[int, dict[str, Any]]:
    request: dict[str, Any] = {
        "protocol_version": POLICY_RUNNER_PROTOCOL_VERSION,
        "provider_id": provider["id"],
        "provider_revision": provider["revision"],
        "policy": {
            "id": policy["id"],
            "training_provider": policy["training_provider"],
            "resolved_artifact_path": f"/proc/self/fd/{artifact_descriptor}",
            "artifact_sha256": policy["artifact_sha256"],
            "model_runtime_contract": policy["model_runtime_contract"],
            "stochastic_actions": policy["stochastic_actions"],
            "execution_identity": policy["execution_identity"],
        },
        "seeds": seeds,
    }
    if fixture_failure_seed is not None:
        request["fixture_failure_seed"] = fixture_failure_seed
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = subprocess.run(
                [sys.executable, f"/proc/self/fd/{runner_descriptor}"],
                input=json.dumps(request, allow_nan=False).encode(),
                check=False,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_seconds,
                pass_fds=(runner_descriptor, artifact_descriptor),
                preexec_fn=_limit_runner_output_files,
            )
        except (OSError, subprocess.SubprocessError) as error:
            code = (
                "runner_timeout"
                if isinstance(error, subprocess.TimeoutExpired)
                else "runner_failure"
            )
            return {seed: _failure_outcome(seed, code=code, message=str(error)) for seed in seeds}
        stdout_size = stdout.tell()
        stderr_size = stderr.tell()
        stdout.seek(0)
        stderr.seek(0)
        stdout_payload = stdout.read(_RUNNER_CAPTURE_LIMIT)
        stderr_payload = stderr.read(_RUNNER_CAPTURE_LIMIT)
    if stdout_size >= _RUNNER_CAPTURE_LIMIT or stderr_size >= _RUNNER_CAPTURE_LIMIT:
        return {
            seed: _failure_outcome(
                seed,
                code="runner_output_limit",
                message="Policy runner output reached the bounded capture limit.",
            )
            for seed in seeds
        }
    if result.returncode != 0:
        message = _bounded_failure_message(stderr_payload)
        return {
            seed: _failure_outcome(
                seed,
                code="runner_process_failure",
                message=message or f"policy runner exited with status {result.returncode}",
            )
            for seed in seeds
        }
    return _normalize_runner_outcomes(stdout_payload, requested_seeds=seeds)


def _load_reusable_outcomes(
    merge_path: Path | None,
    *,
    evaluation_identity: str,
    expected_units: dict[str, tuple[str, str, int, int]],
    expected_unit_order: list[str],
    expected_binding: dict[str, Any],
    expected_report_binding: dict[str, Any],
    expected_evidence_entries: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    if merge_path is None:
        return {}
    report = _parse_json_document(merge_path.read_bytes(), document="merge report")
    if not isinstance(report, dict):
        raise EvidenceError("merge report must be an object")
    _validate_string_content(report, document="merge report")
    _exact_fields(report, _POLICY_REPORT_FIELDS, document="merge report")
    report_identity = _validate_sha256(report.get("run_identity"), "merge report run_identity")
    if report_identity != evaluation_identity:
        raise EvidenceError("cannot merge unlike policy evaluation identities")
    for field, expected in expected_report_binding.items():
        if not _same_json_value(report.get(field), expected):
            raise EvidenceError(f"merge report {field} does not match current evidence")
    status = report.get("status")
    if type(status) is not str or status not in {
        "evaluation_in_progress",
        "evaluation_complete",
    }:
        raise EvidenceError("merge report has an invalid policy evaluation status")
    evaluation = report.get("policy_evaluation")
    if not isinstance(evaluation, dict):
        raise EvidenceError("merge report policy_evaluation must be an object")
    retained_evaluation_identity = _validate_sha256(
        evaluation.get("evaluation_identity"),
        "merge report policy_evaluation.evaluation_identity",
    )
    if retained_evaluation_identity != evaluation_identity:
        raise EvidenceError("cannot merge unlike policy evaluation identities")
    required_evaluation_fields = frozenset(
        {
            "protocol_version",
            "evaluation_identity",
            "corpus_manifest_sha256",
            "corpus",
            "seed_manifest_sha256",
            "seed_manifest",
            "providers",
            "outcomes",
            "expected_outcome_count",
            "failure_count",
        }
    )
    _exact_fields(evaluation, required_evaluation_fields, document="merge policy_evaluation")
    for field, expected in expected_binding.items():
        if not _same_json_value(evaluation.get(field), expected):
            raise EvidenceError(
                f"merge report policy_evaluation.{field} does not match current evidence"
            )
    if not isinstance(evaluation.get("corpus"), dict):
        raise EvidenceError("merge report policy_evaluation.corpus must be an object")
    evidence_index = report.get("evidence_index")
    if not isinstance(evidence_index, dict):
        raise EvidenceError("merge report evidence_index is invalid")
    _exact_fields(evidence_index, _EVIDENCE_INDEX_FIELDS, document="merge evidence_index")
    entries = evidence_index.get("entries")
    if not isinstance(entries, list):
        raise EvidenceError("merge report evidence_index is invalid")
    if evidence_index.get("algorithm") != "sha256":
        raise EvidenceError("merge report evidence_index algorithm is invalid")
    stored_index = evidence_index.get("sha256")
    if stored_index != _canonical_sha256(entries, document="merge report"):
        raise EvidenceError("merge report evidence_index SHA-256 mismatch")
    evaluation_digest = _canonical_sha256(evaluation, document="merge report policy evaluation")
    expected_entries = [
        *expected_evidence_entries,
        {"name": "policy_evaluation", "sha256": evaluation_digest},
    ]
    if not _same_json_value(entries, expected_entries):
        raise EvidenceError("merge report policy_evaluation SHA-256 mismatch")
    outcomes = evaluation.get("outcomes")
    if not isinstance(outcomes, list):
        raise EvidenceError("merge report policy_evaluation.outcomes must be an array")
    reusable: dict[str, dict[str, Any]] = {}
    observed_order: list[str] = []
    for index, item in enumerate(outcomes):
        if not isinstance(item, dict):
            raise EvidenceError("merge report contains an invalid policy outcome")
        try:
            _exact_fields(
                item,
                _RETAINED_OUTCOME_FIELDS,
                document=f"merge policy outcome[{index}]",
            )
        except EvidenceError as error:
            raise EvidenceError("merge report contains an invalid policy outcome") from error
        identity = _validate_sha256(
            item.get("unit_identity"),
            f"merge policy outcome[{index}].unit_identity",
        )
        expected = expected_units.get(identity)
        actual = (
            item.get("provider_id"),
            item.get("policy_id"),
            item.get("seed_index"),
            item.get("seed"),
        )
        if (
            expected is None
            or not all(
                _same_json_value(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected, strict=True)
            )
            or identity in reusable
        ):
            raise EvidenceError("merge report contains mismatched or duplicate policy evidence")
        try:
            normalized = _canonical_outcome(
                {field: item[field] for field in _OUTCOME_FIELDS},
                expected_seed=expected[3],
                document=f"merge policy outcome[{index}]",
            )
        except EvidenceError as error:
            raise EvidenceError("merge report contains an invalid policy outcome") from error
        reusable[identity] = {
            "unit_identity": identity,
            "provider_id": expected[0],
            "policy_id": expected[1],
            "seed_index": expected[2],
            **normalized,
        }
        observed_order.append(identity)
    if observed_order != expected_unit_order[: len(observed_order)]:
        raise EvidenceError("completed policy outcomes must be an exact leading prefix")
    batch_size = len(expected_binding["seed_manifest"]["seeds"])
    if len(observed_order) % batch_size != 0:
        raise EvidenceError(
            "completed policy outcomes must contain complete provider-policy batches"
        )
    if status == "evaluation_complete" and len(observed_order) != len(expected_unit_order):
        raise EvidenceError("complete policy evaluation is missing required outcomes")
    if status == "evaluation_in_progress" and len(observed_order) >= len(expected_unit_order):
        raise EvidenceError("in-progress policy evaluation cannot contain the complete grid")
    failure_count = sum(item["execution_failure"] is not None for item in reusable.values())
    if not _same_json_value(evaluation.get("failure_count"), failure_count):
        raise EvidenceError("merge report policy_evaluation.failure_count is invalid")
    return reusable


def build_policy_evaluation_report(
    manifest_path: Path,
    *,
    merge_path: Path | None = None,
    output_path: Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    manifest, manifest_payload = _load_manifest(manifest_path)
    _exact_fields(manifest, _POLICY_MANIFEST_FIELDS, document="manifest")
    _validate_schema_version(manifest.get("schema_version"), document="manifest")
    if manifest.get("workflow") != "parity_certification":
        raise EvidenceError("this command path requires workflow parity_certification")
    if manifest.get("evidence_level") != "formal":
        raise EvidenceError("parity certification requires formal evidence")
    if type(manifest.get("fixture")) is not bool:
        raise EvidenceError("fixture is required and must be a boolean")
    if isinstance(manifest.get("code_provenance"), dict):
        _exact_fields(
            manifest["code_provenance"],
            _CODE_PROVENANCE_FIELDS,
            document="code_provenance",
        )
    code_provenance = _validate_code_provenance(manifest.get("code_provenance"))
    if isinstance(manifest.get("declared_inputs"), list):
        for index, declared_input in enumerate(manifest["declared_inputs"]):
            if isinstance(declared_input, dict):
                _exact_fields(
                    declared_input,
                    _DECLARED_INPUT_FIELDS,
                    document=f"declared_inputs[{index}]",
                )
    declared_inputs = _validate_declared_inputs(
        manifest.get("declared_inputs"), base_directory=manifest_path.parent
    )
    inputs_by_name = {item["name"]: item for item in declared_inputs}
    configuration = manifest.get("policy_evaluation")
    if not isinstance(configuration, dict):
        raise EvidenceError("policy_evaluation must be an object")
    _exact_fields(
        configuration,
        _POLICY_EVALUATION_FIELDS,
        optional=_POLICY_EVALUATION_OPTIONAL_FIELDS,
        document="policy_evaluation",
    )
    if (
        type(configuration.get("protocol_version")) is not int
        or configuration.get("protocol_version") != POLICY_RUNNER_PROTOCOL_VERSION
    ):
        raise EvidenceError("policy_evaluation uses an unsupported protocol_version")
    corpus_input = _declared_input(
        configuration.get("corpus_input"),
        inputs_by_name=inputs_by_name,
        field="policy_evaluation.corpus_input",
    )
    seeds_input = _declared_input(
        configuration.get("seed_manifest_input"),
        inputs_by_name=inputs_by_name,
        field="policy_evaluation.seed_manifest_input",
    )
    runner_input = _declared_input(
        configuration.get("runner_input"),
        inputs_by_name=inputs_by_name,
        field="policy_evaluation.runner_input",
    )
    verified: dict[str, tuple[Path, bytes]] = {
        item["name"]: _read_verified_input(item, base_directory=manifest_path.parent)
        for item in declared_inputs
    }
    corpus_path, corpus_payload = verified[corpus_input["name"]]
    corpus, policies, artifact_payloads = _load_corpus(corpus_path, corpus_payload)
    evidence_document_paths = [manifest_path.resolve(strict=False)] + [
        item[0] for item in verified.values()
    ]
    for policy in policies:
        artifact_path = Path(policy["resolved_artifact_path"])
        if any(_paths_alias(artifact_path, path) for path in evidence_document_paths):
            raise EvidenceError(
                f"policy {policy['id']!r} artifact aliases an evidence input document"
            )
        if output_path is not None:
            resolved_output = _resolve_evidence_path(output_path, base_directory=Path.cwd())
            if _paths_alias(resolved_output, artifact_path):
                raise EvidenceError(f"output path aliases policy artifact {policy['id']!r}")
    seeds = _load_seeds(verified[seeds_input["name"]][1])
    providers = _providers(configuration.get("providers"))
    timeout_seconds = configuration.get("timeout_seconds", 120)
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_TIMEOUT_SECONDS
    ):
        raise EvidenceError(
            "policy_evaluation.timeout_seconds must be a positive finite number within "
            "the supported range"
        )
    fixture_failure_seed = configuration.get("fixture_failure_seed")
    if fixture_failure_seed is not None:
        if not manifest["fixture"] or type(fixture_failure_seed) is not int:
            raise EvidenceError("policy_evaluation.fixture_failure_seed is fixture-only")
        if fixture_failure_seed not in seeds["seeds"]:
            raise EvidenceError("policy_evaluation.fixture_failure_seed must be a declared seed")

    evaluation_identity = _canonical_sha256(
        {
            "schema_version": 1,
            "workflow": "parity_certification",
            "evidence_level": "formal",
            "fixture": manifest["fixture"],
            "code_provenance": code_provenance,
            "protocol_version": POLICY_RUNNER_PROTOCOL_VERSION,
            "declared_inputs": sorted(
                ({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
                key=lambda item: item["name"],
            ),
            "corpus_manifest_sha256": corpus_input["sha256"],
            "seed_manifest_sha256": seeds_input["sha256"],
            "runner_sha256": runner_input["sha256"],
            "providers": providers,
            "timeout_seconds": timeout_seconds,
            "fixture_failure_seed": fixture_failure_seed,
        },
        document="policy evaluation manifest",
    )
    expected_units: dict[str, tuple[str, str, int, int]] = {}
    expected_unit_order: list[str] = []
    for provider in providers:
        for policy in policies:
            for seed_index, seed in enumerate(seeds["seeds"]):
                identity = _unit_identity(
                    evaluation_identity,
                    provider_id=provider["id"],
                    policy_id=policy["id"],
                    seed_index=seed_index,
                    seed=seed,
                )
                expected_units[identity] = (provider["id"], policy["id"], seed_index, seed)
                expected_unit_order.append(identity)
    expected_binding = {
        "protocol_version": POLICY_RUNNER_PROTOCOL_VERSION,
        "evaluation_identity": evaluation_identity,
        "corpus_manifest_sha256": corpus_input["sha256"],
        "corpus": corpus,
        "seed_manifest_sha256": seeds_input["sha256"],
        "seed_manifest": seeds,
        "providers": providers,
        "expected_outcome_count": len(expected_units),
    }
    reasons = [
        {
            "code": "parity_verdict_pending",
            "message": (
                "Policy outcomes have not yet been evaluated by the parity verdict workflow."
            ),
        }
    ]
    if manifest["fixture"]:
        reasons.insert(
            0,
            {
                "code": "fixture_evidence",
                "message": "Fixture evidence cannot support public claims.",
            },
        )
    expected_report_binding = {
        "schema_version": 1,
        "workflow": "parity_certification",
        "evidence_level": "formal",
        "fixture": manifest["fixture"],
        "claim_eligible": False,
        "claim_reasons": reasons,
        "run_identity": evaluation_identity,
        "code_provenance": code_provenance,
        "declared_inputs": declared_inputs,
    }
    expected_evidence_entries = [
        {"name": "manifest", "sha256": _sha256_bytes(manifest_payload)},
        *({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
        *(
            {
                "name": f"policy_artifact.{policy['id']}",
                "sha256": policy["artifact_sha256"],
            }
            for policy in policies
        ),
    ]
    reusable = _load_reusable_outcomes(
        merge_path,
        evaluation_identity=evaluation_identity,
        expected_units=expected_units,
        expected_unit_order=expected_unit_order,
        expected_binding=expected_binding,
        expected_report_binding=expected_report_binding,
        expected_evidence_entries=expected_evidence_entries,
    )

    def build_report(outcomes: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
        evaluation = {
            **expected_binding,
            "outcomes": outcomes,
            "failure_count": sum(item["execution_failure"] is not None for item in outcomes),
        }
        evidence_entries = [
            *expected_evidence_entries,
            {
                "name": "policy_evaluation",
                "sha256": _canonical_sha256(evaluation, document="policy evaluation report"),
            },
        ]
        return {
            **expected_report_binding,
            "status": "evaluation_complete" if complete else "evaluation_in_progress",
            "policy_evaluation": evaluation,
            "evidence_index": {
                "algorithm": "sha256",
                "entries": evidence_entries,
                "sha256": _canonical_sha256(evidence_entries, document="policy evaluation report"),
            },
        }

    outcomes: list[dict[str, Any]] = []
    with contextlib.ExitStack() as stack:
        runner_descriptor = _sealed_memfd(
            verified[runner_input["name"]][1],
            name="gradoom-policy-runner",
            stack=stack,
        )
        artifact_descriptors = {
            policy["id"]: _sealed_memfd(
                artifact_payloads[policy["id"]],
                name="gradoom-policy-artifact",
                stack=stack,
            )
            for policy in policies
        }
        for provider in providers:
            for policy in policies:
                pending_seeds = []
                for seed_index, seed in enumerate(seeds["seeds"]):
                    identity = _unit_identity(
                        evaluation_identity,
                        provider_id=provider["id"],
                        policy_id=policy["id"],
                        seed_index=seed_index,
                        seed=seed,
                    )
                    if identity not in reusable:
                        pending_seeds.append(seed)
                executed = (
                    _execute_batch(
                        runner_descriptor,
                        artifact_descriptors[policy["id"]],
                        provider=provider,
                        policy=policy,
                        seeds=pending_seeds,
                        fixture_failure_seed=fixture_failure_seed,
                        timeout_seconds=float(timeout_seconds),
                    )
                    if pending_seeds
                    else {}
                )
                if pending_seeds:
                    _validate_sources_unchanged(
                        declared_inputs=declared_inputs,
                        verified=verified,
                        policies=policies,
                    )
                for seed_index, seed in enumerate(seeds["seeds"]):
                    identity = _unit_identity(
                        evaluation_identity,
                        provider_id=provider["id"],
                        policy_id=policy["id"],
                        seed_index=seed_index,
                        seed=seed,
                    )
                    outcome = reusable[identity] if identity in reusable else executed[seed]
                    outcomes.append(
                        {
                            "unit_identity": identity,
                            "provider_id": provider["id"],
                            "policy_id": policy["id"],
                            "seed_index": seed_index,
                            **outcome,
                        }
                    )
                if pending_seeds and progress_callback is not None:
                    progress_callback(
                        build_report(
                            outcomes,
                            complete=len(outcomes) == len(expected_unit_order),
                        )
                    )
    _validate_sources_unchanged(
        declared_inputs=declared_inputs,
        verified=verified,
        policies=policies,
    )
    return build_report(outcomes, complete=True)


__all__ = ["POLICY_RUNNER_PROTOCOL_VERSION", "build_policy_evaluation_report"]
