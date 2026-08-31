from __future__ import annotations

import json
import math
import subprocess
import sys
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

POLICY_RUNNER_PROTOCOL_VERSION = 1
PROVIDER_IDS = ("gradoom", "env-vizdoom-turbo")


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


def _artifact_sha256(path: Path, *, policy_id: str) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as error:
        raise EvidenceError(f"policy {policy_id!r} artifact is unavailable: {error}") from error


def _validate_contract(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{field} must be an object")
    if (
        value.get("contract_version") != 1
        or value.get("artifact_format") != CHECKPOINT_FORMAT
        or value.get("runtime") != "torch"
    ):
        raise EvidenceError(f"{field} uses an unsupported model/runtime contract")
    _required_string(value.get("architecture"), f"{field}.architecture")
    return value


def _load_corpus(path: Path, payload: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    corpus = _parse_json_document(payload, document="policy corpus manifest")
    if not isinstance(corpus, dict):
        raise EvidenceError("policy corpus manifest must be a JSON object")
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
    origins: set[str] = set()
    for index, raw_policy in enumerate(raw_policies):
        field = f"policy corpus manifest policies[{index}]"
        if not isinstance(raw_policy, dict):
            raise EvidenceError(f"{field} must be an object")
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
        actual_sha256 = _artifact_sha256(artifact_path, policy_id=policy_id)
        if actual_sha256 != expected_sha256:
            raise EvidenceError(
                f"policy {policy_id!r} artifact SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
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
    return normalized, policies


def _load_seeds(payload: bytes) -> dict[str, Any]:
    document = _parse_json_document(payload, document="episode seed manifest")
    if not isinstance(document, dict):
        raise EvidenceError("episode seed manifest must be a JSON object")
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
    return {
        "seed": seed,
        "player_killcount": None,
        "termination_state": None,
        "episode_length": None,
        "execution_failure": {"code": code, "message": message},
    }


def _normalize_runner_outcomes(
    payload: bytes, *, requested_seeds: list[int]
) -> dict[int, dict[str, Any]]:
    try:
        response = _parse_json_document(payload, document="policy runner response")
        if not isinstance(response, dict) or response.get("protocol_version") != 1:
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
            failure = raw.get("execution_failure")
            if failure is not None:
                if not isinstance(failure, dict):
                    raise EvidenceError("policy runner execution_failure must be an object or null")
                code = _required_string(failure.get("code"), "policy runner failure code")
                message = _required_string(failure.get("message"), "policy runner failure message")
                normalized[seed] = _failure_outcome(seed, code=code, message=message)
                continue
            killcount = raw.get("player_killcount")
            length = raw.get("episode_length")
            termination = raw.get("termination_state")
            if (
                type(killcount) not in {int, float}
                or not math.isfinite(killcount)
                or killcount < 0
                or type(length) is not int
                or length < 0
                or not isinstance(termination, str)
                or not termination
            ):
                raise EvidenceError("policy runner response contains an invalid successful outcome")
            normalized[seed] = {
                "seed": seed,
                "player_killcount": killcount,
                "termination_state": termination,
                "episode_length": length,
                "execution_failure": None,
            }
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
        return {
            seed: _failure_outcome(
                seed,
                code="invalid_runner_response",
                message=str(error),
            )
            for seed in requested_seeds
        }


def _execute_batch(
    runner_path: Path,
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
            key: policy[key]
            for key in (
                "id",
                "training_provider",
                "resolved_artifact_path",
                "artifact_sha256",
                "model_runtime_contract",
                "stochastic_actions",
                "execution_identity",
            )
        },
        "seeds": seeds,
    }
    if fixture_failure_seed is not None:
        request["fixture_failure_seed"] = fixture_failure_seed
    try:
        result = subprocess.run(
            [sys.executable, str(runner_path)],
            input=json.dumps(request, allow_nan=False).encode(),
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        code = (
            "runner_timeout" if isinstance(error, subprocess.TimeoutExpired) else "runner_failure"
        )
        return {seed: _failure_outcome(seed, code=code, message=str(error)) for seed in seeds}
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip()
        return {
            seed: _failure_outcome(
                seed,
                code="runner_process_failure",
                message=message or f"policy runner exited with status {result.returncode}",
            )
            for seed in seeds
        }
    return _normalize_runner_outcomes(result.stdout, requested_seeds=seeds)


def _load_reusable_outcomes(
    merge_path: Path | None,
    *,
    evaluation_identity: str,
    expected_units: dict[str, tuple[str, str, int, int]],
) -> dict[str, dict[str, Any]]:
    if merge_path is None:
        return {}
    report = _parse_json_document(merge_path.read_bytes(), document="merge report")
    if not isinstance(report, dict) or report.get("workflow") != "parity_certification":
        raise EvidenceError("merge report workflow must be parity_certification")
    evaluation = report.get("policy_evaluation")
    if not isinstance(evaluation, dict):
        raise EvidenceError("merge report policy_evaluation must be an object")
    if evaluation.get("evaluation_identity") != evaluation_identity:
        raise EvidenceError("cannot merge unlike policy evaluation identities")
    evidence_index = report.get("evidence_index")
    if not isinstance(evidence_index, dict) or not isinstance(evidence_index.get("entries"), list):
        raise EvidenceError("merge report evidence_index is invalid")
    stored_index = evidence_index.get("sha256")
    if stored_index != _canonical_sha256(evidence_index["entries"], document="merge report"):
        raise EvidenceError("merge report evidence_index SHA-256 mismatch")
    evaluation_digest = _canonical_sha256(evaluation, document="merge report policy evaluation")
    indexed = [
        entry
        for entry in evidence_index["entries"]
        if isinstance(entry, dict) and entry.get("name") == "policy_evaluation"
    ]
    if len(indexed) != 1 or indexed[0].get("sha256") != evaluation_digest:
        raise EvidenceError("merge report policy_evaluation SHA-256 mismatch")
    outcomes = evaluation.get("outcomes")
    if not isinstance(outcomes, list):
        raise EvidenceError("merge report policy_evaluation.outcomes must be an array")
    reusable: dict[str, dict[str, Any]] = {}
    for item in outcomes:
        if not isinstance(item, dict):
            raise EvidenceError("merge report contains an invalid policy outcome")
        identity = item.get("unit_identity")
        expected = expected_units.get(identity)
        actual = (
            item.get("provider_id"),
            item.get("policy_id"),
            item.get("seed_index"),
            item.get("seed"),
        )
        if expected is None or actual != expected or identity in reusable:
            raise EvidenceError("merge report contains mismatched or duplicate policy evidence")
        reusable[identity] = item
    return reusable


def build_policy_evaluation_report(
    manifest_path: Path, *, merge_path: Path | None = None
) -> dict[str, Any]:
    manifest, manifest_payload = _load_manifest(manifest_path)
    _validate_schema_version(manifest.get("schema_version"), document="manifest")
    if manifest.get("workflow") != "parity_certification":
        raise EvidenceError("this command path requires workflow parity_certification")
    if manifest.get("evidence_level") != "formal":
        raise EvidenceError("parity certification requires formal evidence")
    if type(manifest.get("fixture")) is not bool:
        raise EvidenceError("fixture is required and must be a boolean")
    code_provenance = _validate_code_provenance(manifest.get("code_provenance"))
    declared_inputs = _validate_declared_inputs(
        manifest.get("declared_inputs"), base_directory=manifest_path.parent
    )
    inputs_by_name = {item["name"]: item for item in declared_inputs}
    configuration = manifest.get("policy_evaluation")
    if not isinstance(configuration, dict):
        raise EvidenceError("policy_evaluation must be an object")
    if configuration.get("protocol_version") != POLICY_RUNNER_PROTOCOL_VERSION:
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
    corpus, policies = _load_corpus(corpus_path, corpus_payload)
    evidence_document_paths = [manifest_path.resolve(strict=False)] + [
        item[0] for item in verified.values()
    ]
    for policy in policies:
        artifact_path = Path(policy["resolved_artifact_path"])
        if any(_paths_alias(artifact_path, path) for path in evidence_document_paths):
            raise EvidenceError(
                f"policy {policy['id']!r} artifact aliases an evidence input document"
            )
    seeds = _load_seeds(verified[seeds_input["name"]][1])
    providers = _providers(configuration.get("providers"))
    runner_path = verified[runner_input["name"]][0]
    timeout_seconds = configuration.get("timeout_seconds", 120)
    if (
        type(timeout_seconds) not in {int, float}
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise EvidenceError("policy_evaluation.timeout_seconds must be a positive finite number")
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
    reusable = _load_reusable_outcomes(
        merge_path,
        evaluation_identity=evaluation_identity,
        expected_units=expected_units,
    )
    outcomes: list[dict[str, Any]] = []
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
                    runner_path,
                    provider=provider,
                    policy=policy,
                    seeds=pending_seeds,
                    fixture_failure_seed=fixture_failure_seed,
                    timeout_seconds=float(timeout_seconds),
                )
                if pending_seeds
                else {}
            )
            if (
                _artifact_sha256(Path(policy["resolved_artifact_path"]), policy_id=policy["id"])
                != policy["artifact_sha256"]
            ):
                raise EvidenceError(
                    f"policy {policy['id']!r} artifact changed after the corpus was sealed"
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
    evaluation = {
        "protocol_version": POLICY_RUNNER_PROTOCOL_VERSION,
        "evaluation_identity": evaluation_identity,
        "corpus_manifest_sha256": corpus_input["sha256"],
        "corpus": corpus,
        "seed_manifest_sha256": seeds_input["sha256"],
        "seed_manifest": seeds,
        "providers": providers,
        "outcomes": outcomes,
        "expected_outcome_count": len(expected_units),
        "failure_count": sum(item["execution_failure"] is not None for item in outcomes),
    }
    evidence_entries = [
        {"name": "manifest", "sha256": _sha256_bytes(manifest_payload)},
        *({"name": item["name"], "sha256": item["sha256"]} for item in declared_inputs),
        *(
            {"name": f"policy_artifact.{policy['id']}", "sha256": policy["artifact_sha256"]}
            for policy in policies
        ),
        {
            "name": "policy_evaluation",
            "sha256": _canonical_sha256(evaluation, document="policy evaluation report"),
        },
    ]
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
    return {
        "schema_version": 1,
        "workflow": "parity_certification",
        "evidence_level": "formal",
        "fixture": manifest["fixture"],
        "status": "evaluation_complete",
        "claim_eligible": False,
        "claim_reasons": reasons,
        "run_identity": evaluation_identity,
        "code_provenance": code_provenance,
        "declared_inputs": declared_inputs,
        "policy_evaluation": evaluation,
        "evidence_index": {
            "algorithm": "sha256",
            "entries": evidence_entries,
            "sha256": _canonical_sha256(evidence_entries, document="policy evaluation report"),
        },
    }


__all__ = ["POLICY_RUNNER_PROTOCOL_VERSION", "build_policy_evaluation_report"]
