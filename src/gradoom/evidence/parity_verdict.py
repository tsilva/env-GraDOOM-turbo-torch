from __future__ import annotations

import math
from typing import Any

import numpy as np

from .report import EvidenceError, _canonical_sha256, _validate_sha256

PARITY_VERDICT_PROTOCOL_VERSION = 1
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RANDOM_GENERATOR = "numpy-pcg64"
BOOTSTRAP_PERCENTILE_INTERPOLATION = "linear"
_PROVIDER_IDS = ("gradoom", "env-vizdoom-turbo")
_MAX_BOOTSTRAP_SEED = 2**63 - 1
REQUIRED_FAST_INVARIANTS = frozenset(
    {
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
        "gradoom.tensor_inputs",
        "gradoom.tensor_outputs",
        "gradoom.device",
    }
)


def _provider_revisions(evaluation: dict[str, Any]) -> dict[str, str]:
    providers = evaluation.get("providers")
    if not isinstance(providers, list):
        raise EvidenceError("policy evaluation providers must be an array")
    revisions: dict[str, str] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            raise EvidenceError("policy evaluation provider must be an object")
        provider_id = provider.get("id")
        revision = provider.get("revision")
        if provider_id not in _PROVIDER_IDS or not isinstance(revision, str) or not revision:
            raise EvidenceError("policy evaluation provider identity is invalid")
        if provider_id in revisions:
            raise EvidenceError("policy evaluation provider identity is duplicated")
        revisions[provider_id] = revision
    if set(revisions) != set(_PROVIDER_IDS):
        raise EvidenceError("policy evaluation must bind both parity providers")
    return revisions


def _linear_percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * (position - lower_index)


def _paired_bootstrap(differences: list[float], *, seed: int) -> tuple[float, float]:
    generator = np.random.Generator(np.random.PCG64(seed))
    values = np.asarray(differences, dtype=np.float64)
    sample_count = len(differences)
    estimates: list[float] = []
    while len(estimates) < BOOTSTRAP_RESAMPLES:
        batch_size = min(1024, BOOTSTRAP_RESAMPLES - len(estimates))
        indexes = generator.integers(
            0,
            sample_count,
            size=(batch_size, sample_count),
            dtype=np.int64,
        )
        estimates.extend(float(value) for value in values[indexes].mean(axis=1))
    estimates.sort()
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    return (
        _linear_percentile(estimates, tail),
        _linear_percentile(estimates, 1.0 - tail),
    )


def _bootstrap_interval(
    differences: list[float] | None,
    *,
    seed: int,
) -> dict[str, Any]:
    interval: dict[str, Any] = {
        "method": "paired_percentile",
        "random_generator": BOOTSTRAP_RANDOM_GENERATOR,
        "percentile_interpolation": BOOTSTRAP_PERCENTILE_INTERPOLATION,
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": seed,
        "status": "unavailable" if differences is None else "available",
        "lower_mean_difference": None,
        "upper_mean_difference": None,
        "affects_verdict": False,
    }
    if differences is not None:
        lower, upper = _paired_bootstrap(differences, seed=seed)
        interval["lower_mean_difference"] = lower
        interval["upper_mean_difference"] = upper
    return interval


def _policy_outcomes(
    evaluation: dict[str, Any],
    *,
    policy_id: str,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    outcomes = evaluation.get("outcomes")
    if not isinstance(outcomes, list):
        raise EvidenceError("policy evaluation outcomes must be an array")
    grouped: dict[str, dict[int, dict[str, Any]]] = {
        provider_id: {} for provider_id in _PROVIDER_IDS
    }
    for outcome in outcomes:
        if not isinstance(outcome, dict) or outcome.get("policy_id") != policy_id:
            continue
        provider_id = outcome.get("provider_id")
        seed_index = outcome.get("seed_index")
        if provider_id not in grouped or type(seed_index) is not int:
            raise EvidenceError("policy evaluation contains an invalid verdict outcome")
        if seed_index in grouped[provider_id]:
            raise EvidenceError("policy evaluation contains duplicate verdict outcomes")
        grouped[provider_id][seed_index] = outcome
    expected_indexes = set(range(len(seeds)))
    if any(set(grouped[provider_id]) != expected_indexes for provider_id in _PROVIDER_IDS):
        return None
    ordered = tuple(
        [grouped[provider_id][index] for index in range(len(seeds))]
        for provider_id in _PROVIDER_IDS
    )
    for provider_outcomes in ordered:
        if any(item.get("seed") != seeds[index] for index, item in enumerate(provider_outcomes)):
            raise EvidenceError("policy verdict outcomes do not match the seed manifest")
    return ordered


def build_parity_verdict(
    evaluation: dict[str, Any],
    *,
    bootstrap_seed: int,
    invariant_suite: dict[str, Any] | None,
    complete: bool,
) -> dict[str, Any]:
    """Compute the bounded policy-level verdict from an authenticated corpus report."""

    if type(bootstrap_seed) is not int or not 0 <= bootstrap_seed <= _MAX_BOOTSTRAP_SEED:
        raise EvidenceError(
            "policy_evaluation.bootstrap_seed must be a non-negative 64-bit integer"
        )
    _provider_revisions(evaluation)
    corpus = evaluation.get("corpus")
    seed_manifest = evaluation.get("seed_manifest")
    if not isinstance(corpus, dict) or not isinstance(corpus.get("policies"), list):
        raise EvidenceError("policy evaluation corpus is invalid for a parity verdict")
    if not isinstance(seed_manifest, dict) or not isinstance(seed_manifest.get("seeds"), list):
        raise EvidenceError("policy evaluation seed manifest is invalid for a parity verdict")
    seeds = seed_manifest["seeds"]
    if len(seeds) != 256:
        raise EvidenceError("parity verdict requires exactly 256 paired episode seeds")

    policy_reports: list[dict[str, Any]] = []
    for policy in corpus["policies"]:
        if not isinstance(policy, dict) or not isinstance(policy.get("id"), str):
            raise EvidenceError("policy evaluation corpus contains an invalid policy")
        policy_id = policy["id"]
        paired = _policy_outcomes(evaluation, policy_id=policy_id, seeds=seeds)
        outcomes = evaluation["outcomes"]
        failures = sum(
            isinstance(item, dict)
            and item.get("policy_id") == policy_id
            and item.get("execution_failure") is not None
            for item in outcomes
        )
        successful = paired is not None and failures == 0
        if successful:
            assert paired is not None
            gradoom_values = [float(item["player_killcount"]) for item in paired[0]]
            reference_values = [float(item["player_killcount"]) for item in paired[1]]
            gradoom_mean = math.fsum(gradoom_values) / len(gradoom_values)
            reference_mean = math.fsum(reference_values) / len(reference_values)
            mean_difference = gradoom_mean - reference_mean
            absolute_difference = abs(mean_difference)
            maximum_difference = max(2.0, 0.1 * reference_mean)
            threshold_passed: bool | None = absolute_difference <= maximum_difference
            differences = [
                gradoom - reference
                for gradoom, reference in zip(gradoom_values, reference_values, strict=True)
            ]
        else:
            gradoom_mean = None
            reference_mean = None
            mean_difference = None
            absolute_difference = None
            maximum_difference = None
            threshold_passed = None
            differences = None
        policy_reports.append(
            {
                "policy_id": policy_id,
                "paired_episode_count": len(seeds) if paired is not None else 0,
                "execution_failure_count": failures,
                "gradoom_mean_player_killcount": gradoom_mean,
                "env_vizdoom_turbo_mean_player_killcount": reference_mean,
                "mean_difference": mean_difference,
                "absolute_mean_difference": absolute_difference,
                "maximum_allowed_difference": maximum_difference,
                "threshold_passed": threshold_passed,
                "bootstrap_interval": _bootstrap_interval(differences, seed=bootstrap_seed),
            }
        )

    all_available = all(item["threshold_passed"] is not None for item in policy_reports)
    all_passed = all(item["threshold_passed"] is True for item in policy_reports)
    invariant_checks = invariant_suite.get("checks") if isinstance(invariant_suite, dict) else None
    invariant_passed = (
        isinstance(invariant_suite, dict)
        and invariant_suite.get("configured") is True
        and invariant_suite.get("status") == "passed"
        and isinstance(invariant_checks, list)
        and len(invariant_checks) == len(REQUIRED_FAST_INVARIANTS)
        and {check.get("behavior") for check in invariant_checks if isinstance(check, dict)}
        == REQUIRED_FAST_INVARIANTS
        and all(
            isinstance(check, dict) and check.get("status") == "passed"
            for check in invariant_checks
        )
    )
    status = (
        "incomplete"
        if not complete
        else "unavailable"
        if not all_available
        else "passed"
        if all_passed
        else "failed"
    )
    return {
        "protocol_version": PARITY_VERDICT_PROTOCOL_VERSION,
        "status": status,
        "bootstrap": {
            "method": "paired_percentile",
            "random_generator": BOOTSTRAP_RANDOM_GENERATOR,
            "percentile_interpolation": BOOTSTRAP_PERCENTILE_INTERPOLATION,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": bootstrap_seed,
            "affects_verdict": False,
        },
        "policies": policy_reports,
        "all_policies_passed": all_passed and all_available,
        "all_invariants_passed": invariant_passed,
        "would_issue": complete and all_passed and all_available and invariant_passed,
    }


def _invariant_revisions(invariant_suite: dict[str, Any]) -> dict[str, str] | None:
    providers = invariant_suite.get("providers")
    if not isinstance(providers, list):
        return None
    revisions: dict[str, str] = {}
    for provider in providers:
        if not isinstance(provider, dict):
            return None
        provider_id = provider.get("id")
        revision = provider.get("revision")
        if provider_id not in _PROVIDER_IDS or not isinstance(revision, str):
            return None
        revisions[provider_id] = revision
    return revisions if set(revisions) == set(_PROVIDER_IDS) else None


def issue_parity_certificate(
    evaluation: dict[str, Any],
    *,
    verdict: dict[str, Any],
    fixture: bool,
    code_provenance: dict[str, Any],
    wad_profile: dict[str, Any] | None,
    invariant_suite: dict[str, Any] | None,
    report_schema_version: int,
) -> dict[str, Any] | None:
    revisions = _provider_revisions(evaluation)
    invariant_revisions = (
        _invariant_revisions(invariant_suite) if isinstance(invariant_suite, dict) else None
    )
    if (
        fixture
        or verdict.get("would_issue") is not True
        or code_provenance.get("dirty") is not False
        or not isinstance(code_provenance.get("repository"), str)
        or code_provenance.get("revision") != revisions["gradoom"]
        or not isinstance(wad_profile, dict)
        or wad_profile.get("status") != "matched"
        or not isinstance(invariant_suite, dict)
        or invariant_suite.get("status") != "passed"
        or invariant_revisions != revisions
    ):
        return None
    profile_identity = wad_profile.get("profile_identity")
    binding_sha256 = wad_profile.get("binding_sha256")
    if not isinstance(profile_identity, str) or not isinstance(binding_sha256, str):
        return None
    _validate_sha256(profile_identity, "wad profile identity")
    _validate_sha256(binding_sha256, "wad profile binding")
    binding = {
        "gradoom": {
            "repository": code_provenance["repository"],
            "revision": revisions["gradoom"],
            "clean": True,
        },
        "reference_revision": revisions["env-vizdoom-turbo"],
        "wad_profile": {
            "profile_identity": profile_identity,
            "binding_sha256": binding_sha256,
        },
        "corpus_manifest_sha256": evaluation["corpus_manifest_sha256"],
        "seed_manifest_sha256": evaluation["seed_manifest_sha256"],
        "invariant_suite_version": invariant_suite["version"],
        "evaluation_protocol": {
            "policy_runner_protocol_version": evaluation["protocol_version"],
            "parity_verdict_protocol_version": verdict["protocol_version"],
            "bootstrap_method": verdict["bootstrap"]["method"],
            "bootstrap_random_generator": verdict["bootstrap"]["random_generator"],
            "bootstrap_percentile_interpolation": verdict["bootstrap"]["percentile_interpolation"],
            "bootstrap_confidence_level": verdict["bootstrap"]["confidence_level"],
            "bootstrap_resamples": verdict["bootstrap"]["resamples"],
            "bootstrap_seed": verdict["bootstrap"]["seed"],
        },
        "report_schema_version": report_schema_version,
        "evaluation_identity": evaluation["evaluation_identity"],
        "parity_verdict_sha256": _canonical_sha256(verdict, document="parity verdict"),
    }
    certificate = {
        "schema_version": 1,
        "scope": "bounded_policy_level_proxy",
        "binding": binding,
        "certificate_identity": _canonical_sha256(binding, document="parity certificate"),
    }
    validate_parity_certificate(certificate, expected_binding=binding)
    return certificate


def parity_claim_reasons(
    evaluation: dict[str, Any],
    *,
    verdict: dict[str, Any],
    fixture: bool,
    code_provenance: dict[str, Any],
    wad_profile: dict[str, Any] | None,
    invariant_suite: dict[str, Any] | None,
) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    if fixture:
        reasons.append(
            {
                "code": "fixture_evidence",
                "message": "Fixture evidence cannot support public claims.",
            }
        )
    if verdict.get("status") == "incomplete":
        reasons.append(
            {
                "code": "incomplete_policy_evidence",
                "message": "The complete provider-policy-seed grid is not available.",
            }
        )
    elif verdict.get("status") == "unavailable":
        reasons.append(
            {
                "code": "policy_evaluation_failure",
                "message": "At least one required policy outcome is unavailable or failed.",
            }
        )
    elif verdict.get("status") == "failed":
        reasons.append(
            {
                "code": "parity_threshold_failure",
                "message": "At least one corpus policy exceeded the approved parity threshold.",
            }
        )
    if verdict.get("all_invariants_passed") is not True:
        reasons.append(
            {
                "code": "invariant_suite_not_passed",
                "message": "Every fast invariant must pass before a certificate can issue.",
            }
        )
    if not isinstance(wad_profile, dict) or wad_profile.get("status") != "matched":
        reasons.append(
            {
                "code": "wad_profile_not_matched",
                "message": "A matched certified WAD profile is required for a real certificate.",
            }
        )
    if code_provenance.get("dirty") is not False:
        reasons.append(
            {
                "code": "dirty_code_provenance",
                "message": "A real certificate requires a clean GraDOOM revision.",
            }
        )
    try:
        revisions = _provider_revisions(evaluation)
    except EvidenceError:
        revisions = {}
    if code_provenance.get("revision") != revisions.get("gradoom"):
        reasons.append(
            {
                "code": "gradoom_revision_mismatch",
                "message": "Code provenance and evaluated GraDOOM revisions do not match.",
            }
        )
    if (
        isinstance(invariant_suite, dict)
        and invariant_suite.get("status") == "passed"
        and _invariant_revisions(invariant_suite) != revisions
    ):
        reasons.append(
            {
                "code": "invariant_revision_mismatch",
                "message": (
                    "Invariant and policy evaluations do not bind the same provider revisions."
                ),
            }
        )
    return reasons


def validate_parity_certificate(
    certificate: object,
    *,
    expected_binding: dict[str, Any],
) -> None:
    if not isinstance(certificate, dict) or set(certificate) != {
        "schema_version",
        "scope",
        "binding",
        "certificate_identity",
    }:
        raise EvidenceError("parity certificate has invalid fields")
    if certificate.get("schema_version") != 1:
        raise EvidenceError("parity certificate has an unsupported schema version")
    if certificate.get("scope") != "bounded_policy_level_proxy":
        raise EvidenceError("parity certificate has an invalid scope")
    binding = certificate.get("binding")
    if not isinstance(binding, dict):
        raise EvidenceError("parity certificate binding must be an object")
    identity = _validate_sha256(
        certificate.get("certificate_identity"), "parity certificate identity"
    )
    if identity != _canonical_sha256(binding, document="parity certificate"):
        raise EvidenceError("parity certificate identity does not match its binding")
    if _canonical_sha256(binding, document="parity certificate") != _canonical_sha256(
        expected_binding, document="expected parity certificate"
    ):
        raise EvidenceError("parity certificate binding does not match current evidence")


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "PARITY_VERDICT_PROTOCOL_VERSION",
    "REQUIRED_FAST_INVARIANTS",
    "build_parity_verdict",
    "issue_parity_certificate",
    "parity_claim_reasons",
    "validate_parity_certificate",
]
