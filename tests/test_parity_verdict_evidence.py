from __future__ import annotations

import copy
import hashlib
import json

import pytest

from gradoom.evidence.parity_verdict import (
    BOOTSTRAP_RESAMPLES,
    REQUIRED_FAST_INVARIANTS,
    build_parity_verdict,
    issue_parity_certificate,
    parity_claim_reasons,
    validate_parity_certificate,
)
from gradoom.evidence.report import EvidenceError


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _evaluation(policy_values: list[tuple[str, float, float]]) -> dict[str, object]:
    policies = [
        {
            "id": policy_id,
            "artifact_sha256": f"{index + 1:064x}",
        }
        for index, (policy_id, _gradoom, _reference) in enumerate(policy_values)
    ]
    outcomes = []
    for provider_id, value_index in (("gradoom", 1), ("env-vizdoom-turbo", 2)):
        for policy_id, gradoom_value, reference_value in policy_values:
            value = (policy_id, gradoom_value, reference_value)[value_index]
            outcomes.extend(
                {
                    "provider_id": provider_id,
                    "policy_id": policy_id,
                    "seed_index": seed_index,
                    "seed": seed_index,
                    "player_killcount": value,
                    "execution_failure": None,
                }
                for seed_index in range(256)
            )
    return {
        "protocol_version": 2,
        "evaluation_identity": "a" * 64,
        "corpus_manifest_sha256": "b" * 64,
        "corpus": {"corpus_version": "sealed-v1", "policies": policies},
        "seed_manifest_sha256": "c" * 64,
        "seed_manifest": {
            "schema_version": 1,
            "seed_set_id": "held-out-v1",
            "seeds": list(range(256)),
        },
        "providers": [
            {"id": "gradoom", "revision": "d" * 40},
            {"id": "env-vizdoom-turbo", "revision": "e" * 40},
        ],
        "outcomes": outcomes,
        "expected_outcome_count": len(outcomes),
        "failure_count": 0,
    }


def _passing_invariants() -> dict[str, object]:
    return {
        "version": "1.0.0",
        "configured": True,
        "status": "passed",
        "checks": [
            {"behavior": behavior, "status": "passed"}
            for behavior in sorted(REQUIRED_FAST_INVARIANTS)
        ],
        "failures": [],
        "unavailable_reasons": [],
        "providers": [
            {"id": "gradoom", "revision": "d" * 40},
            {"id": "env-vizdoom-turbo", "revision": "e" * 40},
        ],
        "diagnostics": {"affects_verdict": False, "tools": []},
    }


def test_exact_threshold_zero_and_low_reference_means_use_the_approved_formula() -> None:
    evaluation = _evaluation(
        [
            ("zero-exact", 2.0, 0.0),
            ("low-exact", 3.0, 1.0),
            ("ten-percent-exact", 33.0, 30.0),
            ("over", 33.000_001, 30.0),
        ]
    )

    verdict = build_parity_verdict(
        evaluation,
        bootstrap_seed=8675309,
        invariant_suite=_passing_invariants(),
        complete=True,
    )

    by_id = {item["policy_id"]: item for item in verdict["policies"]}
    assert by_id["zero-exact"]["maximum_allowed_difference"] == 2.0
    assert by_id["zero-exact"]["threshold_passed"] is True
    assert by_id["low-exact"]["maximum_allowed_difference"] == 2.0
    assert by_id["low-exact"]["threshold_passed"] is True
    assert by_id["ten-percent-exact"]["maximum_allowed_difference"] == 3.0
    assert by_id["ten-percent-exact"]["threshold_passed"] is True
    assert by_id["over"]["threshold_passed"] is False
    assert verdict["status"] == "failed"
    assert verdict["would_issue"] is False


def test_paired_bootstrap_is_deterministic_exactly_10000_and_diagnostic_only() -> None:
    evaluation = _evaluation([("boundary", 2.0, 0.0)])
    for outcome in evaluation["outcomes"]:
        if outcome["provider_id"] == "gradoom":
            outcome["player_killcount"] = 1.5 if outcome["seed_index"] % 2 == 0 else 2.5

    first = build_parity_verdict(
        evaluation,
        bootstrap_seed=12345,
        invariant_suite=_passing_invariants(),
        complete=True,
    )
    second = build_parity_verdict(
        evaluation,
        bootstrap_seed=12345,
        invariant_suite=_passing_invariants(),
        complete=True,
    )

    interval = first["policies"][0]["bootstrap_interval"]
    assert interval == second["policies"][0]["bootstrap_interval"]
    assert {
        key: interval[key]
        for key in (
            "method",
            "random_generator",
            "percentile_interpolation",
            "confidence_level",
            "resamples",
            "seed",
            "status",
            "affects_verdict",
        )
    } == {
        "method": "paired_percentile",
        "random_generator": "numpy-pcg64",
        "percentile_interpolation": "linear",
        "confidence_level": 0.95,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": 12345,
        "status": "available",
        "affects_verdict": False,
    }
    assert interval["lower_mean_difference"] < 2.0
    assert interval["upper_mean_difference"] > 2.0
    assert first["policies"][0]["threshold_passed"] is True
    assert first["would_issue"] is True


def test_failed_or_incomplete_policy_evidence_cannot_would_issue() -> None:
    evaluation = _evaluation([("policy", 1.0, 1.0)])
    evaluation["outcomes"][0]["player_killcount"] = None
    evaluation["outcomes"][0]["execution_failure"] = {
        "code": "fixture_failure",
        "message": "retained",
    }

    failed = build_parity_verdict(
        evaluation,
        bootstrap_seed=7,
        invariant_suite=_passing_invariants(),
        complete=True,
    )
    incomplete = build_parity_verdict(
        evaluation,
        bootstrap_seed=7,
        invariant_suite=_passing_invariants(),
        complete=False,
    )

    assert failed["status"] == "unavailable"
    assert failed["policies"][0]["bootstrap_interval"]["status"] == "unavailable"
    assert failed["would_issue"] is False
    assert incomplete["status"] == "incomplete"
    assert incomplete["would_issue"] is False


def test_invariant_gate_requires_every_named_fast_invariant() -> None:
    evaluation = _evaluation([("policy", 1.0, 1.0)])
    invariants = _passing_invariants()
    invariants["checks"].pop()

    verdict = build_parity_verdict(
        evaluation,
        bootstrap_seed=7,
        invariant_suite=invariants,
        complete=True,
    )

    assert verdict["status"] == "passed"
    assert verdict["all_policies_passed"] is True
    assert verdict["all_invariants_passed"] is False
    assert verdict["would_issue"] is False
    assert "invariant_suite_not_passed" in {
        reason["code"]
        for reason in parity_claim_reasons(
            evaluation,
            verdict=verdict,
            fixture=False,
            code_provenance={"revision": "d" * 40, "dirty": False},
            wad_profile={"status": "matched"},
            invariant_suite=invariants,
        )
    }


def test_real_certificate_binds_every_material_identity_and_rejects_tampering() -> None:
    evaluation = _evaluation([("policy", 1.0, 1.0)])
    invariants = _passing_invariants()
    verdict = build_parity_verdict(
        evaluation,
        bootstrap_seed=20260827,
        invariant_suite=invariants,
        complete=True,
    )
    wad_profile = {
        "status": "matched",
        "profile_identity": "f" * 64,
        "binding_sha256": "1" * 64,
    }
    provenance = {
        "repository": "tsilva/env-GraDOOM-turbo-torch",
        "revision": "d" * 40,
        "dirty": False,
    }

    certificate = issue_parity_certificate(
        evaluation,
        verdict=verdict,
        fixture=False,
        code_provenance=provenance,
        wad_profile=wad_profile,
        invariant_suite=invariants,
        report_schema_version=1,
    )

    assert certificate is not None
    binding = certificate["binding"]
    assert binding["gradoom"] == {
        "repository": "tsilva/env-GraDOOM-turbo-torch",
        "revision": "d" * 40,
        "clean": True,
    }
    assert binding["reference_revision"] == "e" * 40
    assert binding["wad_profile"] == {
        "profile_identity": "f" * 64,
        "binding_sha256": "1" * 64,
    }
    assert binding["corpus_manifest_sha256"] == "b" * 64
    assert binding["seed_manifest_sha256"] == "c" * 64
    assert binding["invariant_suite_version"] == "1.0.0"
    assert binding["evaluation_protocol"]["bootstrap_resamples"] == 10_000
    assert binding["report_schema_version"] == 1
    validate_parity_certificate(certificate, expected_binding=binding)

    reseeded_verdict = build_parity_verdict(
        evaluation,
        bootstrap_seed=20260828,
        invariant_suite=invariants,
        complete=True,
    )
    reseeded_certificate = issue_parity_certificate(
        evaluation,
        verdict=reseeded_verdict,
        fixture=False,
        code_provenance=provenance,
        wad_profile=wad_profile,
        invariant_suite=invariants,
        report_schema_version=1,
    )
    assert reseeded_certificate is not None
    assert reseeded_certificate["certificate_identity"] != certificate["certificate_identity"]

    tampered = copy.deepcopy(certificate)
    tampered["binding"]["reference_revision"] = "0" * 40
    with pytest.raises(EvidenceError, match="certificate identity"):
        validate_parity_certificate(tampered, expected_binding=tampered["binding"])

    self_consistent_change = copy.deepcopy(certificate)
    self_consistent_change["binding"]["seed_manifest_sha256"] = "0" * 64
    self_consistent_change["certificate_identity"] = _canonical_sha256(
        self_consistent_change["binding"]
    )
    with pytest.raises(EvidenceError, match="does not match current evidence"):
        validate_parity_certificate(self_consistent_change, expected_binding=binding)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda context: context.update(fixture=True),
        lambda context: context["code_provenance"].update(dirty=True),
        lambda context: context["code_provenance"].update(revision="0" * 40),
        lambda context: context.update(wad_profile={"status": "failed"}),
        lambda context: context["invariant_suite"].update(status="failed"),
    ],
)
def test_fixture_dirty_stale_mismatched_or_failed_evidence_emits_no_certificate(
    mutation: object,
) -> None:
    evaluation = _evaluation([("policy", 1.0, 1.0)])
    context = {
        "fixture": False,
        "code_provenance": {
            "repository": "tsilva/env-GraDOOM-turbo-torch",
            "revision": "d" * 40,
            "dirty": False,
        },
        "wad_profile": {
            "status": "matched",
            "profile_identity": "f" * 64,
            "binding_sha256": "1" * 64,
        },
        "invariant_suite": _passing_invariants(),
    }
    verdict = build_parity_verdict(
        evaluation,
        bootstrap_seed=1,
        invariant_suite=context["invariant_suite"],
        complete=True,
    )
    mutation(context)

    certificate = issue_parity_certificate(
        evaluation,
        verdict=verdict,
        fixture=context["fixture"],
        code_provenance=context["code_provenance"],
        wad_profile=context["wad_profile"],
        invariant_suite=context["invariant_suite"],
        report_schema_version=1,
    )

    assert certificate is None
