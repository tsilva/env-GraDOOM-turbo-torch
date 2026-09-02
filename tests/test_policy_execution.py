from __future__ import annotations

from types import SimpleNamespace

import pytest

from gradoom.evidence import certification_policy_runner
from gradoom.evidence.certification_policy_runner import _dispatch_loaded_checkpoint
from gradoom.evidence.policy_execution import (
    canonical_policy_execution_identity,
    policy_execution_identity,
)


def test_policy_execution_identity_forbids_provider_specific_modifications() -> None:
    artifact_sha256 = "a" * 64

    gradoom = policy_execution_identity(
        artifact_sha256=artifact_sha256,
        model_runtime_contract={
            "architecture": "nature-quarter",
            "compile_policy": False,
            "precision": "fp32",
        },
        stochastic_actions=True,
    )
    reference = policy_execution_identity(
        artifact_sha256=artifact_sha256,
        model_runtime_contract={
            "architecture": "nature-quarter",
            "compile_policy": False,
            "precision": "fp32",
        },
        stochastic_actions=True,
    )

    assert (
        gradoom
        == reference
        == {
            "artifact_sha256": artifact_sha256,
            "model_runtime_contract": {
                "architecture": "nature-quarter",
                "compile_policy": False,
                "precision": "fp32",
            },
            "preprocessing_sha256": (
                "6ff033ce02585302f78e84c16f6a86da99690e0d861092cab41e55b4257e08d0"
            ),
            "action_sampling": "stochastic",
            "provider_specific_modifications": [],
        }
    )


def test_real_checkpoint_contract_has_one_canonical_identity_for_both_providers() -> None:
    artifact_sha256 = "b" * 64
    artifact_contract = {
        "contract_version": 1,
        "artifact_format": "standalone-gradoom-ppo-v1",
        "runtime": "torch",
        "architecture": "nature",
    }
    runtime_contract = {
        "architecture": "nature",
        "memory_format": "contiguous",
        "observation_blur_kernel": 1,
        "frozen_encoder_custom_conv": False,
        "precision": "fp32",
        "compile_policy": False,
        "float32_matmul_precision": "highest",
    }
    identity = canonical_policy_execution_identity(
        artifact_sha256=artifact_sha256,
        artifact_contract=artifact_contract,
        stochastic_actions=True,
    )
    loaded = SimpleNamespace(
        artifact_sha256=artifact_sha256,
        contract=SimpleNamespace(as_dict=lambda: runtime_contract),
    )
    reached: list[str] = []

    for provider in ("gradoom", "env-vizdoom-turbo"):
        request = {
            "provider_id": provider,
            "policy": {
                "model_runtime_contract": artifact_contract,
                "execution_identity": identity,
            },
        }
        result = _dispatch_loaded_checkpoint(
            request,
            loaded,
            lambda provider=provider: reached.append(provider) or f"{provider}-outcomes",
        )
        assert result == f"{provider}-outcomes"

    assert reached == ["gradoom", "env-vizdoom-turbo"]

    mismatched = {
        "provider_id": "gradoom",
        "policy": {
            "model_runtime_contract": {**artifact_contract, "architecture": "nature-half"},
            "execution_identity": identity,
        },
    }
    with pytest.raises(ValueError, match="does not satisfy"):
        _dispatch_loaded_checkpoint(mismatched, loaded, lambda: reached.append("invalid"))
    assert "invalid" not in reached


def test_runtime_closure_accepts_bound_transitive_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = [
        {"name": name, "content_sha256": character * 64}
        for name, character in (
            ("torch", "a"),
            ("numpy", "b"),
            ("gymnasium", "c"),
            ("nvidia-cublas", "d"),
        )
    ]
    by_name = {identity["name"]: identity for identity in identities}
    monkeypatch.setattr(
        certification_policy_runner,
        "installed_distribution_identity",
        lambda name: by_name[name],
    )

    certification_policy_runner._validate_runtime_distributions(identities)

    mutated = [
        {**identity, "content_sha256": "0" * 64} if identity["name"] == "torch" else identity
        for identity in identities
    ]
    with pytest.raises(ValueError, match="bytes do not match"):
        certification_policy_runner._validate_runtime_distributions(mutated)
