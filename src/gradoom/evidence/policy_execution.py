from __future__ import annotations

from collections.abc import Mapping

POLICY_PREPROCESSING_SHA256 = "6ff033ce02585302f78e84c16f6a86da99690e0d861092cab41e55b4257e08d0"
POLICY_ARTIFACT_CONTRACT_FIELDS = frozenset(
    {"contract_version", "artifact_format", "runtime", "architecture"}
)


def policy_execution_identity(
    *,
    artifact_sha256: str,
    model_runtime_contract: Mapping[str, object],
    stochastic_actions: bool,
) -> dict[str, object]:
    if len(artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in artifact_sha256
    ):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
    return {
        "artifact_sha256": artifact_sha256,
        "model_runtime_contract": dict(model_runtime_contract),
        "preprocessing_sha256": POLICY_PREPROCESSING_SHA256,
        "action_sampling": "stochastic" if stochastic_actions else "argmax",
        "provider_specific_modifications": [],
    }


def canonical_policy_execution_identity(
    *,
    artifact_sha256: str,
    artifact_contract: Mapping[str, object],
    stochastic_actions: bool,
) -> dict[str, object]:
    contract = dict(artifact_contract)
    if set(contract) != POLICY_ARTIFACT_CONTRACT_FIELDS:
        raise ValueError("artifact_contract must be the canonical artifact contract")
    return policy_execution_identity(
        artifact_sha256=artifact_sha256,
        model_runtime_contract=contract,
        stochastic_actions=stochastic_actions,
    )


def validate_loaded_policy_contract(
    declared_contract: Mapping[str, object],
    loaded_runtime_contract: Mapping[str, object],
) -> dict[str, object]:
    """Validate runtime details without conflating them with the artifact contract."""

    declared = dict(declared_contract)
    runtime = dict(loaded_runtime_contract)
    if set(declared) != POLICY_ARTIFACT_CONTRACT_FIELDS:
        raise ValueError("declared policy contract is not canonical")
    if (
        declared.get("contract_version") != 1
        or declared.get("artifact_format") != "standalone-gradoom-ppo-v1"
        or declared.get("runtime") != "torch"
        or runtime.get("architecture") != declared.get("architecture")
    ):
        raise ValueError("loaded checkpoint does not satisfy its declared artifact contract")
    return runtime


__all__ = [
    "POLICY_ARTIFACT_CONTRACT_FIELDS",
    "POLICY_PREPROCESSING_SHA256",
    "canonical_policy_execution_identity",
    "policy_execution_identity",
    "validate_loaded_policy_contract",
]
