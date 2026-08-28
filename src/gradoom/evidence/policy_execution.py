from __future__ import annotations

from collections.abc import Mapping

POLICY_PREPROCESSING_SHA256 = "6ff033ce02585302f78e84c16f6a86da99690e0d861092cab41e55b4257e08d0"


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


__all__ = ["POLICY_PREPROCESSING_SHA256", "policy_execution_identity"]
