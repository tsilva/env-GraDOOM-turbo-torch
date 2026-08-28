from __future__ import annotations

from gradoom.evidence.policy_execution import policy_execution_identity


def test_policy_execution_identity_forbids_provider_specific_modifications() -> None:
    artifact_sha256 = "a" * 64

    gradoom = policy_execution_identity(
        artifact_sha256=artifact_sha256,
        stochastic_actions=True,
    )
    reference = policy_execution_identity(
        artifact_sha256=artifact_sha256,
        stochastic_actions=True,
    )

    assert (
        gradoom
        == reference
        == {
            "artifact_sha256": artifact_sha256,
            "preprocessing_sha256": (
                "6ff033ce02585302f78e84c16f6a86da99690e0d861092cab41e55b4257e08d0"
            ),
            "action_sampling": "stochastic",
            "provider_specific_modifications": [],
        }
    )
