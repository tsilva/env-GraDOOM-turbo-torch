"""Versioned evidence manifests and reports."""

from typing import Any

__all__ = [
    "INVARIANT_SUITE_VERSION",
    "POLICY_RUNNER_PROTOCOL_VERSION",
    "EvidenceError",
    "build_policy_evaluation_report",
    "build_readiness_report",
]


def __getattr__(name: str) -> Any:
    """Keep training submodule imports independent from readiness execution machinery."""
    if name == "INVARIANT_SUITE_VERSION":
        from .invariant_contract import INVARIANT_SUITE_VERSION

        return INVARIANT_SUITE_VERSION
    if name in {"POLICY_RUNNER_PROTOCOL_VERSION", "build_policy_evaluation_report"}:
        from .policy_corpus import POLICY_RUNNER_PROTOCOL_VERSION, build_policy_evaluation_report

        return {
            "POLICY_RUNNER_PROTOCOL_VERSION": POLICY_RUNNER_PROTOCOL_VERSION,
            "build_policy_evaluation_report": build_policy_evaluation_report,
        }[name]
    if name in {"EvidenceError", "build_readiness_report"}:
        from .report import EvidenceError, build_readiness_report

        return {"EvidenceError": EvidenceError, "build_readiness_report": build_readiness_report}[
            name
        ]
    raise AttributeError(name)
