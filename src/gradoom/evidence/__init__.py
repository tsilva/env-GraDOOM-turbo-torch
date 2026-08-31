"""Versioned evidence manifests and reports."""

from typing import Any

__all__ = ["INVARIANT_SUITE_VERSION", "EvidenceError", "build_readiness_report"]


def __getattr__(name: str) -> Any:
    """Keep training submodule imports independent from readiness execution machinery."""
    if name == "INVARIANT_SUITE_VERSION":
        from .invariant_contract import INVARIANT_SUITE_VERSION

        return INVARIANT_SUITE_VERSION
    if name in {"EvidenceError", "build_readiness_report"}:
        from .report import EvidenceError, build_readiness_report

        return {"EvidenceError": EvidenceError, "build_readiness_report": build_readiness_report}[
            name
        ]
    raise AttributeError(name)
