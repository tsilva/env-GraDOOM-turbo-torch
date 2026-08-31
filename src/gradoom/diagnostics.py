"""Public, diagnostic-only actor provenance records.

These immutable values sit outside the policy-facing transition data plane.
They are used by the parity invariant runner to inspect an explicitly staged
semantic case without adding host work to ordinary reset or step calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ActorKind = Literal["player", "enemy"]


class ActorAttributionDiagnostics:
    """Marker for production providers implementing staged actor diagnostics."""


@dataclass(frozen=True)
class ActorSnapshot:
    """Identity and liveness of one actor in a diagnostic stage."""

    actor_id: int
    kind: ActorKind
    alive: bool


@dataclass(frozen=True)
class ActorAttributionStage:
    """The exact actors present before a diagnostic kill is attempted."""

    token: str
    actors: tuple[ActorSnapshot, ...]


@dataclass(frozen=True)
class ActorKillEvent:
    """One engine-observed source/target death relationship."""

    stage_token: str
    attacker_id: int
    attacker_kind: ActorKind
    target_id: int
    target_kind: ActorKind


__all__ = [
    "ActorAttributionDiagnostics",
    "ActorAttributionStage",
    "ActorKillEvent",
    "ActorKind",
    "ActorSnapshot",
]
