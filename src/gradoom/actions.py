"""Pinned action semantics for the first certified deathmatch profile."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

DEATHMATCH_BUTTONS = (
    "ATTACK",
    "SPEED",
    "STRAFE",
    "MOVE_RIGHT",
    "MOVE_LEFT",
    "MOVE_BACKWARD",
    "MOVE_FORWARD",
    "TURN_RIGHT",
    "TURN_LEFT",
    "SELECT_WEAPON1",
    "SELECT_WEAPON2",
    "SELECT_WEAPON3",
    "SELECT_WEAPON4",
    "SELECT_WEAPON5",
    "SELECT_WEAPON6",
    "SELECT_NEXT_WEAPON",
    "SELECT_PREV_WEAPON",
    "LOOK_UP_DOWN_DELTA",
    "TURN_LEFT_RIGHT_DELTA",
    "MOVE_LEFT_RIGHT_DELTA",
)

DEATHMATCH_ACTIONS = (
    (),
    ("ATTACK",),
    ("MOVE_FORWARD",),
    ("MOVE_BACKWARD",),
    ("MOVE_LEFT",),
    ("MOVE_RIGHT",),
    ("TURN_LEFT",),
    ("TURN_RIGHT",),
    ("SPEED", "MOVE_FORWARD"),
    ("ATTACK", "MOVE_FORWARD"),
    ("ATTACK", "MOVE_BACKWARD"),
    ("ATTACK", "MOVE_LEFT"),
    ("ATTACK", "MOVE_RIGHT"),
    ("ATTACK", "TURN_LEFT"),
    ("ATTACK", "TURN_RIGHT"),
    ("SELECT_NEXT_WEAPON",),
    ("SELECT_PREV_WEAPON",),
)


def normalize_action_table(
    actions: Sequence[Sequence[str]],
    *,
    buttons: Sequence[str] = DEATHMATCH_BUTTONS,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...], str]:
    """Normalize and hash actions exactly like the Turbo Vector API v1 contract."""

    if not actions:
        raise ValueError("action table must contain at least one action")
    indices = {name: index for index, name in enumerate(buttons)}
    normalized: list[tuple[str, ...]] = []
    meanings: list[str] = []
    masks: list[tuple[int]] = []
    seen: set[int] = set()
    for action_index, raw_action in enumerate(actions):
        if isinstance(raw_action, (str, bytes, bytearray)):
            raise TypeError(f"action {action_index} must be a sequence of button labels")
        labels = tuple(str(label) for label in raw_action)
        if len(labels) != len(set(labels)):
            raise ValueError(f"action {action_index} contains duplicate buttons")
        try:
            mask = sum(1 << indices[label] for label in labels)
        except KeyError as exc:
            raise ValueError(
                f"action {action_index} contains unknown button {exc.args[0]!r}"
            ) from exc
        if mask in seen:
            raise ValueError(f"action {action_index} duplicates an earlier action")
        seen.add(mask)
        normalized.append(labels)
        meanings.append("noop" if not labels else "_".join(label.lower() for label in labels))
        masks.append((mask,))
    payload = json.dumps(masks, separators=(",", ":"), ensure_ascii=True)
    return tuple(normalized), tuple(meanings), hashlib.sha256(payload.encode("ascii")).hexdigest()


DEATHMATCH_ACTIONS, DEATHMATCH_ACTION_MEANINGS, DEATHMATCH_ACTION_TABLE_SHA256 = (
    normalize_action_table(DEATHMATCH_ACTIONS)
)
