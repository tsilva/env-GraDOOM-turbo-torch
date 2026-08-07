from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from gradoom.actions import DEATHMATCH_ACTIONS

_PLAY = runpy.run_path(str(Path(__file__).parents[1] / "play.py"))
ControlState = _PLAY["ControlState"]
_parser = _PLAY["_parser"]
_select_action = _PLAY["_select_action"]


def _action(*buttons: str) -> int:
    return DEATHMATCH_ACTIONS.index(buttons)


@pytest.mark.parametrize(
    ("controls", "expected"),
    [
        (ControlState(), _action()),
        (ControlState(forward=True), _action("MOVE_FORWARD")),
        (ControlState(forward=True, run=True), _action("SPEED", "MOVE_FORWARD")),
        (ControlState(attack=True, turn_right=True), _action("ATTACK", "TURN_RIGHT")),
        (ControlState(attack=True, strafe_left=True), _action("ATTACK", "MOVE_LEFT")),
        (ControlState(forward=True, backward=True), _action()),
    ],
)
def test_select_action_uses_pinned_deathmatch_actions(
    controls: ControlState,
    expected: int,
) -> None:
    assert _select_action(controls) == expected


def test_weapon_event_overrides_held_movement() -> None:
    previous_weapon = _action("SELECT_PREV_WEAPON")
    selected = _select_action(ControlState(forward=True, attack=True), previous_weapon)
    assert selected == previous_weapon


def test_parser_rejects_non_positive_window_scale() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--scale", "0"])
