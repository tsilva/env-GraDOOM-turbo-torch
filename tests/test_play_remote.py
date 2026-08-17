from __future__ import annotations

import runpy
import socket
import struct
import zlib
from pathlib import Path

import pytest

from gradoom.actions import DEATHMATCH_ACTIONS

_ROOT = Path(__file__).parents[1]
_SERVER = runpy.run_path(str(_ROOT / "tools" / "stream_server.py"))
_CLIENT = runpy.run_path(str(_ROOT / "play_remote.py"))

_ACTION_INDEX = {buttons: index for index, buttons in enumerate(DEATHMATCH_ACTIONS)}
ControlState = _CLIENT["ControlState"]
_select_action = _CLIENT["_select_action"]

_NEXT_WEAPON = _SERVER["_NEXT_WEAPON"]
_PREVIOUS_WEAPON = _SERVER["_PREVIOUS_WEAPON"]
_NOOP = _SERVER["_NOOP"]
_SIGNAL_COUNT = _SERVER["_SIGNAL_COUNT"]
_SELECTED_WEAPON = _SERVER["_SELECTED_WEAPON_SIGNAL"]
_WEAPON_OWNED = _SERVER["_WEAPON_OWNED_SIGNAL"]
_MAX_WEAPON_PRESSES = _SERVER["_MAX_WEAPON_PRESSES"]

_weapon_direction = _SERVER["_weapon_direction"]
_weapon_select_action = _SERVER["_weapon_select_action"]
_WeaponSelect = _SERVER["_WeaponSelect"]


def _signals(slot: int, owned: tuple[bool, ...] = (True,) * 6) -> list[float]:
    signals = [0.0] * _SIGNAL_COUNT
    signals[_SELECTED_WEAPON] = float(slot)
    for index, is_owned in enumerate(owned):
        signals[_WEAPON_OWNED + index] = float(is_owned)
    return signals


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
def test_select_action_matches_pinned_deathmatch_actions(
    controls: ControlState,
    expected: int,
) -> None:
    assert _select_action(controls, _ACTION_INDEX) == expected


@pytest.mark.parametrize(
    ("current", "target", "owned", "expected"),
    [
        (2, 4, (True,) * 6, 1),
        (4, 2, (True,) * 6, -1),
        (2, 5, (True,) * 6, 1),  # tie breaks toward next
        (2, 2, (True,) * 6, 0),
        (2, 6, (True, True, True, True, True, False), 0),
        (0, 4, (True,) * 6, 0),
        (2, 5, (True, True, False, False, True, True), 1),  # skips unowned slots
        (5, 2, (True, True, False, False, True, True), -1),  # prev skips unowned slots
    ],
)
def test_weapon_direction(
    current: int,
    target: int,
    owned: tuple[bool, ...],
    expected: int,
) -> None:
    assert _weapon_direction(current, target, list(owned)) == expected


def test_weapon_select_alternates_press_and_release() -> None:
    state = _WeaponSelect(target=4)
    assert _weapon_select_action(state, _signals(2)) == _NEXT_WEAPON
    assert _weapon_select_action(state, _signals(2)) == _NOOP
    assert _weapon_select_action(state, _signals(2)) == _NEXT_WEAPON


def test_weapon_select_uses_shorter_direction() -> None:
    state = _WeaponSelect(target=2)
    assert _weapon_select_action(state, _signals(4)) == _PREVIOUS_WEAPON


def test_weapon_select_stops_on_target_slot() -> None:
    state = _WeaponSelect(target=4)
    assert _weapon_select_action(state, _signals(4)) is None
    assert state.target == 0


def test_weapon_select_ignores_unowned_target() -> None:
    owned = (True, True, True, True, True, False)
    state = _WeaponSelect(target=6)
    assert _weapon_select_action(state, _signals(2, owned)) is None
    assert state.target == 0


def test_weapon_select_gives_up_after_bounded_presses() -> None:
    state = _WeaponSelect(target=4)
    signals = _signals(2)  # slot never advances
    for _ in range(_MAX_WEAPON_PRESSES * 2 + 2):
        result = _weapon_select_action(state, signals)
    assert result is None
    assert state.target == 0


def test_hello_round_trip() -> None:
    server, client = socket.socketpair()
    try:
        metadata = {
            "width": 320,
            "height": 240,
            "channels": 3,
            "frame_bytes": 320 * 240 * 3,
            "fps": 17.5,
            "signals": ["a"],
            "actions": [[], ["ATTACK"]],
        }
        server.sendall(_SERVER["_encode_hello"](metadata))
        assert _CLIENT["_recv_hello"](client) == metadata
    finally:
        server.close()
        client.close()


def test_request_round_trip() -> None:
    server, client = socket.socketpair()
    try:
        client.sendall(_CLIENT["_encode_request"](7, _SERVER["_FLAG_RESET"], 4))
        data = _SERVER["_recv_exact"](server, _SERVER["_REQUEST"].size)
        assert _SERVER["_REQUEST"].unpack(data) == (7, _SERVER["_FLAG_RESET"], 4)
    finally:
        server.close()
        client.close()


def test_parse_requests_applies_latest_state() -> None:
    state = _SERVER["_ClientInput"]()
    data = _CLIENT["_encode_request"](3, 0, 0) + _CLIENT["_encode_request"](
        5, _SERVER["_FLAG_RESET"], 2
    )
    _SERVER["_parse_requests"](state, data)
    assert state.action == 5
    assert state.reset is True
    assert state.quit is False
    assert state.weapon_key == 2


def test_parse_requests_latches_quit_and_ignores_partial() -> None:
    state = _SERVER["_ClientInput"]()
    quit_message = _CLIENT["_encode_request"](1, _SERVER["_FLAG_QUIT"], 0)
    _SERVER["_parse_requests"](state, quit_message + b"\x00")
    assert state.quit is True
    assert state.action == 1

    state = _SERVER["_ClientInput"]()
    _SERVER["_parse_requests"](state, b"\x01")
    assert state.action == _SERVER["_NOOP"]
    assert state.quit is False


def test_step_reply_round_trip() -> None:
    server, client = socket.socketpair()
    try:
        header = _SERVER["_step_reply_header"](_SIGNAL_COUNT)
        signals = [float(index) for index in range(_SIGNAL_COUNT)]
        frame = bytes(range(12))
        server.sendall(_SERVER["_encode_step_reply"](header, True, signals, frame))
        done, received_signals, received_frame = _CLIENT["_recv_reply"](
            client,
            _CLIENT["_step_reply_header"](_SIGNAL_COUNT),
            len(frame),
            compressed=True,
        )
        assert done is True
        assert received_signals == signals
        assert received_frame == frame
    finally:
        server.close()
        client.close()


def test_step_reply_compresses_frame() -> None:
    header = _SERVER["_step_reply_header"](_SIGNAL_COUNT)
    frame = bytes(320 * 240 * 3)
    reply = _SERVER["_encode_step_reply"](header, False, [0.0] * _SIGNAL_COUNT, frame)
    (payload_length,) = struct.unpack("!I", reply[header.size : header.size + 4])
    assert payload_length == len(reply) - header.size - 4
    assert payload_length < len(frame)
    assert zlib.decompress(reply[header.size + 4 :]) == frame


def test_pressed_controls_original_doom_bindings() -> None:
    pygame = pytest.importorskip("pygame")
    from collections import defaultdict

    def keys(*held: int) -> defaultdict[int, bool]:
        pressed: defaultdict[int, bool] = defaultdict(bool)
        for key in held:
            pressed[key] = True
        return pressed

    pressed_controls = _CLIENT["_pressed_controls"]

    controls = pressed_controls(keys(pygame.K_LEFT), pygame)
    assert controls.turn_left and not controls.strafe_left

    controls = pressed_controls(keys(pygame.K_LEFT, pygame.K_LALT), pygame)
    assert controls.strafe_left and not controls.turn_left

    controls = pressed_controls(keys(pygame.K_RIGHT, pygame.K_RALT), pygame)
    assert controls.strafe_right and not controls.turn_right

    controls = pressed_controls(keys(pygame.K_COMMA), pygame)
    assert controls.strafe_left

    controls = pressed_controls(keys(pygame.K_PERIOD), pygame)
    assert controls.strafe_right

    controls = pressed_controls(keys(pygame.K_LCTRL), pygame)
    assert controls.attack

    controls = pressed_controls(keys(pygame.K_SPACE), pygame)
    assert controls.attack

    controls = pressed_controls(keys(pygame.K_UP, pygame.K_LSHIFT), pygame)
    assert controls.forward and controls.run


def test_client_parser_rejects_non_positive_port() -> None:
    with pytest.raises(SystemExit):
        _CLIENT["_parser"]().parse_args(["--port", "0"])


def test_server_parser_rejects_non_positive_port() -> None:
    with pytest.raises(SystemExit):
        _SERVER["_parser"]().parse_args(["--port", "0"])
