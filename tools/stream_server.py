"""Stream one GraDOOM deathmatch lane to a remote keyboard player.

The server steps the environment on its own real-time clock and pushes
zlib-compressed frames; the client sends input only when it changes. This
keeps the frame rate independent of the connection's round-trip latency.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from gradoom import GraDoomVecEnv
from gradoom.actions import DEATHMATCH_ACTIONS
from gradoom.engine import DEVICE_SIGNAL_NAMES

_REQUEST = struct.Struct("!BBB")
_FLAG_RESET = 0x01
_FLAG_QUIT = 0x02

_NOOP = DEATHMATCH_ACTIONS.index(())
_NEXT_WEAPON = DEATHMATCH_ACTIONS.index(("SELECT_NEXT_WEAPON",))
_PREVIOUS_WEAPON = DEATHMATCH_ACTIONS.index(("SELECT_PREV_WEAPON",))

_WEAPON_SLOTS = 6
_MAX_WEAPON_PRESSES = _WEAPON_SLOTS * 8
_SIGNAL_COUNT = len(DEVICE_SIGNAL_NAMES)
_SELECTED_WEAPON_SIGNAL = DEVICE_SIGNAL_NAMES.index("selected_weapon")
_WEAPON_OWNED_SIGNAL = DEVICE_SIGNAL_NAMES.index("weapon1")


@dataclass(slots=True)
class _ClientInput:
    """Latest client input state, applied at every Doom tic until replaced."""

    action: int = _NOOP
    quit: bool = False
    reset: bool = False
    weapon_key: int = 0


def _parse_requests(state: _ClientInput, data: bytes) -> None:
    """Apply complete 3-byte client requests to the input state."""

    for offset in range(0, len(data) - len(data) % _REQUEST.size, _REQUEST.size):
        action, flags, weapon_key = _REQUEST.unpack(data[offset : offset + _REQUEST.size])
        state.action = action
        state.quit = state.quit or bool(flags & _FLAG_QUIT)
        state.reset = state.reset or bool(flags & _FLAG_RESET)
        if weapon_key:
            state.weapon_key = weapon_key


def _drain_requests(
    connection: socket.socket, state: _ClientInput, pending: bytearray
) -> bytearray:
    """Read whatever the client sent and apply complete requests."""

    try:
        chunk = connection.recv(4096)
    except BlockingIOError:
        return pending
    if not chunk:
        raise ConnectionError("client closed the connection")
    pending.extend(chunk)
    complete = len(pending) - len(pending) % _REQUEST.size
    _parse_requests(state, bytes(pending[:complete]))
    del pending[:complete]
    return pending


@dataclass(slots=True)
class _WeaponSelect:
    """Number-key weapon selection pressed one edge-latched tic at a time."""

    target: int = 0
    presses: int = 0
    release_next: bool = True


def _recv_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _encode_hello(metadata: dict[str, Any]) -> bytes:
    payload = json.dumps(metadata).encode()
    return struct.pack("!I", len(payload)) + payload


def _step_reply_header(signal_count: int) -> struct.Struct:
    return struct.Struct(f"!B{signal_count}f")


def _encode_step_reply(
    header: struct.Struct,
    done: bool,
    signals: list[float],
    frame: bytes,
) -> bytes:
    payload = zlib.compress(frame, 1)
    return header.pack(done, *signals) + struct.pack("!I", len(payload)) + payload


def _slot_presses(current: int, target: int, owned: list[bool], direction: int) -> int:
    """Count cycle presses needed to reach a target slot, skipping unowned slots."""

    slot = current - 1
    goal = target - 1
    presses = 0
    for _ in range(_WEAPON_SLOTS):
        slot = (slot + direction) % _WEAPON_SLOTS
        if owned[slot]:
            presses += 1
        if slot == goal:
            return presses
    return 0


def _weapon_direction(current: int, target: int, owned: list[bool]) -> int:
    """Pick the shorter cycle direction toward an owned slot: +1 next, -1 prev, 0 give up."""

    if not 1 <= current <= _WEAPON_SLOTS:
        return 0
    if not 1 <= target <= _WEAPON_SLOTS or not owned[target - 1] or target == current:
        return 0
    forward = _slot_presses(current, target, owned, 1)
    backward = _slot_presses(current, target, owned, -1)
    return 1 if forward <= backward else -1


def _weapon_select_action(state: _WeaponSelect, signals: list[float]) -> int | None:
    """Resolve an in-flight number-key selection into this tic's action override."""

    if not state.target:
        return None
    current = int(signals[_SELECTED_WEAPON_SIGNAL])
    owned = [value > 0 for value in signals[_WEAPON_OWNED_SIGNAL : _WEAPON_OWNED_SIGNAL + 6]]
    direction = _weapon_direction(current, state.target, owned)
    if direction == 0 or state.presses >= _MAX_WEAPON_PRESSES:
        state.target = 0
        return None
    state.release_next = not state.release_next
    if state.release_next:
        return _NOOP  # the engine's weapon latch only accepts fresh presses
    state.presses += 1
    return _NEXT_WEAPON if direction > 0 else _PREVIOUS_WEAPON


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream GraDOOM's deathmatch-p1-v1 environment to a remote player.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iwad",
        type=Path,
        help="Doom II or Freedoom IWAD (or set GRADOOM_IWAD)",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        help="ViZDoom deathmatch.wad (or set GRADOOM_DEATHMATCH_WAD)",
    )
    parser.add_argument("--device", help="Torch device; defaults to CUDA when available")
    parser.add_argument("--seed", type=int, default=0, help="initial episode seed")
    parser.add_argument("--bind", default="0.0.0.0", help="interface to listen on")
    parser.add_argument("--port", type=_positive_int, default=6666, help="TCP port to listen on")
    parser.add_argument(
        "--compile-engine",
        action="store_true",
        help="compile the engine with torch.compile (CUDA only)",
    )
    parser.add_argument(
        "--allow-unpinned-scenario",
        action="store_true",
        help="allow a non-certified deathmatch scenario WAD",
    )
    return parser


def _create_env(args: argparse.Namespace) -> GraDoomVecEnv:
    return GraDoomVecEnv(
        game="VizdoomDeathmatch-v1",
        scenario=args.scenario,
        rom_path=None if args.iwad is None else str(args.iwad),
        num_envs=1,
        device=args.device,
        transport="torch",
        use_restricted_actions=DEATHMATCH_ACTIONS,
        render_mode="rgb_array",
        obs_copy="unsafe_view",
        obs_resize=(84, 84),
        obs_crop=(0, 32, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        frame_skip=2,
        frame_stack=4,
        maxpool_last_two=False,
        noop_reset_max=0,
        use_fire_reset=False,
        sticky_action_prob=0.0,
        reward_clip=False,
        compile_engine=args.compile_engine,
        require_pinned_scenario=not args.allow_unpinned_scenario,
    )


def _reset_lane(env: GraDoomVecEnv, seed: int) -> list[float]:
    mask = torch.ones(1, device=env.device, dtype=torch.bool)
    seeds = torch.tensor([seed], device=env.device, dtype=torch.int64)
    _, signals = env.reset_device(mask, seeds)
    return signals[0].detach().to("cpu").tolist()


def _render_frame(env: GraDoomVecEnv) -> Any:
    frame = env.render()
    if frame is None:  # pragma: no cover - explicit render mode invariant
        raise RuntimeError("rgb_array rendering did not produce a frame")
    return frame


def _serve_connection(
    env: GraDoomVecEnv,
    connection: socket.socket,
    seed: int,
) -> int:
    """Play one client until quit or disconnect; return the next episode seed."""

    signals = _reset_lane(env, seed)
    frame = _render_frame(env)
    header = _step_reply_header(_SIGNAL_COUNT)
    connection.sendall(
        _encode_hello(
            {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2],
                "frame_bytes": frame.nbytes,
                "encoding": "zlib",
                "protocol": 2,
                "fps": env.metadata["render_fps"] / env.frame_skip,
                "signals": list(DEVICE_SIGNAL_NAMES),
                "actions": [list(buttons) for buttons in DEATHMATCH_ACTIONS],
            }
        )
    )
    connection.sendall(_encode_step_reply(header, False, signals, frame.tobytes()))

    action = torch.zeros(1, dtype=torch.int64, device=env.device)
    state = _ClientInput()
    weapon_select = _WeaponSelect()
    pending = bytearray()
    done = False
    tic_period = env.frame_skip / env.metadata["render_fps"]
    next_tic = time.monotonic()
    while not state.quit:
        connection.setblocking(False)
        pending = _drain_requests(connection, state, pending)
        connection.setblocking(True)

        if state.reset or done:
            seed += 1
            signals = _reset_lane(env, seed)
            weapon_select = _WeaponSelect()
            state.reset = False
            done = False
        else:
            if state.weapon_key:
                weapon_select = _WeaponSelect(target=state.weapon_key)
                state.weapon_key = 0
            override = _weapon_select_action(weapon_select, signals)
            action.fill_(override if override is not None else state.action)
            transition = env.step_device(action)
            signals = transition.signals[0].detach().to("cpu").tolist()
            done = bool((transition.terminated | transition.truncated)[0].item())

        reply = _encode_step_reply(header, done, signals, _render_frame(env).tobytes())
        connection.sendall(reply)
        next_tic += tic_period
        delay = next_tic - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tic = time.monotonic()
    return seed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    env = _create_env(args)
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind, args.port))
            listener.listen(1)
            print(f"GraDOOM stream server listening on {args.bind}:{args.port}", flush=True)
            seed = args.seed
            while True:
                connection, address = listener.accept()
                print(f"player connected from {address[0]}:{address[1]}", flush=True)
                try:
                    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    seed = _serve_connection(env, connection, seed)
                except (ConnectionError, OSError) as exc:
                    print(f"player connection dropped: {exc}", flush=True)
                finally:
                    connection.close()
                print("player disconnected", flush=True)
    except KeyboardInterrupt:  # pragma: no cover - operator shutdown path
        pass
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
