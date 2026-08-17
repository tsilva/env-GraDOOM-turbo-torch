"""Play GraDOOM hosted on a remote stream server with original Doom keybindings.

This client runs where the gradoom package itself cannot be imported (the
engine's Triton dependency is Linux-only); the stream server sends the pinned
action table in its hello message. Receive throughput, stream lag, and display
timings are logged every --metrics-interval seconds.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any

_REQUEST = struct.Struct("!BBB")
_FLAG_RESET = 0x01
_FLAG_QUIT = 0x02

_CONTROLS = """Controls (original Doom):
  Up / Down             move forward / backward
  Left / Right          turn left / right
  Alt + Left / Right    strafe left / right
  , / .                 strafe left / right
  Ctrl / Space          fire
  Shift + Up            run forward
  1-6                   select weapon
  Q / E                 previous / next weapon
  R                     restart with a new seed
  Esc                   quit
"""


@dataclass(frozen=True, slots=True)
class ControlState:
    """Held controls used to select one action from the certified profile."""

    attack: bool = False
    forward: bool = False
    backward: bool = False
    strafe_left: bool = False
    strafe_right: bool = False
    turn_left: bool = False
    turn_right: bool = False
    run: bool = False


def _select_action(
    controls: ControlState,
    action_index: dict[tuple[str, ...], int],
    weapon_action: int | None = None,
) -> int:
    """Resolve held keys into the closest action in the pinned 17-action table."""

    if weapon_action is not None:
        return weapon_action

    forward = controls.forward and not controls.backward
    backward = controls.backward and not controls.forward
    strafe_left = controls.strafe_left and not controls.strafe_right
    strafe_right = controls.strafe_right and not controls.strafe_left
    turn_left = controls.turn_left and not controls.turn_right
    turn_right = controls.turn_right and not controls.turn_left

    if controls.attack:
        if forward:
            return action_index[("ATTACK", "MOVE_FORWARD")]
        if backward:
            return action_index[("ATTACK", "MOVE_BACKWARD")]
        if strafe_left:
            return action_index[("ATTACK", "MOVE_LEFT")]
        if strafe_right:
            return action_index[("ATTACK", "MOVE_RIGHT")]
        if turn_left:
            return action_index[("ATTACK", "TURN_LEFT")]
        if turn_right:
            return action_index[("ATTACK", "TURN_RIGHT")]
        return action_index[("ATTACK",)]

    if forward:
        buttons = ("SPEED", "MOVE_FORWARD") if controls.run else ("MOVE_FORWARD",)
        return action_index[buttons]
    if backward:
        return action_index[("MOVE_BACKWARD",)]
    if strafe_left:
        return action_index[("MOVE_LEFT",)]
    if strafe_right:
        return action_index[("MOVE_RIGHT",)]
    if turn_left:
        return action_index[("TURN_LEFT",)]
    if turn_right:
        return action_index[("TURN_RIGHT",)]
    return action_index[()]


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def _recv_hello(connection: socket.socket) -> dict[str, Any]:
    (length,) = struct.unpack("!I", _recv_exact(connection, 4))
    return json.loads(_recv_exact(connection, length))


def _encode_request(action: int, flags: int, weapon_key: int) -> bytes:
    return _REQUEST.pack(action, flags, weapon_key)


def _step_reply_header(signal_count: int) -> struct.Struct:
    # done byte, signal floats, and the server's send timestamp (protocol 3).
    return struct.Struct(f"!B{signal_count}fd")


class _StreamClock:
    """Estimates stream lag from server send timestamps.

    The machines' monotonic clocks are unrelated, so anchor on the best
    observed delay: lag above that floor is queueing backlog on the wire.
    """

    def __init__(self) -> None:
        self._best_delay: float | None = None

    def lag(self, sent_at: float) -> float:
        delay = time.monotonic() - sent_at
        if self._best_delay is None or delay < self._best_delay:
            self._best_delay = delay
        return delay - self._best_delay


def _recv_reply(
    connection: socket.socket,
    header: struct.Struct,
    frame_bytes: int,
    *,
    compressed: bool = False,
) -> tuple[bool, list[float], bytes, float]:
    head = _recv_exact(connection, header.size + 4)
    values = header.unpack(head[: header.size])
    (payload_length,) = struct.unpack("!I", head[header.size :])
    payload = _recv_exact(connection, payload_length)
    if compressed:
        payload = zlib.decompress(payload)
    if len(payload) != frame_bytes:
        raise ConnectionError(f"bad frame payload: {len(payload)} != {frame_bytes}")
    return bool(values[0]), list(values[1:-1]), payload, float(values[-1])


def _pressed_controls(keys: Any, pygame: Any) -> ControlState:
    strafe_modifier = bool(keys[pygame.K_LALT] or keys[pygame.K_RALT])
    return ControlState(
        attack=bool(keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL] or keys[pygame.K_SPACE]),
        forward=bool(keys[pygame.K_UP]),
        backward=bool(keys[pygame.K_DOWN]),
        strafe_left=bool(keys[pygame.K_COMMA] or (keys[pygame.K_LEFT] and strafe_modifier)),
        strafe_right=bool(keys[pygame.K_PERIOD] or (keys[pygame.K_RIGHT] and strafe_modifier)),
        turn_left=bool(keys[pygame.K_LEFT]) and not strafe_modifier,
        turn_right=bool(keys[pygame.K_RIGHT]) and not strafe_modifier,
        run=bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]),
    )


def _draw_frame(
    pygame: Any,
    screen: Any,
    frame: bytes,
    width: int,
    height: int,
    palette: list[tuple[int, int, int]] | None,
) -> None:
    if palette is None:
        native = pygame.image.frombuffer(frame, (width, height), "RGB")
    else:
        indexed = pygame.image.frombuffer(frame, (width, height), "P")
        indexed.set_palette(palette)
        native = indexed.convert()
    scaled = pygame.transform.scale(native, screen.get_size())
    screen.blit(scaled, (0, 0))
    pygame.display.flip()


def _caption(signal_names: list[str], signals: list[float]) -> str:
    by_name = dict(zip(signal_names, signals, strict=True))
    return (
        "GraDOOM | "
        f"kills {int(by_name['killcount'])}  "
        f"health {int(by_name['health'])}  "
        f"armor {int(by_name['armor'])}  "
        f"ammo {int(by_name['selected_weapon_ammo'])}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play GraDOOM hosted on a remote stream server (tools/stream_server.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="beast-3.local", help="stream server host")
    parser.add_argument("--port", type=_positive_int, default=6666, help="stream server port")
    parser.add_argument("--scale", type=_positive_int, default=3, help="integer window scale")
    parser.add_argument(
        "--fps",
        type=_positive_float,
        default=60.0,
        help="input/display poll rate; the server streams at real-time Doom tics",
    )
    parser.add_argument(
        "--metrics-interval",
        type=_nonnegative_float,
        default=5.0,
        help="seconds between [play] timing/throughput log lines; 0 disables",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import pygame
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise SystemExit("play_remote.py requires pygame-ce; run `uv sync --group dev`") from exc

    pygame.init()
    try:
        with socket.create_connection((args.host, args.port)) as connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            hello = _recv_hello(connection)
            if hello.get("protocol") != 3:
                raise SystemExit(
                    f"server speaks protocol {hello.get('protocol')}, expected 3; "
                    "update play_remote.py and tools/stream_server.py together"
                )
            width, height = hello["width"], hello["height"]
            action_index = {tuple(buttons): index for index, buttons in enumerate(hello["actions"])}
            next_weapon = action_index[("SELECT_NEXT_WEAPON",)]
            previous_weapon = action_index[("SELECT_PREV_WEAPON",)]
            header = _step_reply_header(len(hello["signals"]))
            compressed = hello.get("encoding", "").startswith("zlib")
            palette_list = hello.get("palette")
            palette = (
                [tuple(palette_list[i : i + 3]) for i in range(0, len(palette_list), 3)]
                if palette_list is not None
                else None
            )
            screen = pygame.display.set_mode((width * args.scale, height * args.scale))
            pygame.display.set_caption("GraDOOM")

            latest: dict[str, Any] = {"frame": None, "signals": None, "dead": None}
            stats = {
                "frames": 0,
                "bytes": 0,
                "lag_s": 0.0,
                "lag_max": 0.0,
                "draws": 0,
                "draw_s": 0.0,
                "inputs": 0,
            }
            lock = threading.Lock()
            stream_clock = _StreamClock()

            def receive() -> None:
                try:
                    while True:
                        _, signals, frame, sent_at = _recv_reply(
                            connection, header, hello["frame_bytes"], compressed=compressed
                        )
                        lag = stream_clock.lag(sent_at)
                        with lock:
                            latest["signals"] = signals
                            latest["frame"] = frame
                            stats["frames"] += 1
                            stats["bytes"] += len(frame)
                            stats["lag_s"] += lag
                            stats["lag_max"] = max(stats["lag_max"], lag)
                except (ConnectionError, OSError) as exc:
                    with lock:
                        latest["dead"] = exc

            receiver = threading.Thread(target=receive, daemon=True)
            receiver.start()
            while True:
                with lock:
                    frame, dead = latest["frame"], latest["dead"]
                if frame is not None:
                    break
                if dead is not None:
                    raise dead
                time.sleep(0.005)

            print(_CONTROLS)
            clock = pygame.time.Clock()
            last_sent: tuple[int, int, int] | None = None
            metrics_start = time.monotonic()
            running = True
            while running:
                flags = 0
                weapon_key = 0
                weapon_action: int | None = None
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            flags |= _FLAG_RESET
                        elif event.key == pygame.K_q:
                            weapon_action = previous_weapon
                        elif event.key == pygame.K_e:
                            weapon_action = next_weapon
                        elif pygame.K_1 <= event.key <= pygame.K_6:
                            weapon_key = event.key - pygame.K_0
                if not running:
                    flags |= _FLAG_QUIT

                controls = _pressed_controls(pygame.key.get_pressed(), pygame)
                action = _select_action(controls, action_index, weapon_action)
                message = (action, flags, weapon_key)
                if message != last_sent:
                    connection.sendall(_encode_request(*message))
                    last_sent = message
                    with lock:
                        stats["inputs"] += 1
                if not running:
                    break

                with lock:
                    frame = latest["frame"]
                    signals = latest["signals"]
                    dead = latest["dead"]
                if dead is not None:
                    raise dead
                if frame is not None:
                    draw_start = time.perf_counter()
                    _draw_frame(pygame, screen, frame, width, height, palette)
                    draw_end = time.perf_counter()
                    with lock:
                        stats["draws"] += 1
                        stats["draw_s"] += draw_end - draw_start
                if signals is not None:
                    pygame.display.set_caption(_caption(hello["signals"], signals))
                clock.tick(args.fps)

                now = time.monotonic()
                elapsed = now - metrics_start
                if args.metrics_interval > 0 and elapsed >= args.metrics_interval:
                    with lock:
                        frames, nbytes = stats["frames"], stats["bytes"]
                        lag_s, lag_max = stats["lag_s"], stats["lag_max"]
                        draws, draw_s, inputs = stats["draws"], stats["draw_s"], stats["inputs"]
                        stats["frames"] = stats["bytes"] = stats["inputs"] = 0
                        stats["lag_s"] = stats["lag_max"] = 0.0
                        stats["draws"] = 0
                        stats["draw_s"] = 0.0
                    print(
                        f"[play] recv {frames / elapsed:.1f} fps "
                        f"{nbytes / elapsed / 1024:.0f} KiB/s | "
                        f"lag {1e3 * lag_s / max(frames, 1):.0f}ms avg "
                        f"{1e3 * lag_max:.0f}ms max | "
                        f"draw {1e3 * draw_s / max(draws, 1):.2f}ms "
                        f"{draws / elapsed:.1f} presents/s | "
                        f"inputs {inputs / elapsed:.1f}/s",
                        flush=True,
                    )
                    metrics_start = now
    except (ConnectionError, OSError) as exc:
        print(f"connection to {args.host}:{args.port} failed: {exc}")
        return 1
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
