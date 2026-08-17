"""Stream one GraDOOM deathmatch lane to a remote keyboard player.

The server steps the environment on its own real-time clock, replaying one
CUDA-graphed step+reset transaction per Doom tic, and pushes zlib-compressed
frames; the client sends input only when it changes. This keeps the frame
rate independent of the connection's round-trip latency. Compression and
sending run on a worker thread that keeps only the latest frame, so a slow
connection drops frames instead of slowing the game clock. Per-tic phase
timings and stream throughput are logged every --metrics-interval seconds.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import struct
import threading
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


class _StreamSender:
    """Compresses and sends replies on a worker thread, keeping only the latest.

    The tic loop never waits on zlib or the network: when the sender falls
    behind, pending frames are coalesced (dropped) so the game keeps real time.
    """

    def __init__(self, connection: socket.socket, header: struct.Struct) -> None:
        self._connection = connection
        self._header = header
        self._lock = threading.Lock()
        self._wakeup = threading.Event()
        self._latest: tuple[bool, list[float], bytes] | None = None
        self._stop = False
        self._dead: OSError | None = None
        self.sent = 0
        self.drops = 0
        self.compress_s = 0.0
        self.send_s = 0.0
        self.bytes_sent = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="stream-sender")

    def start(self) -> None:
        self._thread.start()

    def submit(self, done: bool, signals: list[float], frame: bytes) -> None:
        with self._lock:
            if self._latest is not None:
                self.drops += 1
            self._latest = (done, signals, frame)
        self._wakeup.set()

    @property
    def dead(self) -> OSError | None:
        with self._lock:
            return self._dead

    def snapshot(self) -> tuple[int, int, float, float, int]:
        """Return and reset the window counters: sent, drops, compress_s, send_s, bytes."""
        with self._lock:
            stats = (self.sent, self.drops, self.compress_s, self.send_s, self.bytes_sent)
            self.sent = self.drops = 0
            self.compress_s = self.send_s = 0.0
            self.bytes_sent = 0
        return stats

    def stop(self) -> None:
        with self._lock:
            self._stop = True
        self._wakeup.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            self._wakeup.wait()
            self._wakeup.clear()
            with self._lock:
                item = self._latest
                self._latest = None
                stop = self._stop
            if item is not None:
                done, signals, frame = item
                compress_start = time.perf_counter()
                reply = _encode_step_reply(self._header, done, signals, frame)
                compress_end = time.perf_counter()
                try:
                    self._connection.sendall(reply)
                except OSError as exc:
                    with self._lock:
                        self._dead = exc
                    return
                send_end = time.perf_counter()
                with self._lock:
                    self.compress_s += compress_end - compress_start
                    self.send_s += send_end - compress_end
                    self.sent += 1
                    self.bytes_sent += len(reply)
            if stop:
                return


class _TicMetrics:
    """Windowed accumulator that logs where each Doom tic's time goes."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self._window_start = now
        self.tics = 0
        self.step_s = 0.0
        self.render_s = 0.0
        self.submit_s = 0.0
        self.sleep_s = 0.0
        self.late = 0

    def record(
        self, step_s: float, render_s: float, submit_s: float, sleep_s: float, late: bool
    ) -> None:
        self.tics += 1
        self.step_s += step_s
        self.render_s += render_s
        self.submit_s += submit_s
        self.sleep_s += sleep_s
        self.late += int(late)

    def maybe_report(self, sender: _StreamSender) -> None:
        now = time.monotonic()
        if self.interval <= 0 or now - self._window_start < self.interval:
            return
        self._report(now, sender.snapshot())

    def _report(self, now: float, sender_stats: tuple[int, int, float, float, int]) -> None:
        elapsed = now - self._window_start
        sent, drops, compress_s, send_s, bytes_sent = sender_stats
        tics = max(self.tics, 1)
        sent_denom = max(sent, 1)
        print(
            f"[stream] {self.tics / elapsed:.1f} tics/s (target ~17.5) "
            f"step {1e3 * self.step_s / tics:.2f}ms "
            f"render {1e3 * self.render_s / tics:.2f}ms "
            f"submit {1e3 * self.submit_s / tics:.2f}ms "
            f"sleep {1e3 * self.sleep_s / tics:.2f}ms "
            f"late {self.late} | "
            f"sent {sent / elapsed:.1f}/s drops {drops / elapsed:.1f}/s "
            f"compress {1e3 * compress_s / sent_denom:.2f}ms "
            f"send {1e3 * send_s / sent_denom:.2f}ms "
            f"{bytes_sent / elapsed / 1024:.0f} KiB/s",
            flush=True,
        )
        self._reset(now)


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


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile engine phases and replay the CUDA-graphed step transaction; "
        "roughly 25x faster single-lane stepping, at the cost of a one-time warmup "
        "(CUDA only; automatically disabled otherwise)",
    )
    parser.add_argument(
        "--allow-unpinned-scenario",
        action="store_true",
        help="allow a non-certified deathmatch scenario WAD",
    )
    parser.add_argument(
        "--metrics-interval",
        type=_nonnegative_float,
        default=5.0,
        help="seconds between [stream] timing/throughput log lines; 0 disables",
    )
    return parser


def _create_env(args: argparse.Namespace) -> GraDoomVecEnv:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    compile_engine = args.compile_engine
    if compile_engine and not device.startswith("cuda"):
        print("CUDA unavailable; --no-compile-engine is implied", flush=True)
        compile_engine = False
    return GraDoomVecEnv(
        game="VizdoomDeathmatch-v1",
        scenario=args.scenario,
        rom_path=None if args.iwad is None else str(args.iwad),
        num_envs=1,
        device=device,
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
        compile_engine=compile_engine,
        require_pinned_scenario=not args.allow_unpinned_scenario,
    )


def _warm_up(env: GraDoomVecEnv) -> None:
    """Pay the one-time compile/capture cost before the first client connects."""

    if env.engine_backend != "torch-compiled-cudagraph":
        return
    print(
        "compiling engine phases and capturing the CUDA graph (one-time warmup)...",
        flush=True,
    )
    start = time.monotonic()
    mask = torch.ones(1, device=env.device, dtype=torch.bool)
    seeds = torch.zeros(1, device=env.device, dtype=torch.int64)
    env.reset_device(mask, seeds)
    actions = torch.zeros(1, device=env.device, dtype=torch.int64)  # noop
    env.step_and_reset_device(actions, seeds)
    torch.cuda.synchronize(env.device)
    print(f"warmup done in {time.monotonic() - start:.0f}s", flush=True)


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
    metrics_interval: float,
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
    reset_seeds = torch.zeros(1, dtype=torch.int64, device=env.device)
    state = _ClientInput()
    weapon_select = _WeaponSelect()
    pending = bytearray()
    done = False
    last_seed = seed
    next_episode_seed = seed + 1
    tic_period = env.frame_skip / env.metadata["render_fps"]
    next_tic = time.monotonic()
    sender = _StreamSender(connection, header)
    sender.start()
    metrics = _TicMetrics(metrics_interval)
    try:
        while not state.quit:
            dead_error = sender.dead
            if dead_error is not None:
                raise ConnectionError(f"stream sender failed: {dead_error}")
            tic_start = time.perf_counter()
            connection.setblocking(False)
            pending = _drain_requests(connection, state, pending)
            connection.setblocking(True)

            if state.reset:
                signals = _reset_lane(env, next_episode_seed)
                last_seed = next_episode_seed
                next_episode_seed += 1
                weapon_select = _WeaponSelect()
                state.reset = False
                done = False
            else:
                if state.weapon_key:
                    weapon_select = _WeaponSelect(target=state.weapon_key)
                    state.weapon_key = 0
                override = _weapon_select_action(weapon_select, signals)
                action.fill_(override if override is not None else state.action)
                reset_seeds.fill_(next_episode_seed)
                # One CUDA graph replay: step plus atomic reset of terminal lanes.
                transition = env.step_and_reset_device(action, reset_seeds)
                signals = transition.signals[0].detach().to("cpu").tolist()
                done = bool((transition.terminated | transition.truncated)[0].item())
                if done:
                    last_seed = next_episode_seed
                    next_episode_seed += 1

            step_end = time.perf_counter()
            reply_frame = _render_frame(env)
            render_end = time.perf_counter()
            sender.submit(done, signals, reply_frame.tobytes())
            submit_end = time.perf_counter()
            next_tic += tic_period
            delay = next_tic - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tic = time.monotonic()
            metrics.record(
                step_end - tic_start,
                render_end - step_end,
                submit_end - render_end,
                max(delay, 0.0),
                delay <= 0,
            )
            metrics.maybe_report(sender)
    finally:
        sender.stop()
    return last_seed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    env = _create_env(args)
    _warm_up(env)
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
                    with contextlib.suppress(OSError):  # best effort; OS may cap the buffer
                        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
                    seed = _serve_connection(env, connection, seed, args.metrics_interval)
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
