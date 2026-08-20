"""Record the unprocessed 320x240 RGB24 env-Doom-turbo-torch deathmatch renderer."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import torch

from env_doom_turbo_torch.actions import DEATHMATCH_BUTTONS
from env_doom_turbo_torch.engine import TorchDeathmatchEngine
from env_doom_turbo_torch.scenario import compile_deathmatch_scenario


def _buttons(frame: int, frame_count: int) -> torch.Tensor:
    buttons = torch.zeros((1, len(DEATHMATCH_BUTTONS)), dtype=torch.bool)
    progress = frame / frame_count
    if 0.12 <= progress < 0.42:
        buttons[0, DEATHMATCH_BUTTONS.index("MOVE_FORWARD")] = True
    elif 0.42 <= progress < 0.62:
        buttons[0, DEATHMATCH_BUTTONS.index("ATTACK")] = True
        buttons[0, DEATHMATCH_BUTTONS.index("TURN_RIGHT")] = True
    elif 0.62 <= progress < 0.76:
        buttons[0, DEATHMATCH_BUTTONS.index("ATTACK")] = True
    elif progress >= 0.76:
        buttons[0, DEATHMATCH_BUTTONS.index("ATTACK")] = True
        buttons[0, DEATHMATCH_BUTTONS.index("MOVE_FORWARD")] = True
        buttons[0, DEATHMATCH_BUTTONS.index("TURN_LEFT")] = True
    return buttons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=17.5)
    args = parser.parse_args()

    scenario = compile_deathmatch_scenario(args.scenario, args.iwad)
    engine = TorchDeathmatchEngine(
        scenario,
        1,
        device=torch.device("cpu"),
        frame_skip=1,
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([args.seed]))
    # Begin at the reference pit-comparison pose used during renderer parity work.
    engine.x.fill_(668.9710083007812)
    engine.y.fill_(393.1371307373047)
    engine.z.zero_()
    engine.view_z.fill_(41.0)
    engine.view_height.fill_(41.0)
    engine.delta_view_height.zero_()
    engine.angle.fill_(math.radians(145.95336917460742))

    frame_count = round(args.seconds * args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = (
        "ffmpeg",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "320x240",
        "-framerate",
        str(args.fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "10",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
        "-y",
        str(args.output),
    )
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg did not expose its input pipe")
    try:
        for frame_index in range(frame_count):
            buttons = _buttons(frame_index, frame_count)
            for _ in range(2):
                engine.step(buttons)
                if bool(engine.pending_reset[0]):
                    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([args.seed]))
            frame = engine.render_native_frame(include_hud=True)[0]
            encoder.stdin.write(frame.numpy().tobytes())
    finally:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the native renderer capture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
