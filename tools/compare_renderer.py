"""Compare GraDOOM and ViZDoom rendering at identical seeded player poses."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from gradoom.engine import TorchDeathmatchEngine
from gradoom.scenario import compile_deathmatch_scenario


def _write_comparison(
    output: Path,
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> None:
    difference = torch.abs(reference - actual).mul(3).clamp(0, 255)
    comparison = torch.cat((reference, actual, difference), dim=1).to(torch.uint8)
    subprocess.run(
        (
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "960x240",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-y",
            str(output),
        ),
        input=comparison.numpy().tobytes(),
        check=True,
    )


def _reference_frame(
    config: Path,
    iwad: Path,
    seed: int,
    settle_tics: int,
) -> tuple[torch.Tensor, float, float, float, float, float]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("compare_renderer.py requires the reference vizdoom package") from exc

    game = vzd.DoomGame()
    game.load_config(str(config))
    game.set_doom_game_path(str(iwad))
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    variables = (
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.POSITION_Z,
        vzd.GameVariable.ANGLE,
        vzd.GameVariable.CAMERA_POSITION_Z,
    )
    for variable in variables:
        game.add_available_game_variable(variable)
    game.set_seed(seed)
    game.init()
    try:
        game.new_episode()
        noop = [0.0] * len(game.get_available_buttons())
        for _ in range(settle_tics):
            game.make_action(noop, 1)
        state = game.get_state()
        if state is None:
            raise RuntimeError("ViZDoom did not expose an initial state")
        raw = np.asarray(state.screen_buffer).copy()
        if raw.shape != (240, 320, 3):
            raise RuntimeError(f"expected a 240x320 RGB24 frame, got {raw.shape}")
        frame = torch.from_numpy(raw).to(torch.float32)
        x, y, z, angle, camera_z = (
            float(game.get_game_variable(variable)) for variable in variables
        )
        return frame, x, y, z, angle, camera_z
    finally:
        game.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1_337))
    parser.add_argument("--settle-tics", type=int, default=16)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-unpinned-scenario", action="store_true")
    args = parser.parse_args()

    scenario = compile_deathmatch_scenario(
        args.scenario,
        args.iwad,
        require_pinned_scenario=not args.allow_unpinned_scenario,
    )
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    mask = torch.ones(1, dtype=torch.bool)
    records: list[dict[str, float | int]] = []
    for seed in args.seeds:
        reference, x, y, z, angle_degrees, camera_z = _reference_frame(
            args.config,
            args.iwad,
            seed,
            args.settle_tics,
        )
        engine.reset(mask, torch.tensor([seed], dtype=torch.int64))
        engine.x.fill_(x)
        engine.y.fill_(y)
        engine.z.fill_(z)
        engine.view_z.fill_(camera_z)
        engine.view_height.fill_(camera_z - z)
        engine.angle.fill_(angle_degrees * math.pi / 180.0)
        engine.episode_time.fill_(args.settle_tics + 1)
        engine.weapon_raise_cooldown.zero_()
        actual = engine.render_native_frame(include_hud=True)[0].to(torch.float32)
        flattened = torch.stack((reference.flatten(), actual.flatten()))
        absolute_error = torch.abs(reference - actual)
        if args.output_dir is not None:
            _write_comparison(
                args.output_dir / f"seed-{seed}-reference-actual-diff.png",
                reference,
                actual,
            )
        records.append(
            {
                "actual_mean": float(actual.mean()),
                "angle": angle_degrees,
                "camera_z": camera_z,
                "channel_mae_b": float(absolute_error[..., 2].mean()),
                "channel_mae_g": float(absolute_error[..., 1].mean()),
                "channel_mae_r": float(absolute_error[..., 0].mean()),
                "correlation": float(torch.corrcoef(flattened)[0, 1]),
                "mae": float(absolute_error.mean()),
                "mae_ceiling": float(absolute_error[:104].mean()),
                "mae_floor": float(absolute_error[104:208].mean()),
                "mae_hud": float(absolute_error[208:].mean()),
                "reference_mean": float(reference.mean()),
                "seed": seed,
                "x": x,
                "y": y,
                "z": z,
            }
        )

    correlations = np.asarray([record["correlation"] for record in records], dtype=np.float64)
    errors = np.asarray([record["mae"] for record in records], dtype=np.float64)
    print(
        json.dumps(
            {
                "iwad_sha256": scenario.iwad_sha256,
                "mean_correlation": float(correlations.mean()),
                "mean_mae": float(errors.mean()),
                "median_correlation": float(np.median(correlations)),
                "median_mae": float(np.median(errors)),
                "records": records,
                "scenario_sha256": scenario.scenario_sha256,
                "schema": "gradoom.renderer-parity.raw-rgb-hud.v1",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
