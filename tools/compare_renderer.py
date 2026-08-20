"""Compare env-Doom-turbo-torch and ViZDoom rendering at identical seeded player poses."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from env_doom_turbo_torch.engine import TorchDeathmatchEngine
from env_doom_turbo_torch.scenario import compile_deathmatch_scenario


def _reference_policy_frame(reference_rgb: torch.Tensor) -> torch.Tensor:
    """Apply the pinned env-ViZDoom-turbo deathmatch observation transform."""

    try:
        from vizdoom_turbo._vizdoom_turbo import preprocess_into
    except ImportError as exc:
        raise RuntimeError(
            "compare_renderer.py requires the reference vizdoom_turbo package"
        ) from exc
    current = reference_rgb.to(torch.uint8).numpy()[None]
    output = np.empty((1, 84, 84, 1), dtype=np.uint8)
    preprocess_into(current, output, [0, 32, 0, 0], True, 0, "area")
    return torch.from_numpy(output[0, ..., 0]).to(torch.float32)


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


def _write_policy_comparison(
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
            "gray",
            "-video_size",
            "252x84",
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
    look_delta: float,
) -> tuple[torch.Tensor, float, float, float, float, float, float]:
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
        vzd.GameVariable.PITCH,
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
        if look_delta:
            look = noop.copy()
            look[game.get_available_buttons().index(vzd.Button.LOOK_UP_DOWN_DELTA)] = look_delta
            game.make_action(look, 1)
        state = game.get_state()
        if state is None:
            raise RuntimeError("ViZDoom did not expose an initial state")
        raw = np.asarray(state.screen_buffer).copy()
        if raw.shape != (240, 320, 3):
            raise RuntimeError(f"expected a 240x320 RGB24 frame, got {raw.shape}")
        frame = torch.from_numpy(raw).to(torch.float32)
        x, y, z, angle, camera_z, pitch = (
            float(game.get_game_variable(variable)) for variable in variables
        )
        return frame, x, y, z, angle, camera_z, pitch
    finally:
        game.close()


def _match_reference_mugshot(
    engine: TorchDeathmatchEngine,
    reference: torch.Tensor,
) -> int:
    """Synchronize Doom's permitted visual-only neutral-face randomness."""

    scores: list[float] = []
    for face_index in range(3):
        engine.mugshot_face_index.fill_(face_index)
        indexed_hud = engine._native_render_hud()[0]
        rgb_hud = engine.map.playpal[indexed_hud.to(torch.int64)].to(torch.float32)
        scores.append(float(torch.abs(reference[-32:] - rgb_hud).sum()))
    matched = min(range(3), key=scores.__getitem__)
    engine.mugshot_face_index.fill_(matched)
    return matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1_337))
    parser.add_argument("--settle-tics", type=int, default=16)
    parser.add_argument(
        "--look-delta",
        type=float,
        default=0.0,
        help="apply one raw LOOK_UP_DOWN_DELTA action after settling",
    )
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
        reference, x, y, z, angle_degrees, camera_z, pitch_degrees = _reference_frame(
            args.config,
            args.iwad,
            seed,
            args.settle_tics,
            args.look_delta,
        )
        engine.reset(mask, torch.tensor([seed], dtype=torch.int64))
        engine.x.fill_(x)
        engine.y.fill_(y)
        engine.z.fill_(z)
        engine.view_z.fill_(camera_z)
        engine.view_height.fill_(camera_z - z)
        engine.angle.fill_(angle_degrees * math.pi / 180.0)
        engine._pitch_bam.fill_(round(pitch_degrees / 360.0 * (1 << 32)))
        engine.pitch.fill_(pitch_degrees * math.pi / 180.0)
        engine.episode_time.fill_(args.settle_tics + 1 + int(bool(args.look_delta)))
        engine.weapon_raise_cooldown.zero_()
        matched_mugshot_face_index = _match_reference_mugshot(engine, reference)
        actual = engine.render_native_frame(include_hud=True)[0].to(torch.float32)
        expected_policy = _reference_policy_frame(reference)
        actual_policy = engine.render_reference_frame()[0].to(torch.float32)
        approximate_policy = engine.render_approximate_frame()[0].to(torch.float32)
        flattened = torch.stack((reference.flatten(), actual.flatten()))
        policy_flattened = torch.stack((expected_policy.flatten(), actual_policy.flatten()))
        approximate_policy_flattened = torch.stack(
            (expected_policy.flatten(), approximate_policy.flatten())
        )
        absolute_error = torch.abs(reference - actual)
        policy_absolute_error = torch.abs(expected_policy - actual_policy)
        approximate_policy_absolute_error = torch.abs(expected_policy - approximate_policy)
        if args.output_dir is not None:
            _write_comparison(
                args.output_dir / f"seed-{seed}-reference-actual-diff.png",
                reference,
                actual,
            )
            _write_policy_comparison(
                args.output_dir / f"seed-{seed}-policy-reference-actual-diff.png",
                expected_policy,
                actual_policy,
            )
            _write_policy_comparison(
                args.output_dir / f"seed-{seed}-policy-reference-approximate-diff.png",
                expected_policy,
                approximate_policy,
            )
        records.append(
            {
                "actual_mean": float(actual.mean()),
                "angle": angle_degrees,
                "approximate_policy_correlation": float(
                    torch.corrcoef(approximate_policy_flattened)[0, 1]
                ),
                "approximate_policy_mae": float(approximate_policy_absolute_error.mean()),
                "camera_z": camera_z,
                "channel_mae_b": float(absolute_error[..., 2].mean()),
                "channel_mae_g": float(absolute_error[..., 1].mean()),
                "channel_mae_r": float(absolute_error[..., 0].mean()),
                "correlation": float(torch.corrcoef(flattened)[0, 1]),
                "mae": float(absolute_error.mean()),
                "mae_ceiling": float(absolute_error[:104].mean()),
                "mae_floor": float(absolute_error[104:208].mean()),
                "mae_hud": float(absolute_error[208:].mean()),
                "matched_mugshot_face_index": matched_mugshot_face_index,
                "pitch": pitch_degrees,
                "policy_actual_mean": float(actual_policy.mean()),
                "policy_correlation": float(torch.corrcoef(policy_flattened)[0, 1]),
                "policy_mae": float(policy_absolute_error.mean()),
                "policy_max_error": float(policy_absolute_error.max()),
                "policy_reference_mean": float(expected_policy.mean()),
                "reference_mean": float(reference.mean()),
                "seed": seed,
                "x": x,
                "y": y,
                "z": z,
            }
        )

    correlations = np.asarray([record["correlation"] for record in records], dtype=np.float64)
    errors = np.asarray([record["mae"] for record in records], dtype=np.float64)
    approximate_policy_correlations = np.asarray(
        [record["approximate_policy_correlation"] for record in records], dtype=np.float64
    )
    approximate_policy_errors = np.asarray(
        [record["approximate_policy_mae"] for record in records], dtype=np.float64
    )
    policy_correlations = np.asarray(
        [record["policy_correlation"] for record in records], dtype=np.float64
    )
    policy_errors = np.asarray([record["policy_mae"] for record in records], dtype=np.float64)
    print(
        json.dumps(
            {
                "approximate_policy_mean_correlation": float(
                    approximate_policy_correlations.mean()
                ),
                "approximate_policy_mean_mae": float(approximate_policy_errors.mean()),
                "iwad_sha256": scenario.iwad_sha256,
                "mean_correlation": float(correlations.mean()),
                "mean_mae": float(errors.mean()),
                "median_correlation": float(np.median(correlations)),
                "median_mae": float(np.median(errors)),
                "policy_mean_correlation": float(policy_correlations.mean()),
                "policy_mean_mae": float(policy_errors.mean()),
                "policy_median_correlation": float(np.median(policy_correlations)),
                "policy_median_mae": float(np.median(policy_errors)),
                "records": records,
                "scenario_sha256": scenario.scenario_sha256,
                "schema": "env_doom_turbo_torch.renderer-parity.raw-and-policy.v2",
                "stochastic_state_alignment": ["mugshot_face_index"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
