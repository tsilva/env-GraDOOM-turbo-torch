"""Compare GraDOOM and ViZDoom rendering at identical seeded player poses."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from gradoom.engine import TorchDeathmatchEngine
from gradoom.scenario import compile_deathmatch_scenario


def _reference_frame(
    config: Path,
    iwad: Path,
    seed: int,
    settle_tics: int,
) -> tuple[torch.Tensor, float, float, float, float]:
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise RuntimeError("compare_renderer.py requires the reference vizdoom package") from exc

    game = vzd.DoomGame()
    game.load_config(str(config))
    game.set_doom_game_path(str(iwad))
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    variables = (
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.POSITION_Z,
        vzd.GameVariable.ANGLE,
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
        if raw.shape != (240, 320):
            raise RuntimeError(f"expected a 240x320 GRAY8 frame, got {raw.shape}")
        raw[-32:] = 0
        frame = functional.interpolate(
            torch.from_numpy(raw)[None, None].to(torch.float32),
            size=(84, 84),
            mode="area",
        )[0, 0]
        x, y, z, angle = (float(game.get_game_variable(variable)) for variable in variables)
        return frame, x, y, z, angle
    finally:
        game.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=(123, 456, 789, 1_337))
    parser.add_argument("--settle-tics", type=int, default=16)
    parser.add_argument("--allow-unpinned-scenario", action="store_true")
    args = parser.parse_args()

    scenario = compile_deathmatch_scenario(
        args.scenario,
        args.iwad,
        require_pinned_scenario=not args.allow_unpinned_scenario,
    )
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    mask = torch.ones(1, dtype=torch.bool)
    records: list[dict[str, float | int]] = []
    for seed in args.seeds:
        reference, x, y, z, angle_degrees = _reference_frame(
            args.config,
            args.iwad,
            seed,
            args.settle_tics,
        )
        engine.reset(mask, torch.tensor([seed], dtype=torch.int64))
        engine.x.fill_(x)
        engine.y.fill_(y)
        engine.z.fill_(z)
        engine.angle.fill_(angle_degrees * math.pi / 180.0)
        actual = engine.render_frame()[0].to(torch.float32)
        flattened = torch.stack((reference.flatten(), actual.flatten()))
        records.append(
            {
                "actual_mean": float(actual.mean()),
                "angle": angle_degrees,
                "correlation": float(torch.corrcoef(flattened)[0, 1]),
                "mae": float(torch.mean(torch.abs(reference - actual))),
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
                "schema": "gradoom.renderer-parity.v1",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
