"""Exercise the device API on CUDA without collecting performance timings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env_doom_turbo_torch import EnvDoomTurboTorchVecEnv
from env_doom_turbo_torch.scenario import CompiledScenario


def _square_scenario() -> CompiledScenario:
    vertices = np.asarray(
        [(-256, -256), (256, -256), (256, 256), (-256, 256)],
        dtype=np.float32,
    )
    walls = np.asarray(
        [
            (-256, -256, 256, -256),
            (256, -256, 256, 256),
            (256, 256, -256, 256),
            (-256, 256, -256, -256),
        ],
        dtype=np.float32,
    )
    return CompiledScenario(
        scenario_sha256="0" * 64,
        iwad_sha256="1" * 64,
        namespace="zdoom",
        vertices=vertices,
        wall_segments=walls,
        blocking_segments=walls.copy(),
        blocking_wall_indices=np.arange(4, dtype=np.int32),
        wall_texture_ids=np.zeros(4, dtype=np.int32),
        wall_texture_offsets=np.zeros((4, 2), dtype=np.float32),
        wall_side_texture_ids=np.concatenate(
            (
                np.zeros((4, 1, 1), dtype=np.int32),
                np.full((4, 1, 1), -1, dtype=np.int32),
            ),
            axis=1,
        ).repeat(3, axis=2),
        wall_side_texture_offsets=np.zeros((4, 2, 2), dtype=np.float32),
        wall_sectors=np.zeros((4, 2), dtype=np.int32),
        sector_edge_mask=np.ones((1, 4), dtype=np.bool_),
        sector_heights=np.asarray([(0, 128)], dtype=np.float32),
        sector_lights=np.asarray([192], dtype=np.int16),
        sector_floor_texture_ids=np.zeros(1, dtype=np.int32),
        sector_ceiling_texture_ids=np.zeros(1, dtype=np.int32),
        player_starts=np.asarray(
            [(-128, -128, 45), (128, -128, 135), (0, 128, 270)],
            dtype=np.float32,
        ),
        item_spawns=np.empty((0, 3), dtype=np.float32),
        item_types=np.empty((0,), dtype=np.int32),
        playpal=np.zeros((256, 3), dtype=np.uint8),
        texture_names=("TEST",),
        texture_atlas=np.full((1, 1, 1), 192, dtype=np.uint8),
        texture_widths=np.ones(1, dtype=np.int32),
        texture_heights=np.ones(1, dtype=np.int32),
        sprite_names=tuple(f"TEST{index}" for index in range(26)),
        sprite_atlas=np.full((26, 1, 1), 224, dtype=np.uint8),
        sprite_opaque=np.ones((26, 1, 1), dtype=np.bool_),
        sprite_widths=np.ones(26, dtype=np.int32),
        sprite_heights=np.ones(26, dtype=np.int32),
        sprite_left_offsets=np.zeros(26, dtype=np.int32),
        sprite_top_offsets=np.full(26, 42, dtype=np.int32),
        weapon_sprite_names=tuple(f"WEAPON{index}" for index in range(8)),
        weapon_screen_values=np.zeros((8, 84, 84), dtype=np.float32),
        weapon_screen_alpha=np.zeros((8, 84, 84), dtype=np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--compile-engine", action="store_true")
    parser.add_argument("--iwad", type=Path, default=None)
    parser.add_argument("--scenario", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    device = torch.device("cuda")
    env_kwargs = (
        {"compiled_scenario": _square_scenario()}
        if args.iwad is None
        else {"rom_path": str(args.iwad), "scenario": args.scenario}
    )
    env = EnvDoomTurboTorchVecEnv(
        game="VizdoomDeathmatch-v1",
        **env_kwargs,
        num_envs=args.num_envs,
        device=device,
        transport="torch",
        render_mode="rgb_array",
        obs_copy="unsafe_view",
        obs_crop=(0, 32, 0, 0),
        obs_crop_mode="mask",
        frame_skip=2,
        compile_engine=args.compile_engine,
    )
    try:
        mask = torch.ones(args.num_envs, device=device, dtype=torch.bool)
        seeds = torch.arange(1, args.num_envs + 1, device=device, dtype=torch.int64)
        observations, signals = env.reset_device(mask, seeds)
        actions = torch.arange(args.num_envs, device=device) % 17
        for _ in range(args.steps):
            transition = env.step_device(actions)
            observations = transition.observations
            signals = transition.signals
        torch.cuda.synchronize(device)
        expected_time = 1 + args.steps * env.frame_skip
        checks = {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "device_api_version": env.metadata["env_doom_turbo_torch_device_api_version"],
            "engine_backend": env.engine_backend,
            "episode_time": int(signals[0, 17].item()),
            "finite_signals": bool(torch.isfinite(signals).all().item()),
            "iwad_sha256": env.iwad_sha256,
            "observation_device": observations.device.type,
            "observation_dtype": str(observations.dtype),
            "observation_shape": list(observations.shape),
            "scenario_sha256": env.scenario_sha256,
            "torch": torch.__version__,
        }
        if checks["episode_time"] != expected_time:
            raise RuntimeError(
                f"episode time mismatch: expected {expected_time}, got {checks['episode_time']}"
            )
        if checks["observation_device"] != "cuda" or not checks["finite_signals"]:
            raise RuntimeError(f"device-residency check failed: {checks}")
        print(json.dumps(checks, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
