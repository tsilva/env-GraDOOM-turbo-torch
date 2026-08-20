"""Measure steady-state env-Doom-turbo-torch device throughput after an untimed warmup.

This is an operator-run benchmark, not a correctness test. Beast-3 runs require
an explicitly confirmed quiet window before launching this tool.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from env_doom_turbo_torch import EnvDoomTurboTorchVecEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iwad", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument(
        "--observation-renderer",
        choices=("approximate", "native-fused", "reference"),
        default="approximate",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.num_envs, args.warmup_steps, args.sample_steps, args.repeats) <= 0:
        raise ValueError("all count arguments must be positive")

    device = torch.device("cuda")
    env = EnvDoomTurboTorchVecEnv(
        game="VizdoomDeathmatch-v1",
        scenario=args.scenario,
        rom_path=str(args.iwad),
        num_envs=args.num_envs,
        device=device,
        transport="torch",
        render_mode="rgb_array",
        obs_copy="unsafe_view",
        obs_crop=(0, 32, 0, 0),
        obs_crop_mode="mask",
        frame_skip=2,
        compile_engine=not args.eager,
        observation_renderer=args.observation_renderer,
    )
    try:
        lane = torch.arange(args.num_envs, device=device, dtype=torch.int64)
        mask = torch.ones(args.num_envs, device=device, dtype=torch.bool)
        env.reset_device(mask, lane + 1)
        reset_generation = 1

        def advance(step: int) -> None:
            nonlocal reset_generation
            actions = torch.remainder(lane + step, env.single_action_space.n)
            reset_generation += 1
            reset_seeds = lane + reset_generation * args.num_envs + 1
            env.step_and_reset_device(actions, reset_seeds)

        for step in range(args.warmup_steps):
            advance(step)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        samples: list[float] = []
        elapsed_samples: list[float] = []
        for repeat in range(args.repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            for step in range(args.sample_steps):
                advance(repeat * args.sample_steps + step)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            elapsed_samples.append(elapsed)
            samples.append(args.num_envs * args.sample_steps / elapsed)

        median_env_steps = statistics.median(samples)
        median_elapsed = statistics.median(elapsed_samples)
        result = {
            "backend": env.engine_backend,
            "cuda": torch.version.cuda,
            "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(device),
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "device": torch.cuda.get_device_name(device),
            "doom_tics_per_second_median": median_env_steps * env.frame_skip,
            "environment_batch_latency_seconds_median": median_elapsed / args.sample_steps,
            "environment_transitions_per_second_median": median_env_steps,
            "environment_transitions_per_second_samples": samples,
            "elapsed_seconds_samples": elapsed_samples,
            "frame_skip": env.frame_skip,
            "iwad_sha256": env.iwad_sha256,
            "num_envs": args.num_envs,
            "observation_shape": list(env.observation_space.shape),
            "observation_renderer": args.observation_renderer,
            "repeats": args.repeats,
            "sample_steps_per_repeat": args.sample_steps,
            "scenario_sha256": env.scenario_sha256,
            "torch": torch.__version__,
            "warmup_steps_excluded": args.warmup_steps,
        }
        print(json.dumps(result, sort_keys=True))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
