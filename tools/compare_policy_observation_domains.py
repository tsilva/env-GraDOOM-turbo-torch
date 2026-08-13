"""Measure policy drift between paired exact and fast GraDOOM observations.

Both observations are rendered from the same device-resident engine state.  The
exact path matches ViZDoom's native RGB/area/grayscale pipeline; the fast path
is the training renderer.  Comparing the unchanged checkpoint on these pairs
separates observation-induced control drift from gameplay stochasticity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _load_train() -> ModuleType:
    path = Path(__file__).parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("gradoom_domain_compare_train", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"cannot load standalone trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument(
        "--state-source",
        choices=("trajectory", "random-pose"),
        default="trajectory",
    )
    parser.add_argument(
        "--fast-renderer",
        choices=("direct", "native-fused", "native-fused-exact-weapon"),
        default="direct",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "iwad", "scenario"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            raise FileNotFoundError(f"{name} does not exist: {value}")
        setattr(args, name, value)
    if args.updates <= 0 or args.num_envs <= 0:
        raise ValueError("updates and num-envs must be positive")


def _policy_outputs(
    policy: Any,
    observations: torch.Tensor,
    context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = policy.encode_observations(observations)
    features = policy.features_from_encoded(encoded, context)
    logits = policy.action_head(features)
    return encoded, logits, F.softmax(logits, dim=1)


def _random_pose_pair(
    env: Any,
    all_lanes: torch.Tensor,
    frame_stack: int,
    fast_renderer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    engine = env._engine
    x, y, angle, _valid = engine._random_spawn_positions(
        all_lanes,
        avoid_player=False,
        candidate_count=32,
    )
    engine.x.copy_(x)
    engine.y.copy_(y)
    engine.angle.copy_(angle)
    engine._x_fixed.copy_(torch.round(x * 65_536.0).to(torch.int64))
    engine._y_fixed.copy_(torch.round(y * 65_536.0).to(torch.int64))
    engine._angle_bam.copy_(
        torch.bitwise_and(
            torch.round(angle * ((1 << 32) / (2.0 * np.pi))).to(torch.int64),
            (1 << 32) - 1,
        )
    )
    sector = engine._sector_at(x, y)
    floor = engine.map.sector_heights[sector, 0]
    ceiling = engine.map.sector_heights[sector, 1]
    engine.z.copy_(floor)
    engine.player_floor_z.copy_(floor)
    engine.previous_player_floor_z.copy_(floor)
    engine.player_ceiling_z.copy_(ceiling)
    engine.view_height.fill_(41.0)
    engine.delta_view_height.zero_()
    engine.view_z.copy_(floor + engine.view_height)
    exact_frame = engine.render_reference_frame()
    if fast_renderer == "direct":
        fast_frame = engine.render_approximate_frame()
    else:
        fast_frame = engine.render_fast_native_policy_frame(
            exact_weapon=fast_renderer == "native-fused-exact-weapon"
        )
    return (
        exact_frame[:, None].expand(-1, frame_stack, -1, -1).clone(),
        fast_frame[:, None].expand(-1, frame_stack, -1, -1).clone(),
    )


def _kl_from_reference(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    return float(
        F.kl_div(
            F.log_softmax(candidate_logits, dim=1),
            F.softmax(reference_logits, dim=1),
            reduction="batchmean",
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    train = _load_train()
    args = _parser().parse_args(argv)
    _validate(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    loaded = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not isinstance(loaded, Mapping) or loaded.get("format") != "standalone-gradoom-ppo-v1":
        raise ValueError(f"unsupported standalone checkpoint: {args.checkpoint}")
    config = loaded.get("config", {})
    policy_config = config.get("policy_model", {}) if isinstance(config, Mapping) else {}
    effective = config.get("effective_recipe", {}) if isinstance(config, Mapping) else {}
    architecture = str(policy_config.get("architecture", "nature"))
    memory_format = str(policy_config.get("memory_format", "contiguous"))
    blur_kernel = int(
        effective.get(
            "observation_blur_kernel",
            policy_config.get("observation_blur_kernel", 1),
        )
    )
    policy = train.NatureActorCritic(
        architecture,
        memory_format,
        blur_kernel,
    ).to(device)
    policy.load_state_dict(loaded["policy_state_dict"])
    policy.eval()

    env_args = argparse.Namespace(
        scenario=args.scenario,
        iwad=args.iwad,
        num_envs=args.num_envs,
        wall_contact_damage_scale=1.0,
        observation_renderer="reference",
        compile_engine=True,
    )
    env = train._make_env(env_args, device, num_envs=args.num_envs)
    context_encoder = train.CombatContextEncoder(env.device_info_history_names, device)
    episode_indices = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
    episode_seeds = train.GradLabEpisodeSeeds(args.seed, args.num_envs, device)
    all_lanes = torch.ones(args.num_envs, dtype=torch.bool, device=device)
    exact, _signals = env.reset_device(all_lanes, episode_seeds.lookup(episode_indices))
    if args.fast_renderer == "direct":
        fast_frame = env._engine.render_approximate_frame()
    else:
        fast_frame = env._engine.render_fast_native_policy_frame(
            exact_weapon=args.fast_renderer == "native-fused-exact-weapon"
        )
    fast = fast_frame[:, None].expand(-1, train.FRAME_STACK, -1, -1).clone()
    context = context_encoder.encode(env.device_info_histories())

    region_slices = {
        "top": (slice(0, 28), slice(0, 84)),
        "middle": (slice(28, 56), slice(0, 84)),
        "bottom": (slice(56, 84), slice(0, 84)),
        "left": (slice(0, 84), slice(0, 28)),
        "center": (slice(0, 84), slice(28, 56)),
        "right": (slice(0, 84), slice(56, 84)),
    }
    scalar_sums = {
        "feature_cosine": 0.0,
        "feature_l1": 0.0,
        "frame_mae": 0.0,
        "policy_kl_exact_to_fast": 0.0,
        "argmax_agreement": 0.0,
    }
    hybrid_kl_sums = dict.fromkeys(region_slices, 0.0)
    exact_probability_sum = torch.zeros(len(train.RESTRICTED_ACTIONS), device=device)
    fast_probability_sum = torch.zeros_like(exact_probability_sum)
    samples = 0
    try:
        for _update in range(args.updates):
            if args.state_source == "random-pose":
                exact, fast = _random_pose_pair(
                    env,
                    all_lanes,
                    train.FRAME_STACK,
                    args.fast_renderer,
                )
            with torch.no_grad():
                exact_encoded, exact_logits, exact_probabilities = _policy_outputs(
                    policy,
                    exact,
                    context,
                )
                fast_encoded, fast_logits, fast_probabilities = _policy_outputs(
                    policy,
                    fast,
                    context,
                )
                scalar_sums["feature_cosine"] += float(
                    F.cosine_similarity(exact_encoded, fast_encoded, dim=1).sum()
                )
                scalar_sums["feature_l1"] += float(
                    torch.abs(exact_encoded - fast_encoded).mean(dim=1).sum()
                )
                scalar_sums["frame_mae"] += float(
                    torch.abs(exact.float() - fast.float()).mean(dim=(1, 2, 3)).sum()
                )
                scalar_sums["policy_kl_exact_to_fast"] += (
                    _kl_from_reference(exact_logits, fast_logits) * args.num_envs
                )
                scalar_sums["argmax_agreement"] += float(
                    (torch.argmax(exact_logits, dim=1) == torch.argmax(fast_logits, dim=1))
                    .float()
                    .sum()
                )
                exact_probability_sum.add_(exact_probabilities.sum(dim=0))
                fast_probability_sum.add_(fast_probabilities.sum(dim=0))
                for name, (rows, columns) in region_slices.items():
                    hybrid = exact.clone()
                    hybrid[:, :, rows, columns] = fast[:, :, rows, columns]
                    _encoded, hybrid_logits, _probabilities = _policy_outputs(
                        policy,
                        hybrid,
                        context,
                    )
                    hybrid_kl_sums[name] += (
                        _kl_from_reference(exact_logits, hybrid_logits) * args.num_envs
                    )
                actions = torch.distributions.Categorical(logits=exact_logits).sample()
            samples += args.num_envs

            if args.state_source == "trajectory":
                next_episode_indices = episode_indices + 1
                transition = env.step_and_reset_device(
                    actions,
                    episode_seeds.lookup(next_episode_indices),
                )
                done = transition.terminated | transition.truncated
                episode_indices.add_(done.to(torch.int64))
                exact = transition.observations
                if args.fast_renderer == "direct":
                    fast_frame = env._engine.render_approximate_frame()
                else:
                    fast_frame = env._engine.render_fast_native_policy_frame(
                        exact_weapon=args.fast_renderer == "native-fused-exact-weapon"
                    )
                rolled = torch.roll(fast, shifts=-1, dims=1)
                rolled[:, -1].copy_(fast_frame)
                fast = torch.where(
                    done[:, None, None, None],
                    fast_frame[:, None].expand(-1, train.FRAME_STACK, -1, -1),
                    rolled,
                )
                context = context_encoder.encode(transition.info_histories)
    finally:
        env.close()

    exact_probabilities = (exact_probability_sum / samples).cpu().tolist()
    fast_probabilities = (fast_probability_sum / samples).cpu().tolist()
    action_probabilities = []
    for index, labels in enumerate(train.RESTRICTED_ACTIONS):
        action_probabilities.append(
            {
                "index": index,
                "labels": list(labels),
                "exact": exact_probabilities[index],
                "fast": fast_probabilities[index],
                "fast_minus_exact": fast_probabilities[index] - exact_probabilities[index],
            }
        )
    result: dict[str, Any] = {
        "schema": "gradoom.policy-observation-domain-comparison.v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": train._file_sha256(args.checkpoint),
        "state_source": args.state_source,
        "fast_renderer": args.fast_renderer,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "updates": args.updates,
        "samples": samples,
        **{name: value / samples for name, value in scalar_sums.items()},
        "hybrid_region_kl_exact_to_candidate": {
            name: value / samples for name, value in hybrid_kl_sums.items()
        },
        "action_probabilities": action_probabilities,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
