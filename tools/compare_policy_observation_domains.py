"""Measure policy drift between paired exact and fast env-Doom-turbo-torch observations.

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

_DEATH_VISIBILITY_STATE = ("enemy_death_tics",)
_LIVE_ENEMY_VISIBILITY_STATE = ("enemy_alive",)
_DECAL_VISIBILITY_STATE = ("hitscan_decal_serial",)
_EFFECT_VISIBILITY_STATE = (
    "projectile_alive",
    "projectile_impact_tics",
    "enemy_projectile_alive",
    "enemy_projectile_impact_tics",
    "teleport_fog_tics",
    "hitscan_puff_tics",
    "hitscan_decal_serial",
)
_COMBAT_VISIBILITY_STATE = (
    "enemy_alive",
    "enemy_death_tics",
    "projectile_alive",
    "projectile_impact_tics",
    "enemy_projectile_alive",
    "enemy_projectile_impact_tics",
    "teleport_fog_tics",
    "hitscan_puff_tics",
    "hitscan_decal_serial",
    "drop_spawned",
)


def _load_train() -> ModuleType:
    path = Path(__file__).parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("env_doom_turbo_torch_domain_compare_train", path)
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
    parser.add_argument(
        "--death-ablation",
        action="store_true",
        help="also compare matched frame stacks with death/corpse sprites hidden",
    )
    parser.add_argument(
        "--combat-actor-ablation",
        action="store_true",
        help="also compare matched frame stacks with combat actors and effects hidden",
    )
    parser.add_argument(
        "--live-enemy-ablation",
        action="store_true",
        help="also compare matched frame stacks with live enemy sprites hidden",
    )
    parser.add_argument(
        "--effect-ablation",
        action="store_true",
        help="also compare matched frame stacks with combat effects hidden",
    )
    parser.add_argument(
        "--decal-ablation",
        action="store_true",
        help="also compare matched frame stacks with persistent hitscan decals hidden",
    )
    parser.add_argument("--output", type=Path)
    return parser


def _validate(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "iwad", "scenario"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            raise FileNotFoundError(f"{name} does not exist: {value}")
        setattr(args, name, value)
    if args.updates <= 0 or args.num_envs <= 0:
        raise ValueError("updates and num-envs must be positive")
    if (
        sum(
            (
                args.death_ablation,
                args.combat_actor_ablation,
                args.live_enemy_ablation,
                args.effect_ablation,
                args.decal_ablation,
            )
        )
        > 1
    ):
        raise ValueError("observation-ablation modes are mutually exclusive")


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


def _visibility_state_active(state_name: str, state: torch.Tensor) -> torch.Tensor:
    if state_name == "hitscan_decal_serial":
        return state >= 0
    return state != 0


def _per_sample_metrics(
    exact: torch.Tensor,
    fast: torch.Tensor,
    exact_encoded: torch.Tensor,
    fast_encoded: torch.Tensor,
    exact_logits: torch.Tensor,
    fast_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    exact_log_probabilities = F.log_softmax(exact_logits, dim=1)
    fast_log_probabilities = F.log_softmax(fast_logits, dim=1)
    exact_probabilities = torch.exp(exact_log_probabilities)
    return {
        "feature_cosine": F.cosine_similarity(exact_encoded, fast_encoded, dim=1),
        "feature_l1": torch.abs(exact_encoded - fast_encoded).mean(dim=1),
        "frame_mae": torch.abs(exact.float() - fast.float()).mean(dim=(1, 2, 3)),
        "policy_kl_exact_to_fast": torch.sum(
            exact_probabilities * (exact_log_probabilities - fast_log_probabilities),
            dim=1,
        ),
        "argmax_agreement": (
            torch.argmax(exact_logits, dim=1) == torch.argmax(fast_logits, dim=1)
        ).to(torch.float32),
    }


def _current_ablation_frames(
    env: Any,
    fast_renderer: str,
    state_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    engine = env._engine
    saved_state = {name: getattr(engine, name).clone() for name in state_names}
    for name in state_names:
        state = getattr(engine, name)
        if name == "hitscan_decal_serial":
            state.fill_(-1)
        else:
            state.zero_()
    try:
        exact_frame = engine.render_reference_frame()
        if fast_renderer == "direct":
            fast_frame = engine.render_approximate_frame()
        else:
            fast_frame = engine.render_fast_native_policy_frame(
                exact_weapon=fast_renderer == "native-fused-exact-weapon"
            )
    finally:
        for name, value in saved_state.items():
            getattr(engine, name).copy_(value)
    return exact_frame, fast_frame


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
    if (
        not isinstance(loaded, Mapping)
        or loaded.get("format") != "standalone-env_doom_turbo_torch-ppo-v1"
    ):
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
    ablation_state_names: tuple[str, ...] | None = None
    ablation_mode: str | None = None
    if args.death_ablation:
        ablation_state_names = _DEATH_VISIBILITY_STATE
        ablation_mode = "death_and_corpse_sprites"
    elif args.live_enemy_ablation:
        ablation_state_names = _LIVE_ENEMY_VISIBILITY_STATE
        ablation_mode = "live_enemy_sprites"
    elif args.effect_ablation:
        ablation_state_names = _EFFECT_VISIBILITY_STATE
        ablation_mode = "combat_effects"
    elif args.decal_ablation:
        ablation_state_names = _DECAL_VISIBILITY_STATE
        ablation_mode = "persistent_hitscan_decals"
    elif args.combat_actor_ablation:
        ablation_state_names = _COMBAT_VISIBILITY_STATE
        ablation_mode = "combat_actors_effects_and_drops"
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
    death_hidden_exact: torch.Tensor | None = None
    death_hidden_fast: torch.Tensor | None = None
    if ablation_state_names is not None:
        hidden_exact_frame, hidden_fast_frame = _current_ablation_frames(
            env,
            args.fast_renderer,
            ablation_state_names,
        )
        death_hidden_exact = hidden_exact_frame[:, None].expand_as(exact).clone()
        death_hidden_fast = hidden_fast_frame[:, None].expand_as(fast).clone()
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
    cohort_sums: dict[str, dict[str, float]] = {}
    cohort_counts: dict[str, int] = {}
    ablation_sums = {
        comparison: dict.fromkeys(scalar_sums, 0.0)
        for comparison in (
            "hidden_exact_to_fast",
            "reference_visible_to_hidden",
            "fast_visible_to_hidden",
        )
    }
    ablation_samples = 0
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
                if ablation_state_names is not None:
                    hidden_exact_frame, hidden_fast_frame = _current_ablation_frames(
                        env,
                        args.fast_renderer,
                        ablation_state_names,
                    )
                    death_hidden_exact = hidden_exact_frame[:, None].expand_as(exact).clone()
                    death_hidden_fast = hidden_fast_frame[:, None].expand_as(fast).clone()
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
                per_sample = _per_sample_metrics(
                    exact,
                    fast,
                    exact_encoded,
                    fast_encoded,
                    exact_logits,
                    fast_logits,
                )
                for name, values in per_sample.items():
                    scalar_sums[name] += float(values.sum())

                death_tics = env._engine.enemy_death_tics
                corpse_count = torch.sum(death_tics > 0, dim=1)
                cohort_masks = {
                    "no_death_or_corpse": corpse_count == 0,
                    "any_death_or_corpse": corpse_count > 0,
                    "active_death_animation": torch.any(death_tics > 1, dim=1),
                    "persistent_corpse": torch.any(death_tics == 1, dim=1),
                    "one_to_four_deaths_or_corpses": (corpse_count >= 1) & (corpse_count <= 4),
                    "five_or_more_deaths_or_corpses": corpse_count >= 5,
                }
                for cohort_name, mask in cohort_masks.items():
                    count = int(mask.sum())
                    if not count:
                        continue
                    cohort_counts[cohort_name] = cohort_counts.get(cohort_name, 0) + count
                    sums = cohort_sums.setdefault(
                        cohort_name,
                        dict.fromkeys(per_sample, 0.0),
                    )
                    for metric_name, values in per_sample.items():
                        sums[metric_name] += float(values[mask].sum())
                if ablation_state_names is not None:
                    if death_hidden_exact is None or death_hidden_fast is None:
                        raise RuntimeError("death-ablation frame stacks are unavailable")
                    hidden_exact_encoded, hidden_exact_logits, _hidden_exact_probabilities = (
                        _policy_outputs(policy, death_hidden_exact, context)
                    )
                    hidden_fast_encoded, hidden_fast_logits, _hidden_fast_probabilities = (
                        _policy_outputs(policy, death_hidden_fast, context)
                    )
                    ablation_comparisons = {
                        "hidden_exact_to_fast": _per_sample_metrics(
                            death_hidden_exact,
                            death_hidden_fast,
                            hidden_exact_encoded,
                            hidden_fast_encoded,
                            hidden_exact_logits,
                            hidden_fast_logits,
                        ),
                        "reference_visible_to_hidden": _per_sample_metrics(
                            exact,
                            death_hidden_exact,
                            exact_encoded,
                            hidden_exact_encoded,
                            exact_logits,
                            hidden_exact_logits,
                        ),
                        "fast_visible_to_hidden": _per_sample_metrics(
                            fast,
                            death_hidden_fast,
                            fast_encoded,
                            hidden_fast_encoded,
                            fast_logits,
                            hidden_fast_logits,
                        ),
                    }
                    ablation_mask = cohort_masks["any_death_or_corpse"]
                    if args.live_enemy_ablation:
                        ablation_mask = torch.any(env._engine.enemy_alive, dim=1)
                    elif args.effect_ablation:
                        ablation_mask = torch.zeros_like(ablation_mask)
                        for state_name in _EFFECT_VISIBILITY_STATE:
                            state = getattr(env._engine, state_name)
                            visible = _visibility_state_active(state_name, state)
                            ablation_mask |= torch.any(visible, dim=1)
                    elif args.decal_ablation:
                        ablation_mask = torch.any(env._engine.hitscan_decal_serial >= 0, dim=1)
                    elif args.combat_actor_ablation:
                        ablation_mask = torch.zeros_like(ablation_mask)
                        for state_name in _COMBAT_VISIBILITY_STATE:
                            state = getattr(env._engine, state_name)
                            visible = _visibility_state_active(state_name, state)
                            ablation_mask |= torch.any(visible, dim=1)
                    ablation_count = int(ablation_mask.sum())
                    ablation_samples += ablation_count
                    for comparison, metrics in ablation_comparisons.items():
                        for metric_name, values in metrics.items():
                            ablation_sums[comparison][metric_name] += float(
                                values[ablation_mask].sum()
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
                if ablation_state_names is not None:
                    if death_hidden_exact is None or death_hidden_fast is None:
                        raise RuntimeError("death-ablation frame stacks are unavailable")
                    hidden_exact_frame, hidden_fast_frame = _current_ablation_frames(
                        env,
                        args.fast_renderer,
                        ablation_state_names,
                    )
                    rolled_hidden_exact = torch.roll(death_hidden_exact, shifts=-1, dims=1)
                    rolled_hidden_exact[:, -1].copy_(hidden_exact_frame)
                    death_hidden_exact = torch.where(
                        done[:, None, None, None],
                        hidden_exact_frame[:, None].expand_as(death_hidden_exact),
                        rolled_hidden_exact,
                    )
                    rolled_hidden_fast = torch.roll(death_hidden_fast, shifts=-1, dims=1)
                    rolled_hidden_fast[:, -1].copy_(hidden_fast_frame)
                    death_hidden_fast = torch.where(
                        done[:, None, None, None],
                        hidden_fast_frame[:, None].expand_as(death_hidden_fast),
                        rolled_hidden_fast,
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
        "schema": "env_doom_turbo_torch.policy-observation-domain-comparison.v4",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": train._file_sha256(args.checkpoint),
        "state_source": args.state_source,
        "fast_renderer": args.fast_renderer,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "updates": args.updates,
        "samples": samples,
        **{name: value / samples for name, value in scalar_sums.items()},
        "cohorts": {
            cohort_name: {
                "samples": cohort_counts[cohort_name],
                **{
                    metric_name: value / cohort_counts[cohort_name]
                    for metric_name, value in sums.items()
                },
            }
            for cohort_name, sums in cohort_sums.items()
        },
        "observation_ablation": (
            {
                "mode": ablation_mode,
                "comparisons": {
                    comparison: {
                        "samples": ablation_samples,
                        **{
                            metric_name: value / ablation_samples
                            for metric_name, value in sums.items()
                        },
                    }
                    for comparison, sums in ablation_sums.items()
                },
            }
            if ablation_samples
            else None
        ),
        "hybrid_region_kl_exact_to_candidate": {
            name: value / samples for name, value in hybrid_kl_sums.items()
        },
        "action_probabilities": action_probabilities,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
