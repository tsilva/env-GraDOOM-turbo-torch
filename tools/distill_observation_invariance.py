"""Adapt a policy encoder to paired exact and fast GraDOOM observations.

The downstream policy is frozen.  The visual encoder learns to reproduce the
reference encoder features from the selected fast renderer while rehearsing the
reference frames, producing one policy that can consume either observation
domain without a provider flag.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _load_train() -> ModuleType:
    path = Path(__file__).parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("gradoom_distill_train", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"cannot load standalone trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iwad", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--reference-retention", type=float, default=1.0)
    parser.add_argument(
        "--policy-logit-coef",
        type=float,
        default=0.0,
        help="Weight for matching the frozen teacher's exact-frame action distribution.",
    )
    parser.add_argument(
        "--reference-policy-logit-coef",
        type=float,
        default=0.0,
        help="Weight for retaining the teacher's action distribution on exact frames.",
    )
    parser.add_argument(
        "--state-source",
        choices=("trajectory", "random-pose"),
        default="trajectory",
        help="Collect paired frames from teacher trajectories or randomized valid map poses.",
    )
    parser.add_argument(
        "--fast-renderer",
        choices=("direct", "native-fused", "native-fused-exact-weapon"),
        default="native-fused-exact-weapon",
        help="Policy-facing renderer to align with the exact reference domain.",
    )
    parser.add_argument("--observation-blur-kernel", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12_345)
    parser.add_argument("--log-every", type=int, default=25)
    return parser


def _validate(args: argparse.Namespace) -> None:
    for name in ("checkpoint", "iwad", "scenario"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            raise FileNotFoundError(f"{name} does not exist: {value}")
        setattr(args, name, value)
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    if args.updates <= 0 or args.num_envs <= 0 or args.log_every <= 0:
        raise ValueError("updates, num-envs, and log-every must be positive")
    if (
        args.learning_rate <= 0.0
        or args.reference_retention < 0.0
        or args.policy_logit_coef < 0.0
        or args.reference_policy_logit_coef < 0.0
    ):
        raise ValueError(
            "learning-rate must be positive and retention/logit coefficients non-negative"
        )
    if args.observation_blur_kernel <= 0 or args.observation_blur_kernel % 2 == 0:
        raise ValueError("observation-blur-kernel must be a positive odd integer")


def _policy_logits_from_encoded(
    policy: Any,
    encoded: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    features = policy.features_from_encoded(encoded, context)
    return policy.action_head(features)


def _random_pose_pair(
    env: Any,
    all_lanes: torch.Tensor,
    frame_stack: int,
    fast_renderer: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a paired repeated-frame stack from broad valid map poses."""

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
    engine.selected_weapon.copy_(torch.randint(1, 7, engine.selected_weapon.shape, device=x.device))
    engine.selected_weapon_variant.copy_(
        torch.randint(
            0,
            2,
            engine.selected_weapon_variant.shape,
            device=x.device,
            dtype=torch.int64,
        ).bool()
    )
    engine.weapons.fill_(True)
    engine.weapon_raise_cooldown.zero_()
    engine.weapon_lower_cooldown.zero_()
    engine.weapon_state_cooldown.zero_()
    engine.pending_weapon.fill_(-1)
    exact_frame = engine.render_reference_frame()
    approximate_frame = _render_fast_frame(engine, fast_renderer)
    exact = exact_frame[:, None].expand(-1, frame_stack, -1, -1).clone()
    approximate = approximate_frame[:, None].expand(-1, frame_stack, -1, -1).clone()
    return exact, approximate


def _render_fast_frame(engine: Any, fast_renderer: str) -> torch.Tensor:
    if fast_renderer == "direct":
        return engine.render_approximate_frame()
    return engine.render_fast_native_policy_frame(
        exact_weapon=fast_renderer == "native-fused-exact-weapon"
    )


def _save(
    args: argparse.Namespace,
    train: ModuleType,
    loaded: Mapping[str, Any],
    student: Any,
    summary: Mapping[str, Any],
) -> None:
    payload = dict(loaded)
    payload["policy_state_dict"] = {
        name: value.detach().to("cpu") for name, value in student.state_dict().items()
    }
    config = copy.deepcopy(loaded.get("config"))
    if not isinstance(config, dict):
        config = {}
    config["operation"] = "distill-observation-invariance"
    config["observation_invariance"] = dict(summary)
    effective_recipe = copy.deepcopy(config.get("effective_recipe"))
    if not isinstance(effective_recipe, dict):
        effective_recipe = {}
    effective_recipe["observation_blur_kernel"] = int(args.observation_blur_kernel)
    config["effective_recipe"] = effective_recipe
    canonical = json.dumps(effective_recipe, sort_keys=True, separators=(",", ":"))
    config["recipe_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    policy_model = copy.deepcopy(config.get("policy_model"))
    if not isinstance(policy_model, dict):
        policy_model = {}
    policy_model["observation_blur_kernel"] = int(args.observation_blur_kernel)
    config["policy_model"] = policy_model
    payload["config"] = config
    payload["observation_invariance"] = dict(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "checkpoint": str(args.output),
                "checkpoint_sha256": train._file_sha256(args.output),
                **summary,
            },
            sort_keys=True,
        ),
        flush=True,
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
        raise ValueError(f"unsupported checkpoint: {args.checkpoint}")
    config = loaded.get("config", {})
    policy_config = config.get("policy_model", {}) if isinstance(config, Mapping) else {}
    effective = config.get("effective_recipe", {}) if isinstance(config, Mapping) else {}
    architecture = str(policy_config.get("architecture", "nature"))
    memory_format = str(policy_config.get("memory_format", "contiguous"))
    source_blur_kernel = int(
        effective.get(
            "observation_blur_kernel",
            policy_config.get("observation_blur_kernel", 1),
        )
    )
    if source_blur_kernel != 1 and args.observation_blur_kernel != source_blur_kernel:
        raise ValueError(
            "changing the blur kernel cannot preserve source checkpoint weights: "
            f"source={source_blur_kernel}, requested={args.observation_blur_kernel}"
        )
    teacher = train.NatureActorCritic(
        architecture,
        memory_format,
        source_blur_kernel,
    ).to(device)
    student = train.NatureActorCritic(
        architecture,
        memory_format,
        observation_blur_kernel=args.observation_blur_kernel
    ).to(device)
    teacher.load_state_dict(loaded["policy_state_dict"])
    student.load_state_dict(loaded["policy_state_dict"])
    teacher.eval()
    student.train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name.startswith("observation_encoder."))
    optimizer = torch.optim.Adam(
        student.observation_encoder.parameters(),
        lr=args.learning_rate,
        eps=1e-5,
    )

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
    approximate_frame = _render_fast_frame(env._engine, args.fast_renderer)
    approximate = approximate_frame[:, None].expand(-1, train.FRAME_STACK, -1, -1).clone()
    context = context_encoder.encode(env.device_info_histories())
    started = time.perf_counter()
    cumulative: dict[str, float] = {
        "approximate_feature_loss": 0.0,
        "reference_feature_loss": 0.0,
        "approximate_action_agreement": 0.0,
        "reference_action_agreement": 0.0,
        "approximate_policy_kl": 0.0,
        "reference_policy_kl": 0.0,
    }
    try:
        for update in range(1, args.updates + 1):
            if args.state_source == "random-pose":
                with torch.no_grad():
                    exact, approximate = _random_pose_pair(
                        env,
                        all_lanes,
                        train.FRAME_STACK,
                        args.fast_renderer,
                    )
            with torch.no_grad():
                target_encoded = teacher.encode_observations(exact)
                teacher_logits = _policy_logits_from_encoded(teacher, target_encoded, context)
                actions = torch.distributions.Categorical(logits=teacher_logits).sample()

            approximate_encoded = student.encode_observations(approximate)
            retained_encoded = student.encode_observations(exact)
            approximate_logits = _policy_logits_from_encoded(
                student,
                approximate_encoded,
                context,
            )
            retained_logits = _policy_logits_from_encoded(
                student,
                retained_encoded,
                context,
            )
            approximate_feature_loss = F.smooth_l1_loss(
                approximate_encoded,
                target_encoded,
            )
            reference_feature_loss = F.smooth_l1_loss(
                retained_encoded,
                target_encoded,
            )
            approximate_policy_kl = F.kl_div(
                F.log_softmax(approximate_logits, dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction="batchmean",
            )
            reference_policy_kl = F.kl_div(
                F.log_softmax(retained_logits, dim=1),
                F.softmax(teacher_logits, dim=1),
                reduction="batchmean",
            )
            loss = (
                approximate_feature_loss
                + args.reference_retention * reference_feature_loss
                + args.policy_logit_coef * approximate_policy_kl
                + args.reference_policy_logit_coef * reference_policy_kl
            )
            with torch.no_grad():
                teacher_actions = torch.argmax(teacher_logits, dim=1)
                approximate_agreement = (
                    torch.argmax(approximate_logits, dim=1) == teacher_actions
                ).float().mean()
                reference_agreement = (
                    torch.argmax(retained_logits, dim=1) == teacher_actions
                ).float().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.observation_encoder.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                cumulative["approximate_feature_loss"] += float(approximate_feature_loss)
                cumulative["reference_feature_loss"] += float(reference_feature_loss)
                cumulative["approximate_action_agreement"] += float(approximate_agreement)
                cumulative["reference_action_agreement"] += float(reference_agreement)
                cumulative["approximate_policy_kl"] += float(approximate_policy_kl)
                cumulative["reference_policy_kl"] += float(reference_policy_kl)

            if args.state_source == "trajectory":
                next_episode_indices = episode_indices + 1
                transition = env.step_and_reset_device(
                    actions,
                    episode_seeds.lookup(next_episode_indices),
                )
                done = transition.terminated | transition.truncated
                episode_indices.add_(done.to(torch.int64))
                exact = transition.observations
                approximate_frame = _render_fast_frame(env._engine, args.fast_renderer)
                rolled = torch.roll(approximate, shifts=-1, dims=1)
                rolled[:, -1].copy_(approximate_frame)
                approximate = torch.where(
                    done[:, None, None, None],
                    approximate_frame[:, None].expand(-1, train.FRAME_STACK, -1, -1),
                    rolled,
                )
                context = context_encoder.encode(transition.info_histories)
            if update % args.log_every == 0 or update == args.updates:
                window = min(args.log_every, update)
                print(
                    json.dumps(
                        {
                            "type": "distillation",
                            "update": update,
                            "paired_frames": update * args.num_envs,
                            **{name: value / window for name, value in cumulative.items()},
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                cumulative = dict.fromkeys(cumulative, 0.0)
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": train._file_sha256(args.checkpoint),
        "paired_frames": args.updates * args.num_envs,
        "updates": args.updates,
        "num_envs": args.num_envs,
        "learning_rate": args.learning_rate,
        "reference_retention": args.reference_retention,
        "policy_logit_coef": args.policy_logit_coef,
        "reference_policy_logit_coef": args.reference_policy_logit_coef,
        "state_source": args.state_source,
        "fast_renderer": args.fast_renderer,
        "observation_blur_kernel": args.observation_blur_kernel,
        "seed": args.seed,
        "seconds": elapsed,
        "paired_frames_per_second": args.updates * args.num_envs / elapsed,
        "wall_contact_damage_scale": 1.0,
    }
    _save(args, train, loaded, student, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
