#!/usr/bin/env python3
"""Standalone PPO trainer for GraDOOM's certified Deathmatch fast path.

The defaults reproduce the successful GradLab VizdoomDeathmatch-v1 PPO recipe,
but this script has no GradLab or Stable-Baselines3 runtime dependency. It uses
only GraDOOM, PyTorch, NumPy, and the Python standard library so GradLab and
GraDOOM can be optimized independently against a fixed learning baseline.

Run training with::

    uv run python train.py --iwad /path/to/doom2.wad

Use ``--config-only`` to print the complete benchmark contract without CUDA or
game assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import statistics
import time
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gradoom._triton_kernels import frozen_nature_conv1

REFERENCE_NAME = "GradLab VizdoomDeathmatch-v1/ppo"
REFERENCE_CAPTURED_AT = "2026-08-11"
ROLLING_EPISODES = 100
REFERENCE_KILLS_TARGET = 31.78
GRADLAB_WANDB_PROJECT = "VizdoomDeathmatch-v1"
GRADLAB_RETURN_METRIC = "train/episode/return/shaped/origin/target/rolling/mean"
GRADLAB_KILLS_METRIC = "train/progress/kills/origin/target/rolling/mean"
GRADLAB_PPO_DIAGNOSTIC_METRICS = (
    "train/algorithm/ppo/policy/dominant/action/rate",
    "train/algorithm/ppo/policy/entropy",
    "train/algorithm/ppo/rollout/advantage/mean",
    "train/algorithm/ppo/rollout/advantage/std",
    "train/algorithm/ppo/rollout/value/prediction/mean",
    "train/algorithm/ppo/rollout/value/prediction/std",
    "train/algorithm/ppo/update/approx_kl",
    "train/algorithm/ppo/update/clip_fraction",
    "train/algorithm/ppo/update/learning_rate",
    "train/algorithm/ppo/update/policy_gradient_loss",
    "train/algorithm/ppo/update/value_loss",
    "train/algorithm/ppo/value/explained_variance",
)
GRADLAB_WANDB_METRICS = (
    GRADLAB_RETURN_METRIC,
    GRADLAB_KILLS_METRIC,
    *GRADLAB_PPO_DIAGNOSTIC_METRICS,
)
GRADOOM_WANDB_TAG = "env_provider:gradoom"
UINT32_MASK = (1 << 32) - 1
SEED_TABLE_INITIAL_EPISODES = 64
NATIVE_MONSTER_KILL_REWARDS = (1.0, 3.0, 3.0, 4.0, 3.0, 10.0)

GAME_VARIABLES = (
    "killcount",
    "deathcount",
    "hitcount",
    "damagecount",
    "health",
    "armor",
    "selected_weapon",
    "selected_weapon_ammo",
    "weapon1",
    "weapon2",
    "weapon3",
    "weapon4",
    "weapon5",
    "weapon6",
    "ammo1",
    "ammo2",
    "ammo3",
    "ammo4",
    "ammo5",
    "ammo6",
)
INFO_SIGNALS = (*GAME_VARIABLES, "player_dead")
RESTRICTED_ACTIONS = (
    (),
    ("ATTACK",),
    ("MOVE_FORWARD",),
    ("MOVE_BACKWARD",),
    ("MOVE_LEFT",),
    ("MOVE_RIGHT",),
    ("TURN_LEFT",),
    ("TURN_RIGHT",),
    ("SPEED", "MOVE_FORWARD"),
    ("ATTACK", "MOVE_FORWARD"),
    ("ATTACK", "MOVE_BACKWARD"),
    ("ATTACK", "MOVE_LEFT"),
    ("ATTACK", "MOVE_RIGHT"),
    ("ATTACK", "TURN_LEFT"),
    ("ATTACK", "TURN_RIGHT"),
    ("SELECT_NEXT_WEAPON",),
    ("SELECT_PREV_WEAPON",),
)
MODEL_HISTORY_SIGNALS = (
    "armor",
    "health",
    "selected_weapon",
    "selected_weapon_ammo",
    "ammo1",
    "ammo2",
    "ammo3",
    "ammo4",
    "ammo5",
    "ammo6",
    "weapon1",
    "weapon2",
    "weapon3",
    "weapon4",
    "weapon5",
    "weapon6",
)
FRAME_STACK = 4
CONTEXT_FEATURES_PER_FRAME = 21
# Preserve a short temporal trace for the standalone learner. The 4-frame
# context learned measurably faster than current-state-only context in the
# faithful native-reward gates, and every value is policy-facing at transfer.
MODEL_CONTEXT_FRAMES = FRAME_STACK
CONTEXT_FEATURES = MODEL_CONTEXT_FRAMES * CONTEXT_FEATURES_PER_FRAME


@dataclass(frozen=True)
class Recipe:
    timesteps: int = 500_000_000
    seed: int = 123
    num_envs: int = 128
    n_steps: int = 32
    batch_size: int = 256
    n_epochs: int = 2
    learning_rate: float = 6.25e-5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    clip_range: float = 0.1
    max_grad_norm: float = 0.5
    adam_eps: float = 1e-5
    precision: str = "fp32"
    frame_skip: int = 2
    episode_timeout: int = 4200
    doom_skill: int = 1
    action_count: int = 17
    observation_channels: int = 4
    observation_height: int = 84
    observation_width: int = 84
    cnn_features: int = 512
    fusion_features: int = 256


REFERENCE_RECIPE = Recipe()


@dataclass(frozen=True)
class PolicyArchitecture:
    """Static NatureCNN widths used by an audited training run."""

    convolution_channels: tuple[int, int, int]
    observation_features: int
    fusion_features: int


POLICY_ARCHITECTURES = {
    "nature": PolicyArchitecture((32, 64, 64), 512, 256),
    # Preserve the successful policy/value trunk while reducing only the
    # convolutional work.  The earlier half/quarter profiles also narrowed the
    # learned observation embedding and fusion trunk, so they could not
    # distinguish visual-encoder capacity from decision-trunk capacity.
    "nature-pyramid": PolicyArchitecture((16, 32, 64), 512, 256),
    "nature-waist": PolicyArchitecture((32, 32, 64), 512, 256),
    "nature-flat": PolicyArchitecture((32, 32, 32), 512, 256),
    "nature-thin": PolicyArchitecture((16, 32, 32), 512, 256),
    "nature-half": PolicyArchitecture((16, 32, 32), 128, 128),
    "nature-quarter": PolicyArchitecture((8, 16, 16), 128, 128),
}

# Keep the immutable GradLab recipe above as evidence, while using the measured
# RTX 4090 sweet spot for new standalone runs.  Both shapes contain 4,096
# transitions; 256x16 was faster and learned at least as well in the short
# acceptance gates as the historical 128x32 shape.
DEFAULT_NUM_ENVS = 256
DEFAULT_N_STEPS = 16


@dataclass(frozen=True)
class SampleFactoryRewardConfig:
    """Registered GradLab ``sample-factory-v0`` Deathmatch reward contract."""

    kill_reward: float = 1.0
    kill_loss_penalty: float = 1.5
    death_penalty: float = 0.75
    death_count_decrease_reward: float = 0.75
    hit_reward: float = 0.01
    hit_count_decrease_penalty: float = 0.01
    damage_reward: float = 0.003
    damage_count_decrease_penalty: float = 0.003
    health_gain_reward: float = 0.005
    health_loss_penalty: float = 0.003
    armor_gain_reward: float = 0.005
    armor_loss_penalty: float = 0.001
    weapon_preferences: tuple[float, ...] = (1.0, 1.0, 5.0, 5.0, 5.0, 10.0)
    weapon_gain_reward_scale: float = 0.02
    weapon_loss_penalty_scale: float = 0.01
    ammo_gain_reward_scale: float = 0.0002
    ammo_loss_penalty_scale: float = 0.0001
    selected_weapon_hold_reward_scale: float = 0.0002
    selected_weapon_hold_steps: int = 5
    hit_delta_cap: int = 5
    damage_delta_cap: int = 200


SAMPLE_FACTORY_REWARD = SampleFactoryRewardConfig()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train standalone PPO on GraDOOM's GPU-native Deathmatch runtime.",
    )
    parser.add_argument(
        "--iwad",
        type=Path,
        default=os.environ.get("GRADOOM_IWAD"),
        help="Doom II or Freedoom IWAD (default: GRADOOM_IWAD).",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=os.environ.get("GRADOOM_DEATHMATCH_WAD"),
        help=(
            "Pinned ViZDoom deathmatch WAD (default: GRADOOM_DEATHMATCH_WAD or the "
            "installed ViZDoom scenario)."
        ),
    )
    parser.add_argument("--timesteps", type=int, default=REFERENCE_RECIPE.timesteps)
    parser.add_argument("--seed", type=int, default=REFERENCE_RECIPE.seed)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--batch-size", type=int, default=REFERENCE_RECIPE.batch_size)
    parser.add_argument("--n-epochs", type=int, default=REFERENCE_RECIPE.n_epochs)
    parser.add_argument("--learning-rate", type=float, default=REFERENCE_RECIPE.learning_rate)
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=REFERENCE_RECIPE.ent_coef,
        help=("PPO entropy coefficient (default: 0.01, matching the registered reference recipe)."),
    )
    parser.add_argument(
        "--wall-contact-damage-scale",
        type=float,
        default=1.0,
        help=(
            "Experimental enemy-damage multiplier while the player touches blocking "
            "geometry (default: 1.0, disabled)."
        ),
    )
    parser.add_argument(
        "--reward-shape",
        choices=(
            "native-v1",
            "native-death-v1",
            "killcount-v1",
            "sample-factory-v0",
        ),
        default="native-v1",
        help=(
            "Use scenario-native rewards, native rewards plus an explicit death cost, "
            "uniform kill-count deltas, or the registered GradLab Sample Factory "
            "shaping contract."
        ),
    )
    parser.add_argument(
        "--death-penalty",
        type=float,
        default=2.0,
        help="Terminal cost used only by native-death-v1 (default: 2.0).",
    )
    parser.add_argument(
        "--privileged-imitation-coef",
        type=float,
        default=0.0,
        help=(
            "Training-only cross-entropy coefficient for GraDOOM's visible-enemy "
            "combat teacher (default: 0, disabled). The saved policy remains "
            "pixels/context-only."
        ),
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp-fp16", "amp-bf16"),
        default=REFERENCE_RECIPE.precision,
        help="FP32 matches the registered reference recipe.",
    )
    parser.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help=(
            "PyTorch float32 matrix-multiply mode (default: high, enabling audited "
            "TF32 acceleration on RTX 4090 while retaining float32 storage)."
        ),
    )
    parser.add_argument(
        "--policy-architecture",
        choices=tuple(POLICY_ARCHITECTURES),
        default="nature",
        help=(
            "Audited NatureCNN width profile. 'nature' preserves checkpoint compatibility; "
            "the half- and quarter-width profiles trade capacity for training throughput."
        ),
    )
    parser.add_argument(
        "--policy-memory-format",
        choices=("contiguous", "channels-last"),
        default="channels-last",
        help=(
            "CUDA convolution memory format (default: channels-last, including the "
            "policy-input conversion in the compiled graph)."
        ),
    )
    parser.add_argument(
        "--freeze-observation-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Freeze the visual encoder and cache its rollout features during PPO updates. "
            "Intended for a resumed late training stage after the encoder has learned."
        ),
    )
    parser.add_argument(
        "--frozen-encoder-custom-conv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the fused uint8 first-convolution kernel while the observation "
            "encoder is frozen (default: enabled)."
        ),
    )
    parser.add_argument(
        "--compile-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-engine",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torch-permutation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--steady-state-after-rollouts",
        type=int,
        default=1,
        help="Exclude this many compile/warmup rollouts from steady-state aggregates.",
    )
    parser.add_argument(
        "--metrics-jsonl",
        type=Path,
        help="Optionally append every emitted JSON record to this file.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log GradLab-compatible training metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        default=GRADLAB_WANDB_PROJECT,
        help=(
            "W&B project (default: VizdoomDeathmatch-v1, matching GradLab's "
            "environment-project convention)."
        ),
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-tags",
        default="",
        help=(
            "Comma-separated additional W&B tags. The env_provider:gradoom tag "
            "is always included."
        ),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-description", default="")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optionally save the final standalone policy and optimizer checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every-rollouts",
        type=int,
        default=0,
        help="Also save unique step-suffixed recovery checkpoints at this rollout interval.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume policy, optimizer, counters, and RNG state from a trusted checkpoint.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Initialize policy weights from a trusted standalone checkpoint while "
            "starting fresh optimizer, episode, and timestep state."
        ),
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        help=(
            "Evaluate a trusted standalone checkpoint without learning and emit an exact "
            "episode-level result."
        ),
    )
    parser.add_argument(
        "--evaluation-episodes",
        type=int,
        default=ROLLING_EPISODES,
        help=f"Number of completed checkpoint-evaluation episodes (default: {ROLLING_EPISODES}).",
    )
    parser.add_argument(
        "--evaluation-num-envs",
        type=int,
        default=16,
        help=(
            "Parallel evaluation lanes; episodes are balanced across lane seed streams "
            "(default: 16, matching GradLab and zero-shot ViZDoom evaluation)."
        ),
    )
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=REFERENCE_RECIPE.seed,
        help="Independent GradLab-compatible evaluation seed (default: 123).",
    )
    parser.add_argument(
        "--evaluation-stochastic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample policy actions during evaluation, matching GradLab (default: enabled).",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Print the effective contract and exit before CUDA/environment setup.",
    )
    return parser


def _validate_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _checkpoint_destination(path: Path) -> Path:
    destination = path.expanduser().resolve()
    return destination if destination.suffix == ".pt" else Path(str(destination) + ".pt")


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "timesteps",
        "num_envs",
        "n_steps",
        "batch_size",
        "n_epochs",
        "evaluation_episodes",
        "evaluation_num_envs",
    ):
        _validate_positive(int(getattr(args, name)), name.replace("_", "-"))
    if not 0 <= int(args.seed) <= UINT32_MASK:
        raise ValueError(f"seed must be in [0, {UINT32_MASK}]")
    if not 0 <= int(args.evaluation_seed) <= UINT32_MASK:
        raise ValueError(f"evaluation-seed must be in [0, {UINT32_MASK}]")
    if int(args.evaluation_num_envs) > int(args.evaluation_episodes):
        raise ValueError("evaluation-num-envs cannot exceed evaluation-episodes")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be finite and positive")
    if not math.isfinite(args.ent_coef) or args.ent_coef < 0.0:
        raise ValueError("ent-coef must be finite and non-negative")
    if not math.isfinite(args.death_penalty) or args.death_penalty < 0.0:
        raise ValueError("death-penalty must be finite and non-negative")
    if not math.isfinite(args.privileged_imitation_coef) or args.privileged_imitation_coef < 0.0:
        raise ValueError("privileged-imitation-coef must be finite and non-negative")
    if (
        not math.isfinite(args.wall_contact_damage_scale)
        or args.wall_contact_damage_scale < 0.0
        or args.wall_contact_damage_scale > 1.0
    ):
        raise ValueError("wall-contact-damage-scale must be finite and in [0, 1]")
    if args.steady_state_after_rollouts < 0:
        raise ValueError("steady-state-after-rollouts must be non-negative")
    if args.checkpoint_every_rollouts < 0:
        raise ValueError("checkpoint-every-rollouts must be non-negative")
    rollout_transitions = int(args.num_envs) * int(args.n_steps)
    if int(args.batch_size) > rollout_transitions:
        raise ValueError("batch-size cannot exceed num-envs * n-steps")
    if args.checkpoint is not None:
        destination = _checkpoint_destination(args.checkpoint)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    elif args.checkpoint_every_rollouts:
        raise ValueError("checkpoint-every-rollouts requires --checkpoint")
    if args.resume is not None:
        args.resume = _checkpoint_destination(args.resume)
        if not args.resume.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
    if args.initialize_from is not None:
        args.initialize_from = _checkpoint_destination(args.initialize_from)
        if not args.initialize_from.is_file():
            raise FileNotFoundError(
                f"initialization checkpoint does not exist: {args.initialize_from}"
            )
        if args.resume is not None:
            raise ValueError("initialize-from cannot be combined with resume")
    if args.evaluate_checkpoint is not None:
        args.evaluate_checkpoint = _checkpoint_destination(args.evaluate_checkpoint)
        if not args.evaluate_checkpoint.is_file():
            raise FileNotFoundError(
                f"evaluation checkpoint does not exist: {args.evaluate_checkpoint}"
            )
        if args.resume is not None:
            raise ValueError("evaluate-checkpoint cannot be combined with resume")
        if args.initialize_from is not None:
            raise ValueError("evaluate-checkpoint cannot be combined with initialize-from")
        if args.checkpoint is not None:
            raise ValueError("evaluate-checkpoint cannot be combined with checkpoint")


def _runtime_paths(args: argparse.Namespace) -> None:
    if args.iwad is None:
        raise FileNotFoundError("pass --iwad or set GRADOOM_IWAD")
    args.iwad = args.iwad.expanduser().resolve()
    if not args.iwad.is_file():
        raise FileNotFoundError(f"IWAD does not exist: {args.iwad}")
    if args.scenario is not None:
        args.scenario = args.scenario.expanduser().resolve()
        if not args.scenario.is_file():
            raise FileNotFoundError(f"scenario does not exist: {args.scenario}")


def _execution_timesteps(args: argparse.Namespace) -> int:
    quantum = int(args.num_envs) * int(args.n_steps)
    return math.ceil(int(args.timesteps) / quantum) * quantum


def _audit_config(args: argparse.Namespace) -> dict[str, Any]:
    initialization_checkpoint = None if args.initialize_from is None else str(args.initialize_from)
    initialization_sha256 = (
        None if args.initialize_from is None else _file_sha256(args.initialize_from)
    )
    effective = {
        **asdict(REFERENCE_RECIPE),
        "timesteps": int(args.timesteps),
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "n_steps": int(args.n_steps),
        "batch_size": int(args.batch_size),
        "n_epochs": int(args.n_epochs),
        "learning_rate": float(args.learning_rate),
        "ent_coef": float(args.ent_coef),
        "death_penalty": float(args.death_penalty),
        "privileged_imitation_coef": float(args.privileged_imitation_coef),
        "wall_contact_damage_scale": float(args.wall_contact_damage_scale),
        "reward_shape": str(args.reward_shape),
        "precision": str(args.precision),
        "float32_matmul_precision": str(args.float32_matmul_precision),
        "policy_architecture": str(args.policy_architecture),
        "policy_memory_format": str(args.policy_memory_format),
        "freeze_observation_encoder": bool(args.freeze_observation_encoder),
        "frozen_encoder_custom_conv": bool(args.frozen_encoder_custom_conv),
        "compile_policy": bool(args.compile_policy),
        "compile_engine": bool(args.compile_engine),
        "fused_optimizer": bool(args.fused_optimizer),
        "torch_permutation": bool(args.torch_permutation),
        "initialize_from": initialization_checkpoint,
        "initialize_from_sha256": initialization_sha256,
    }
    canonical = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    return {
        "type": "config",
        "contract": "standalone-gradoom-deathmatch-ppo-v2",
        "operation": "evaluate" if args.evaluate_checkpoint is not None else "train",
        "standalone": True,
        "runtime_dependencies": [
            "gradoom",
            "torch",
            "numpy",
            *(["wandb"] if bool(args.wandb) else []),
        ],
        "reference": REFERENCE_NAME,
        "reference_captured_at": REFERENCE_CAPTURED_AT,
        "recipe_sha256": hashlib.sha256(canonical.encode("ascii")).hexdigest(),
        "reward_shape": str(args.reward_shape),
        "reward_config": {
            "native-v1": {
                "source": "scenario-native",
                "monster_kill_rewards": list(NATIVE_MONSTER_KILL_REWARDS),
            },
            "native-death-v1": {
                "source": "scenario-native",
                "monster_kill_rewards": list(NATIVE_MONSTER_KILL_REWARDS),
                "terminal_death_penalty": float(args.death_penalty),
            },
            "killcount-v1": {"killcount_delta_reward": 1.0},
            "sample-factory-v0": asdict(SAMPLE_FACTORY_REWARD),
        }[str(args.reward_shape)],
        "return_comparability": (
            {
                "native-v1": "scenario-native return and kills",
                "native-death-v1": "native-plus-death-cost return and kills",
                "killcount-v1": "uniform kill-count return and kills",
                "sample-factory-v0": "exact sample-factory-v0 shaped return and kills",
            }[str(args.reward_shape)]
        ),
        "requested_timesteps": int(args.timesteps),
        "execution_timesteps": _execution_timesteps(args),
        "rollout_transitions": int(args.num_envs) * int(args.n_steps),
        "initialization": {
            "checkpoint": initialization_checkpoint,
            "checkpoint_sha256": initialization_sha256,
            "mode": "policy-weights-only" if initialization_checkpoint is not None else "random",
        },
        "evaluation": {
            "checkpoint": (
                None if args.evaluate_checkpoint is None else str(args.evaluate_checkpoint)
            ),
            "episodes": int(args.evaluation_episodes),
            "num_envs": int(args.evaluation_num_envs),
            "seed": int(args.evaluation_seed),
            "stochastic_actions": bool(args.evaluation_stochastic),
            "kills_target": REFERENCE_KILLS_TARGET,
        },
        "effective_recipe": effective,
        "policy_model": {
            "architecture": str(args.policy_architecture),
            "memory_format": str(args.policy_memory_format),
            "observation_encoder": "nature_cnn",
            "convolution_channels": list(
                POLICY_ARCHITECTURES[str(args.policy_architecture)].convolution_channels
            ),
            "observation_features": POLICY_ARCHITECTURES[
                str(args.policy_architecture)
            ].observation_features,
            "context_history_frames": MODEL_CONTEXT_FRAMES,
            "context_features": CONTEXT_FEATURES,
            "fusion_features": POLICY_ARCHITECTURES[str(args.policy_architecture)].fusion_features,
            "fusion_activation": "tanh",
            "shared_actor_critic_features": True,
            "normalize_images": True,
            "orthogonal_init": True,
            "observation_encoder_trainable": not bool(args.freeze_observation_encoder),
            "frozen_encoder_custom_conv": bool(
                args.freeze_observation_encoder and args.frozen_encoder_custom_conv
            ),
            "ppo_update_input": (
                "cached_observation_features" if bool(args.freeze_observation_encoder) else "pixels"
            ),
            "training_only_privileged_imitation": (float(args.privileged_imitation_coef) > 0.0),
        },
        "environment": {
            "provider": "gradoom",
            "game": "VizdoomDeathmatch-v1",
            "doom_skill": REFERENCE_RECIPE.doom_skill,
            "wall_contact_damage_scale": float(args.wall_contact_damage_scale),
            "episode_timeout": REFERENCE_RECIPE.episode_timeout,
            "frame_skip": REFERENCE_RECIPE.frame_skip,
            "frame_stack": FRAME_STACK,
            "episode_seed_protocol": "gradlab-vizdoom-turbo-v1",
            "observation_shape": [4, 84, 84],
            "observation_grayscale": True,
            "observation_layout": "chw",
            "observation_resize_algorithm": "area",
            "hud_mask": [0, 32, 0, 0],
            "action_count": REFERENCE_RECIPE.action_count,
            "action_table": [list(action) for action in RESTRICTED_ACTIONS],
        },
        "tracking": {
            "wandb_enabled": bool(args.wandb),
            "wandb_project": str(args.wandb_project),
            "wandb_entity": args.wandb_entity,
            "wandb_group": args.wandb_group,
            "wandb_mode": str(args.wandb_mode),
            "wandb_provider_tag": GRADOOM_WANDB_TAG,
            "wandb_metrics": list(GRADLAB_WANDB_METRICS),
        },
    }


class JsonEmitter:
    def __init__(self, path: Path | None) -> None:
        self.path = None if path is None else path.expanduser().resolve()
        self.wandb_run: Any | None = None

    def attach_wandb(self, run: Any) -> None:
        self.wandb_run = run

    def emit(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        print(line, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        if self.wandb_run is not None and payload.get("type") == "rollout":
            wandb_payload: dict[str, int | float] = {
                "global_step": int(payload["train/global_step"]),
            }
            for metric in GRADLAB_WANDB_METRICS:
                value = payload.get(metric)
                if value is not None:
                    wandb_payload[metric] = float(value)
            self.wandb_run.log(wandb_payload)


def _wandb_tags(additional: str) -> list[str]:
    requested = [tag.strip() for tag in additional.split(",") if tag.strip()]
    standard = [
        "goal_id:VizdoomDeathmatch-v1",
        "recipe_id:ppo",
        "env_id:VizdoomDeathmatch-v1",
        GRADOOM_WANDB_TAG,
    ]
    return list(dict.fromkeys([*standard, *requested]))


def _init_wandb(args: argparse.Namespace, audit: Mapping[str, Any]) -> Any | None:
    if not bool(args.wandb):
        return None
    try:
        import wandb
    except ImportError as error:  # pragma: no cover - packaging invariant
        raise RuntimeError("--wandb requires the project's wandb dependency") from error

    init_kwargs: dict[str, Any] = {
        "project": str(args.wandb_project),
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "name": args.run_name,
        "notes": args.run_description or None,
        "tags": _wandb_tags(str(args.wandb_tags)),
        "config": dict(audit),
        "mode": str(args.wandb_mode),
        "job_type": "train",
        "save_code": True,
    }
    if args.metrics_jsonl is not None:
        init_kwargs["dir"] = str(args.metrics_jsonl.expanduser().resolve().parent)
    run = wandb.init(**init_kwargs)
    run.define_metric("global_step")
    for metric in GRADLAB_WANDB_METRICS:
        run.define_metric(metric, step_metric="global_step")
    return run


class CombatContextEncoder:
    """Encode the standalone policy's short combat history on device."""

    def __init__(self, history_names: Sequence[str], device: torch.device) -> None:
        indices = {name: index for index, name in enumerate(history_names)}
        missing = sorted(set(MODEL_HISTORY_SIGNALS) - set(indices))
        if missing:
            raise ValueError(f"GraDOOM context histories are missing: {missing}")
        self.armor = indices["armor"]
        self.health = indices["health"]
        self.selected_weapon = indices["selected_weapon"]
        self.selected_weapon_ammo = indices["selected_weapon_ammo"]
        self.ammo = torch.tensor(
            [indices[f"ammo{slot}"] for slot in range(1, 7)],
            dtype=torch.int64,
            device=device,
        )
        self.weapons = torch.tensor(
            [indices[f"weapon{slot}"] for slot in range(1, 7)],
            dtype=torch.int64,
            device=device,
        )
        self.categories = torch.arange(1, 7, dtype=torch.float32, device=device)
        self.ammo_scale = torch.tensor(
            [1.0, 0.005, 0.02, 0.005, 0.02, 1.0 / 300.0],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 6)

    def encode(self, histories: torch.Tensor) -> torch.Tensor:
        if histories.ndim != 3 or histories.shape[2] != FRAME_STACK:
            raise ValueError(f"context histories must have shape (N, signals, {FRAME_STACK})")
        current = histories[:, :, -MODEL_CONTEXT_FRAMES:]
        armor = (current[:, self.armor] * 0.005).clamp_(0.0, 1.0)
        health = (current[:, self.health] * 0.01).clamp_(0.0, 2.0)
        selected_raw = current[:, self.selected_weapon]
        selected_indices = torch.argmax(
            (selected_raw[..., None] == self.categories).to(torch.int64),
            dim=-1,
        )
        selected_one_hot = F.one_hot(selected_indices, num_classes=6).to(torch.float32)
        selected_ammo = (current[:, self.selected_weapon_ammo] / 300.0).clamp_(0.0, 1.0)
        ammo = (current.index_select(1, self.ammo).transpose(1, 2) * self.ammo_scale).clamp_(
            0.0, 1.0
        )
        weapons = current.index_select(1, self.weapons).transpose(1, 2).clamp_(0.0, 1.0)
        per_frame = torch.cat(
            (
                armor[..., None],
                health[..., None],
                selected_one_hot,
                selected_ammo[..., None],
                ammo,
                weapons,
            ),
            dim=2,
        )
        return per_frame.flatten(1)


class PrivilegedCombatTeacher:
    """Label visible combat states without changing the deployed policy inputs."""

    def __init__(self, env: Any, device: torch.device) -> None:
        self.engine = env._engine
        self.rows = torch.arange(env.num_envs, dtype=torch.int64, device=device)
        self.turn_threshold = math.radians(7.0)

    def actions(self) -> tuple[torch.Tensor, torch.Tensor]:
        engine = self.engine
        delta_x = engine.enemy_x - engine.x[:, None]
        delta_y = engine.enemy_y - engine.y[:, None]
        distance_squared = delta_x.square() + delta_y.square()
        blocked = engine._sight_blocked(
            engine.x[:, None],
            engine.y[:, None],
            engine.z[:, None] + 36.0,
            engine.enemy_x,
            engine.enemy_y,
            engine.enemy_z,
            engine._effective_enemy_height(),
        )
        visible = engine.enemy_alive & ~blocked
        valid = torch.any(visible, dim=1)
        target_scores = torch.where(
            visible,
            distance_squared,
            torch.full_like(distance_squared, torch.inf),
        )
        target = torch.argmin(target_scores, dim=1)
        target_delta = (
            torch.atan2(
                delta_y[self.rows, target],
                delta_x[self.rows, target],
            )
            - engine.angle
        )
        target_delta = torch.atan2(torch.sin(target_delta), torch.cos(target_delta))
        target_distance = torch.sqrt(distance_squared[self.rows, target])

        turn_left = target_delta > self.turn_threshold
        turn_right = target_delta < -self.turn_threshold
        strafe_left = torch.full_like(target, 11)
        strafe_right = torch.full_like(target, 12)
        strafe = torch.where(
            torch.bitwise_and(engine.episode_time // 32, 1) == 0,
            strafe_left,
            strafe_right,
        )
        aligned = torch.where(target_distance < 192.0, torch.full_like(target, 10), strafe)
        actions = torch.where(
            turn_left,
            torch.full_like(target, 13),
            torch.where(turn_right, torch.full_like(target, 14), aligned),
        )
        return torch.where(valid, actions, torch.full_like(actions, 8)), valid


class SampleFactoryDeathmatchReward:
    """GPU-resident port of GradLab's registered Deathmatch reward kernel."""

    _SCALAR_NAMES = ("killcount", "deathcount", "hitcount", "damagecount", "health", "armor")

    def __init__(
        self,
        signal_names: Sequence[str],
        num_envs: int,
        device: torch.device,
        *,
        compile_reward: bool,
    ) -> None:
        indices = {name: index for index, name in enumerate(signal_names)}
        required = {
            *self._SCALAR_NAMES,
            "selected_weapon",
            "selected_weapon_ammo",
            "player_dead",
            *(f"weapon{slot}" for slot in range(1, 7)),
            *(f"ammo{slot}" for slot in range(1, 7)),
        }
        missing = sorted(required - set(indices))
        if missing:
            raise ValueError(f"sample-factory-v0 signals are missing: {missing}")
        self.scalar_indices = torch.tensor(
            [indices[name] for name in self._SCALAR_NAMES],
            dtype=torch.int64,
            device=device,
        )
        self.weapon_indices = torch.tensor(
            [indices[f"weapon{slot}"] for slot in range(1, 7)],
            dtype=torch.int64,
            device=device,
        )
        self.ammo_indices = torch.tensor(
            [indices[f"ammo{slot}"] for slot in range(1, 7)],
            dtype=torch.int64,
            device=device,
        )
        self.selected_weapon_index = indices["selected_weapon"]
        self.selected_weapon_ammo_index = indices["selected_weapon_ammo"]
        self.player_dead_index = indices["player_dead"]
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.previous_dead = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.previous_scalars = torch.zeros((num_envs, 6), device=device)
        self.previous_weapons = torch.zeros((num_envs, 6), device=device)
        self.previous_ammo = torch.zeros((num_envs, 6), device=device)
        self.held_weapon = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.held_steps = torch.zeros(num_envs, dtype=torch.int64, device=device)

        config = SAMPLE_FACTORY_REWARD
        self.increase_coefficients = torch.tensor(
            (
                config.kill_reward,
                -config.death_penalty,
                config.hit_reward,
                config.damage_reward,
                config.health_gain_reward,
                config.armor_gain_reward,
            ),
            device=device,
        )
        self.decrease_coefficients = torch.tensor(
            (
                -config.kill_loss_penalty,
                config.death_count_decrease_reward,
                -config.hit_count_decrease_penalty,
                -config.damage_count_decrease_penalty,
                -config.health_loss_penalty,
                -config.armor_loss_penalty,
            ),
            device=device,
        )
        self.increase_caps = torch.tensor(
            (math.inf, math.inf, config.hit_delta_cap, config.damage_delta_cap, math.inf, math.inf),
            device=device,
        )
        self.weapon_preferences = torch.tensor(config.weapon_preferences, device=device)
        self.preference_lookup = torch.tensor(
            (0.0, *config.weapon_preferences),
            device=device,
        )
        process = self._process
        self.process = (
            torch.compile(process, dynamic=False, fullgraph=True) if compile_reward else process
        )

    @staticmethod
    def _delta_component(
        current: torch.Tensor,
        previous: torch.Tensor,
        increase_coefficients: torch.Tensor,
        decrease_coefficients: torch.Tensor,
        increase_caps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = current - previous
        increase = torch.clamp_min(delta, 0.0)
        if increase_caps is not None:
            increase = torch.minimum(increase, increase_caps)
        decrease = torch.clamp_min(-delta, 0.0)
        return increase * increase_coefficients + decrease * decrease_coefficients

    def _process(
        self,
        final_signals: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        current_scalars = final_signals.index_select(1, self.scalar_indices)
        current_weapons = final_signals.index_select(1, self.weapon_indices)
        current_ammo = final_signals.index_select(1, self.ammo_indices)
        selected_weapon = final_signals[:, self.selected_weapon_index].to(torch.int64).clamp_min(0)
        selected_weapon_ammo = final_signals[:, self.selected_weapon_ammo_index]
        current_dead = final_signals[:, self.player_dead_index] != 0
        pure_timeout = truncated & ~terminated
        first_after_reset = self.previous_dead & ~current_dead
        active = self.initialized & ~first_after_reset & ~pure_timeout

        scalar_components = self._delta_component(
            current_scalars,
            self.previous_scalars,
            self.increase_coefficients,
            self.decrease_coefficients,
            self.increase_caps,
        ).sum(dim=1)
        config = SAMPLE_FACTORY_REWARD
        weapon_components = self._delta_component(
            current_weapons,
            self.previous_weapons,
            self.weapon_preferences * config.weapon_gain_reward_scale,
            -self.weapon_preferences * config.weapon_loss_penalty_scale,
        ).sum(dim=1)
        ammo_components = self._delta_component(
            current_ammo,
            self.previous_ammo,
            self.weapon_preferences * config.ammo_gain_reward_scale,
            -self.weapon_preferences * config.ammo_loss_penalty_scale,
        ).sum(dim=1)

        next_held_steps = torch.where(
            selected_weapon == self.held_weapon,
            self.held_steps + 1,
            torch.ones_like(self.held_steps),
        )
        valid_hold = (
            active
            & (selected_weapon >= 1)
            & (selected_weapon <= 6)
            & (selected_weapon_ammo > 0)
            & (next_held_steps >= config.selected_weapon_hold_steps)
        )
        safe_weapon = selected_weapon.clamp(0, 6)
        hold_component = torch.where(
            valid_hold,
            self.preference_lookup[safe_weapon] * config.selected_weapon_hold_reward_scale,
            torch.zeros_like(scalar_components),
        )
        reward = torch.where(
            active,
            scalar_components + weapon_components + ammo_components + hold_component,
            torch.zeros_like(scalar_components),
        ).to(torch.float32)

        done = terminated | truncated
        self.previous_scalars.copy_(
            torch.where(done[:, None], torch.zeros_like(current_scalars), current_scalars)
        )
        self.previous_weapons.copy_(
            torch.where(done[:, None], torch.zeros_like(current_weapons), current_weapons)
        )
        self.previous_ammo.copy_(
            torch.where(done[:, None], torch.zeros_like(current_ammo), current_ammo)
        )
        self.previous_dead.copy_(torch.where(done, torch.ones_like(current_dead), current_dead))
        self.initialized.copy_(~done)
        self.held_weapon.copy_(
            torch.where(done, torch.zeros_like(selected_weapon), selected_weapon)
        )
        self.held_steps.copy_(torch.where(done, torch.zeros_like(next_held_steps), next_held_steps))
        return reward


class KillcountReward:
    """GPU-resident uniform reward for each KILLCOUNT increment."""

    def __init__(
        self,
        signal_names: Sequence[str],
        num_envs: int,
        device: torch.device,
        *,
        compile_reward: bool,
    ) -> None:
        try:
            self.kill_index = tuple(signal_names).index("killcount")
        except ValueError as exc:
            raise ValueError("killcount-v1 signals are missing: ['killcount']") from exc
        self.previous_kills = torch.zeros(num_envs, dtype=torch.float32, device=device)
        process = self._process
        self.process = (
            torch.compile(process, dynamic=False, fullgraph=True) if compile_reward else process
        )

    def _process(
        self,
        final_signals: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        current_kills = final_signals[:, self.kill_index]
        reward = torch.clamp_min(current_kills - self.previous_kills, 0.0).to(torch.float32)
        done = terminated | truncated
        self.previous_kills.copy_(torch.where(done, torch.zeros_like(current_kills), current_kills))
        return reward


class NatureActorCritic(nn.Module):
    """Shared NatureCNN actor-critic with a fixed, audited width profile."""

    def __init__(
        self,
        architecture: str = "nature",
        memory_format: str = "contiguous",
    ) -> None:
        super().__init__()
        if memory_format not in ("contiguous", "channels-last"):
            raise ValueError(f"unsupported policy memory format: {memory_format}")
        self.channels_last = memory_format == "channels-last"
        self.use_frozen_encoder_custom_conv = False
        profile = POLICY_ARCHITECTURES[architecture]
        self.observation_feature_count = profile.observation_features
        first_channels, second_channels, third_channels = profile.convolution_channels
        self.observation_encoder = nn.Sequential(
            nn.Conv2d(4, first_channels, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(first_channels, second_channels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(second_channels, third_channels, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(third_channels * 7 * 7, profile.observation_features),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                profile.observation_features + CONTEXT_FEATURES,
                profile.fusion_features,
            ),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(
            profile.fusion_features,
            REFERENCE_RECIPE.action_count,
        )
        self.value_head = nn.Linear(profile.fusion_features, 1)
        self._orthogonal_initialize()

    @staticmethod
    def _initialize_module(module: nn.Module, gain: float) -> None:
        if isinstance(module, nn.Conv2d | nn.Linear):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _orthogonal_initialize(self) -> None:
        gain = math.sqrt(2.0)
        self.observation_encoder.apply(lambda module: self._initialize_module(module, gain))
        self.fusion.apply(lambda module: self._initialize_module(module, gain))
        self._initialize_module(self.action_head, 0.01)
        self._initialize_module(self.value_head, 1.0)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if self.use_frozen_encoder_custom_conv:
            first_convolution = self.observation_encoder[0]
            if not isinstance(first_convolution, nn.Conv2d):  # pragma: no cover - invariant
                raise RuntimeError("NatureCNN first encoder layer is not a convolution")
            if first_convolution.bias is None:  # pragma: no cover - invariant
                raise RuntimeError("NatureCNN first convolution requires a bias")
            encoded = frozen_nature_conv1(
                observations,
                first_convolution.weight,
                first_convolution.bias,
            )
            encoded = F.relu(self.observation_encoder[2](encoded))
            encoded = F.relu(self.observation_encoder[4](encoded))
            encoded = torch.flatten(encoded, start_dim=1)
            return F.relu(self.observation_encoder[7](encoded))
        normalized = observations.float() / 255.0
        if self.channels_last:
            normalized = normalized.contiguous(memory_format=torch.channels_last)
        return self.observation_encoder(normalized)

    def features_from_encoded(
        self,
        encoded_observations: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.fusion(torch.cat((encoded_observations, context), dim=1))

    def features(self, observations: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.features_from_encoded(self.encode_observations(observations), context)

    def _act_from_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = torch.distributions.Categorical(logits=self.action_head(features))
        actions = distribution.sample()
        values = self.value_head(features).flatten()
        return actions, values, distribution.log_prob(actions)

    def act(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._act_from_features(self.features(observations, context))

    def act_and_encode(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_observations = self.encode_observations(observations)
        actions, values, log_probs = self._act_from_features(
            self.features_from_encoded(encoded_observations, context)
        )
        return actions, values, log_probs, encoded_observations

    def evaluate_encoded_actions(
        self,
        encoded_observations: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features_from_encoded(encoded_observations, context)
        logits = self.action_head(features)
        distribution = torch.distributions.Categorical(logits=logits)
        values = self.value_head(features).flatten()
        return values, distribution.log_prob(actions), distribution.entropy(), logits

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.evaluate_encoded_actions(
            self.encode_observations(observations),
            context,
            actions,
        )

    def value(self, observations: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(observations, context)).flatten()

    def deterministic_action(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(observations, context)
        return torch.argmax(self.action_head(features), dim=1)


class PolicyCalls:
    def __init__(self, policy: NatureActorCritic, *, compile_policy: bool) -> None:
        if compile_policy:
            self.act = torch.compile(policy.act, dynamic=False, fullgraph=False)
            self.act_and_encode = torch.compile(
                policy.act_and_encode,
                dynamic=False,
                fullgraph=False,
            )
            self.evaluate_actions = torch.compile(
                policy.evaluate_actions,
                dynamic=False,
                fullgraph=False,
            )
            self.evaluate_encoded_actions = torch.compile(
                policy.evaluate_encoded_actions,
                dynamic=False,
                fullgraph=False,
            )
            self.value = torch.compile(policy.value, dynamic=True, fullgraph=False)
            self.deterministic_action = torch.compile(
                policy.deterministic_action,
                dynamic=False,
                fullgraph=False,
            )
        else:
            self.act = policy.act
            self.act_and_encode = policy.act_and_encode
            self.evaluate_actions = policy.evaluate_actions
            self.evaluate_encoded_actions = policy.evaluate_encoded_actions
            self.value = policy.value
            self.deterministic_action = policy.deterministic_action


class Precision:
    def __init__(self, name: str, device: torch.device) -> None:
        if name != "fp32" and device.type != "cuda":
            raise ValueError(f"{name} precision requires CUDA")
        self.name = name
        self.device = device
        self.dtype = {
            "fp32": torch.float32,
            "amp-fp16": torch.float16,
            "amp-bf16": torch.bfloat16,
        }[name]
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=name == "amp-fp16" and device.type == "cuda",
        )

    def autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.name != "fp32",
        )


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        device: torch.device,
        *,
        observation_feature_count: int | None = None,
    ) -> None:
        batch = (n_steps, n_envs)
        observations = (n_steps, n_envs, 4, 84, 84)
        contexts = (n_steps, n_envs, CONTEXT_FEATURES)
        histories = (n_steps, n_envs, len(MODEL_HISTORY_SIGNALS), FRAME_STACK)
        self.observations = (
            torch.empty(observations, dtype=torch.uint8, device=device)
            if observation_feature_count is None
            else None
        )
        self.observation_features = (
            None
            if observation_feature_count is None
            else torch.empty(
                (n_steps, n_envs, int(observation_feature_count)),
                dtype=torch.float32,
                device=device,
            )
        )
        self.context = torch.empty(contexts, dtype=torch.float32, device=device)
        self.final_observations = torch.empty(observations, dtype=torch.uint8, device=device)
        self.final_histories = torch.empty(histories, dtype=torch.float32, device=device)
        self.actions = torch.empty(batch, dtype=torch.int64, device=device)
        self.teacher_actions = torch.empty(batch, dtype=torch.int64, device=device)
        self.teacher_valid = torch.empty(batch, dtype=torch.bool, device=device)
        self.rewards = torch.empty(batch, dtype=torch.float32, device=device)
        self.episode_starts = torch.empty(batch, dtype=torch.bool, device=device)
        self.values = torch.empty(batch, dtype=torch.float32, device=device)
        self.log_probs = torch.empty(batch, dtype=torch.float32, device=device)
        self.advantages = torch.empty(batch, dtype=torch.float32, device=device)
        self.returns = torch.empty(batch, dtype=torch.float32, device=device)
        self.truncated = torch.empty(batch, dtype=torch.bool, device=device)
        self.completed = torch.empty(batch, dtype=torch.bool, device=device)
        self.completed_returns = torch.empty(batch, dtype=torch.float32, device=device)
        self.completed_kills = torch.empty(batch, dtype=torch.float32, device=device)
        self.completed_lengths = torch.empty(batch, dtype=torch.int32, device=device)
        self.completed_success = torch.empty(batch, dtype=torch.bool, device=device)
        self.position = 0

    @property
    def n_steps(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def n_envs(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def size(self) -> int:
        return self.n_steps * self.n_envs

    def reset(self) -> None:
        self.position = 0

    def stage(
        self,
        observations: torch.Tensor,
        context: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position >= self.n_steps:
            raise RuntimeError("rollout buffer overflow")
        if self.observations is not None:
            self.observations[self.position].copy_(observations)
            staged_observations = self.observations[self.position]
        else:
            staged_observations = observations
        self.context[self.position].copy_(context)
        self.episode_starts[self.position].copy_(episode_starts)
        return staged_observations, self.context[self.position]

    def add(
        self,
        *,
        actions: torch.Tensor,
        teacher_actions: torch.Tensor,
        teacher_valid: torch.Tensor,
        rewards: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        observation_features: torch.Tensor | None,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        final_observations: torch.Tensor,
        final_histories: torch.Tensor,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
        final_kills: torch.Tensor,
    ) -> None:
        step = self.position
        self.actions[step].copy_(actions)
        self.teacher_actions[step].copy_(teacher_actions)
        self.teacher_valid[step].copy_(teacher_valid)
        self.rewards[step].copy_(rewards)
        self.values[step].copy_(values.float())
        self.log_probs[step].copy_(log_probs.float())
        if self.observation_features is None:
            if observation_features is not None:
                raise ValueError("rollout buffer was not configured for observation features")
        elif observation_features is None:
            raise ValueError("cached observation features are required by this rollout buffer")
        else:
            self.observation_features[step].copy_(observation_features.float())
        self.truncated[step].copy_(truncated)
        self.final_observations[step].copy_(final_observations)
        self.final_histories[step].copy_(final_histories)
        torch.logical_or(terminated, truncated, out=self.completed[step])
        self.completed_returns[step].copy_(episode_returns)
        self.completed_kills[step].copy_(final_kills)
        self.completed_lengths[step].copy_(episode_lengths)
        torch.logical_and(truncated, ~terminated, out=self.completed_success[step])
        self.position += 1

    def finish(
        self,
        *,
        last_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.n_steps:
            raise RuntimeError("cannot finish an incomplete rollout")
        last_gae = torch.zeros(self.n_envs, dtype=torch.float32, device=self.rewards.device)
        for step in range(self.n_steps - 1, -1, -1):
            if step == self.n_steps - 1:
                next_non_terminal = ~dones
                next_values = last_values.float()
            else:
                next_non_terminal = ~self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + float(gamma) * next_values * next_non_terminal.float()
                - self.values[step]
            )
            last_gae = (
                delta + float(gamma) * float(gae_lambda) * next_non_terminal.float() * last_gae
            )
            self.advantages[step].copy_(last_gae)
        self.returns.copy_(self.advantages + self.values)

    def completed_episode_rows(self) -> list[list[float]]:
        values = torch.stack(
            (
                self.completed_returns,
                self.completed_kills,
                self.completed_lengths.float(),
                self.completed_success.float(),
            ),
            dim=2,
        )
        return values[self.completed].detach().cpu().tolist()


def _bootstrap_time_limits(
    buffer: RolloutBuffer,
    *,
    calls: PolicyCalls,
    context_encoder: CombatContextEncoder,
    precision: Precision,
    gamma: float,
) -> None:
    flat_truncated = buffer.truncated.flatten()
    indices = torch.nonzero(flat_truncated, as_tuple=False).flatten()
    safe_indices = torch.cat((indices, torch.zeros(1, dtype=torch.int64, device=indices.device)))
    flat_observations = buffer.final_observations.flatten(0, 1)
    flat_histories = buffer.final_histories.flatten(0, 1)
    selected_observations = flat_observations.index_select(0, safe_indices)
    selected_histories = flat_histories.index_select(0, safe_indices)
    selected_context = context_encoder.encode(selected_histories)
    with torch.no_grad(), precision.autocast():
        selected_values = calls.value(selected_observations, selected_context).float()
    bootstrap = torch.zeros_like(flat_truncated, dtype=torch.float32)
    bootstrap.index_copy_(0, indices, selected_values[:-1])
    buffer.rewards.add_(bootstrap.view_as(buffer.rewards) * float(gamma))


def _flatten(value: torch.Tensor, *, env_major: bool) -> torch.Tensor:
    return value.transpose(0, 1).flatten(0, 1) if env_major else value.flatten(0, 1)


def _ppo_update(
    policy: NatureActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    *,
    calls: PolicyCalls,
    precision: Precision,
    args: argparse.Namespace,
) -> dict[str, float]:
    policy.train()
    env_major = not bool(args.torch_permutation)
    observations = (
        None if buffer.observations is None else _flatten(buffer.observations, env_major=env_major)
    )
    observation_features = (
        None
        if buffer.observation_features is None
        else _flatten(buffer.observation_features, env_major=env_major)
    )
    context = _flatten(buffer.context, env_major=env_major)
    actions = _flatten(buffer.actions, env_major=env_major)
    imitation_enabled = float(args.privileged_imitation_coef) > 0.0
    teacher_actions = (
        _flatten(buffer.teacher_actions, env_major=env_major) if imitation_enabled else None
    )
    teacher_valid = (
        _flatten(buffer.teacher_valid, env_major=env_major) if imitation_enabled else None
    )
    old_values = _flatten(buffer.values, env_major=env_major)
    old_log_probs = _flatten(buffer.log_probs, env_major=env_major)
    advantages = _flatten(buffer.advantages, env_major=env_major)
    returns = _flatten(buffer.returns, env_major=env_major)
    metric_sums = torch.zeros(6, dtype=torch.float32, device=buffer.rewards.device)
    metric_count = 0
    last_epoch_kl_sum = torch.zeros((), dtype=torch.float32, device=buffer.rewards.device)
    last_epoch_kl_count = 0

    for _epoch in range(int(args.n_epochs)):
        last_epoch_kl_sum.zero_()
        last_epoch_kl_count = 0
        if args.torch_permutation:
            indices = torch.randperm(buffer.size, device=buffer.rewards.device)
        else:
            indices = torch.as_tensor(
                np.random.permutation(buffer.size),
                dtype=torch.int64,
                device=buffer.rewards.device,
            )
        for start in range(0, buffer.size, int(args.batch_size)):
            batch = indices[start : start + int(args.batch_size)]
            batch_context = context.index_select(0, batch)
            batch_actions = actions.index_select(0, batch)
            batch_old_log_probs = old_log_probs.index_select(0, batch)
            batch_advantages = advantages.index_select(0, batch)
            batch_returns = returns.index_select(0, batch)
            if batch_advantages.numel() > 1:
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )

            with precision.autocast():
                if observation_features is None:
                    if observations is None:  # pragma: no cover - constructor invariant
                        raise RuntimeError("rollout has neither pixels nor encoded observations")
                    values, log_probs, entropy, logits = calls.evaluate_actions(
                        observations.index_select(0, batch),
                        batch_context,
                        batch_actions,
                    )
                else:
                    values, log_probs, entropy, logits = calls.evaluate_encoded_actions(
                        observation_features.index_select(0, batch),
                        batch_context,
                        batch_actions,
                    )
                ratio = torch.exp(log_probs - batch_old_log_probs)
                policy_loss = -torch.min(
                    batch_advantages * ratio,
                    batch_advantages
                    * torch.clamp(
                        ratio,
                        1.0 - REFERENCE_RECIPE.clip_range,
                        1.0 + REFERENCE_RECIPE.clip_range,
                    ),
                ).mean()
                value_loss = F.mse_loss(batch_returns, values)
                entropy_loss = -entropy.mean()
                if imitation_enabled:
                    if teacher_actions is None or teacher_valid is None:  # pragma: no cover
                        raise RuntimeError("imitation tensors were not prepared")
                    batch_teacher_actions = teacher_actions.index_select(0, batch)
                    batch_teacher_valid = teacher_valid.index_select(0, batch)
                    imitation_per_sample = F.cross_entropy(
                        logits,
                        batch_teacher_actions,
                        reduction="none",
                    )
                    imitation_weight = batch_teacher_valid.to(imitation_per_sample.dtype)
                    imitation_count = imitation_weight.sum().clamp_min(1.0)
                    imitation_loss = (imitation_per_sample * imitation_weight).sum() / (
                        imitation_count
                    )
                    imitation_accuracy = (
                        (torch.argmax(logits, dim=1) == batch_teacher_actions).to(
                            imitation_per_sample.dtype
                        )
                        * batch_teacher_valid.to(imitation_per_sample.dtype)
                    ).sum() / batch_teacher_valid.sum().clamp_min(1)
                else:
                    imitation_loss = logits.sum() * 0.0
                    imitation_accuracy = logits.new_zeros(())
                loss = (
                    policy_loss
                    + float(args.ent_coef) * entropy_loss
                    + REFERENCE_RECIPE.vf_coef * value_loss
                    + float(args.privileged_imitation_coef) * imitation_loss
                )

            with torch.no_grad():
                log_ratio = log_probs - batch_old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > REFERENCE_RECIPE.clip_range).float().mean()
                last_epoch_kl_sum.add_(approx_kl.detach().float())
                last_epoch_kl_count += 1
                metric_sums.add_(
                    torch.stack(
                        (
                            policy_loss.detach().float(),
                            value_loss.detach().float(),
                            entropy.detach().mean().float(),
                            clip_fraction,
                            imitation_loss.detach().float(),
                            imitation_accuracy.detach().float(),
                        )
                    )
                )
                metric_count += 1

            optimizer.zero_grad(set_to_none=True)
            if precision.scaler.is_enabled():
                precision.scaler.scale(loss).backward()
                precision.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), REFERENCE_RECIPE.max_grad_norm)
                precision.scaler.step(optimizer)
                precision.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), REFERENCE_RECIPE.max_grad_norm)
                optimizer.step()

    returns_variance = torch.var(returns, correction=0)
    explained_variance = torch.where(
        returns_variance == 0.0,
        torch.full_like(returns_variance, float("nan")),
        1.0 - torch.var(returns - old_values, correction=0) / returns_variance,
    )
    means = metric_sums / max(metric_count, 1)
    tensors = (
        torch.stack(
            (
                last_epoch_kl_sum / max(last_epoch_kl_count, 1),
                means[3],
                means[1],
                means[0],
                means[2],
                explained_variance,
            )
        )
        .detach()
        .float()
        .cpu()
        .tolist()
    )
    metrics = {
        "train/algorithm/ppo/update/approx_kl": float(tensors[0]),
        "train/algorithm/ppo/update/clip_fraction": float(tensors[1]),
        "train/algorithm/ppo/update/value_loss": float(tensors[2]),
        "train/algorithm/ppo/update/policy_gradient_loss": float(tensors[3]),
        "train/algorithm/ppo/policy/entropy": float(tensors[4]),
        "train/algorithm/ppo/update/learning_rate": _optimizer_learning_rate(optimizer),
        "train/algorithm/imitation/loss": float(means[4].detach().cpu()),
        "train/algorithm/imitation/accuracy": float(means[5].detach().cpu()),
        "train/algorithm/imitation/valid_rate": float(buffer.teacher_valid.float().mean().cpu()),
    }
    if math.isfinite(tensors[5]):
        metrics["train/algorithm/ppo/value/explained_variance"] = float(tensors[5])
    return metrics


def _rollout_diagnostics(buffer: RolloutBuffer) -> dict[str, float]:
    action_counts = torch.bincount(
        buffer.actions.flatten(),
        minlength=REFERENCE_RECIPE.action_count,
    )
    values = (
        torch.stack(
            (
                buffer.values.mean(),
                buffer.values.std(correction=0),
                buffer.advantages.mean(),
                buffer.advantages.std(correction=0),
                action_counts.max().float() / max(buffer.actions.numel(), 1),
            )
        )
        .detach()
        .float()
        .cpu()
        .tolist()
    )
    return {
        "train/algorithm/ppo/rollout/value/prediction/mean": float(values[0]),
        "train/algorithm/ppo/rollout/value/prediction/std": float(values[1]),
        "train/algorithm/ppo/rollout/advantage/mean": float(values[2]),
        "train/algorithm/ppo/rollout/advantage/std": float(values[3]),
        "train/algorithm/ppo/policy/dominant/action/rate": float(values[4]),
    }


def _make_optimizer(
    policy: NatureActorCritic,
    *,
    learning_rate: float,
    fused: bool,
) -> torch.optim.Adam:
    return torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
        eps=REFERENCE_RECIPE.adam_eps,
        fused=fused,
        foreach=False if fused else None,
        capturable=False,
    )


def _load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    state_dict: Mapping[str, Any],
    *,
    learning_rate: float,
) -> None:
    """Restore optimizer moments while honoring the resumed run's explicit LR."""

    optimizer.load_state_dict(state_dict)
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(learning_rate)


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    learning_rates = {float(group["lr"]) for group in optimizer.param_groups}
    if len(learning_rates) != 1:
        raise RuntimeError(
            f"optimizer parameter groups disagree on learning rate: {learning_rates}"
        )
    return learning_rates.pop()


def _make_env(
    args: argparse.Namespace,
    device: torch.device,
    *,
    num_envs: int | None = None,
):
    from gradoom import GraDoomVecEnv

    return GraDoomVecEnv(
        game="VizdoomDeathmatch-v1",
        scenario=None if args.scenario is None else str(args.scenario),
        use_restricted_actions=RESTRICTED_ACTIONS,
        rom_path=str(args.iwad),
        num_envs=int(args.num_envs) if num_envs is None else int(num_envs),
        num_threads=1,
        device=device,
        info="data",
        obs_resize=(84, 84),
        obs_crop=(0, 32, 0, 0),
        obs_crop_mode="mask",
        obs_crop_fill=0,
        obs_grayscale=True,
        obs_resize_algorithm="area",
        obs_layout="chw",
        obs_copy="safe_view",
        frame_skip=REFERENCE_RECIPE.frame_skip,
        frame_stack=FRAME_STACK,
        maxpool_last_two=False,
        noop_reset_max=0,
        sticky_action_prob=0.0,
        reward_clip=False,
        info_filter={"mode": "all", "keys": list(INFO_SIGNALS)},
        info_frame_stack_keys=MODEL_HISTORY_SIGNALS,
        doom_skill=REFERENCE_RECIPE.doom_skill,
        wall_contact_damage_scale=float(args.wall_contact_damage_scale),
        game_variables=GAME_VARIABLES,
        treat_episode_timeout_as_truncation=True,
        vizdoom_config={"episode_timeout": REFERENCE_RECIPE.episode_timeout},
        require_pinned_scenario=True,
        compile_engine=bool(args.compile_engine),
    )


class GradLabEpisodeSeeds:
    """Reproduce BatchRuntime + ViZDoom-turbo's per-episode game seeds."""

    def __init__(self, run_seed: int, n_envs: int, device: torch.device) -> None:
        self.run_seed = int(run_seed)
        self.n_envs = int(n_envs)
        self.device = device
        self.capacity = 0
        self.table = torch.empty((self.n_envs, 0), dtype=torch.int64, device=device)
        self.ensure(SEED_TABLE_INITIAL_EPISODES - 1)

    def _episode_seed(self, lane: int, episode_index: int) -> int:
        if episode_index == 0:
            provider_seed = self.run_seed + lane
        else:
            sequence = np.random.SeedSequence([self.run_seed, lane, episode_index])
            provider_seed = int(sequence.generate_state(1, dtype=np.uint32)[0])
        generator = np.random.default_rng(provider_seed)
        return int(
            generator.integers(
                0,
                UINT32_MASK + 1,
                dtype=np.uint32,
            )
        )

    def ensure(self, episode_index: int) -> None:
        required = int(episode_index) + 1
        if required <= self.capacity:
            return
        new_capacity = max(required, SEED_TABLE_INITIAL_EPISODES, self.capacity * 2)
        extension = np.empty(
            (self.n_envs, new_capacity - self.capacity),
            dtype=np.int64,
        )
        for lane in range(self.n_envs):
            for index in range(self.capacity, new_capacity):
                extension[lane, index - self.capacity] = self._episode_seed(lane, index)
        extension_device = torch.from_numpy(extension).to(self.device)
        self.table = torch.cat((self.table, extension_device), dim=1)
        self.capacity = new_capacity

    def lookup(self, episode_indices: torch.Tensor) -> torch.Tensor:
        if episode_indices.shape != (self.n_envs,):
            raise ValueError(f"episode indices must have shape ({self.n_envs},)")
        return self.table.gather(1, episode_indices[:, None]).flatten()


def _rolling_mean(values: Sequence[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def _episode_quotas(episodes: int, num_envs: int) -> tuple[int, ...]:
    """Balance an exact episode count over stable lane seed streams."""

    quotient, remainder = divmod(int(episodes), int(num_envs))
    return tuple(quotient + int(lane < remainder) for lane in range(int(num_envs)))


def _restore_episode_indices(
    destination: torch.Tensor,
    saved: torch.Tensor | None,
    *,
    fallback_index: int,
) -> int:
    """Restore stable lane streams while permitting an explicit env-count change."""
    if saved is None:
        destination.fill_(int(fallback_index))
        return 0
    if saved.ndim != 1:
        raise ValueError("checkpoint episode_index must be one-dimensional")
    destination.zero_()
    preserved = min(destination.numel(), saved.numel())
    destination[:preserved].copy_(
        saved[:preserved].to(device=destination.device, dtype=torch.int64)
    )
    return preserved


def _save_checkpoint(
    path: Path,
    *,
    policy: NatureActorCritic,
    optimizer: torch.optim.Optimizer,
    step: int,
    audit: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    destination = _checkpoint_destination(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "standalone-gradoom-ppo-v1",
            "step": int(step),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(audit),
            "training_state": dict(training_state or {}),
        },
        destination,
    )
    return destination


def _periodic_checkpoint_path(path: Path, step: int) -> Path:
    destination = _checkpoint_destination(path)
    return destination.with_name(f"{destination.stem}.step-{int(step)}{destination.suffix}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("evaluation requires at least one completed episode")
    kills = [float(record["kills"]) for record in records]
    returns = [float(record["return"]) for record in records]
    lengths = [float(record["length"]) for record in records]
    mean_kills = statistics.fmean(kills)
    return {
        "evaluation/episode/count": len(records),
        "evaluation/kills/mean": mean_kills,
        "evaluation/kills/median": statistics.median(kills),
        "evaluation/kills/std": statistics.pstdev(kills),
        "evaluation/kills/min": min(kills),
        "evaluation/kills/max": max(kills),
        "evaluation/return/native/mean": statistics.fmean(returns),
        "evaluation/episode/length/mean": statistics.fmean(lengths),
        "evaluation/target/kills/mean": REFERENCE_KILLS_TARGET,
        "evaluation/target/passed": mean_kills >= REFERENCE_KILLS_TARGET,
    }


def _evaluate(
    args: argparse.Namespace,
    emitter: JsonEmitter,
    audit: Mapping[str, Any],
    *,
    process_started: float,
) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GraDOOM checkpoint evaluation")
    if args.evaluate_checkpoint is None:  # pragma: no cover - caller invariant
        raise ValueError("evaluate-checkpoint is required")
    random.seed(int(args.evaluation_seed))
    np.random.seed(int(args.evaluation_seed))
    torch.manual_seed(int(args.evaluation_seed))
    torch.cuda.manual_seed_all(int(args.evaluation_seed))
    device = torch.device("cuda")
    evaluation_envs = int(args.evaluation_num_envs)
    env = _make_env(args, device, num_envs=evaluation_envs)
    try:
        loaded = torch.load(args.evaluate_checkpoint, map_location=device, weights_only=False)
        if not isinstance(loaded, Mapping) or loaded.get("format") != "standalone-gradoom-ppo-v1":
            raise ValueError(f"unsupported evaluation checkpoint: {args.evaluate_checkpoint}")
        policy = NatureActorCritic(
            str(args.policy_architecture),
            str(args.policy_memory_format),
        ).to(
            device=device,
            memory_format=(
                torch.channels_last
                if str(args.policy_memory_format) == "channels-last"
                else torch.contiguous_format
            ),
        )
        policy.load_state_dict(loaded["policy_state_dict"])
        policy.use_frozen_encoder_custom_conv = bool(
            args.freeze_observation_encoder and args.frozen_encoder_custom_conv
        )
        policy.eval()
        calls = PolicyCalls(policy, compile_policy=bool(args.compile_policy))
        precision = Precision(str(args.precision), device)
        context_encoder = CombatContextEncoder(env.device_info_history_names, device)
        episode_index = torch.zeros(evaluation_envs, dtype=torch.int64, device=device)
        episode_seeds = GradLabEpisodeSeeds(
            int(args.evaluation_seed),
            evaluation_envs,
            device,
        )
        current_seeds = episode_seeds.lookup(episode_index)
        observations, _signals = env.reset_device(
            torch.ones(evaluation_envs, dtype=torch.bool, device=device),
            current_seeds,
        )
        context = context_encoder.encode(env.device_info_histories())
        episode_returns = torch.zeros(evaluation_envs, dtype=torch.float32, device=device)
        episode_lengths = torch.zeros(evaluation_envs, dtype=torch.int32, device=device)
        kill_index = {name: index for index, name in enumerate(env.device_signal_names)}[
            "killcount"
        ]
        episode_quotas = _episode_quotas(int(args.evaluation_episodes), evaluation_envs)
        decisions_per_episode = math.ceil(
            REFERENCE_RECIPE.episode_timeout / REFERENCE_RECIPE.frame_skip
        )
        maximum_decisions = max(episode_quotas) * (decisions_per_episode + 1)
        quota_tensor = torch.tensor(episode_quotas, dtype=torch.int64, device=device)
        episode_seeds.ensure(maximum_decisions + 1)
        completed = torch.empty(
            (maximum_decisions, evaluation_envs),
            dtype=torch.bool,
            device=device,
        )
        completed_kills = torch.empty(
            (maximum_decisions, evaluation_envs),
            dtype=torch.float32,
            device=device,
        )
        completed_returns = torch.empty_like(completed_kills)
        completed_lengths = torch.empty(
            (maximum_decisions, evaluation_envs),
            dtype=torch.int32,
            device=device,
        )
        completed_seeds = torch.empty(
            (maximum_decisions, evaluation_envs),
            dtype=torch.int64,
            device=device,
        )
        completed_episode_indices = torch.empty_like(completed_seeds)
        completed_terminated = torch.empty_like(completed)
        completed_truncated = torch.empty_like(completed)
        evaluation_started = time.perf_counter()
        emitter.emit(
            {
                "type": "event",
                "event": "evaluation_started",
                "checkpoint": str(args.evaluate_checkpoint),
                "checkpoint_step": int(loaded.get("step", 0)),
                "episodes": int(args.evaluation_episodes),
                "num_envs": evaluation_envs,
                "episode_quotas": list(episode_quotas),
                "seed_grid": "gradlab-vizdoom-turbo-v1 lanes x episode-index",
                "deterministic_actions": not bool(args.evaluation_stochastic),
            }
        )
        executed_decisions = maximum_decisions
        for decision in range(maximum_decisions):
            with torch.no_grad(), precision.autocast():
                if args.evaluation_stochastic:
                    actions, _values, _log_probs = calls.act(observations, context)
                else:
                    actions = calls.deterministic_action(observations, context)
            next_episode_index = episode_index + 1
            next_seeds = episode_seeds.lookup(next_episode_index)
            transition = env.step_and_reset_device(actions, next_seeds)
            episode_returns.add_(transition.rewards)
            episode_lengths.add_(1)
            done = transition.terminated | transition.truncated
            completed[decision].copy_(done)
            completed_kills[decision].copy_(transition.final_signals[:, kill_index])
            completed_returns[decision].copy_(episode_returns)
            completed_lengths[decision].copy_(episode_lengths)
            completed_seeds[decision].copy_(current_seeds)
            completed_episode_indices[decision].copy_(episode_index)
            completed_terminated[decision].copy_(transition.terminated)
            completed_truncated[decision].copy_(transition.truncated)
            episode_returns.masked_fill_(done, 0.0)
            episode_lengths.masked_fill_(done, 0)
            episode_index.add_(done.to(torch.int64))
            current_seeds.copy_(torch.where(done, next_seeds, current_seeds))
            observations = transition.observations
            context = context_encoder.encode(transition.info_histories)
            if decision % int(args.n_steps) == int(args.n_steps) - 1 and bool(
                torch.all(episode_index >= quota_tensor)
            ):
                executed_decisions = decision + 1
                break

        torch.cuda.synchronize(device)
        evaluation_seconds = time.perf_counter() - evaluation_started
        completed_cpu = completed[:executed_decisions].cpu().numpy()
        kills_cpu = completed_kills[:executed_decisions].cpu().numpy()
        returns_cpu = completed_returns[:executed_decisions].cpu().numpy()
        lengths_cpu = completed_lengths[:executed_decisions].cpu().numpy()
        seeds_cpu = completed_seeds[:executed_decisions].cpu().numpy()
        episode_indices_cpu = completed_episode_indices[:executed_decisions].cpu().numpy()
        terminated_cpu = completed_terminated[:executed_decisions].cpu().numpy()
        truncated_cpu = completed_truncated[:executed_decisions].cpu().numpy()
        records_by_seed_grid: dict[tuple[int, int], dict[str, Any]] = {}
        for completion_decision in range(executed_decisions):
            for lane in np.flatnonzero(completed_cpu[completion_decision]).tolist():
                lane_episode = int(episode_indices_cpu[completion_decision, lane])
                if lane_episode >= episode_quotas[lane]:
                    continue
                key = (int(lane), lane_episode)
                if key in records_by_seed_grid:
                    raise RuntimeError(f"duplicate evaluation seed-grid completion: {key}")
                records_by_seed_grid[key] = {
                    "lane": int(lane),
                    "lane_episode": lane_episode,
                    "game_seed": int(seeds_cpu[completion_decision, lane]),
                    "kills": float(kills_cpu[completion_decision, lane]),
                    "return": float(returns_cpu[completion_decision, lane]),
                    "length": int(lengths_cpu[completion_decision, lane]),
                    "terminated": bool(terminated_cpu[completion_decision, lane]),
                    "truncated": bool(truncated_cpu[completion_decision, lane]),
                    "completion_decision": completion_decision + 1,
                }
        expected_grid = [
            (lane, lane_episode)
            for lane in range(evaluation_envs)
            for lane_episode in range(episode_quotas[lane])
        ]
        missing_grid = [key for key in expected_grid if key not in records_by_seed_grid]
        if missing_grid:
            raise RuntimeError(f"evaluation did not complete fixed seed grid: {missing_grid}")
        records = []
        for index, key in enumerate(expected_grid):
            records.append({"index": index, **records_by_seed_grid[key]})
        emitter.emit(
            {
                "type": "evaluation",
                "status": "completed",
                "protocol": "standalone-gradoom-deathmatch-checkpoint-eval-v3-balanced-seed-grid",
                "checkpoint": str(args.evaluate_checkpoint),
                "checkpoint_sha256": _file_sha256(args.evaluate_checkpoint),
                "checkpoint_step": int(loaded.get("step", 0)),
                "checkpoint_config": loaded.get("config"),
                "evaluation_config": audit["evaluation"],
                "deterministic_actions": not bool(args.evaluation_stochastic),
                "episode_quotas": list(episode_quotas),
                "evaluation_seconds": evaluation_seconds,
                "process_elapsed_seconds": time.perf_counter() - process_started,
                "environment_steps": executed_decisions * evaluation_envs,
                "environment_backend": env.engine_backend,
                "iwad_sha256": env.iwad_sha256,
                "scenario_sha256": env.scenario_sha256,
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                **_evaluation_aggregate(records),
                "episodes": records,
            }
        )
        return 0
    finally:
        env.close()


def _train(
    args: argparse.Namespace,
    emitter: JsonEmitter,
    audit: Mapping[str, Any],
    *,
    process_started: float,
) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GraDOOM training fast path")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    device = torch.device("cuda")
    env = _make_env(args, device)
    interrupted = False
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        emitter.emit({"type": "event", "event": "stop_requested", "signal": signum})

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, request_stop)
        if env.single_action_space.n != REFERENCE_RECIPE.action_count:
            raise RuntimeError(
                f"expected {REFERENCE_RECIPE.action_count} actions, got {env.single_action_space.n}"
            )
        context_encoder = CombatContextEncoder(env.device_info_history_names, device)
        episode_index = torch.zeros(int(args.num_envs), dtype=torch.int64, device=device)
        episode_seeds = GradLabEpisodeSeeds(int(args.seed), int(args.num_envs), device)
        reset_mask = torch.ones(int(args.num_envs), dtype=torch.bool, device=device)
        observations, _signals = env.reset_device(
            reset_mask,
            episode_seeds.lookup(episode_index),
        )
        context = context_encoder.encode(env.device_info_histories())
        expected_observations = (int(args.num_envs), 4, 84, 84)
        if observations.shape != expected_observations or observations.device.type != "cuda":
            raise RuntimeError(
                f"expected CUDA observations {expected_observations}, got "
                f"{tuple(observations.shape)} on {observations.device}"
            )

        policy = NatureActorCritic(
            str(args.policy_architecture),
            str(args.policy_memory_format),
        ).to(
            device=device,
            memory_format=(
                torch.channels_last
                if str(args.policy_memory_format) == "channels-last"
                else torch.contiguous_format
            ),
        )
        if bool(args.freeze_observation_encoder):
            policy.observation_encoder.requires_grad_(False)
        policy.use_frozen_encoder_custom_conv = bool(
            args.freeze_observation_encoder and args.frozen_encoder_custom_conv
        )
        optimizer = _make_optimizer(
            policy,
            learning_rate=float(args.learning_rate),
            fused=bool(args.fused_optimizer),
        )
        resume_payload: Mapping[str, Any] | None = None
        if args.initialize_from is not None:
            loaded = torch.load(args.initialize_from, map_location=device, weights_only=False)
            if (
                not isinstance(loaded, Mapping)
                or loaded.get("format") != "standalone-gradoom-ppo-v1"
            ):
                raise ValueError(f"unsupported initialization checkpoint: {args.initialize_from}")
            policy.load_state_dict(loaded["policy_state_dict"])
            emitter.emit(
                {
                    "type": "event",
                    "event": "policy_initialized",
                    "checkpoint": str(args.initialize_from),
                    "checkpoint_sha256": _file_sha256(args.initialize_from),
                    "source_step": int(loaded.get("step", 0)),
                    "mode": "policy-weights-only",
                }
            )
        if args.resume is not None:
            loaded = torch.load(args.resume, map_location=device, weights_only=False)
            if (
                not isinstance(loaded, Mapping)
                or loaded.get("format") != "standalone-gradoom-ppo-v1"
            ):
                raise ValueError(f"unsupported resume checkpoint: {args.resume}")
            policy.load_state_dict(loaded["policy_state_dict"])
            _load_optimizer_state(
                optimizer,
                loaded["optimizer_state_dict"],
                learning_rate=float(args.learning_rate),
            )
            resume_payload = loaded
        calls = PolicyCalls(policy, compile_policy=bool(args.compile_policy))
        precision = Precision(str(args.precision), device)
        buffer = RolloutBuffer(
            int(args.n_steps),
            int(args.num_envs),
            device,
            observation_feature_count=(
                policy.observation_feature_count if bool(args.freeze_observation_encoder) else None
            ),
        )
        episode_starts = torch.ones(int(args.num_envs), dtype=torch.bool, device=device)
        dones = torch.zeros(int(args.num_envs), dtype=torch.bool, device=device)
        episode_returns = torch.zeros(int(args.num_envs), dtype=torch.float32, device=device)
        episode_lengths = torch.zeros(int(args.num_envs), dtype=torch.int32, device=device)
        signal_indices = {name: index for index, name in enumerate(env.device_signal_names)}
        kill_index = signal_indices["killcount"]
        reward_shaper = {
            "native-v1": None,
            "native-death-v1": None,
            "killcount-v1": KillcountReward(
                env.device_signal_names,
                int(args.num_envs),
                device,
                compile_reward=bool(args.compile_engine),
            ),
            "sample-factory-v0": SampleFactoryDeathmatchReward(
                env.device_signal_names,
                int(args.num_envs),
                device,
                compile_reward=bool(args.compile_engine),
            ),
        }[str(args.reward_shape)]
        combat_teacher = (
            PrivilegedCombatTeacher(env, device)
            if float(args.privileged_imitation_coef) > 0.0
            else None
        )
        disabled_teacher_actions = torch.zeros(int(args.num_envs), dtype=torch.int64, device=device)
        disabled_teacher_valid = torch.zeros(int(args.num_envs), dtype=torch.bool, device=device)
        saved_training_state = (
            resume_payload.get("training_state", {}) if resume_payload is not None else {}
        )
        if not isinstance(saved_training_state, Mapping):
            raise ValueError("checkpoint training_state must be a mapping")
        rolling_returns: deque[float] = deque(
            (float(value) for value in saved_training_state.get("rolling_returns", ())),
            maxlen=ROLLING_EPISODES,
        )
        rolling_kills: deque[float] = deque(
            (float(value) for value in saved_training_state.get("rolling_kills", ())),
            maxlen=ROLLING_EPISODES,
        )
        rolling_lengths: deque[float] = deque(
            (float(value) for value in saved_training_state.get("rolling_lengths", ())),
            maxlen=ROLLING_EPISODES,
        )
        rolling_success: deque[float] = deque(
            (float(value) for value in saved_training_state.get("rolling_success", ())),
            maxlen=ROLLING_EPISODES,
        )
        loop_rates: list[float] = []
        steady_loop_rates: list[float] = []
        completed_episodes = int(saved_training_state.get("completed_episodes", 0))
        global_step = int(resume_payload.get("step", 0)) if resume_payload is not None else 0
        executed_rollouts = int(
            saved_training_state.get(
                "executed_rollouts",
                global_step // (int(args.num_envs) * int(args.n_steps)),
            )
        )
        resume_step = global_step
        if resume_payload is not None:
            saved_episode_index = saved_training_state.get("episode_index")
            saved_episode_index_tensor = (
                saved_episode_index if isinstance(saved_episode_index, torch.Tensor) else None
            )
            preserved_lanes = _restore_episode_indices(
                episode_index,
                saved_episode_index_tensor,
                fallback_index=global_step // int(args.num_envs),
            )
            if (
                saved_episode_index_tensor is not None
                and saved_episode_index_tensor.numel() != episode_index.numel()
            ):
                emitter.emit(
                    {
                        "type": "event",
                        "event": "resume_lanes_migrated",
                        "source_num_envs": saved_episode_index_tensor.numel(),
                        "target_num_envs": episode_index.numel(),
                        "preserved_lanes": preserved_lanes,
                        "new_lanes_start_at_episode": 0,
                    }
                )
            episode_seeds.ensure(int(episode_index.max().item()))
            observations, _signals = env.reset_device(
                reset_mask,
                episode_seeds.lookup(episode_index),
            )
            context = context_encoder.encode(env.device_info_histories())
            episode_starts.fill_(True)
            dones.zero_()
            episode_returns.zero_()
            episode_lengths.zero_()
            python_rng_state = saved_training_state.get("python_rng_state")
            numpy_rng_state = saved_training_state.get("numpy_rng_state")
            torch_rng_state = saved_training_state.get("torch_rng_state")
            cuda_rng_state = saved_training_state.get("cuda_rng_state")
            if python_rng_state is not None:
                random.setstate(python_rng_state)
            if numpy_rng_state is not None:
                np.random.set_state(numpy_rng_state)
            if isinstance(torch_rng_state, torch.Tensor):
                torch.set_rng_state(torch_rng_state.cpu())
            if isinstance(cuda_rng_state, Sequence):
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in cuda_rng_state if isinstance(state, torch.Tensor)]
                )
            emitter.emit(
                {
                    "type": "event",
                    "event": "resumed",
                    "checkpoint": str(args.resume),
                    "train/global_step": global_step,
                }
            )
        last_metrics: dict[str, Any] = {}
        rollout_transitions = int(args.num_envs) * int(args.n_steps)
        training_started = time.perf_counter()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        def checkpoint_training_state() -> dict[str, Any]:
            return {
                "completed_episodes": completed_episodes,
                "executed_rollouts": executed_rollouts,
                "episode_index": episode_index.detach().cpu(),
                "rolling_returns": list(rolling_returns),
                "rolling_kills": list(rolling_kills),
                "rolling_lengths": list(rolling_lengths),
                "rolling_success": list(rolling_success),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            }

        while global_step < int(args.timesteps) and not interrupted:
            executed_rollouts += 1
            episode_seeds.ensure(int(episode_index.max().item()) + int(args.n_steps) + 1)
            buffer.reset()
            policy.eval()
            torch.cuda.synchronize(device)
            rollout_started = time.perf_counter()
            for _step in range(int(args.n_steps)):
                if combat_teacher is None:
                    teacher_actions = disabled_teacher_actions
                    teacher_valid = disabled_teacher_valid
                else:
                    with torch.no_grad():
                        teacher_actions, teacher_valid = combat_teacher.actions()
                staged_observations, staged_context = buffer.stage(
                    observations,
                    context,
                    episode_starts,
                )
                with torch.no_grad(), precision.autocast():
                    if bool(args.freeze_observation_encoder):
                        actions, values, log_probs, observation_features = calls.act_and_encode(
                            staged_observations,
                            staged_context,
                        )
                    else:
                        actions, values, log_probs = calls.act(
                            staged_observations,
                            staged_context,
                        )
                        observation_features = None
                next_episode_index = episode_index + 1
                transition = env.step_and_reset_device(
                    actions,
                    episode_seeds.lookup(next_episode_index),
                )
                if str(args.reward_shape) == "native-death-v1":
                    policy_rewards = transition.rewards - (
                        float(args.death_penalty) * transition.terminated.to(torch.float32)
                    )
                elif reward_shaper is None:
                    policy_rewards = transition.rewards
                else:
                    policy_rewards = reward_shaper.process(
                        transition.final_signals,
                        transition.terminated,
                        transition.truncated,
                    )
                episode_returns.add_(policy_rewards)
                episode_lengths.add_(1)
                buffer.add(
                    actions=actions,
                    teacher_actions=teacher_actions,
                    teacher_valid=teacher_valid,
                    rewards=policy_rewards,
                    values=values,
                    log_probs=log_probs,
                    observation_features=observation_features,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    final_observations=transition.final_observations,
                    final_histories=transition.final_info_histories,
                    episode_returns=episode_returns,
                    episode_lengths=episode_lengths,
                    final_kills=transition.final_signals[:, kill_index],
                )
                observations = transition.observations
                context = context_encoder.encode(transition.info_histories)
                torch.logical_or(
                    transition.terminated,
                    transition.truncated,
                    out=dones,
                )
                episode_starts = dones
                episode_returns.masked_fill_(dones, 0.0)
                episode_lengths.masked_fill_(dones, 0)
                episode_index.add_(dones.to(torch.int64))
                global_step += int(args.num_envs)

            for episode_return, kills, length, success in buffer.completed_episode_rows():
                rolling_returns.append(float(episode_return))
                rolling_kills.append(float(kills))
                rolling_lengths.append(float(length))
                rolling_success.append(float(success))
                completed_episodes += 1
            with torch.no_grad(), precision.autocast():
                last_values = calls.value(observations, context)
            _bootstrap_time_limits(
                buffer,
                calls=calls,
                context_encoder=context_encoder,
                precision=precision,
                gamma=REFERENCE_RECIPE.gamma,
            )
            buffer.finish(
                last_values=last_values,
                dones=dones,
                gamma=REFERENCE_RECIPE.gamma,
                gae_lambda=REFERENCE_RECIPE.gae_lambda,
            )
            torch.cuda.synchronize(device)
            rollout_seconds = time.perf_counter() - rollout_started

            update_started = time.perf_counter()
            update_metrics = _ppo_update(
                policy,
                optimizer,
                buffer,
                calls=calls,
                precision=precision,
                args=args,
            )
            torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_started
            loop_seconds = rollout_seconds + update_seconds
            loop_rate = rollout_transitions / loop_seconds
            loop_rates.append(loop_rate)
            if executed_rollouts > int(args.steady_state_after_rollouts):
                steady_loop_rates.append(loop_rate)
            last_metrics = {
                "type": "rollout",
                "rollout": executed_rollouts,
                "train/global_step": global_step,
                "train/throughput/rollout/rate": rollout_transitions / rollout_seconds,
                "train/throughput/update/rate": rollout_transitions / update_seconds,
                "train/throughput/loop/rate": loop_rate,
                "train/throughput/rollout/seconds": rollout_seconds,
                "train/throughput/update/seconds": update_seconds,
                "train/episode/completed/count": completed_episodes,
                "train/episode/return/shaped/origin/target/rolling/mean": _rolling_mean(
                    rolling_returns
                ),
                "train/progress/kills/origin/target/rolling/mean": _rolling_mean(rolling_kills),
                "train/episode/length/origin/all/rolling/mean": _rolling_mean(rolling_lengths),
                "train/outcome/success/starts/all/rolling/rate/min": _rolling_mean(rolling_success),
                **update_metrics,
                **_rollout_diagnostics(buffer),
            }
            emitter.emit(last_metrics)
            checkpoint_interval = int(args.checkpoint_every_rollouts)
            if (
                args.checkpoint is not None
                and checkpoint_interval
                and executed_rollouts % checkpoint_interval == 0
            ):
                recovery_path = _save_checkpoint(
                    _periodic_checkpoint_path(args.checkpoint, global_step),
                    policy=policy,
                    optimizer=optimizer,
                    step=global_step,
                    audit=audit,
                    training_state=checkpoint_training_state(),
                )
                emitter.emit(
                    {
                        "type": "event",
                        "event": "checkpoint_saved",
                        "checkpoint": str(recovery_path),
                        "train/global_step": global_step,
                    }
                )

        torch.cuda.synchronize(device)
        training_elapsed_seconds = time.perf_counter() - training_started
        checkpoint_path = None
        if args.checkpoint is not None:
            checkpoint_path = _save_checkpoint(
                args.checkpoint,
                policy=policy,
                optimizer=optimizer,
                step=global_step,
                audit=audit,
                training_state=checkpoint_training_state(),
            )
        torch.cuda.synchronize(device)
        process_elapsed_seconds = time.perf_counter() - process_started
        emitter.emit(
            {
                "type": "summary",
                "status": "interrupted" if interrupted else "completed",
                "train/global_step": global_step,
                "requested_timesteps": int(args.timesteps),
                "execution_timesteps": _execution_timesteps(args),
                "executed_rollouts": executed_rollouts,
                "rollout_transitions": rollout_transitions,
                "initialization_seconds": training_started - process_started,
                "training_elapsed_seconds": training_elapsed_seconds,
                "process_elapsed_seconds": process_elapsed_seconds,
                "training_transitions_per_second": (
                    (global_step - resume_step) / training_elapsed_seconds
                ),
                "end_to_end_transitions_per_second": (
                    (global_step - resume_step) / process_elapsed_seconds
                ),
                "resumed_from_step": resume_step,
                "median_loop_transitions_per_second": (
                    statistics.median(loop_rates) if loop_rates else None
                ),
                "steady_state_transitions_per_second": (
                    statistics.median(steady_loop_rates) if steady_loop_rates else None
                ),
                "steady_state_after_rollouts": int(args.steady_state_after_rollouts),
                "train/episode/completed/count": completed_episodes,
                "train/episode/return/shaped/origin/target/rolling/mean": _rolling_mean(
                    rolling_returns
                ),
                "train/progress/kills/origin/target/rolling/mean": _rolling_mean(rolling_kills),
                "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "environment_backend": env.engine_backend,
                "iwad_sha256": env.iwad_sha256,
                "scenario_sha256": env.scenario_sha256,
                "last_rollout": last_metrics,
            }
        )
        return 130 if interrupted else 0
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    process_started = time.perf_counter()
    args = _parser().parse_args(argv)
    _validate_args(args)
    torch.set_float32_matmul_precision(str(args.float32_matmul_precision))
    audit = _audit_config(args)
    emitter = JsonEmitter(args.metrics_jsonl)
    emitter.emit(audit)
    if args.config_only:
        return 0
    _runtime_paths(args)
    if args.evaluate_checkpoint is not None:
        return _evaluate(args, emitter, audit, process_started=process_started)
    wandb_run = _init_wandb(args, audit)
    if wandb_run is not None:
        emitter.attach_wandb(wandb_run)
    exit_code = 1
    try:
        exit_code = _train(args, emitter, audit, process_started=process_started)
        return exit_code
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
