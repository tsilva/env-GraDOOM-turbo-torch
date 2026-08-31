from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

_TRAIN_PATH = Path(__file__).parents[1] / "train.py"
_TRAIN_SPEC = importlib.util.spec_from_file_location("standalone_train", _TRAIN_PATH)
assert _TRAIN_SPEC is not None and _TRAIN_SPEC.loader is not None
train = importlib.util.module_from_spec(_TRAIN_SPEC)
sys.modules[_TRAIN_SPEC.name] = train
_TRAIN_SPEC.loader.exec_module(train)


def _args(*arguments: str):
    return train._parser().parse_args(arguments)


def test_checkpoint_evaluation_defaults_to_exact_stochastic_100() -> None:
    args = _args()
    audit = train._audit_config(args)

    assert args.ent_coef == train.REFERENCE_RECIPE.ent_coef
    assert args.num_envs == 256
    assert args.n_steps == 16
    assert args.num_envs * args.n_steps == (
        train.REFERENCE_RECIPE.num_envs * train.REFERENCE_RECIPE.n_steps
    )
    assert args.evaluation_episodes == 100
    assert args.evaluation_num_envs == 16
    assert args.evaluation_seed == train.REFERENCE_RECIPE.seed
    assert args.evaluation_stochastic is True
    assert args.wandb is False
    assert args.wandb_project == "VizdoomDeathmatch-v1"
    assert args.wandb_mode == "online"
    assert args.observation_blur_kernel == 1
    assert args.observation_augmentation == "none"
    assert audit["state_initialization"] == {
        "policy_state": "fresh_random",
        "optimizer_state": "fresh",
    }
    assert audit["evaluation"]["kills_signal"] == "player_killcount"
    assert audit["evaluation"]["compatibility_killcount_signal"] == "killcount"
    assert audit["evaluation"]["kills_target_signal"] == "player_killcount"


def test_cuda_residency_acceptance_is_opt_in_on_the_real_trainer_contract() -> None:
    disabled = train._audit_config(_args())
    enabled = train._audit_config(_args("--cuda-residency-acceptance"))

    assert "cuda_residency_acceptance" not in disabled
    assert enabled["cuda_residency_acceptance"] == {
        "contract": "gradoom-cuda-residency-v1",
        "enabled": True,
        "steady_state_after_rollouts": 1,
    }


@pytest.mark.parametrize(
    "fault_site",
    (
        "augmentation",
        "staging",
        "transition",
        "reward",
        "rollout_write",
        "context_update",
        "reset_update",
        "bootstrap",
        "finalization",
        "update",
    ),
)
def test_real_trainer_guard_rejects_host_round_trip_across_the_data_plane(
    monkeypatch: pytest.MonkeyPatch,
    fault_site: str,
) -> None:
    """Exercise the trainer's actual steady-state boundary without allocating CUDA."""

    class FakeEnv:
        single_action_space = SimpleNamespace(n=train.REFERENCE_RECIPE.action_count)
        device_info_history_names = train.MODEL_HISTORY_SIGNALS
        device_signal_names = train.INFO_SIGNALS
        engine_backend = "fake"
        iwad_sha256 = "fixture"
        scenario_sha256 = "fixture"
        num_envs = 1

        def reset_device(
            self, _reset_mask: torch.Tensor, _seeds: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                torch.zeros((1, 4, 84, 84), dtype=torch.uint8, device="cuda"),
                torch.zeros((1, len(train.INFO_SIGNALS)), device="cuda"),
            )

        def device_info_histories(self) -> torch.Tensor:
            return torch.zeros(
                (1, len(train.MODEL_HISTORY_SIGNALS), train.FRAME_STACK),
                device="cuda",
            )

        def step_and_reset_device(
            self, actions: torch.Tensor, _seeds: torch.Tensor
        ) -> SimpleNamespace:
            if fault_site == "transition":
                host_round_trip(actions)
            observations = torch.zeros((1, 4, 84, 84), dtype=torch.uint8, device="cuda")
            histories = torch.zeros(
                (1, len(train.MODEL_HISTORY_SIGNALS), train.FRAME_STACK),
                device="cuda",
            )
            return SimpleNamespace(
                observations=observations,
                rewards=torch.zeros(1, device="cuda"),
                terminated=torch.zeros(1, dtype=torch.bool, device="cuda"),
                truncated=torch.zeros(1, dtype=torch.bool, device="cuda"),
                final_signals=torch.zeros((1, len(train.INFO_SIGNALS)), device="cuda"),
                final_observations=observations,
                final_info_histories=histories,
                info_histories=histories,
            )

        def close(self) -> None:
            pass

    class FakePolicy(torch.nn.Module):
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones((), device="cuda"))
            self.observation_encoder = torch.nn.Sequential()
            self.observation_feature_count = 1

        def to(self, *_args: object, **_kwargs: object) -> FakePolicy:
            return self

    class FakeCalls:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def act(
            self, observations: torch.Tensor, _context: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            size = observations.shape[0]
            return (
                torch.zeros(size, dtype=torch.int64, device="cuda"),
                torch.zeros(size, device="cuda"),
                torch.zeros(size, device="cuda"),
            )

        def value(self, observations: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
            return torch.zeros(observations.shape[0], device="cuda")

    class FakePrecision:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def autocast(self):
            return train._NOOP_CONTEXT

    class FakeReward:
        def process(
            self,
            final_signals: torch.Tensor,
            _terminated: torch.Tensor,
            _truncated: torch.Tensor,
        ) -> torch.Tensor:
            rewards = final_signals[:, 0]
            return host_round_trip(rewards) if fault_site == "reward" else rewards

    def host_round_trip(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.cpu().to("cuda")

    monkeypatch.setattr(train, "_make_env", lambda *_args, **_kwargs: FakeEnv())
    monkeypatch.setattr(train, "NatureActorCritic", FakePolicy)
    monkeypatch.setattr(train, "PolicyCalls", FakeCalls)
    monkeypatch.setattr(train, "Precision", FakePrecision)
    monkeypatch.setattr(train, "KillcountReward", lambda *_args, **_kwargs: FakeReward())
    monkeypatch.setattr(train.torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(train.torch.cuda, "reset_peak_memory_stats", lambda *_args: None)
    monkeypatch.setattr(train, "_max_episode_index", lambda _indices: 0, raising=False)

    original_augment = train._augment_training_observations

    def augment(observations: torch.Tensor, mode: str) -> torch.Tensor:
        if fault_site == "augmentation":
            return host_round_trip(observations)
        return original_augment(observations, mode)

    monkeypatch.setattr(train, "_augment_training_observations", augment)
    original_stage = train.RolloutBuffer.stage

    def stage(buffer, observations, context, episode_starts):
        if fault_site == "staging":
            host_round_trip(observations)
        return original_stage(buffer, observations, context, episode_starts)

    monkeypatch.setattr(train.RolloutBuffer, "stage", stage)
    original_add = train.RolloutBuffer.add

    def add(buffer, **kwargs):
        if fault_site == "rollout_write":
            host_round_trip(kwargs["actions"])
        return original_add(buffer, **kwargs)

    monkeypatch.setattr(train.RolloutBuffer, "add", add)
    original_encode = train.CombatContextEncoder.encode
    encode_calls = 0

    def encode(encoder, histories):
        nonlocal encode_calls
        encode_calls += 1
        if fault_site == "context_update" and encode_calls > 1:
            host_round_trip(histories)
        return original_encode(encoder, histories)

    monkeypatch.setattr(train.CombatContextEncoder, "encode", encode)
    original_logical_or = train.torch.logical_or

    def logical_or(left, right, *, out=None):
        if fault_site == "reset_update":
            host_round_trip(left)
        return original_logical_or(left, right, out=out)

    monkeypatch.setattr(train.torch, "logical_or", logical_or)
    original_bootstrap = train._bootstrap_time_limits

    def bootstrap(*bootstrap_args, **bootstrap_kwargs):
        if fault_site == "bootstrap":
            host_round_trip(bootstrap_args[0].rewards)
        if fault_site in {"finalization", "update"}:
            return None
        return original_bootstrap(*bootstrap_args, **bootstrap_kwargs)

    monkeypatch.setattr(train, "_bootstrap_time_limits", bootstrap)
    original_finish = train.RolloutBuffer.finish

    def finish(buffer, **kwargs):
        if fault_site == "finalization":
            host_round_trip(buffer.rewards)
        return original_finish(buffer, **kwargs)

    monkeypatch.setattr(train.RolloutBuffer, "finish", finish)
    if fault_site == "update":
        monkeypatch.setattr(train.RolloutBuffer, "completed_episode_rows", lambda _buffer: [])

        def update_fault(*_args, **_kwargs):
            return host_round_trip(_args[2].rewards)

        monkeypatch.setattr(train, "_ppo_update_device", update_fault)
    args = _args(
        "--cuda-residency-acceptance",
        "--num-envs",
        "1",
        "--n-steps",
        "1",
        "--timesteps",
        "1",
        "--steady-state-after-rollouts",
        "0",
        "--batch-size",
        "1",
        "--n-epochs",
        "1",
        "--reward-shape",
        "killcount-v1",
        "--no-compile-policy",
        "--no-compile-engine",
        "--no-fused-optimizer",
    )

    with FakeTensorMode(allow_non_fake_inputs=True):
        real_cuda_available = train.torch.cuda.is_available

        def pass_initial_cuda_check() -> bool:
            monkeypatch.setattr(train.torch.cuda, "is_available", real_cuda_available)
            return True

        monkeypatch.setattr(train.torch.cuda, "is_available", pass_initial_cuda_check)
        with pytest.raises(
            RuntimeError,
            match=rf"steady_state_{'update' if fault_site == 'update' else 'rollout'}.*"
            r"accelerator-to-host",
        ):
            train._train(
                args,
                train.JsonEmitter(None),
                train._audit_config(args),
                process_started=time.perf_counter(),
            )


def test_training_command_rejects_ten_step_budget_without_overshoot(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(_TRAIN_PATH),
            "--config-only",
            "--timesteps",
            "10",
            "--metrics-jsonl",
            str(metrics),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )

    assert result.returncode != 0
    assert "at least one rollout transition quantum" in result.stderr
    assert not metrics.exists()


def test_checkpoint_evaluation_accepts_exact_predeclared_episode_seed_grid(
    tmp_path: Path,
) -> None:
    seed_file = tmp_path / "evaluation-seeds.json"
    seeds = list(range(10_000, 10_100))
    seed_file.write_text(json.dumps(seeds), encoding="utf-8")
    args = _args("--config-only", "--evaluation-seeds-file", str(seed_file))

    train._validate_args(args)
    evaluation = train._audit_config(args)["evaluation"]

    assert evaluation["episode_seed_protocol"] == "predeclared-game-seeds-v1"
    assert evaluation["episode_seeds"] == seeds
    assert evaluation["episode_seeds_sha256"] == train._file_sha256(seed_file)


@pytest.mark.parametrize(
    "seeds",
    [list(range(99)), [1] * 100],
)
def test_checkpoint_evaluation_rejects_invalid_predeclared_episode_seed_grid(
    tmp_path: Path,
    seeds: list[int],
) -> None:
    seed_file = tmp_path / "evaluation-seeds.json"
    seed_file.write_text(json.dumps(seeds), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 100 unique"):
        train._validate_args(_args("--config-only", "--evaluation-seeds-file", str(seed_file)))


def test_observation_blur_is_audited_and_rejects_even_kernels() -> None:
    args = _args("--observation-blur-kernel", "9")
    train._validate_args(args)

    audit = train._audit_config(args)

    assert audit["effective_recipe"]["observation_blur_kernel"] == 9
    assert audit["policy_model"]["observation_blur_kernel"] == 9
    with pytest.raises(ValueError, match="positive odd"):
        train._validate_args(_args("--observation-blur-kernel", "4"))


def test_bounded_observation_augmentation_is_training_only_and_audited() -> None:
    args = _args("--observation-augmentation", "bounded-shift-gray-v1")

    train._validate_args(args)
    audit = train._audit_config(args)

    assert audit["effective_recipe"]["observation_augmentation"] == "bounded-shift-gray-v1"
    assert (
        audit["policy_model"]["training_only_observation_augmentation"] == "bounded-shift-gray-v1"
    )


def test_encoder_anchor_is_audited_and_rejects_frozen_encoder() -> None:
    args = _args("--encoder-anchor-coef", "0.001")

    train._validate_args(args)
    audit = train._audit_config(args)

    assert audit["effective_recipe"]["encoder_anchor_coef"] == 0.001
    assert audit["policy_model"]["encoder_anchor"] == {
        "coefficient": 0.001,
        "penalty": "sum_squared_distance_from_training_start",
    }
    with pytest.raises(ValueError, match="requires a trainable"):
        train._validate_args(
            _args("--encoder-anchor-coef", "0.001", "--freeze-observation-encoder")
        )


def test_encoder_anchor_loss_measures_sum_squared_distance() -> None:
    parameter = torch.nn.Parameter(torch.tensor((1.0, 2.0)))
    anchor = parameter.detach().clone()

    parameter.data.add_(torch.tensor((2.0, -1.0)))
    actual = train._encoder_anchor_loss(((parameter, anchor),), fallback=parameter)

    assert actual.item() == 5.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bounded_observation_augmentation_is_stack_consistent() -> None:
    device = torch.device("cuda")
    base = torch.arange(84, dtype=torch.uint8, device=device).view(1, 1, 1, 84)
    observations = base.expand(2, 4, 84, 84).contiguous()
    randoms = torch.tensor(
        (
            (0.5, 0.5, 0.5, 0.5),
            (0.999, 0.5, 0.5, 0.5),
        ),
        dtype=torch.float32,
        device=device,
    )

    actual = train.bounded_observation_augment(observations, randoms)

    torch.testing.assert_close(actual[0], observations[0])
    assert torch.count_nonzero(actual[1, :, :, :2]) == 0
    torch.testing.assert_close(actual[1, :, :, 2:], observations[1, :, :, :-2])


def test_reference_observations_allow_compiled_gameplay_phases() -> None:
    args = _args("--config-only", "--observation-renderer", "reference")

    train._validate_args(args)

    assert args.compile_engine is True
    assert train._audit_config(args)["environment"]["observation_renderer"] == "reference"


def test_wandb_uses_gradlab_project_metrics_and_doom_turbo_torch_provider_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.defined: list[tuple[tuple[object, ...], dict[str, object]]] = []
            self.logged: list[dict[str, int | float]] = []

        def define_metric(self, *args: object, **kwargs: object) -> None:
            self.defined.append((args, kwargs))

        def log(self, payload: dict[str, int | float]) -> None:
            self.logged.append(payload)

    run = FakeRun()
    init_calls: list[dict[str, object]] = []

    def init(**kwargs: object) -> FakeRun:
        init_calls.append(kwargs)
        return run

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    args = _args(
        "--wandb",
        "--wandb-mode",
        "disabled",
        "--wandb-tags",
        "experiment:throughput,env_provider:env-gradoom-turbo-torch",
    )
    audit = train._audit_config(args)

    actual_run = train._init_wandb(args, audit)
    emitter = train.JsonEmitter(None)
    emitter.attach_wandb(actual_run)
    rollout = {
        "type": "rollout",
        "train/global_step": 4096,
        train.GRADLAB_RETURN_METRIC: 12.5,
        train.GRADLAB_KILLS_METRIC: 4.25,
        **{
            metric: float(index)
            for index, metric in enumerate(train.GRADLAB_PPO_DIAGNOSTIC_METRICS)
        },
    }
    emitter.emit(rollout)

    assert init_calls[0]["project"] == "VizdoomDeathmatch-v1"
    assert init_calls[0]["job_type"] == "train"
    assert init_calls[0]["tags"] == [
        "goal_id:VizdoomDeathmatch-v1",
        "recipe_id:ppo",
        "env_id:VizdoomDeathmatch-v1",
        "env_provider:env-gradoom-turbo-torch",
        "experiment:throughput",
    ]
    assert audit["tracking"]["wandb_metrics"] == list(train.GRADLAB_WANDB_METRICS)
    assert run.logged == [
        {
            "global_step": 4096,
            train.GRADLAB_RETURN_METRIC: 12.5,
            train.GRADLAB_KILLS_METRIC: 4.25,
            **{
                metric: float(index)
                for index, metric in enumerate(train.GRADLAB_PPO_DIAGNOSTIC_METRICS)
            },
        }
    ]
    for metric in train.GRADLAB_WANDB_METRICS:
        assert ((metric,), {"step_metric": "global_step"}) in run.defined


def test_partial_final_ppo_minibatch_is_supported() -> None:
    args = _args("--num-envs", "2048", "--n-steps", "16", "--batch-size", "12288")

    train._validate_args(args)

    assert args.num_envs * args.n_steps == 32768
    assert (args.num_envs * args.n_steps) % args.batch_size != 0


def test_policy_context_preserves_four_policy_facing_history_frames() -> None:
    encoder = train.CombatContextEncoder(train.MODEL_HISTORY_SIGNALS, torch.device("cpu"))
    histories = torch.zeros((2, len(train.MODEL_HISTORY_SIGNALS), train.FRAME_STACK))
    histories[:, :, :-1] = 999.0

    context = encoder.encode(histories)

    assert context.shape == (2, 84)
    assert train.NatureActorCritic().fusion[0].in_features == 512 + 84


def test_tf32_matmul_mode_is_explicitly_audited() -> None:
    args = _args("--config-only", "--float32-matmul-precision", "high")

    train._validate_args(args)

    assert train._audit_config(args)["effective_recipe"]["float32_matmul_precision"] == "high"


def test_channels_last_policy_format_is_explicitly_audited() -> None:
    args = _args("--config-only")
    policy = train.NatureActorCritic(memory_format=args.policy_memory_format)
    config = train._audit_config(args)

    assert args.policy_memory_format == "channels-last"
    assert policy.channels_last is True
    assert config["effective_recipe"]["policy_memory_format"] == "channels-last"
    assert config["policy_model"]["memory_format"] == "channels-last"


def test_frozen_encoder_feature_cache_is_explicitly_audited() -> None:
    disabled = _args("--config-only")
    enabled = _args("--config-only", "--freeze-observation-encoder")

    disabled_config = train._audit_config(disabled)
    enabled_config = train._audit_config(enabled)

    assert disabled.freeze_observation_encoder is False
    assert disabled_config["policy_model"]["observation_encoder_trainable"] is True
    assert disabled_config["policy_model"]["ppo_update_input"] == "pixels"
    assert enabled.freeze_observation_encoder is True
    assert enabled_config["effective_recipe"]["freeze_observation_encoder"] is True
    assert enabled_config["effective_recipe"]["frozen_encoder_custom_conv"] is True
    assert enabled_config["policy_model"]["observation_encoder_trainable"] is False
    assert enabled_config["policy_model"]["frozen_encoder_custom_conv"] is True
    assert enabled_config["policy_model"]["ppo_update_input"] == "cached_observation_features"


def test_projection_only_encoder_mode_freezes_convolutions_and_is_audited() -> None:
    args = _args("--train-observation-projection-only")
    policy = train.NatureActorCritic()

    train._validate_args(args)
    train._configure_observation_encoder_trainability(
        policy,
        freeze=False,
        projection_only=True,
    )

    trainable = {
        name
        for name, parameter in policy.observation_encoder.named_parameters()
        if parameter.requires_grad
    }
    assert trainable == {"7.weight", "7.bias"}
    assert train._audit_config(args)["policy_model"]["observation_encoder_train_mode"] == (
        "projection-only"
    )
    with pytest.raises(ValueError, match="are exclusive"):
        train._validate_args(
            _args("--freeze-observation-encoder", "--train-observation-projection-only")
        )


def test_encoded_action_evaluation_matches_pixel_path() -> None:
    torch.manual_seed(7)
    policy = train.NatureActorCritic()
    observations = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    context = torch.randn(2, train.CONTEXT_FEATURES)
    actions = torch.tensor((3, 11))

    encoded = policy.encode_observations(observations)
    expected = policy.evaluate_actions(observations, context, actions)
    actual = policy.evaluate_encoded_actions(encoded, context, actions)

    assert encoded.shape == (2, policy.observation_feature_count)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_frozen_encoder_custom_conv_tracks_cudnn_path() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    policy = train.NatureActorCritic(memory_format="channels-last").to(
        device=device,
        memory_format=torch.channels_last,
    )
    policy.eval()
    observations = torch.randint(
        0,
        256,
        (17, 4, 84, 84),
        dtype=torch.uint8,
        device=device,
    )

    with torch.no_grad():
        expected = policy.encode_observations(observations)
        policy.use_frozen_encoder_custom_conv = True
        actual = policy.encode_observations(observations)

    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_feature_cache_buffer_skips_rollout_pixel_storage() -> None:
    buffer = train.RolloutBuffer(
        2,
        3,
        torch.device("cpu"),
        observation_feature_count=5,
    )
    observations = torch.zeros((3, 4, 84, 84), dtype=torch.uint8)
    context = torch.zeros((3, train.CONTEXT_FEATURES))
    starts = torch.ones(3, dtype=torch.bool)

    staged_observations, staged_context = buffer.stage(observations, context, starts)

    assert buffer.observations is None
    assert buffer.observation_features is not None
    assert buffer.observation_features.shape == (2, 3, 5)
    assert staged_observations is observations
    torch.testing.assert_close(staged_context, context)


@pytest.mark.parametrize(
    ("architecture", "channels", "observation_features", "fusion_features"),
    (
        ("nature", (32, 64, 64), 512, 256),
        ("nature-pyramid", (16, 32, 64), 512, 256),
        ("nature-waist", (32, 32, 64), 512, 256),
        ("nature-flat", (32, 32, 32), 512, 256),
        ("nature-thin", (16, 32, 32), 512, 256),
        ("nature-half", (16, 32, 32), 128, 128),
        ("nature-quarter", (8, 16, 16), 128, 128),
    ),
)
def test_policy_architecture_is_static_and_audited(
    architecture: str,
    channels: tuple[int, int, int],
    observation_features: int,
    fusion_features: int,
) -> None:
    args = _args("--config-only", "--policy-architecture", architecture)
    policy = train.NatureActorCritic(architecture)

    convolution_channels = tuple(
        layer.out_channels
        for layer in policy.observation_encoder
        if isinstance(layer, torch.nn.Conv2d)
    )
    config = train._audit_config(args)

    assert convolution_channels == channels
    assert policy.observation_encoder[-2].out_features == observation_features
    assert policy.fusion[0].out_features == fusion_features
    assert config["effective_recipe"]["policy_architecture"] == architecture
    assert config["policy_model"]["convolution_channels"] == list(channels)


def test_checkpoint_evaluation_rejects_more_lanes_than_episodes() -> None:
    args = _args("--evaluation-episodes", "16", "--evaluation-num-envs", "17")

    with pytest.raises(
        ValueError,
        match="evaluation-num-envs cannot exceed evaluation-episodes",
    ):
        train._validate_args(args)


def test_fixed_seed_grid_default_balances_exact_episode_count() -> None:
    args = _args()
    quotas = train._episode_quotas(args.evaluation_episodes, args.evaluation_num_envs)

    grid = [
        (lane, lane_episode)
        for lane in range(args.evaluation_num_envs)
        for lane_episode in range(quotas[lane])
    ]

    assert quotas == (7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    assert len(grid) == 100


def test_resume_episode_indices_preserve_existing_lanes_when_scaling_up() -> None:
    destination = torch.full((5,), -1, dtype=torch.int64)
    saved = torch.tensor((7, 11, 13), dtype=torch.int64)

    preserved = train._restore_episode_indices(destination, saved, fallback_index=99)

    assert preserved == 3
    torch.testing.assert_close(destination, torch.tensor((7, 11, 13, 0, 0)))


def test_legacy_resume_episode_indices_use_fallback() -> None:
    destination = torch.zeros(3, dtype=torch.int64)

    preserved = train._restore_episode_indices(destination, None, fallback_index=17)

    assert preserved == 0
    torch.testing.assert_close(destination, torch.full((3,), 17, dtype=torch.int64))


def test_checkpoint_evaluation_is_mutually_exclusive_with_training_resume(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.touch()
    args = _args(
        "--evaluate-checkpoint",
        str(checkpoint),
        "--resume",
        str(checkpoint),
    )

    with pytest.raises(ValueError, match="evaluate-checkpoint cannot be combined with resume"):
        train._validate_args(args)


def test_weights_only_initialization_is_audited_and_mutually_exclusive(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"standalone checkpoint fixture")
    args = _args("--initialize-from", str(checkpoint), "--config-only")

    train._validate_args(args)
    initialization = train._audit_config(args)["initialization"]
    assert initialization == {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": train._file_sha256(checkpoint),
        "mode": "policy-weights-only",
    }

    with pytest.raises(ValueError, match="initialize-from cannot be combined with resume"):
        train._validate_args(
            _args(
                "--initialize-from",
                str(checkpoint),
                "--resume",
                str(checkpoint),
            )
        )


def test_evaluation_aggregate_uses_player_killcount_quality_target() -> None:
    records = [
        {
            "player_killcount": 30.0,
            "compatibility_killcount": 130.0,
            "return": 30.0,
            "length": 2100,
        },
        {
            "player_killcount": 34.0,
            "compatibility_killcount": 132.0,
            "return": 34.0,
            "length": 2100,
        },
    ]

    result = train._evaluation_aggregate(records)

    assert result["evaluation/episode/count"] == 2
    assert result["evaluation/kills/mean"] == 32.0
    assert result["evaluation/kills/median"] == 32.0
    assert result["evaluation/kills/std"] == 2.0
    assert result["evaluation/kills/signal"] == "player_killcount"
    assert result["evaluation/compatibility_killcount/mean"] == 131.0
    assert result["evaluation/target/kills/mean"] == 31.78
    assert result["evaluation/target/kills/signal"] == "player_killcount"
    assert result["evaluation/target/passed"] is True

    compatibility_only = train._evaluation_aggregate(
        [
            {
                "player_killcount": 29.0,
                "compatibility_killcount": 100.0,
                "return": 29.0,
                "length": 2100,
            }
        ]
    )
    assert compatibility_only["evaluation/target/passed"] is False


def test_evaluation_aggregate_rejects_no_completed_episodes() -> None:
    with pytest.raises(ValueError, match="at least one completed episode"):
        train._evaluation_aggregate([])


def test_evaluation_aggregate_summarizes_action_histograms() -> None:
    first = [0] * len(train.RESTRICTED_ACTIONS)
    second = [0] * len(train.RESTRICTED_ACTIONS)
    first[2] = 3
    second[9] = 1
    records = [
        {"player_killcount": 1, "return": 2, "length": 3, "action_counts": first},
        {"player_killcount": 2, "return": 4, "length": 1, "action_counts": second},
    ]

    result = train._evaluation_aggregate(records)

    assert result["evaluation/actions/2/count"] == 3
    assert result["evaluation/actions/2/fraction"] == pytest.approx(0.75)
    assert result["evaluation/actions/9/count"] == 1
    assert result["evaluation/actions/9/fraction"] == pytest.approx(0.25)


def test_killcount_reward_is_uniform_and_resets_at_episode_boundaries() -> None:
    reward = train.KillcountReward(
        ("health", "killcount"),
        2,
        torch.device("cpu"),
        compile_reward=False,
    )

    first = reward.process(
        torch.tensor(((100.0, 1.0), (100.0, 3.0))),
        torch.tensor((False, True)),
        torch.tensor((False, False)),
    )
    second = reward.process(
        torch.tensor(((100.0, 4.0), (100.0, 0.0))),
        torch.tensor((False, False)),
        torch.tensor((False, False)),
    )

    assert first.tolist() == [1.0, 3.0]
    assert second.tolist() == [3.0, 0.0]


def test_player_killcount_reward_ignores_vizdoom_compatibility_kills() -> None:
    reward = train.KillcountReward(
        ("killcount", "player_killcount"),
        2,
        torch.device("cpu"),
        compile_reward=False,
        signal_name="player_killcount",
    )

    actual = reward.process(
        torch.tensor(((3.0, 1.0), (4.0, 0.0))),
        torch.tensor((False, False)),
        torch.tensor((False, False)),
    )

    assert actual.tolist() == [1.0, 0.0]


def test_player_combat_reward_adds_bounded_outgoing_progress() -> None:
    reward = train.PlayerCombatReward(
        ("damagecount", "player_killcount", "hitcount"),
        2,
        torch.device("cpu"),
        compile_reward=False,
    )

    first = reward.process(
        torch.tensor(((250.0, 1.0, 7.0), (6.0, 0.0, 1.0))),
        torch.tensor((False, True)),
        torch.tensor((False, False)),
    )
    second = reward.process(
        torch.tensor(((260.0, 2.0, 9.0), (0.0, 0.0, 0.0))),
        torch.tensor((False, False)),
        torch.tensor((False, False)),
    )

    assert first.tolist() == pytest.approx([1.65, 0.028])
    assert second.tolist() == pytest.approx([1.05, 0.0])


def test_native_reward_audit_preserves_class_weights() -> None:
    config = train._audit_config(_args("--config-only"))

    assert config["reward_config"] == {
        "source": "scenario-native",
        "monster_kill_rewards": [1.0, 3.0, 3.0, 4.0, 3.0, 10.0],
    }


def test_entropy_coefficient_override_is_validated_and_audited() -> None:
    args = _args("--config-only", "--ent-coef", "0.001")

    train._validate_args(args)

    assert train._audit_config(args)["effective_recipe"]["ent_coef"] == 0.001

    with pytest.raises(ValueError, match="ent-coef must be finite and non-negative"):
        train._validate_args(_args("--ent-coef", "-0.001"))


def test_resume_optimizer_state_honors_explicit_learning_rate() -> None:
    source_parameter = torch.nn.Parameter(torch.tensor(1.0))
    source = torch.optim.Adam((source_parameter,), lr=6.25e-5, eps=1e-5)
    source_parameter.grad = torch.tensor(2.0)
    source.step()

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed = torch.optim.Adam((resumed_parameter,), lr=3.125e-5, eps=1e-5)
    train._load_optimizer_state(
        resumed,
        source.state_dict(),
        learning_rate=3.125e-5,
    )

    assert train._optimizer_learning_rate(resumed) == 3.125e-5
    assert resumed.state[resumed_parameter]["step"] == source.state[source_parameter]["step"]


def test_native_death_reward_is_explicitly_audited() -> None:
    args = _args("--config-only", "--reward-shape", "native-death-v1", "--death-penalty", "3")

    train._validate_args(args)

    config = train._audit_config(args)
    assert config["reward_config"] == {
        "source": "scenario-native",
        "monster_kill_rewards": [1.0, 3.0, 3.0, 4.0, 3.0, 10.0],
        "terminal_death_penalty": 3.0,
    }
    assert config["return_comparability"] == "native-plus-death-cost return and kills"


def test_privileged_imitation_is_disabled_by_default_and_explicitly_audited() -> None:
    disabled = _args("--config-only")
    enabled = _args("--config-only", "--privileged-imitation-coef", "0.5")

    train._validate_args(disabled)
    train._validate_args(enabled)

    assert disabled.privileged_imitation_coef == 0.0
    assert (
        train._audit_config(disabled)["policy_model"]["training_only_privileged_imitation"] is False
    )
    assert train._audit_config(enabled)["effective_recipe"]["privileged_imitation_coef"] == 0.5
    assert (
        train._audit_config(enabled)["policy_model"]["training_only_privileged_imitation"] is True
    )
    with pytest.raises(
        ValueError,
        match="privileged-imitation-coef must be finite and non-negative",
    ):
        train._validate_args(_args("--privileged-imitation-coef", "-0.1"))


def test_privileged_teacher_labels_only_visible_enemy_states() -> None:
    class FakeEngine:
        def __init__(self) -> None:
            self.x = torch.zeros(2)
            self.y = torch.zeros(2)
            self.z = torch.zeros(2)
            self.angle = torch.zeros(2)
            self.enemy_x = torch.tensor(((100.0, 300.0), (100.0, 300.0)))
            self.enemy_y = torch.zeros((2, 2))
            self.enemy_z = torch.zeros((2, 2))
            self.enemy_alive = torch.ones((2, 2), dtype=torch.bool)
            self.episode_time = torch.zeros(2, dtype=torch.int32)

        def _effective_enemy_height(self) -> torch.Tensor:
            return torch.full((2, 2), 56.0)

        def _sight_blocked(self, *_arguments: torch.Tensor) -> torch.Tensor:
            return torch.tensor(((False, True), (True, True)))

    class FakeEnv:
        num_envs = 2
        _engine = FakeEngine()

    actions, valid = train.PrivilegedCombatTeacher(FakeEnv(), torch.device("cpu")).actions()

    assert actions.tolist() == [10, 8]
    assert valid.tolist() == [True, False]
