from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
        audit["policy_model"]["training_only_observation_augmentation"]
        == "bounded-shift-gray-v1"
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


def test_wandb_uses_gradlab_project_metrics_and_gradoom_provider_tag(
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
        "experiment:throughput,env_provider:gradoom",
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
        "env_provider:gradoom",
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


def test_evaluation_aggregate_uses_exact_records_and_reference_target() -> None:
    records = [
        {"kills": 30.0, "return": 30.0, "length": 2100},
        {"kills": 34.0, "return": 34.0, "length": 2100},
    ]

    result = train._evaluation_aggregate(records)

    assert result["evaluation/episode/count"] == 2
    assert result["evaluation/kills/mean"] == 32.0
    assert result["evaluation/kills/median"] == 32.0
    assert result["evaluation/kills/std"] == 2.0
    assert result["evaluation/target/kills/mean"] == 31.78
    assert result["evaluation/target/passed"] is True


def test_evaluation_aggregate_rejects_no_completed_episodes() -> None:
    with pytest.raises(ValueError, match="at least one completed episode"):
        train._evaluation_aggregate([])


def test_evaluation_aggregate_summarizes_action_histograms() -> None:
    first = [0] * len(train.RESTRICTED_ACTIONS)
    second = [0] * len(train.RESTRICTED_ACTIONS)
    first[2] = 3
    second[9] = 1
    records = [
        {"kills": 1, "return": 2, "length": 3, "action_counts": first},
        {"kills": 2, "return": 4, "length": 1, "action_counts": second},
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
