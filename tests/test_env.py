from __future__ import annotations

import pytest
import torch

from gradoom import GraDoomVecEnv
from gradoom.actions import DEATHMATCH_ACTION_TABLE_SHA256


def _env(square_scenario, **kwargs) -> GraDoomVecEnv:
    device = kwargs.pop("device", "cpu")
    return GraDoomVecEnv(
        game="VizdoomDeathmatch-v1",
        compiled_scenario=square_scenario,
        num_envs=2,
        device=device,
        transport="torch",
        render_mode=kwargs.pop("render_mode", "rgb_array"),
        obs_crop=kwargs.pop("obs_crop", (0, 32, 0, 0)),
        obs_crop_mode=kwargs.pop("obs_crop_mode", "mask"),
        frame_skip=kwargs.pop("frame_skip", 2),
        **kwargs,
    )


def test_device_api_is_deterministic_and_resident(square_scenario) -> None:
    first = _env(square_scenario)
    second = _env(square_scenario)
    try:
        mask = torch.ones(2, dtype=torch.bool)
        seeds = torch.tensor([123, 456])
        first_obs, first_signals = first.reset_device(mask, seeds)
        second_obs, second_signals = second.reset_device(mask, seeds)
        assert first_obs.device.type == "cpu"
        assert first_signals.device == first_obs.device
        assert torch.equal(first_obs, second_obs)
        assert torch.equal(first_signals, second_signals)
        actions = torch.tensor([2, 13])
        first_step = first.step_device(actions)
        second_step = second.step_device(actions)
        assert torch.equal(first_step.observations, second_step.observations)
        assert torch.equal(first_step.rewards, second_step.rewards)
        assert torch.equal(first_step.signals, second_step.signals)
    finally:
        first.close()
        second.close()


def test_device_api_validates_concrete_resident_device(square_scenario) -> None:
    env = _env(square_scenario, device="cpu:0")
    resident_device = torch.empty((), device="cpu:0").device
    try:
        assert env.device == torch.device("cpu:0")
        observations, signals = env.reset_device(
            torch.ones(2, device="cpu:0", dtype=torch.bool),
            torch.tensor([123, 456], device="cpu:0"),
        )
        transition = env.step_and_reset_device(
            torch.tensor([2, 13], device="cpu:0"),
            torch.tensor([789, 1011], device="cpu:0"),
        )

        assert observations.device == resident_device
        assert signals.device == resident_device
        assert transition.observations.device == resident_device
        assert transition.rewards.device == resident_device
        assert transition.signals.device == resident_device
    finally:
        env.close()


def test_vizdoom_screen_flash_option_is_static_and_validated(square_scenario) -> None:
    default_env = _env(square_scenario)
    flash_env = _env(
        square_scenario,
        vizdoom_config={"render_screen_flashes": True},
    )
    try:
        assert default_env._engine.render_screen_flashes is False
        assert flash_env._engine.render_screen_flashes is True
    finally:
        default_env.close()
        flash_env.close()

    with pytest.raises(ValueError, match="render_screen_flashes must be a boolean"):
        _env(
            square_scenario,
            vizdoom_config={"render_screen_flashes": 1},
        )


def test_turbo_api_contract_and_torch_transport(square_scenario) -> None:
    env = _env(square_scenario, obs_copy="safe_view")
    try:
        observations, infos = env.reset(seed=[1, 2])
        assert observations.shape == (2, 4, 84, 84)
        assert observations.dtype == torch.uint8
        assert observations.device.type == "cpu"
        assert env.action_table_hash == DEATHMATCH_ACTION_TABLE_SHA256
        assert env.metadata["turbo_api_version"] == 2
        assert env.metadata["gradoom_device_api_version"] == 1
        assert env.engine_backend == "torch-eager"
        assert "engine_backend" not in env.capabilities
        assert env.capabilities["supports_enemy_variants"] is False
        assert env.capabilities["supports_surface_variants"] is False
        assert env.capabilities["supports_info_frame_stack"] is True
        assert not any(key.startswith("native_render_") for key in env.capabilities)
        assert env.observation_ownership == "safe_view"
        assert env.observation_buffer_depth == 2
        assert infos["health"].tolist() == [100.0, 100.0]
        assert infos["noop_reset_count"].tolist() == [0, 0]
        result = env.step(torch.tensor([0, 2], dtype=torch.int64))
        assert result[0].shape == observations.shape
        assert result[1].dtype == torch.float32
        assert result[2].dtype == torch.bool
        assert result[3].dtype == torch.bool
        assert env.render_lane(1).shape == (240, 320, 3)
    finally:
        env.close()


def test_timeout_requires_masked_reset_and_preserves_other_lane(square_scenario) -> None:
    env = _env(square_scenario, vizdoom_config={"episode_timeout": 3})
    try:
        env.reset(seed=[9, 10])
        _, _, terminated, truncated, _ = env.step(torch.zeros(2, dtype=torch.int64))
        assert not torch.any(terminated)
        assert torch.all(truncated)
        with pytest.raises(RuntimeError, match="terminal lanes"):
            env.step(torch.zeros(2, dtype=torch.int64))
        previous_x = env._engine.x[1].clone()
        mask = torch.tensor([True, False])
        env.reset(seed=[11, None], options={"reset_mask": mask})
        assert env._engine.episode_time.tolist() == [1, 3]
        assert torch.equal(env._engine.x[1], previous_x)
    finally:
        env.close()


def test_frame_skip_stops_on_native_timeout_tic(square_scenario) -> None:
    env = _env(square_scenario, vizdoom_config={"episode_timeout": 4})
    try:
        env.reset(seed=[9, 10])
        first = env.step(torch.zeros(2, dtype=torch.int64))
        assert not torch.any(first[3])
        second = env.step(torch.zeros(2, dtype=torch.int64))
        assert torch.all(second[3])
        assert env._engine.episode_time.tolist() == [4, 4]
    finally:
        env.close()


def test_device_step_and_reset_retains_terminal_observation(square_scenario) -> None:
    env = _env(square_scenario, vizdoom_config={"episode_timeout": 3})
    try:
        observations, _ = env.reset_device(torch.ones(2, dtype=torch.bool), torch.tensor([9, 10]))
        transition = env.step_and_reset_device(
            torch.zeros(2, dtype=torch.int64), torch.tensor([11, 12])
        )
        assert torch.all(transition.truncated)
        assert not torch.any(transition.terminated)
        assert transition.final_observations.data_ptr() != observations.data_ptr()
        assert transition.final_observations.data_ptr() != transition.observations.data_ptr()
        assert transition.final_signals[:, 17].tolist() == [3.0, 3.0]
        assert transition.signals[:, 17].tolist() == [1.0, 1.0]
    finally:
        env.close()


def test_info_frame_stacks_stay_on_device_and_reset_per_lane(square_scenario) -> None:
    env = _env(
        square_scenario,
        info_frame_stack_keys=("health", "selected_weapon"),
        vizdoom_config={"episode_timeout": 3},
    )
    try:
        _observations, infos = env.reset(seed=[9, 10])
        assert infos["health_frame_stack"].shape == (2, 4)
        assert torch.all(infos["health_frame_stack"] == 100)
        assert env.signal_schema["health_frame_stack"]["shape"] == (4,)
        assert env.device_info_history_names == ("health", "selected_weapon")

        env._engine.health.copy_(torch.tensor([90.0, 80.0]))
        transition = env.step_and_reset_device(
            torch.zeros(2, dtype=torch.int64),
            torch.tensor([11, 12]),
        )

        assert transition.final_info_histories.device.type == "cpu"
        assert transition.final_info_histories[:, 0].tolist() == [
            [100.0, 100.0, 100.0, 90.0],
            [100.0, 100.0, 100.0, 80.0],
        ]
        assert transition.info_histories[:, 0].tolist() == [
            [100.0, 100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0],
        ]
    finally:
        env.close()


def test_rejects_options_that_would_fake_profile_support(square_scenario) -> None:
    with pytest.raises(ValueError, match="frame-stack 4"):
        _env(square_scenario, frame_stack=2)
    with pytest.raises(ValueError, match="players=1"):
        _env(square_scenario, players=2)


def test_compiled_engine_requires_cuda(square_scenario) -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _env(square_scenario, compile_engine=True)
