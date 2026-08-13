from __future__ import annotations

import inspect
from types import MappingProxyType

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode

from gradoom import GraDoomVecEnv, scenario_buttons


def _env(square_scenario, **kwargs) -> GraDoomVecEnv:
    return GraDoomVecEnv(
        compiled_scenario=square_scenario,
        num_envs=2,
        device="cpu",
        **kwargs,
    )


def test_public_surface_matches_turbo_vector_api_v1(square_scenario) -> None:
    parameters = inspect.signature(GraDoomVecEnv).parameters
    shared_parameters = {
        "game",
        "state",
        "scenario",
        "info",
        "use_restricted_actions",
        "record",
        "players",
        "inttype",
        "obs_type",
        "render_mode",
        "num_envs",
        "num_threads",
        "rom_path",
        "obs_copy",
        "obs_resize",
        "obs_crop",
        "obs_crop_mode",
        "obs_crop_fill",
        "obs_grayscale",
        "obs_resize_algorithm",
        "obs_layout",
        "frame_skip",
        "frame_stack",
        "maxpool_last_two",
        "noop_reset_max",
        "use_fire_reset",
        "sticky_action_prob",
        "reward_clip",
        "info_filter",
        "info_frame_stack_keys",
        "state_catalog",
        "doom_map",
        "doom_skill",
        "game_args",
        "game_variables",
        "enemy_variants",
        "surface_variants",
        "treat_episode_timeout_as_truncation",
        "vizdoom_config",
    }
    assert shared_parameters <= set(parameters)
    assert parameters["transport"].default == "torch"
    assert issubclass(GraDoomVecEnv, gym.vector.VectorEnv)
    assert GraDoomVecEnv.metadata["autoreset_mode"] is AutoresetMode.DISABLED
    assert GraDoomVecEnv.metadata["turbo_api_version"] == 1

    env = _env(square_scenario)
    try:
        expected_capabilities = {
            "supported_action_modes",
            "supported_observation_layouts",
            "supported_resize_algorithms",
            "supported_observation_copy_modes",
            "supports_maxpool_last_two",
            "supports_sticky_action_prob",
            "supports_reward_clipping",
            "supports_noop_reset",
            "supports_state_catalog",
            "supports_live_snapshots",
            "supports_per_lane_rgb",
            "supports_enemy_variants",
            "supports_surface_variants",
            "supports_info_frame_stack",
        }
        assert isinstance(env.capabilities, type(MappingProxyType({})))
        assert set(env.capabilities) == expected_capabilities
        assert env.capabilities["supported_action_modes"] == ("custom_discrete",)
        assert env.capabilities["supported_resize_algorithms"] == ("area",)
        assert env.num_threads == env.num_envs
        assert env.state_catalog == ("default",)
        assert env.info_frame_stack_keys == ()
        assert env.supports_live_snapshots is False
        assert env.live_snapshots_deterministic is False
        assert isinstance(env.signal_schema, type(MappingProxyType({})))
        assert all(
            isinstance(spec, type(MappingProxyType({})))
            for spec in env.signal_schema.values()
        )

        active = env.active_state_indices()
        assert active is env.active_state_indices()
        assert active.shape == (env.num_envs,)
        assert active.dtype == np.dtype(np.int32)
        assert not active.flags.writeable
    finally:
        env.close()


def test_torch_is_the_only_transition_transport(square_scenario) -> None:
    with pytest.raises(ValueError, match="NumPy transition transport"):
        _env(square_scenario, transport="numpy")

    env = _env(square_scenario)
    try:
        with pytest.raises(TypeError, match="Torch tensor"):
            env.reset(options={"reset_mask": np.ones(2, dtype=np.bool_)})

        observations, infos = env.reset(seed=7)
        assert isinstance(observations, torch.Tensor)
        assert all(
            isinstance(value, torch.Tensor)
            for key, value in infos.items()
            if key != "start_source"
        )
        assert isinstance(infos["start_source"], np.ndarray)

        with pytest.raises(TypeError, match="Torch tensor"):
            env.step(np.zeros(2, dtype=np.int64))
        transition = env.step(torch.zeros(2, dtype=torch.int64))
        assert all(isinstance(value, torch.Tensor) for value in transition[:4])
        assert all(isinstance(value, torch.Tensor) for value in transition[4].values())
    finally:
        env.close()


def test_reference_incoming_damage_variables_are_selectable(square_scenario) -> None:
    env = _env(
        square_scenario,
        game_variables=("health", "hits_taken", "damage_taken"),
    )
    try:
        _observations, infos = env.reset(seed=7)
        assert infos["hits_taken"].tolist() == [0.0, 0.0]
        assert infos["damage_taken"].tolist() == [0.0, 0.0]
    finally:
        env.close()


def test_seed_and_manual_lifecycle_match_turbo_semantics(square_scenario) -> None:
    left = _env(square_scenario)
    right = _env(square_scenario)
    try:
        with pytest.raises(RuntimeError, match="all lanes"):
            left.step(torch.zeros(2, dtype=torch.int64))
        with pytest.raises(ValueError, match="at least one lane"):
            left.reset(options={"reset_mask": torch.zeros(2, dtype=torch.bool)})

        assert left.seed(100) == [100, 101]
        left_observations, _ = left.reset()
        right_observations, _ = right.reset(seed=100)
        assert torch.equal(left_observations, right_observations)

        partial = _env(square_scenario)
        try:
            partial.reset(
                seed=[1, None],
                options={"reset_mask": torch.tensor([True, False])},
            )
            with pytest.raises(RuntimeError, match="all lanes"):
                partial.step(torch.zeros(2, dtype=torch.int64))
            partial.reset(
                seed=[None, 2],
                options={"reset_mask": torch.tensor([False, True])},
            )
            partial.step(torch.zeros(2, dtype=torch.int64))
        finally:
            partial.close()
    finally:
        left.close()
        right.close()


def test_info_filter_schema_and_histories_are_exact(square_scenario) -> None:
    empty = _env(square_scenario, info_filter="none")
    try:
        _observations, reset_infos = empty.reset(seed=1)
        assert empty.signal_schema == {}
        assert set(reset_infos) == {
            "state_index",
            "_state_index",
            "start_source",
            "_start_source",
            "noop_reset_count",
            "_noop_reset_count",
        }
        assert empty.step(torch.zeros(2, dtype=torch.int64))[4] == {}
    finally:
        empty.close()

    history = _env(
        square_scenario,
        info_filter={"mode": "all", "keys": ["health"]},
        info_frame_stack_keys=["health"],
    )
    try:
        _observations, infos = history.reset(seed=2)
        assert history.info_frame_stack_keys == ("health",)
        assert history.signal_schema["health_frame_stack"]["dtype"] == np.dtype(np.float64)
        assert infos["health_frame_stack"].dtype == torch.float64
        assert history.device_info_histories().dtype == torch.float32
    finally:
        history.close()

    with pytest.raises(ValueError, match="included by info_filter"):
        _env(
            square_scenario,
            info_filter={"mode": "all", "keys": ["armor"]},
            info_frame_stack_keys=["health"],
        )


def test_safe_view_uses_two_device_buffers(square_scenario) -> None:
    env = _env(square_scenario, obs_copy="safe_view")
    try:
        first, _ = env.reset(seed=3)
        first_value = first.clone()
        second = env.step(torch.zeros(2, dtype=torch.int64))[0]
        assert first.data_ptr() != second.data_ptr()
        assert torch.equal(first, first_value)
        third = env.step(torch.zeros(2, dtype=torch.int64))[0]
        assert first.data_ptr() == third.data_ptr()
    finally:
        env.close()


def test_reward_clip_keeps_episode_return_signal_aligned(square_scenario) -> None:
    env = _env(square_scenario, reward_clip=True)
    try:
        env.reset(seed=4)

        def fake_step(_buttons):
            rewards = torch.full((2,), 5.0)
            terminated = torch.zeros(2, dtype=torch.bool)
            truncated = torch.zeros(2, dtype=torch.bool)
            env._engine.episode_return.fill_(5.0)
            env._engine.signal_buffer[:, 18].fill_(5.0)
            return env._engine.frames, rewards, terminated, truncated

        env._step_engine = fake_step
        _observations, rewards, _terminated, _truncated, infos = env.step(
            torch.zeros(2, dtype=torch.int64)
        )
        assert rewards.tolist() == [1.0, 1.0]
        assert infos["episode_return"].tolist() == [1.0, 1.0]
    finally:
        env.close()


def test_profile_rejects_silently_ignored_options(square_scenario) -> None:
    with pytest.raises(ValueError, match=r"area.*only"):
        _env(square_scenario, obs_resize_algorithm="nearest")
    skill_one = _env(square_scenario, doom_skill=1)
    skill_one.close()
    with pytest.raises(ValueError, match="skill 1 or 3"):
        _env(square_scenario, doom_skill=2)
    wall_scaled = _env(square_scenario, wall_contact_damage_scale=0.5)
    assert wall_scaled.wall_contact_damage_scale == 0.5
    wall_scaled.close()
    with pytest.raises(ValueError, match="wall_contact_damage_scale"):
        _env(square_scenario, wall_contact_damage_scale=1.01)
    with pytest.raises(ValueError, match="unsupported vizdoom_config"):
        _env(square_scenario, vizdoom_config={"render_hud": True})
    with pytest.raises(ValueError, match="only VizdoomDeathmatch-v1"):
        scenario_buttons("VizdoomBasic-v1")


def test_close_and_render_validation(square_scenario) -> None:
    env = _env(square_scenario)
    env.reset(seed=5)
    with pytest.raises(TypeError, match="integer"):
        env.render_lane(True)
    assert env.render_lane(0).shape == (240, 320, 3)
    assert len(env.get_images()) == 2
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.render()
