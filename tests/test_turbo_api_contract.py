from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import EnvSpec
from gymnasium.vector import AutoresetMode

import gradoom
from gradoom import GraDoomVecEnv, scenario_buttons


def _env(square_scenario, **kwargs) -> GraDoomVecEnv:
    return GraDoomVecEnv(
        game="VizdoomDeathmatch-v1",
        compiled_scenario=square_scenario,
        num_envs=2,
        device="cpu",
        transport=kwargs.pop("transport", "torch"),
        render_mode=kwargs.pop("render_mode", "rgb_array"),
        obs_crop=kwargs.pop("obs_crop", (0, 32, 0, 0)),
        obs_crop_mode=kwargs.pop("obs_crop_mode", "mask"),
        frame_skip=kwargs.pop("frame_skip", 2),
        **kwargs,
    )


def test_generic_gymnasium_registration_is_vector_only_and_idempotent(monkeypatch):
    spec = gym.spec(gradoom.GYMNASIUM_ENV_ID)
    assert spec.entry_point is None
    assert spec.vector_entry_point == "gradoom:_make_gymnasium_vec_env"
    assert spec.kwargs == {}
    gradoom._register_gymnasium_env()

    with pytest.raises(gym.error.Error, match="entry_point is not specified"):
        gym.make(gradoom.GYMNASIUM_ENV_ID, game="VizdoomDeathmatch-v1")
    with pytest.raises(TypeError, match="game"):
        gym.make_vec(gradoom.GYMNASIUM_ENV_ID, num_envs=1)

    monkeypatch.setitem(
        gym.registry,
        gradoom.GYMNASIUM_ENV_ID,
        EnvSpec(
            id=gradoom.GYMNASIUM_ENV_ID,
            entry_point=None,
            vector_entry_point="tests:conflicting_factory",
        ),
    )
    with pytest.raises(gym.error.Error, match="conflicting specification"):
        gradoom._register_gymnasium_env()


def test_module_qualified_gymnasium_id_registers_in_a_clean_process():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root / "src"), env.get("PYTHONPATH"))))
    subprocess.run(
        [
            sys.executable,
            "-c",
            'exec("""import gymnasium as gym\n'
            "assert 'GraDOOM-v0' not in gym.registry\n"
            "try:\n"
            "    gym.make_vec('gradoom:GraDOOM-v0', num_envs=1)\n"
            "except TypeError as exc:\n"
            "    assert 'game' in str(exc)\n"
            "else:\n"
            "    raise AssertionError('game was not required')\n"
            "spec = gym.spec('GraDOOM-v0')\n"
            "assert spec.vector_entry_point == "
            '\'gradoom:_make_gymnasium_vec_env\'\n""")',
        ],
        check=True,
        cwd=root,
        env=env,
    )


def test_generic_gymnasium_factory_preserves_torch_transport(square_scenario):
    env = gym.make_vec(
        "gradoom:GraDOOM-v0",
        game="VizdoomDeathmatch-v1",
        num_envs=2,
        device="cpu",
        compiled_scenario=square_scenario,
    )
    try:
        assert isinstance(env, GraDoomVecEnv)
        observations, _infos = env.reset(seed=7)
        assert isinstance(observations, torch.Tensor)
        transition = env.step(torch.zeros(2, dtype=torch.int64))
        assert all(isinstance(value, torch.Tensor) for value in transition[:4])
    finally:
        env.close()


def test_public_surface_matches_turbo_vector_api_v2(square_scenario) -> None:
    parameters = inspect.signature(GraDoomVecEnv).parameters
    common_parameters = (
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
        "transport",
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
    )
    provider_extensions = {
        "device",
        "doom_map",
        "doom_skill",
        "wall_contact_damage_scale",
        "game_args",
        "game_variables",
        "enemy_variants",
        "surface_variants",
        "treat_episode_timeout_as_truncation",
        "vizdoom_config",
        "compiled_scenario",
        "require_pinned_scenario",
        "compile_engine",
    }
    assert tuple(parameters)[: len(common_parameters)] == common_parameters
    assert tuple(
        parameter.default for parameter in tuple(parameters.values())[: len(common_parameters)]
    ) == (
        inspect.Parameter.empty,
        None,
        None,
        None,
        "default",
        False,
        1,
        "stable",
        "image",
        None,
        1,
        None,
        None,
        "default",
        "safe_view",
        (84, 84),
        None,
        "remove",
        0,
        True,
        "area",
        "chw",
        4,
        4,
        False,
        0,
        False,
        0.0,
        False,
        "all",
        None,
        None,
    )
    assert set(parameters) - set(common_parameters) == provider_extensions
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in tuple(parameters.values())[10:]
    )
    assert parameters["game"].default is inspect.Parameter.empty
    assert parameters["transport"].default == "default"
    assert parameters["render_mode"].default is None
    assert parameters["frame_skip"].default == 4
    assert issubclass(GraDoomVecEnv, gym.vector.VectorEnv)
    assert GraDoomVecEnv.metadata["autoreset_mode"] is AutoresetMode.DISABLED
    assert GraDoomVecEnv.metadata["turbo_api_version"] == 2
    assert GraDoomVecEnv.metadata["transition_transport"] == "torch"

    env = _env(square_scenario)
    try:
        expected_capabilities = (
            "supported_action_modes",
            "supported_observation_layouts",
            "supported_observation_color_modes",
            "supported_resize_algorithms",
            "supported_crop_modes",
            "supported_observation_copy_modes",
            "supported_transition_transports",
            "supports_async_step",
            "supports_branching",
            "supports_device_api",
            "supports_emulator_ram",
            "supports_enemy_variants",
            "supports_fire_reset",
            "supports_info_frame_stack",
            "supports_live_snapshots",
            "supports_maxpool_last_two",
            "supports_noop_reset",
            "supports_per_lane_rgb",
            "supports_reward_clipping",
            "supports_snapshot_codec",
            "supports_state_catalog",
            "supports_sticky_action_prob",
            "supports_surface_variants",
        )
        assert isinstance(env.capabilities, type(MappingProxyType({})))
        assert tuple(env.capabilities) == expected_capabilities
        assert env.capabilities["supported_action_modes"] == ("custom_discrete",)
        assert env.capabilities["supported_resize_algorithms"] == ("area",)
        assert env.num_threads is None
        assert env.state_catalog == ("default",)
        assert env.info_frame_stack_keys == ()
        assert env.supports_live_snapshots is False
        assert env.live_snapshots_deterministic is False
        assert isinstance(env.signal_schema, type(MappingProxyType({})))
        assert all(
            isinstance(spec, type(MappingProxyType({}))) for spec in env.signal_schema.values()
        )
        observations, infos = env.reset(seed=19)
        transition = env.step(torch.zeros(env.num_envs, dtype=torch.int64))
        for name, spec in env.signal_schema.items():
            assert isinstance(spec["dtype"], str)
            assert isinstance(spec["shape"], tuple)
            if spec["available_on_reset"]:
                assert getattr(torch, spec["dtype"]) == infos[name].dtype
                assert tuple(infos[name].shape[1:]) == spec["shape"]
        assert all(
            isinstance(value, torch.Tensor)
            for value in (observations, *transition[:4], *infos.values(), *transition[4].values())
        )

        active = env.active_state_indices()
        assert active is env.active_state_indices()
        assert active.shape == (env.num_envs,)
        assert active.dtype == torch.int32
        assert active.device == env.device
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
        assert all(isinstance(value, torch.Tensor) for value in infos.values())
        assert infos["start_source"].dtype == torch.int8

        with pytest.raises(TypeError, match="Torch tensor"):
            env.step(np.zeros(2, dtype=np.int64))
        transition = env.step(torch.zeros(2, dtype=torch.int64))
        assert all(isinstance(value, torch.Tensor) for value in transition[:4])
        assert all(isinstance(value, torch.Tensor) for value in transition[4].values())
    finally:
        env.close()

    with pytest.raises(ValueError, match="num_threads is unsupported"):
        _env(square_scenario, num_threads=1)


def test_v2_defaults_action_resolution_rendering_catalog_and_async(square_scenario) -> None:
    env = GraDoomVecEnv(
        game="VizdoomDeathmatch-v1",
        compiled_scenario=square_scenario,
        num_envs=2,
        device="cpu",
    )
    try:
        assert env.transport == "torch"
        assert env.action_preset == "deathmatch-p1-v1"
        assert env.render_mode is None
        assert env.render() is None
        assert env.get_images() == [None, None]
        assert env.capabilities["supports_per_lane_rgb"] is False
        observations, infos = env.reset(seed=17)
        assert observations.shape == (2, 4, 84, 84)
        assert infos["state_index"].dtype == torch.int32
        assert infos["start_source"].dtype == torch.int8
        assert infos["noop_reset_count"].dtype == torch.int64
        env.step_async(torch.zeros(2, dtype=torch.int64))
        transition = env.step_wait()
        assert all(isinstance(value, torch.Tensor) for value in transition[:4])
    finally:
        env.close()

    with pytest.raises(ValueError, match="mutually exclusive"):
        GraDoomVecEnv(
            game="VizdoomDeathmatch-v1",
            state="default",
            state_catalog=("default",),
            compiled_scenario=square_scenario,
            device="cpu",
        )
    with pytest.raises(ValueError, match="unique"):
        GraDoomVecEnv(
            game="VizdoomDeathmatch-v1",
            state_catalog=("default", "default"),
            compiled_scenario=square_scenario,
            device="cpu",
        )


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
        assert history.signal_schema["health_frame_stack"]["dtype"] == "float64"
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
