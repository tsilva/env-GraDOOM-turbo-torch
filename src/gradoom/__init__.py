"""GPU-native Doom reinforcement-learning environments."""

from typing import Any

import gymnasium as gym

from .actions import DEATHMATCH_ACTIONS, DEATHMATCH_BUTTONS
from .env import DeviceAutoResetTransition, DeviceTransition, GraDoomVecEnv, scenario_buttons
from .scenario import CompiledScenario, compile_deathmatch_scenario

GYMNASIUM_ENV_ID = "GraDOOM-v0"
_GYMNASIUM_VECTOR_ENTRY_POINT = "gradoom:_make_gymnasium_vec_env"


def _make_gymnasium_vec_env(*, game: str, num_envs: int = 1, **kwargs: Any) -> GraDoomVecEnv:
    return GraDoomVecEnv(game=game, num_envs=num_envs, **kwargs)


def _register_gymnasium_env() -> None:
    existing = gym.registry.get(GYMNASIUM_ENV_ID)
    if existing is None:
        gym.register(
            id=GYMNASIUM_ENV_ID,
            entry_point=None,
            vector_entry_point=_GYMNASIUM_VECTOR_ENTRY_POINT,
        )
        return
    if (
        existing.entry_point is None
        and existing.vector_entry_point == _GYMNASIUM_VECTOR_ENTRY_POINT
        and existing.kwargs == {}
        and existing.max_episode_steps is None
        and existing.additional_wrappers == ()
    ):
        return
    raise gym.error.Error(
        f"Gymnasium environment ID {GYMNASIUM_ENV_ID!r} is already registered "
        "with a conflicting specification"
    )


_register_gymnasium_env()

__all__ = [
    "DEATHMATCH_ACTIONS",
    "DEATHMATCH_BUTTONS",
    "GYMNASIUM_ENV_ID",
    "CompiledScenario",
    "DeviceAutoResetTransition",
    "DeviceTransition",
    "GraDoomVecEnv",
    "compile_deathmatch_scenario",
    "scenario_buttons",
]

__version__ = "0.1.0a2"
