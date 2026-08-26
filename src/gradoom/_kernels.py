"""Portable access to optional CUDA-only Triton kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    from ._triton_kernels import (
        bounded_observation_augment,
        enemy_hitscan_trace,
        enemy_projectile_move,
        enemy_sight_blocked,
        enemy_sight_opening,
        first_free_enemy_slot,
        frozen_nature_conv1,
        initialize_enemy_spawn,
        masked_portal_intersections,
        masked_render_portal_walls_,
        move_drops_,
        move_enemy_thrust,
        player_projectile_move,
        policy_area_grayscale,
        portal_intersections,
        random_spawn_candidates,
        render_fast_native_flats,
        render_fast_native_portal_walls_,
        render_fast_native_sprites_,
        render_native_weapon,
        render_portal_walls_,
        rocket_splash_blocked,
        select_enemy_spawn_position,
        try_enemy_chase_step,
    )
except ModuleNotFoundError as exc:
    if exc.name != "triton":
        raise

    def _triton_unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("GraDOOM Triton kernels require a CUDA-capable Torch installation")

    bounded_observation_augment: Callable[..., Any] = _triton_unavailable
    enemy_hitscan_trace: Callable[..., Any] = _triton_unavailable
    enemy_projectile_move: Callable[..., Any] = _triton_unavailable
    enemy_sight_blocked: Callable[..., Any] = _triton_unavailable
    enemy_sight_opening: Callable[..., Any] = _triton_unavailable
    first_free_enemy_slot: Callable[..., Any] = _triton_unavailable
    frozen_nature_conv1: Callable[..., Any] = _triton_unavailable
    initialize_enemy_spawn: Callable[..., Any] = _triton_unavailable
    masked_portal_intersections: Callable[..., Any] = _triton_unavailable
    masked_render_portal_walls_: Callable[..., Any] = _triton_unavailable
    move_drops_: Callable[..., Any] = _triton_unavailable
    move_enemy_thrust: Callable[..., Any] = _triton_unavailable
    player_projectile_move: Callable[..., Any] = _triton_unavailable
    policy_area_grayscale: Callable[..., Any] = _triton_unavailable
    portal_intersections: Callable[..., Any] = _triton_unavailable
    random_spawn_candidates: Callable[..., Any] = _triton_unavailable
    render_fast_native_flats: Callable[..., Any] = _triton_unavailable
    render_fast_native_portal_walls_: Callable[..., Any] = _triton_unavailable
    render_fast_native_sprites_: Callable[..., Any] = _triton_unavailable
    render_native_weapon: Callable[..., Any] = _triton_unavailable
    render_portal_walls_: Callable[..., Any] = _triton_unavailable
    rocket_splash_blocked: Callable[..., Any] = _triton_unavailable
    select_enemy_spawn_position: Callable[..., Any] = _triton_unavailable
    try_enemy_chase_step: Callable[..., Any] = _triton_unavailable
