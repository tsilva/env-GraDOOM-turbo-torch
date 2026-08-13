from __future__ import annotations

import numpy as np
import pytest

from gradoom.scenario import CompiledScenario


@pytest.fixture
def square_scenario() -> CompiledScenario:
    vertices = np.asarray([(-256, -256), (256, -256), (256, 256), (-256, 256)], dtype=np.float32)
    walls = np.asarray(
        [
            (-256, -256, 256, -256),
            (256, -256, 256, 256),
            (256, 256, -256, 256),
            (-256, 256, -256, -256),
        ],
        dtype=np.float32,
    )
    return CompiledScenario(
        scenario_sha256="0" * 64,
        iwad_sha256="1" * 64,
        namespace="zdoom",
        vertices=vertices,
        wall_segments=walls,
        blocking_segments=walls.copy(),
        blocking_wall_indices=np.arange(4, dtype=np.int32),
        wall_texture_ids=np.zeros(4, dtype=np.int32),
        wall_texture_offsets=np.zeros((4, 2), dtype=np.float32),
        wall_side_texture_ids=np.concatenate(
            (
                np.zeros((4, 1, 1), dtype=np.int32),
                np.full((4, 1, 1), -1, dtype=np.int32),
            ),
            axis=1,
        ).repeat(3, axis=2),
        wall_side_texture_offsets=np.zeros((4, 2, 2), dtype=np.float32),
        wall_sectors=np.zeros((4, 2), dtype=np.int32),
        sector_edge_mask=np.ones((1, 4), dtype=np.bool_),
        sector_heights=np.asarray([(0, 128)], dtype=np.float32),
        sector_lights=np.asarray([192], dtype=np.int16),
        sector_floor_texture_ids=np.zeros(1, dtype=np.int32),
        sector_ceiling_texture_ids=np.zeros(1, dtype=np.int32),
        player_starts=np.asarray(
            [(-128, -128, 45), (128, -128, 135), (0, 128, 270)], dtype=np.float32
        ),
        item_spawns=np.empty((0, 3), dtype=np.float32),
        item_types=np.empty((0,), dtype=np.int32),
        playpal=np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1),
        texture_names=("TEST",),
        texture_atlas=np.full((1, 1, 1), 192, dtype=np.uint8),
        texture_widths=np.ones(1, dtype=np.int32),
        texture_heights=np.ones(1, dtype=np.int32),
        sprite_names=tuple(f"TEST{index}" for index in range(26)),
        sprite_atlas=np.full((26, 1, 1), 224, dtype=np.uint8),
        sprite_opaque=np.ones((26, 1, 1), dtype=np.bool_),
        sprite_widths=np.ones(26, dtype=np.int32),
        sprite_heights=np.ones(26, dtype=np.int32),
        sprite_left_offsets=np.zeros(26, dtype=np.int32),
        sprite_top_offsets=np.full(26, 42, dtype=np.int32),
        weapon_sprite_names=tuple(f"WEAPON{index}" for index in range(8)),
        weapon_screen_values=np.zeros((8, 84, 84), dtype=np.float32),
        weapon_screen_alpha=np.zeros((8, 84, 84), dtype=np.float32),
    )
