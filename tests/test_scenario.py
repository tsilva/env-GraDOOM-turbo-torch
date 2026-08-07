from __future__ import annotations

from pathlib import Path

import pytest

from gradoom.scenario import (
    KNOWN_DOOM2_WAD_SHA256,
    PINNED_DEATHMATCH_WAD_SHA256,
    compile_deathmatch_scenario,
)

SCENARIO = Path("/Users/tsilva/repos/tsilva/ViZDoom-turbo/scenarios/deathmatch.wad")
DOOM2 = Path("/Users/tsilva/roms/vizdoom/doom2.wad")
FREEDOOM2 = Path("/Users/tsilva/repos/tsilva/ViZDoom-turbo/bin/python3.14/vizdoom/freedoom2.wad")


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_pinned_deathmatch_compiles_external_doom2_data() -> None:
    scenario = compile_deathmatch_scenario(SCENARIO, DOOM2)
    assert scenario.scenario_sha256 == PINNED_DEATHMATCH_WAD_SHA256
    assert scenario.iwad_sha256 == KNOWN_DOOM2_WAD_SHA256
    assert scenario.namespace == "zdoom"
    assert scenario.vertices.shape == (198, 2)
    assert scenario.wall_segments.shape == (215, 4)
    assert scenario.blocking_segments.shape == (88, 4)
    assert scenario.blocking_wall_indices.shape == (88,)
    assert (scenario.wall_sectors[scenario.blocking_wall_indices, 1] < 0).all()
    assert scenario.texture_names == (
        "BFALL1",
        "BIGBRIK1",
        "BLOOD1",
        "BRICK12",
        "CEIL3_3",
        "CEIL4_1",
        "COMPBLUE",
        "FLAT5_3",
    )
    assert scenario.texture_atlas.shape == (8, 128, 64)
    assert scenario.texture_widths.tolist() == [64] * 8
    assert scenario.texture_heights.tolist() == [128, 128, 64, 128, 64, 64, 128, 64]
    assert (scenario.wall_texture_ids[scenario.blocking_wall_indices] >= 0).all()
    assert scenario.sector_edge_mask.shape == (14, 215)
    assert scenario.sector_floor_texture_ids.shape == (14,)
    assert scenario.sector_ceiling_texture_ids.shape == (14,)
    assert len(scenario.sprite_names) == 26
    assert scenario.sprite_names[5].startswith("BOS2A1")
    assert scenario.sprite_atlas.shape == scenario.sprite_opaque.shape
    assert scenario.sprite_atlas.shape[0] == 26
    assert scenario.sprite_opaque.any(axis=(1, 2)).all()
    assert scenario.wall_side_texture_ids.shape == (215, 2, 3)
    assert scenario.wall_side_texture_offsets.shape == (215, 2, 2)
    assert scenario.weapon_sprite_names == (
        "PUNGA0",
        "SAWGC0",
        "PISGA0",
        "SHTGA0",
        "SHT2A0",
        "CHGGA0",
        "MISGA0",
        "PLSGA0",
    )
    assert scenario.weapon_screen_values.shape == (8, 84, 84)
    assert scenario.weapon_screen_alpha.shape == (8, 84, 84)
    assert scenario.weapon_screen_alpha.any(axis=(1, 2)).all()
    assert scenario.sector_heights.shape == (14, 2)
    assert scenario.player_starts.shape == (3, 3)
    assert scenario.item_spawns.shape == (192, 3)
    assert (scenario.item_spawns[:, 2] == 0).all()
    assert scenario.item_types.shape == (192,)
    assert scenario.bounds == (-256.0, 1280.0, -256.0, 1280.0)


@pytest.mark.skipif(
    not SCENARIO.is_file() or not FREEDOOM2.is_file(),
    reason="operator WADs absent",
)
def test_pinned_deathmatch_compiles_external_freedoom2_data() -> None:
    scenario = compile_deathmatch_scenario(SCENARIO, FREEDOOM2)

    assert scenario.weapon_screen_values.shape == (8, 84, 84)
    assert scenario.weapon_screen_alpha.any(axis=(1, 2)).all()
