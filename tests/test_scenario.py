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
        "BFALL2",
        "BFALL3",
        "BFALL4",
        "BIGBRIK1",
        "BLOOD1",
        "BLOOD2",
        "BLOOD3",
        "BRICK12",
        "CEIL3_3",
        "CEIL4_1",
        "COMPBLUE",
        "FLAT5_3",
    )
    assert scenario.texture_atlas.shape == (13, 128, 64)
    assert scenario.texture_widths.tolist() == [64] * 13
    assert scenario.texture_heights.tolist() == [
        128,
        128,
        128,
        128,
        128,
        64,
        64,
        64,
        128,
        64,
        64,
        128,
        64,
    ]
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
    assert scenario.texture_index_atlas is not None
    assert scenario.texture_index_atlas.shape == scenario.texture_atlas.shape
    assert scenario.texture_animation_ids is not None
    assert scenario.texture_animation_counts is not None
    assert scenario.texture_animation_counts[:4].tolist() == [4, 4, 4, 4]
    assert scenario.texture_animation_counts[5:8].tolist() == [3, 3, 3]
    assert scenario.colormap is not None
    assert scenario.colormap.shape == (34, 256)
    assert scenario.raw_sprite_atlas is not None
    assert scenario.raw_sprite_opaque is not None
    assert scenario.raw_sprite_atlas.shape == scenario.raw_sprite_opaque.shape
    assert scenario.enemy_walk_sprite_ids is not None
    assert scenario.enemy_walk_sprite_ids.shape == (6, 4, 8)
    assert len(set(scenario.enemy_walk_sprite_ids[0, :, 0].tolist())) == 4
    assert scenario.enemy_attack_sprite_ids is not None
    assert scenario.enemy_attack_sprite_ids.shape == (6, 4, 8)
    assert scenario.enemy_death_sprite_ids is not None
    assert scenario.enemy_death_sprite_ids.shape == (6, 7)
    assert scenario.enemy_death_frame_counts is not None
    assert scenario.enemy_death_frame_counts.tolist() == [5, 5, 7, 7, 6, 7]
    assert scenario.raw_static_sprite_ids is not None
    assert scenario.raw_static_sprite_ids.shape == (20,)
    assert scenario.raw_item_animation_sprite_ids is not None
    assert scenario.raw_item_animation_sprite_ids.shape == (8,)
    assert scenario.native_weapon_screen_values is not None
    assert scenario.native_weapon_screen_values.shape == (8, 208, 320)
    assert scenario.native_weapon_screen_alpha is not None
    assert scenario.native_weapon_screen_alpha.any(axis=(1, 2)).all()
    assert scenario.native_weapon_frame_values is not None
    assert scenario.native_weapon_frame_values.shape[1:] == (208, 320)
    assert scenario.native_weapon_frame_alpha is not None
    assert scenario.native_weapon_frame_alpha.any(axis=(1, 2)).all()
    assert scenario.native_weapon_frame_ids is not None
    assert scenario.native_weapon_frame_ids.shape == (8, 2, 62)
    assert scenario.native_weapon_flash_ids is not None
    assert scenario.native_weapon_flash_ids.shape == (8, 2, 62)
    assert scenario.native_weapon_flash_lights is not None
    assert scenario.native_weapon_flash_lights.shape == (8, 2, 62)
    assert scenario.enemy_death_frame_durations is not None
    assert scenario.enemy_death_frame_durations.shape == (6, 7)
    assert scenario.enemy_death_total_tics is not None
    assert scenario.enemy_death_total_tics.tolist() == [21, 21, 61, 31, 29, 49]
    assert scenario.enemy_pain_sprite_ids is not None
    assert scenario.enemy_pain_sprite_ids.shape == (6, 8)
    assert scenario.raw_projectile_flight_sprite_ids is not None
    assert scenario.raw_projectile_flight_sprite_ids.shape == (3, 2, 8)
    assert scenario.raw_projectile_explosion_sprite_ids is not None
    assert scenario.raw_projectile_explosion_sprite_ids.shape == (3, 5)
    assert scenario.raw_teleport_fog_sprite_ids is not None
    assert scenario.raw_teleport_fog_sprite_ids.shape == (12,)
    assert scenario.projectile_explosion_frame_counts is not None
    assert scenario.projectile_explosion_frame_counts.tolist() == [3, 5, 3]
    assert scenario.projectile_explosion_frame_durations is not None
    assert scenario.projectile_explosion_total_tics is not None
    assert scenario.projectile_explosion_total_tics.tolist() == [18, 20, 18]
    assert scenario.projectile_additive_luts is not None
    assert scenario.projectile_additive_luts.shape == (2, 256, 256)
    assert scenario.projectile_additive_luts[0, 0, 3] == 107
    assert scenario.projectile_additive_luts[1, 0, 3] == 3
    assert scenario.projectile_additive_luts[0, 71, 73] == 62
    assert scenario.projectile_additive_luts[1, 71, 73] == 213
    assert scenario.hud_patch_names[0:2] == ("STBAR", "STARMS")
    assert scenario.hud_patch_names[44:49] == tuple(
        f"STFTR{pain}0" for pain in range(5)
    )
    assert scenario.hud_patch_names[69] == "STFDEAD0"
    assert scenario.hud_patch_atlas is not None
    assert scenario.hud_patch_atlas.shape == (70, 32, 320)
    assert scenario.hud_patch_left_offsets is not None
    assert scenario.hud_patch_top_offsets is not None
    assert scenario.hud_patch_left_offsets[14] == -5
    assert scenario.hud_patch_top_offsets[14] == -2
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
    assert scenario.native_weapon_screen_values is not None
    assert scenario.native_weapon_screen_values.shape == (8, 208, 320)
