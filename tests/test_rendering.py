from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gradoom.engine import TorchDeathmatchEngine
from gradoom.scenario import compile_deathmatch_scenario

SCENARIO = Path("/Users/tsilva/repos/tsilva/ViZDoom-turbo/scenarios/deathmatch.wad")
DOOM2 = Path("/Users/tsilva/roms/vizdoom/doom2.wad")


@pytest.fixture(scope="module")
def pinned_deathmatch_scenario():
    if not SCENARIO.is_file() or not DOOM2.is_file():
        pytest.skip("operator WADs absent")
    return compile_deathmatch_scenario(SCENARIO, DOOM2)


def test_doom_sprite_rotation_uses_actor_to_viewer_angle() -> None:
    viewer_angle = torch.tensor((0.0, math.pi / 2, math.pi, -math.pi / 2))
    actor_angle = torch.zeros_like(viewer_angle)

    rotation = TorchDeathmatchEngine._doom_sprite_rotation(
        viewer_angle,
        actor_angle,
    )

    assert rotation.tolist() == [0, 2, 4, 6]


def test_pitch_view_pan_uses_reference_fixed_tangent_projection(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.enemy_alive.zero_()
    engine.item_available.zero_()
    baseline_native = engine.render_native_frame(include_hud=False)
    baseline_training = engine.render_frame()

    engine._pitch_bam.fill_(-(182 << 16) * 10)
    engine.pitch.copy_(engine._pitch_bam.to(torch.float32) * (2.0 * math.pi / float(1 << 32)))

    # ZDoom quantizes view pitch through its 8192-entry finetangent table;
    # ten binary +1 deltas therefore pan a 320-wide view by this exact amount.
    assert engine._pitch_projection_offset(192.0).item() == 33.7705078125
    assert not torch.equal(
        baseline_native[:, :150, :],
        engine.render_native_frame(include_hud=False)[:, :150, :],
    )
    assert not torch.equal(baseline_training, engine.render_frame())


def test_screen_flashes_follow_vizdoom_render_option(
    pinned_deathmatch_scenario,
) -> None:
    default_engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    default_engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    native_without_counters = default_engine.render_native_frame(include_hud=True)
    training_without_counters = default_engine.render_frame()
    default_engine.bonus_count.fill_(6)
    default_engine.damage_count.fill_(13)

    assert torch.equal(
        default_engine.render_native_frame(include_hud=True),
        native_without_counters,
    )
    assert torch.equal(default_engine.render_frame(), training_without_counters)
    assert default_engine.bonus_count.item() == 6
    assert default_engine.damage_count.item() == 13

    flash_engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
        render_screen_flashes=True,
    )
    flash_engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    native_without_flash = flash_engine.render_native_frame(include_hud=True)
    training_without_flash = flash_engine.render_frame()
    flash_engine.bonus_count.fill_(6)
    flash_engine.damage_count.fill_(13)

    assert not torch.equal(
        flash_engine.render_native_frame(include_hud=True),
        native_without_flash,
    )
    assert not torch.equal(flash_engine.render_frame(), training_without_flash)


def test_native_flats_match_reference_span_sampling(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.x.fill_(940.9204254150391)
    engine.y.fill_(826.7186584472656)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(341.29028328258784))
    engine.episode_time.fill_(17)

    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # R_SetupFreelook intersects planes through y + 0.5.
    assert rgb[0, 127, 136].tolist() == [79, 59, 39]
    # R_DrawNormalPlane anchors spans at centerx - 1, independently of walls.
    assert rgb[0, 127, 135].tolist() == [79, 59, 39]
    # Its 16.16/32-bit stepping selects the adjacent texel here; continuous
    # floating-point ray mapping produces [79, 59, 39] instead.
    assert rgb[0, 127, 141].tolist() == [79, 59, 43]


def test_native_walls_use_reference_rounded_texel_length(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([789]))
    engine.x.fill_(569.3474273681641)
    engine.y.fill_(515.9971313476562)
    engine.z.fill_(-64)
    engine.view_z.fill_(-27)
    engine.angle.fill_(math.radians(348.81591804996503))
    engine.episode_time.fill_(17)

    (
        wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # P_FinishLoadingLineDef rounds this diagonal wall's TexelLength. Keeping
    # its exact Euclidean length selects the neighboring red-rock column.
    assert rgb[0, 0, 2].tolist() == [127, 0, 0]
    # At these nested-pit vertices, the integer column ray still intersects the
    # seg whose half-open range just ended. BSP rasterization instead gives the
    # column to the adjacent projected seg with the same sector pair.
    assert geometric_intersections[0, 145, 92]
    assert not projected_intersections[0, 145, 92]
    assert projected_left_edges[0, 145, 100]
    assert torch.isfinite(wall_distance[0, 145, 100])
    assert geometric_intersections[0, 156, 81]
    assert not projected_intersections[0, 156, 81]
    assert projected_left_edges[0, 156, 116]
    assert torch.isfinite(wall_distance[0, 156, 116])
    assert rgb[0, 50, 145].tolist() == [79, 0, 0]
    assert rgb[0, 100, 156].tolist() == [91, 0, 0]


def test_native_walls_use_reference_fine_angle_rays(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(17)

    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # R_RenderBSPNode transforms walls through the 8192-entry fine-angle
    # basis. Continuous sin/cos intersects the neighboring stone column.
    assert rgb[0, 0, 272].tolist() == [43, 35, 15]
    # wallscan interpolates 16.16 visibility between FWallCoords' 20.12
    # endpoint depths. A direct floating-point 1280/distance lookup selects
    # the next brighter colormap at this threshold.
    assert rgb[0, 20, 255].tolist() == [87, 67, 51]
    # Endpoint ownership applies only while the adjacent projected seg remains
    # ahead of the current portal depth. Reusing an owner from an earlier BSP
    # layer incorrectly paints the left edge with the pit's blue ceiling.
    assert rgb[0, 35, 5].tolist() == [119, 95, 75]
    # R_MapPlane lights against the integer row edge even though its texture
    # lookup uses a half-pixel yslope. Reusing the sampling distance chooses
    # the next brighter colormap and produces [83, 63, 47] here.
    assert rgb[0, 119, 10].tolist() == [79, 59, 43]
    # The same floor visplane stays horizontally continuous where independent
    # plane rays fall between the nested pit polygons near the screen edge.
    assert rgb[0, 135, 10].tolist() == [79, 0, 0]


def test_native_portal_clips_bound_solid_wall_against_planes(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(752.9335632324219)
    engine.y.fill_(49.700897216796875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(17)

    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    (
        wall_distance,
        _wall_along,
        _geometric_intersections,
        _projected_intersections,
        _projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections()
    frame, scene_depth = engine._native_render_portal_walls(
        flat_frame.clone(),
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    flat_rgb = engine.map.playpal[flat_frame.to(torch.int64)]
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # R_RenderSegLoop draws this solid wall before its front-sector ceiling
    # visplane. The pixel-center flat intersection is slightly closer, but it
    # must not leave a black diagonal hole through the BIGBRIK1 wall.
    assert wall_distance[0, 137, 196] > surface_depth[0, 37, 137]
    assert flat_rgb[0, 37, 137].tolist() == [0, 0, 0]
    assert rgb[0, 35, 118].tolist() == [0, 0, 23]
    assert rgb[0, 37, 137].tolist() == [159, 135, 111]
    # The accumulated floor clip keeps a farther red lower wall behind this
    # front-sector flat, matching the inverse operation at the ceiling edge.
    assert flat_rgb[0, 122, 260].tolist() == [83, 63, 47]
    assert rgb[0, 122, 260].tolist() == [83, 63, 47]
    # Doom marks floor visplanes as continuous screen-space spans. Independent
    # plane rays fall between every nested pit polygon on this row, but the
    # surrounding span anchors still assign sector 10's BLOOD1 floor.
    assert flat_rgb[0, 124, 290].tolist() == [79, 0, 0]
    assert rgb[0, 124, 290].tolist() == [79, 0, 0]
    # Wall ordering retains the unresolved polygon-ray depth, while sprites
    # see the repaired visplane depth and cannot leak through this pixel.
    assert torch.isinf(surface_depth[0, 124, 290])
    assert scene_surface_depth[0, 124, 290].item() == pytest.approx(458.92681884765625)
    assert scene_depth[0, 124, 290].item() == pytest.approx(458.92681884765625)


def test_native_repaired_visplane_depth_occludes_drops(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(752.9335632324219)
    engine.y.fill_(49.700897216796875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(17)
    engine.item_available.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.drop_spawned.zero_()
    engine.player_dead.fill_(True)

    # Put a clip sprite behind the repaired floor span at screen column 290.
    # Its opaque texels reproduce the item leak that an infinite scene depth
    # permitted even after the floor color itself had been repaired.
    ray = engine._native_wall_ray_directions()[0, 290]
    engine.drop_type[0, 0] = 2007
    engine.drop_spawned[0, 0] = True
    engine.drop_x[0, 0] = engine.x[0] + ray[0] * 600.0
    engine.drop_y[0, 0] = engine.y[0] + ray[1] * 600.0
    engine.drop_z[0, 0] = -32.0

    wall_distance = engine._native_raycast()
    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(), engine.view_z
    )
    portal_frame, scene_depth = engine._native_render_portal_walls(
        flat_frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    with_unrepaired_depth = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        surface_depth,
    )
    with_repaired_depth = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        scene_depth,
    )

    assert torch.isinf(surface_depth[0, 124, 290])
    assert with_unrepaired_depth[0, 124, 290] != portal_frame[0, 124, 290]
    assert torch.equal(with_repaired_depth, portal_frame)


def test_native_walls_use_reference_fixed_vertical_sampling(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(752.9335632324219)
    engine.y.fill_(49.700897216796875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(17)

    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # PrepWall rounds its inverse-depth scale before wallscan advances the
    # vertical column DDA. Continuous world-Z sampling selects [123, 99, 79].
    assert rgb[0, 1, 40].tolist() == [131, 107, 87]


def test_native_walls_use_reference_half_open_screen_bounds(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(752.9335632324219)
    engine.y.fill_(49.700897216796875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(17)

    (
        wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # FWallCoords projects shared endpoints to [sx1, sx2). The right edge of
    # walls 17 and 168 is therefore excluded, while the adjacent walls 60 and
    # 59 own those exact columns. A closed ray/segment test reverses ownership.
    assert torch.isinf(wall_distance[0, 67, 17])
    assert torch.isfinite(wall_distance[0, 67, 60])
    assert torch.isinf(wall_distance[0, 90, 168])
    assert torch.isfinite(wall_distance[0, 90, 59])
    # Wall 184's geometric ray hit falls outside its fixed [106, 107) solid
    # span. The adjacent wall 185 owns [107, 110) even though this column ray
    # misses it, preserving the continuous BFALL1 seam and its x offset.
    assert geometric_intersections[0, 107, 184]
    assert not projected_intersections[0, 107, 184]
    assert torch.isinf(wall_distance[0, 107, 184])
    assert not geometric_intersections[0, 107, 185]
    assert projected_intersections[0, 107, 185]
    assert projected_left_edges[0, 107, 185]
    assert torch.isfinite(wall_distance[0, 107, 185])
    assert rgb[0, 45, 107].tolist() == [107, 15, 15]
    # R_AddLine rejects this one-sided linedef's back face. Sector 0 is
    # non-convex, so incidence alone would incorrectly expose BIGBRIK1 here.
    assert torch.isinf(wall_distance[0, 76, 8])
    # Portal 163 owns this projected endpoint column even though the column's
    # geometric ray misses its segment. Its upper tier renders COMPBLUE, but
    # it must not move traversal into sector 8.
    assert torch.isfinite(wall_distance[0, 76, 163])
    assert not geometric_intersections[0, 76, 163]
    assert rgb[0, 40, 67].tolist() == [0, 0, 71]
    assert rgb[0, 40, 76].tolist() == [0, 0, 35]
    # Sector 8's next geometric boundary is nearer than sector 0's, so the
    # endpoint portal continues into it and exposes the lower COMPBLUE wall.
    assert rgb[0, 100, 76].tolist() == [0, 0, 107]
    # At this shared vertex, solid wall 196 and portal 186 are both projected
    # endpoint-only spans. The solid's line depth rounds slightly nearer, so
    # traversal retains the prior depth and exposes its BRICK12 column.
    assert not geometric_intersections[0, 110, 186]
    assert not geometric_intersections[0, 110, 196]
    assert wall_distance[0, 110, 196] < wall_distance[0, 110, 186]
    assert rgb[0, 82, 110].tolist() == [159, 135, 111]
    # Solid wall 168 and portal 163 meet at equal depth; the solid owns the
    # shared column and selects the reference COMPBLUE texture coordinates.
    assert wall_distance[0, 86, 163] == wall_distance[0, 86, 168]
    assert geometric_intersections[0, 86, 163]
    assert not geometric_intersections[0, 86, 168]
    assert rgb[0, 82, 86].tolist() == [0, 0, 71]
    # Same-sector portal 52 carries a default-pegged BIGBRIK1 middle texture.
    # It starts at the shared ceiling and covers one texture height without
    # terminating traversal like a solid wall.
    assert engine.map.portal_wall_sectors[52].tolist() == [11, 11]
    assert engine.map.portal_side_texture_ids[52, 0, 0] >= 0
    assert rgb[0, 38, 143].tolist() == [55, 35, 19]
    assert rgb[0, 50, 148].tolist() == [51, 43, 19]
    assert rgb[0, 40, 90].tolist() == [95, 75, 55]


def test_native_weapon_uses_reference_fixed_point_vertical_sampling(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.episode_time.fill_(17)
    engine.weapon_raise_cooldown.zero_()

    frame_id, _flash_id, _flash_light = engine._native_weapon_frame_selection()
    value = engine.map.native_weapon_frame_values[frame_id][0]
    alpha = engine.map.native_weapon_frame_alpha[frame_id][0]

    # R_DrawPSprite retains WEAPONTOP's fractional 0x6000 and
    # R_DrawMaskedColumn advances source rows through a 16.16 reciprocal.
    assert alpha.sum().item() == 1783
    assert alpha[152, 157]
    assert value[152, 157].item() == 10
    assert value[152, 159].item() == 6


def test_enemy_fullbright_matches_actor_attack_states() -> None:
    enemy_type = torch.tensor((0, 1, 1, 3, 3, 3, 3, 3, 3))
    attack_phase = torch.tensor((2, 2, 2, 2, 3, 3, 4, 1, 1))
    cooldown = torch.tensor((16, 20, 10, 4, 4, 1, 1, 1, 10))
    attack_recovery = torch.tensor((16, 20, 20, 4, 4, 4, 4, 4, 4))

    fullbright = TorchDeathmatchEngine._native_enemy_fullbright(
        enemy_type,
        attack_phase,
        cooldown,
        attack_recovery,
    )

    # Zombieman's POSS F state is not BRIGHT. ShotgunGuy's SPOS F and
    # ChaingunGuy's CPOS F/E firing states are. The one-tic CPOS F
    # A_CPosRefire gap and the initial CPOS E prefire state are not.
    assert fullbright.tolist() == [
        False,
        True,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_chaingunner_refire_gap_uses_nonbright_f_frame(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 64
    engine.enemy_y[:, 0] = 0
    engine.enemy_angle[:, 0] = 0
    engine.enemy_type[:, 0] = 3
    engine.enemy_alive[:, 0] = True

    # CPOS E 4 BRIGHT A_CPosAttack remains visible until A_CPosRefire.
    engine.enemy_attack_phase[:, 0] = 3
    engine.enemy_cooldown[:, 0] = 1
    firing_e = engine._native_enemy_sprite_ids()[0, 0]

    # The final tic of the initial CPOS E prefire remains E as well.
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 1
    prefire_e = engine._native_enemy_sprite_ids()[0, 0]

    # After the refire action, CPOS F remains for one non-BRIGHT tic before
    # Goto Missile+1 enters the next CPOS F attack state.
    engine.enemy_attack_phase[:, 0] = 4
    engine.enemy_cooldown[:, 0] = 1
    refire_f = engine._native_enemy_sprite_ids()[0, 0]

    assert firing_e == engine.map.enemy_attack_sprite_ids[3, 2, 4]
    assert prefire_e == engine.map.enemy_attack_sprite_ids[3, 0, 4]
    assert refire_f == engine.map.enemy_attack_sprite_ids[3, 1, 4]


def test_native_enemy_rotation_matches_vizdoom_summoned_pose(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.enemy_x[:, 0] = 824.1785278320312
    engine.enemy_y[:, 0] = 446.0887756347656
    engine.enemy_z[:, 0] = 0
    engine.enemy_angle[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -2
    engine.enemy_animation_tics[:, 0] = 0

    sprite = engine._native_enemy_sprite_ids()[0, 0]

    assert sprite == engine.map.enemy_walk_sprite_ids[0, 0, 6]


def test_native_transparent_sprites_reveal_fifth_farther_actor(square_scenario) -> None:
    atlas = np.zeros((2, 3, 3), dtype=np.uint8)
    atlas[0] = 10
    atlas[1] = 20
    opaque = np.ones_like(atlas, dtype=np.bool_)
    opaque[0] = False
    enemy_ids = np.empty((6, 4, 8), dtype=np.int32)
    enemy_ids[:4].fill(0)
    enemy_ids[4:].fill(1)
    scenario = replace(
        square_scenario,
        player_starts=np.asarray([(0, 128, 270)], dtype=np.float32),
        raw_sprite_atlas=atlas,
        raw_sprite_opaque=opaque,
        raw_sprite_widths=np.full(2, 3, dtype=np.int32),
        raw_sprite_heights=np.full(2, 3, dtype=np.int32),
        raw_sprite_left_offsets=np.ones(2, dtype=np.int32),
        raw_sprite_top_offsets=np.full(2, 42, dtype=np.int32),
        enemy_walk_sprite_ids=enemy_ids,
        enemy_attack_sprite_ids=enemy_ids,
        enemy_death_sprite_ids=np.zeros((6, 1), dtype=np.int32),
        raw_static_sprite_ids=np.zeros(20, dtype=np.int32),
    )
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.zero_()
    engine.item_available.zero_()
    engine.enemy_x[0, :5] = torch.tensor([64.0, 72.0, 80.0, 88.0, 96.0])
    engine.enemy_y[0, :5] = 0
    engine.enemy_z[0, :5] = 0
    engine.enemy_type[0, :5] = torch.arange(5)
    engine.enemy_alive[0, :5] = True
    frame = torch.zeros((1, 208, 320), dtype=torch.uint8)

    rendered = engine._native_render_sprites(
        frame,
        torch.full((1, 320), torch.inf),
        engine.view_z,
        torch.full_like(frame, torch.inf, dtype=torch.float32),
    )

    assert rendered[0, 103, 160].item() == 20


def test_native_teleport_fog_uses_reference_animation_and_lifetime(square_scenario) -> None:
    atlas = np.stack(
        [np.full((3, 3), 10 + frame, dtype=np.uint8) for frame in range(12)]
    )
    scenario = replace(
        square_scenario,
        raw_sprite_atlas=atlas,
        raw_sprite_opaque=np.ones_like(atlas, dtype=np.bool_),
        raw_sprite_widths=np.full(12, 3, dtype=np.int32),
        raw_sprite_heights=np.full(12, 3, dtype=np.int32),
        raw_sprite_left_offsets=np.ones(12, dtype=np.int32),
        raw_sprite_top_offsets=np.full(12, 42, dtype=np.int32),
        raw_teleport_fog_sprite_ids=np.arange(12, dtype=np.int32),
    )
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.zero_()
    engine.player_dead.fill_(True)
    engine.item_available.zero_()
    engine.teleport_fog_x[:, 0] = 64
    engine.teleport_fog_y[:, 0] = 0
    engine.teleport_fog_z[:, 0] = 0
    blank = torch.zeros((1, 208, 320), dtype=torch.uint8)
    wall_distance = torch.full((1, 320), torch.inf)
    scene_depth = torch.full_like(blank, torch.inf, dtype=torch.float32)

    def center_pixel(tics: int) -> int:
        engine.teleport_fog_tics[:, 0] = tics
        rendered = engine._native_render_sprites(
            blank,
            wall_distance,
            engine.view_z,
            scene_depth,
        )
        return int(rendered[0, 103, 160])

    assert center_pixel(71) == 10
    assert center_pixel(67) == 10
    assert center_pixel(66) == 11
    assert center_pixel(60) == 12
    assert center_pixel(1) == 21
    assert center_pixel(0) == 0


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_acs_reposition_preserves_start_z_and_idle_pit_floor(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        2,
        device=torch.device("cpu"),
    )
    spawn_x = 569.3474273681641
    spawn_y = 515.9971313476562
    engine.map.spawn_bounds.copy_(
        torch.tensor((spawn_x, spawn_x, spawn_y, spawn_y))
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([789, 790]))

    # SetActorPosition keeps the map-start Z while SetOrigin derives floorz
    # only from the destination's center subsector. The player's box also
    # touches the -48 ledge, but P_XYMovement does not expand the opening
    # until the actor actually has horizontal momentum.
    box_floor, _box_ceiling = engine._player_opening_at(engine.x, engine.y)
    assert engine.z.tolist() == [0.0, 0.0]
    assert engine.player_floor_z.tolist() == [-64.0, -64.0]
    assert box_floor.tolist() == [-48.0, -48.0]

    engine.angle.fill_(math.radians(348.81591804996503))
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[1, 6] = True
    for _ in range(6):
        engine.step(buttons)

    # ViZDoom seed 789, episode time 13: the idle player lands at the bottom
    # while forward movement updates floorz and catches the adjacent ledge.
    assert engine.z.tolist() == [-64.0, -48.0]
    assert engine.player_floor_z.tolist() == [-64.0, -48.0]
    assert engine.x[1].item() == pytest.approx(570.1556396484375)
    assert engine.y[1].item() == pytest.approx(515.8375854492188)


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_player_floor_uses_full_box_across_pit_steps(pinned_deathmatch_scenario) -> None:
    scenario = pinned_deathmatch_scenario
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.x.fill_(569.3977813720703)
    engine.y.fill_(515.9373168945312)
    engine.z.fill_(-45.0)
    engine.velocity_z.fill_(-10.0)

    center_sector = engine._sector_at(engine.x, engine.y)
    floor, _ = engine._player_opening_at(engine.x, engine.y)
    engine.player_floor_z.copy_(floor)
    engine._vertical_player_tick(torch.ones(1, dtype=torch.bool))

    assert scenario.sector_heights[int(center_sector[0]), 0] == -64.0
    assert floor.item() == -48.0
    assert engine.z.item() == -48.0
    assert engine.velocity_z.item() == 0.0


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_acs_monster_spawn_falls_from_absolute_zero_into_center_pit(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([2]))
    spawn_x = 560.7010192871094
    spawn_y = 481.3784484863281
    engine.map.spawn_bounds.copy_(
        torch.tensor((spawn_x, spawn_x, spawn_y, spawn_y))
    )
    # The player occupies the same XY in the pit but ends below the spawned
    # monster. ACS Spawn temporarily enables PASSMOBJ, so this vertical gap is
    # legal even though their 2D boxes overlap.
    engine.x.fill_(spawn_x)
    engine.y.fill_(spawn_y)
    engine.z.fill_(-64)

    engine._spawn_enemy_type(1, torch.ones(1, dtype=torch.bool))

    assert engine.enemy_alive[0, 0]
    assert engine.enemy_z[0, 0].item() == 0.0
    assert engine._enemy_velocity_z_fixed[0, 0].item() == -65536
    assert engine._enemy_floor_z_fixed[0, 0].item() == -64 * 65536
    assert engine.teleport_fog_z[0, 0].item() == 0.0
    opening_floor, _ = engine._actor_opening_at(
        engine.enemy_x[:, 0],
        engine.enemy_y[:, 0],
        engine._enemy_radius[1],
    )
    assert opening_floor.item() == -24.0

    z_trace: list[float] = []
    velocity_trace: list[float] = []
    for _ in range(11):
        engine._move_enemy_thrust(torch.ones(1, dtype=torch.bool))
        z_trace.append(float(engine.enemy_z[0, 0]))
        velocity_trace.append(
            float(engine._enemy_velocity_z_fixed[0, 0]) / 65536.0
        )

    # ViZDoom seed 2, object 196 (ShotgunGuy), episode times 117..127.
    assert z_trace == [-1.0, -3.0, -6.0, -10.0, -15.0, -21.0, -28.0, -36.0, -45.0, -55.0, -64.0]
    assert velocity_trace == [-2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, 0.0]


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_awakened_zombieman_matches_reference_discrete_chase_steps(
    pinned_deathmatch_scenario,
) -> None:
    scenario = pinned_deathmatch_scenario
    engine = TorchDeathmatchEngine(
        scenario,
        1,
        device=torch.device("cpu"),
        frame_skip=1,
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.view_height.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(136)
    engine.weapon_raise_cooldown.zero_()
    engine.selected_weapon.fill_(1)
    engine.selected_weapon_variant.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.teleport_fog_tics.zero_()
    engine.next_spawn_check.fill_(100_000)
    engine.enemy_x[0, 0] = 731.1809539794922
    engine.enemy_y[0, 0] = 953.5846099853516
    engine._enemy_x_fixed[0, 0] = round(float(engine.enemy_x[0, 0]) * 65536)
    engine._enemy_y_fixed[0, 0] = round(float(engine.enemy_y[0, 0]) * 65536)
    engine.enemy_angle[0, 0] = math.radians(181.40625004223693)
    engine.enemy_type[0, 0] = 0
    engine.enemy_health[0, 0] = 20
    engine.enemy_alive[0, 0] = True
    engine.enemy_target_slot[0, 0] = -2
    engine.enemy_move_cooldown[0, 0] = 8
    engine.enemy_cooldown[0, 0] = 0
    engine.enemy_reaction_time[0, 0] = 8

    attack = torch.zeros((1, 20), dtype=torch.bool)
    attack[:, 0] = True
    noop = torch.zeros_like(attack)
    samples: dict[int, tuple[float, float, float]] = {}
    for tick in range(41):
        if int(engine.episode_time[0]) in (145, 149, 153, 157, 176):
            samples[int(engine.episode_time[0])] = (
                float(engine.enemy_x[0, 0]),
                float(engine.enemy_y[0, 0]),
                float(torch.rad2deg(engine.enemy_angle[0, 0])),
            )
        if tick < 40:
            engine.step(attack if tick == 0 else noop)

    expected = {
        145: (736.8378143310547, 947.9277496337891, -135.0),
        149: (742.4946746826172, 942.2708892822266, -90.0),
        153: (748.1515350341797, 936.6140289306641, -45.0),
        157: (753.8083953857422, 930.9571685791016, -45.0),
        176: (776.4358367919922, 908.3297271728516, -45.0),
    }
    for episode_time, reference in expected.items():
        assert samples[episode_time] == pytest.approx(reference, abs=5e-5)


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_ground_monster_refuses_center_pit_dropoff(pinned_deathmatch_scenario) -> None:
    scenario = pinned_deathmatch_scenario
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(1000)
    engine.y.fill_(1000)
    engine.enemy_alive.zero_()
    engine.enemy_x[0, 0] = 617
    engine.enemy_y[0, 0] = 512
    engine.enemy_z[0, 0] = 0
    engine._enemy_x_fixed[0, 0] = 617 * 65536
    engine._enemy_y_fixed[0, 0] = 512 * 65536
    engine.enemy_type[0, 0] = 0
    engine.enemy_health[0, 0] = 20
    engine.enemy_alive[0, 0] = True
    requested = torch.zeros_like(engine.enemy_alive)
    requested[0, 0] = True
    west = torch.full_like(engine.enemy_type, 4)

    moved = engine._try_enemy_chase_step(
        requested,
        west,
        engine.enemy_type.clamp_min(0),
    )

    assert not moved[0, 0]
    assert engine.enemy_x[0, 0].item() == 617


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_renderer_preserves_rgb_hud_and_enemy_animation(
    pinned_deathmatch_scenario,
) -> None:
    scenario = pinned_deathmatch_scenario
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))

    initial = engine.render_native_frame(include_hud=True)

    assert initial.shape == (1, 240, 320, 3)
    assert initial.dtype == torch.uint8
    assert torch.any(initial[:, 208:] != 0)
    assert not torch.equal(initial[..., 0], initial[..., 1])

    blank_view = torch.zeros((1, 208, 320), dtype=torch.uint8)
    engine.episode_time.fill_(7)
    engine.weapon_raise_cooldown.fill_(8)
    first_visible_weapon = engine._native_render_weapon(blank_view)
    assert not torch.any(first_visible_weapon[:, :207])
    assert torch.any(first_visible_weapon[:, 207])
    engine.episode_time.fill_(15)
    engine.weapon_raise_cooldown.zero_()
    settled_weapon = engine._native_render_weapon(blank_view)
    settled_rows = torch.where(settled_weapon[0] != 0)[0]
    assert settled_rows.min().item() == 150

    pistol_frames = engine.map.native_weapon_frame_ids[2, 1]
    pistol_flashes = engine.map.native_weapon_flash_ids[2, 1]
    assert torch.unique(pistol_frames[15:19]).numel() == 1
    assert torch.unique(pistol_frames[9:15]).numel() == 1
    assert torch.unique(pistol_frames[5:9]).numel() == 1
    assert torch.unique(pistol_frames[1:5]).numel() == 1
    assert pistol_frames[1].item() == pistol_frames[9].item()
    assert torch.unique(pistol_frames[15:19]).item() == pistol_frames[0].item()
    assert len(torch.unique(pistol_frames[1:19])) == 3
    assert torch.all(pistol_flashes[8:15] >= 0)
    assert torch.all(pistol_flashes[:8] < 0)
    assert torch.all(pistol_flashes[15:] < 0)
    assert engine.map.native_weapon_flash_lights[2, 1, 10].item() == 1
    assert engine.map.native_weapon_flash_lights[2, 1, 14].item() == 1
    assert engine.map.native_weapon_flash_lights[2, 1, 18].item() == 0

    engine.selected_weapon.fill_(1)
    engine.selected_weapon_variant.fill_(True)
    engine.weapon_state_cooldown.zero_()
    engine.weapon_ready_tics.fill_(1)
    first_idle_chainsaw, _, _ = engine._native_weapon_frame_selection()
    engine.weapon_ready_tics.fill_(4)
    assert torch.equal(engine._native_weapon_frame_selection()[0], first_idle_chainsaw)
    engine.weapon_ready_tics.fill_(5)
    second_idle_chainsaw, _, _ = engine._native_weapon_frame_selection()
    assert not torch.equal(first_idle_chainsaw, second_idle_chainsaw)
    engine.weapon_ready_tics.fill_(8)
    assert torch.equal(engine._native_weapon_frame_selection()[0], second_idle_chainsaw)
    engine.selected_weapon.fill_(2)
    engine.selected_weapon_variant.zero_()

    engine.episode_time.fill_(21)
    engine.weapon_fire_count.fill_(1)
    engine.attack_cooldown.fill_(10)
    engine.weapon_state_cooldown.fill_(14)
    firing = engine.render_native_frame(include_hud=False)
    engine.attack_cooldown.zero_()
    engine.weapon_state_cooldown.zero_()
    ready = engine.render_native_frame(include_hud=False)
    assert not torch.equal(firing, ready)

    engine.mugshot_face_index.fill_(1)
    hud = engine._native_render_hud()[0]
    face_index = 14
    face_width = int(engine.map.hud_patch_widths[face_index].item())
    face_height = int(engine.map.hud_patch_heights[face_index].item())
    face = engine.map.hud_patch_atlas[face_index, :face_height, :face_width]
    opaque = engine.map.hud_patch_opaque[face_index, :face_height, :face_width]
    assert torch.equal(hud[2 : 2 + face_height, 148 : 148 + face_width][opaque], face[opaque])

    number_canvas = torch.zeros((32, 320), dtype=torch.uint8)
    engine._native_draw_hud_number(number_canvas, 50, 44, 3)
    number_y, number_x = torch.where(number_canvas != 0)
    assert (number_x.min().item(), number_x.max().item()) == (16, 43)
    assert (number_y.min().item(), number_y.max().item()) == (3, 18)

    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    noop = torch.zeros((1, 20), dtype=torch.bool)
    assert engine.mugshot_face_index.item() == 0
    for _ in range(8):
        engine.step(noop)
    assert engine.episode_time.item() == 17
    assert engine.mugshot_face_index.item() == 0
    engine.step(noop)
    assert engine.episode_time.item() == 19
    assert engine.mugshot_face_index.item() == 1

    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.weapon_raise_cooldown.zero_()
    attack = torch.zeros((1, 20), dtype=torch.bool)
    attack[:, 0] = True
    for _ in range(4):
        engine.step(attack)
        if engine.ammo[0, 1].item() < 50:
            break
    assert engine.ammo[0, 1].item() == 49
    assert engine.hud_ready_ammo.item() == 50
    fired_hud = engine._native_render_hud()[0]
    expected_ammo = torch.zeros((32, 320), dtype=torch.uint8)
    engine._native_draw_hud_number(expected_ammo, 50, 44, 3)
    expected_pixels = expected_ammo != 0
    assert torch.equal(fired_hud[expected_pixels], expected_ammo[expected_pixels])
    engine.step(torch.zeros_like(attack))
    assert engine.hud_ready_ammo.item() == 49

    engine.x.zero_()
    engine.y.zero_()
    engine.angle.zero_()
    engine.health.fill_(100)
    engine._apply_player_damage(
        torch.tensor([25.0]),
        torch.tensor([0.0]),
        torch.tensor([64.0]),
    )
    assert engine.mugshot_pain_direction.tolist() == [2]
    assert engine.mugshot_pain_tics.tolist() == [35]
    assert engine.mugshot_ouch.tolist() == [True]
    assert engine._native_mugshot_patch_index(0, 75) == 60
    engine.mugshot_grin.fill_(True)
    engine.mugshot_grin_tics.fill_(6)
    engine.bonus_count.fill_(6)
    assert engine._native_mugshot_patch_index(0, 75) == 65
    engine.health.zero_()
    assert engine._native_mugshot_patch_index(0, 0) == 69
    engine.mugshot_grin.zero_()
    engine.mugshot_grin_tics.zero_()
    engine.mugshot_pain_tics.zero_()
    engine.mugshot_ouch.zero_()
    engine.health.fill_(100)
    engine.attack_held_tics.fill_(70)
    assert engine._native_mugshot_patch_index(0, 100) == 49

    engine.x.fill_(668.9710083007812)
    engine.y.fill_(393.1371307373047)
    engine.z.zero_()
    engine.angle.fill_(math.radians(145.95336917460742))
    engine.episode_time.fill_(56)
    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(), engine.z + 41.0
    )
    pit_frame, _scene_depth = engine._native_render_portal_walls(
        flat_frame.clone(), engine.z + 41.0, surface_depth, scene_surface_depth
    )
    assert torch.isinf(surface_depth[0, 131, 160])
    assert pit_frame[0, 131, 160] != flat_frame[0, 131, 160]

    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.enemy_type[:, 0] = 0
    engine.enemy_alive[:, 0] = True
    engine.enemy_x[:, 0] = engine.x + 64
    engine.enemy_y[:, 0] = engine.y
    engine.enemy_angle[:, 0] = 0
    engine.enemy_animation_tics[:, 0] = 0
    first_walk_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.enemy_animation_tics[:, 0] = 8
    second_walk_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.enemy_target_slot[:, 0] = -2
    engine.enemy_animation_tics[:, 0] = 0
    first_idle_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.enemy_animation_tics[:, 0] = 10
    second_idle_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.enemy_target_slot[:, 0] = -1
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 10
    attack_frame = engine._native_enemy_sprite_ids()[:, 0].clone()

    engine.enemy_attack_phase[:, 0] = 2
    engine.enemy_cooldown[:, 0] = 16
    muzzle_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.enemy_cooldown[:, 0] = 8
    recovery_frame = engine._native_enemy_sprite_ids()[:, 0].clone()

    assert not torch.equal(first_walk_frame, second_walk_frame)
    assert not torch.equal(first_idle_frame, second_idle_frame)
    assert not torch.equal(second_walk_frame, attack_frame)
    assert not torch.equal(muzzle_frame, recovery_frame)

    engine.enemy_health[:, 0] = 1
    engine._apply_enemy_damage(torch.ones_like(engine.enemy_health))

    assert not engine.enemy_alive[:, 0].any()
    assert engine.enemy_death_type[:, 0].tolist() == [0]
    assert engine.enemy_death_tics[:, 0].tolist() == [21]
    death_start = engine.render_native_frame(include_hud=False)
    for _ in range(8):
        engine._collect_drops()
    death_progressed = engine.render_native_frame(include_hud=False)
    assert not torch.equal(death_start, death_progressed)
    for _ in range(64):
        engine._collect_drops()
    assert engine.enemy_death_tics[:, 0].tolist() == [1]
    assert engine.enemy_death_type[:, 0].tolist() == [0]
    assert not torch.equal(death_start, engine.render_native_frame(include_hud=False))


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_enemy_overkill_uses_reference_extreme_death_states(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        4,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(4, dtype=torch.bool), torch.arange(4))
    engine.enemy_alive.zero_()
    engine.enemy_type[:, 0] = torch.tensor([0, 0, 2, 4])
    engine.enemy_health[:, 0] = torch.tensor([20.0, 20.0, 100.0, 150.0])
    engine.enemy_alive[:, 0] = True
    damage = torch.zeros_like(engine.enemy_health)
    damage[:, 0] = torch.tensor([40.0, 41.0, 201.0, 301.0])

    engine._apply_enemy_damage(damage)

    # P_DamageMobj selects Death.Extreme only when health is strictly below
    # -SpawnHealth and the actor defines that state. Equality is a normal
    # death, and the Demon falls back to Death despite crossing the threshold.
    assert engine.enemy_death_extreme[:, 0].tolist() == [False, True, True, False]
    assert engine.enemy_death_tics[:, 0].tolist() == [21, 41, 41, 29]
    death_sprites = engine._native_enemy_death_sprite_ids()[:, 0]
    assert death_sprites.tolist() == [
        engine.map.enemy_death_sprite_ids[0, 0].item(),
        engine.map.enemy_xdeath_sprite_ids[0, 0].item(),
        engine.map.enemy_xdeath_sprite_ids[2, 0].item(),
        engine.map.enemy_death_sprite_ids[4, 0].item(),
    ]

    engine.enemy_death_elapsed[:, 0] = 10
    assert engine._enemy_solid_mask()[:, 0].tolist() == [False, False, False, True]


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_renderer_uses_independent_drop_coordinates(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.z.zero_()
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.item_available.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.teleport_fog_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.drop_type[:, 0] = 2007
    engine.drop_spawned[:, 0] = True
    engine.drop_x[:, 0] = engine.x + torch.cos(engine.angle) * 64.0
    engine.drop_y[:, 0] = engine.y + torch.sin(engine.angle) * 64.0
    engine.drop_z[:, 0] = 0
    # The owning corpse is deliberately behind the camera. Rendering at the
    # corpse coordinates would therefore make this drop disappear.
    engine.enemy_x[:, 0] = engine.x - torch.cos(engine.angle) * 64.0
    engine.enemy_y[:, 0] = engine.y - torch.sin(engine.angle) * 64.0

    with_drop = engine.render_native_frame(include_hud=False)
    engine.drop_spawned[:, 0] = False
    without_drop = engine.render_native_frame(include_hud=False)

    assert not torch.equal(with_drop, without_drop)


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_pit_depth_occludes_map_items(pinned_deathmatch_scenario) -> None:
    scenario = pinned_deathmatch_scenario
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([2024]))
    engine.x.fill_(471.74908447265625)
    engine.y.fill_(526.3986206054688)
    engine.angle.fill_(math.radians(145.95336917460742))
    engine.episode_time.fill_(73)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.player_dead.fill_(True)

    for player_z in (-48.0, -56.0, -64.0):
        engine.z.fill_(player_z)
        view_z = engine.z + 41.0
        wall_distance = engine._native_raycast()
        flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
            engine._current_sector(), view_z
        )
        portal_frame, scene_depth = engine._native_render_portal_walls(
            flat_frame, view_z, surface_depth, scene_surface_depth
        )
        without_scene_depth = engine._native_render_sprites(
            portal_frame.clone(),
            wall_distance,
            view_z,
            torch.full_like(scene_depth, torch.inf),
        )
        with_scene_depth = engine._native_render_sprites(
            portal_frame.clone(), wall_distance, view_z, scene_depth
        )

        assert torch.any(without_scene_depth != portal_frame)
        assert torch.equal(with_scene_depth, portal_frame)


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_item_uses_fixed_point_sprite_projection(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.x.fill_(940.9204254150391)
    engine.y.fill_(826.7186584472656)
    engine.z.zero_()
    engine.view_z.fill_(41.0)
    engine.angle.fill_(math.radians(341.29028328258784))
    engine.episode_time.fill_(17)
    engine.weapon_raise_cooldown.zero_()
    engine.player_dead.fill_(True)

    item_index = torch.where(
        (engine.map.item_spawns[:, 0] == 1248)
        & (engine.map.item_spawns[:, 1] == 832)
    )[0].item()
    sprite = engine.map.item_raw_visual_types[item_index].reshape(1, 1)
    sprite_left, sprite_right, texture_step = (
        engine._native_sprite_horizontal_projection(
            engine.map.item_spawns[item_index : item_index + 1, 0].reshape(1, 1),
            engine.map.item_spawns[item_index : item_index + 1, 1].reshape(1, 1),
            sprite,
        )
    )
    assert sprite_left.item() == 85
    assert sprite_right.item() == 119
    assert texture_step.item() == 119506

    engine.item_available.zero_()
    engine.item_available[0, item_index] = True
    with_item = engine.render_native_frame(include_hud=False)
    engine.item_available.zero_()
    without_item = engine.render_native_frame(include_hud=False)
    changed_y, changed_x = torch.where(torch.any(with_item[0] != without_item[0], dim=-1))
    assert (changed_x.min().item(), changed_x.max().item()) == (85, 118)
    assert (changed_y.min().item(), changed_y.max().item()) == (119, 128)


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_renderer_includes_voodoo_dolls(pinned_deathmatch_scenario) -> None:
    scenario = pinned_deathmatch_scenario
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.z.zero_()
    engine.view_z.fill_(41.0)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.item_available.zero_()

    view_z = engine.view_z
    wall_distance = engine._native_raycast()
    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(), view_z
    )
    portal_frame, scene_depth = engine._native_render_portal_walls(
        flat_frame, view_z, surface_depth, scene_surface_depth
    )
    with_dolls = engine._native_render_sprites(
        portal_frame.clone(), wall_distance, view_z, scene_depth
    )

    assert torch.any(with_dolls != portal_frame)
