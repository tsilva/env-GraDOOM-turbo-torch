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


def test_native_hitscan_puff_renders_all_translucent_animation_frames(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.enemy_alive.zero_()
    engine.item_available.zero_()
    baseline = engine.render_native_frame(include_hud=False)
    direction_x, direction_y = engine._fine_direction(engine.angle)
    engine.hitscan_puff_x[:, 0] = engine.x + direction_x * 64
    engine.hitscan_puff_y[:, 0] = engine.y + direction_y * 64
    engine.hitscan_puff_z[:, 0] = engine.z + 36

    frames = []
    for remaining_tics in (13, 12, 8, 4):
        engine.hitscan_puff_tics[:, 0] = remaining_tics
        frames.append(engine.render_native_frame(include_hud=False))

    assert all(not torch.equal(frame, baseline) for frame in frames)
    assert all(
        not torch.equal(frames[index], frames[index + 1])
        for index in range(len(frames) - 1)
    )
    engine.hitscan_puff_tics.zero_()
    assert torch.equal(engine.render_native_frame(include_hud=False), baseline)


def test_native_hitscan_decal_persists_on_its_visible_wall_side(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
        frame_skip=1,
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.enemy_alive.zero_()
    engine.item_available.zero_()
    engine.weapon_raise_cooldown.zero_()
    engine.angle.zero_()
    engine._angle_bam.zero_()
    engine.pitch.zero_()
    engine._pitch_bam.zero_()

    engine._execute_player_attack(
        torch.full((1,), 2, dtype=torch.int64),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
    )
    engine.hitscan_puff_tics.zero_()
    with_decal = engine.render_native_frame(include_hud=False)
    stored_serial = engine.hitscan_decal_serial.clone()
    engine.hitscan_decal_serial.fill_(-1)
    without_decal = engine.render_native_frame(include_hud=False)

    changed = torch.any(with_decal != without_decal, dim=3)
    assert torch.count_nonzero(changed) > 0
    assert engine.hitscan_decal_count.item() == 1

    engine.hitscan_decal_serial.copy_(stored_serial)
    style = engine.hitscan_decal_style[:, 0].to(torch.int16)
    engine.hitscan_decal_style[:, 0] = torch.where(
        style < 20,
        style + 20,
        style - 20,
    ).to(torch.uint8)
    hidden_reverse_side = engine.render_native_frame(include_hud=False)
    assert torch.equal(hidden_reverse_side, without_decal)

    # Impact decals outlive the transient puff actor and are cleared only by
    # the same episode reset that clears the rest of the world state.
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    assert engine.hitscan_decal_count.item() == 0
    assert torch.all(engine.hitscan_decal_serial < 0)


def test_native_hitscan_decal_matches_aligned_vizdoom_wall_pixels(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.enemy_alive.zero_()
    engine.item_available.zero_()
    engine.x.fill_(1026.3348693847656)
    engine.y.fill_(797.8362731933594)
    engine.z.zero_()
    engine.view_z.fill_(46.904083251953125)
    angle_degrees = 341.29028328258784
    engine.angle.fill_(math.radians(angle_degrees))
    engine._angle_bam.fill_(round(angle_degrees / 360.0 * (1 << 32)))
    engine.pitch.zero_()
    engine._pitch_bam.zero_()
    engine.hitscan_decal_wall[0, 0] = 41
    engine.hitscan_decal_along[0, 0] = (712.1397325339216 - 992.0) / (32.0 - 992.0)
    engine.hitscan_decal_z[0, 0] = 36
    # CHIP5, both flips, front sidedef: one of the reference renderer's
    # permitted visual-only random styles for this measured impact.
    engine.hitscan_decal_style[0, 0] = 19
    engine.hitscan_decal_serial[0, 0] = 0
    engine.hitscan_decal_count[0] = 1

    with_decal = engine.render_native_frame(include_hud=False)[0]
    engine.hitscan_decal_serial[0, 0] = -1
    without_decal = engine.render_native_frame(include_hud=False)[0]
    changed = torch.any(with_decal != without_decal, dim=2)
    expected_coordinates = torch.tensor(
        ((111, 160), (112, 158), (112, 159), (112, 160), (113, 160)),
        dtype=torch.int64,
    )

    assert torch.equal(torch.nonzero(changed), expected_coordinates)
    expected_background = torch.tensor(
        ((55, 35, 19), (63, 47, 23), (63, 43, 27), (55, 35, 19), (55, 35, 19)),
        dtype=torch.uint8,
    )
    expected_shaded = torch.tensor(
        ((23, 15, 7), (47, 27, 11), (47, 27, 11), (7, 7, 7), (31, 23, 11)),
        dtype=torch.uint8,
    )
    assert torch.equal(
        without_decal[expected_coordinates[:, 0], expected_coordinates[:, 1]],
        expected_background,
    )
    assert torch.equal(
        with_decal[expected_coordinates[:, 0], expected_coordinates[:, 1]],
        expected_shaded,
    )


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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
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


def test_native_ceiling_visplane_repairs_unresolved_columns(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.x.fill_(337.0191345214844)
    engine.y.fill_(1007.9921722412109)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(341.29028328258784))
    engine.episode_time.fill_(101)

    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # These independent ceiling rays miss every map polygon, but the reference
    # renderer keeps sector 0's lower ceiling visplane continuous above its
    # first resolved row. Falling back to the player's sector instead shifts
    # CEIL4_1 because that ceiling is 32 map units higher.
    assert torch.isinf(surface_depth[0, 11, 158])
    assert torch.isfinite(scene_surface_depth[0, 11, 158])
    assert rgb[0, 11, 158].tolist() == [0, 0, 35]
    assert rgb[0, 20, 146].tolist() == [0, 0, 23]
    assert rgb[0, 27, 129].tolist() == [0, 0, 0]


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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
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


def test_native_bsp_fragments_interpolate_wallscan_light_independently(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(718.8914337158203)
    engine.y.fill_(16.022705078125)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(41)

    *_intersections, parent_visibility = engine._native_portal_intersections()
    fragment_geometry = engine._native_projection_geometry_for_fixed_walls(
        engine.map.portal_projection_fragments_fixed
    )
    fragment_x, fragment_y = engine._native_wall_view_coordinates(
        engine.map.portal_projection_fragments_fixed
    )
    fragment_visibility = engine._native_wall_visibility_from_view_coordinates(
        fragment_x,
        fragment_y,
    )
    frame = engine.render_native_frame(include_hud=False)

    # The node builder splits wall 60 from [576,0]→[544,0] at x=558. Its
    # visible second fragment has an independent FWallCoords span [101,103)
    # and rw_light=552115 at column 101. Interpolating from the parent linedef
    # yields 565287 and selects the next brighter COMPBLUE colormap.
    assert fragment_geometry[0][0, 60, 1].item() == 101
    assert fragment_geometry[1][0, 60, 1].item() == 103
    assert parent_visibility[0, 101, 60, 0].item() == 565287
    assert fragment_visibility[0, 101, 60, 1, 0].item() == 552115
    assert frame[0, 1, 101].tolist() == [0, 0, 83]


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
    frame, scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
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
    # Wall ordering retains the unresolved polygon-ray depth, while the scene
    # buffer records the repaired visplane depth for downstream diagnostics.
    assert torch.isinf(surface_depth[0, 124, 290])
    assert scene_surface_depth[0, 124, 290].item() == pytest.approx(458.92681884765625)
    assert scene_depth[0, 124, 290].item() == pytest.approx(458.92681884765625)


def test_native_two_sided_wall_top_edges_own_flat_boundary(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(801.5532989501953)
    engine.y.fill_(37.320770263671875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(21)

    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            flat_frame.clone(),
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    flat_rgb = engine.map.playpal[flat_frame.to(torch.int64)]
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # R_RenderSegLoop passes wallscan each tier's inclusive integer top row.
    # The independent flat prepass intersects a ceiling slightly nearer here,
    # but it must not punch holes in same-sector BIGBRIK1 or the REDWALL upper
    # tier along their projected top boundaries.
    assert surface_depth[0, 42, 149] < 813.0478515625
    assert flat_rgb[0, 42, 149].tolist() == [0, 0, 23]
    assert rgb[0, 42, 149].tolist() == [43, 35, 15]
    assert surface_depth[0, 47, 202] < 886.530029296875
    assert flat_rgb[0, 47, 202].tolist() == [0, 0, 0]
    assert rgb[0, 47, 202].tolist() == [115, 19, 19]


def test_native_wall_planes_clip_before_fixed_projection(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(801.5532989501953)
    engine.y.fill_(37.320770263671875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(21)

    frame = engine.render_native_frame(include_hud=False)[0]

    # OWallMost clips a horizontal plane against globaluclip/globaldclip and
    # recomputes the crossing depth and screen column before interpolating.
    # Clamping an uncut projection instead exposes BIGBRIK1 above the ceiling
    # at x=104..106 and shifts wall 17's diagonal floor edge by one row.
    assert frame[0, 104].tolist() == [0, 0, 35]
    assert frame[5, 104].tolist() == [63, 43, 27]
    assert frame[9, 105].tolist() == [0, 0, 0]
    assert frame[10, 105].tolist() == [79, 59, 43]
    assert frame[14, 106].tolist() == [0, 0, 35]
    assert frame[15, 106].tolist() == [87, 67, 51]
    assert frame[147, 84].tolist() == [91, 71, 43]

    # A one-column right-side crossing clears xcross, then OWallMost's
    # ix2 == ix1 branch writes the surviving endpoint projection over it.
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(1007.9770812988281)
    engine.y.fill_(453.31781005859375)
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(61)
    frame = engine.render_native_frame(include_hud=False)[0]
    assert frame[0, 200].tolist() == [0, 0, 35]
    assert frame[23, 200].tolist() == [0, 0, 23]
    assert frame[24, 200].tolist() == [119, 95, 75]

    # Generated BSP fragments preserve the reference projection at the right
    # frustum and across portal boundaries.
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(370.07318115234375)
    engine.y.fill_(147.17359924316406)
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(41)
    frame = engine.render_native_frame(include_hud=False)[0]
    assert frame[0, 291].tolist() == [79, 59, 35]
    assert frame[5, 316].tolist() == [79, 59, 35]

    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(883.0337219238281)
    engine.y.fill_(401.4684143066406)
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(21)
    frame = engine.render_native_frame(include_hud=False)[0]
    assert frame[2, 177].tolist() == [0, 0, 35]

    # A terminal wall behind two portals uses its generated BSP fragment. The
    # fragment projection keeps the diagonal sky/wall boundary continuous.
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(340.60516357421875)
    engine.y.fill_(284.878662109375)
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(81)
    frame = engine.render_native_frame(include_hud=False)[0]
    assert frame[0, 78].tolist() == [147, 123, 99]

    # ViZDoom splits linedef 54 at y=667.428558 before projection. Reusing the
    # full linedef moves this distant wall above the sky boundary by one row.
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(496.656982421875)
    engine.y.fill_(318.42120361328125)
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(61)
    frame = engine.render_native_frame(include_hud=False)[0]
    assert frame[0, 25].tolist() == [0, 0, 0]
    assert frame[1, 27].tolist() == [0, 0, 0]
    assert frame[3, 30].tolist() == [0, 0, 35]
    assert frame[6, 35].tolist() == [0, 0, 0]

    # Linedef 41's first generated fragment lies behind the near plane from
    # this view. FWallCoords rejects it, leaving the second fragment to own
    # x=0..56; accepting its unbounded projection paints over the sky opening.
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(288.4954834656081))
    engine.episode_time.fill_(61)
    fragment_geometry = engine._native_projection_geometry_for_fixed_walls(
        engine.map.portal_projection_fragments_fixed
    )
    fragment_left, fragment_right = fragment_geometry[:2]
    assert fragment_left[0, 41, 0].item() == 0
    assert fragment_right[0, 41, 0].item() == 0
    assert fragment_left[0, 41, 1].item() == 0
    assert fragment_right[0, 41, 1].item() == 57
    frame = engine.render_native_frame(include_hud=False)[0]
    assert frame[0, 44].tolist() == [0, 0, 0]
    assert frame[1, 53].tolist() == [0, 0, 35]
    assert frame[2, 54].tolist() == [0, 0, 0]


def test_native_sector_lookup_ignores_self_referencing_linedef(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(-143.1327362060547)
    engine.y.fill_(246.9293975830078)
    engine.z.zero_()
    engine.view_z.fill_(41)
    angle_degrees = 165.67382816357394
    engine.angle.fill_(math.radians(angle_degrees))
    engine._angle_bam.fill_(round(angle_degrees / 360.0 * (1 << 32)))
    engine.episode_time.fill_(81)

    # Linedef 52 has sector 11 on both sides. It carries a visible masked
    # texture elsewhere, but treating it as polygon boundary makes this valid
    # outer-room position fall through to sector 0 and removes wall 48.
    assert engine._current_sector().item() == 11
    frame = engine.render_native_frame(include_hud=False)[0]
    assert frame[50, 100].tolist() == [51, 43, 19]
    assert frame[100, 100].tolist() == [47, 27, 11]


def test_native_portal_silhouette_occludes_drops_behind_pit_floor(
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
    (
        portal_frame,
        scene_depth,
        sprite_clip_depth,
        sprite_clip_wall,
    ) = engine._native_render_portal_walls(
        flat_frame, engine.view_z, surface_depth, scene_surface_depth
    )
    with_unrepaired_depth = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        surface_depth,
    )
    with_portal_clip = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        sprite_clip_depth,
        sprite_clip_wall,
    )

    assert torch.isinf(surface_depth[0, 124, 290])
    assert with_unrepaired_depth[0, 124, 290] != portal_frame[0, 124, 290]
    assert torch.isfinite(scene_depth[0, 124, 290])
    assert torch.isfinite(sprite_clip_depth[0, 124, 290])
    assert torch.equal(with_portal_clip, portal_frame)


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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # PrepWall rounds its inverse-depth scale before wallscan advances the
    # vertical column DDA. Continuous world-Z sampling selects [123, 99, 79].
    assert rgb[0, 1, 40].tolist() == [131, 107, 87]


def test_native_walls_use_reference_fixed_horizontal_sampling(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(660.5986785888672)
    engine.y.fill_(16.022705078125)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(81)

    horizontal_offset_fixed, _vertical_step = engine._native_wall_texture_mapping()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # The continuous column ray reaches wall 17 just below map offset 160.
    # PrepWall instead evaluates its float t-map coefficients in double and
    # rounds the result in 16.16 space before wallscan selects the texel.
    assert (horizontal_offset_fixed[0, 63, 17] >> 16).item() == 160
    assert rgb[0, 5, 63].tolist() == [123, 99, 79]


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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
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


def test_native_projected_endpoint_owner_selects_nested_pit_texture_columns(
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
    engine.view_z.fill_(-23)
    engine.angle.fill_(math.radians(162.48779300658214))
    engine.episode_time.fill_(61)

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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # At these nested-pit corners, the mathematical ray hits one seg just
    # outside its half-open screen span. The adjacent seg's projected left
    # edge owns the column even when the two infinite-line depths differ by
    # more than 1/16 map unit. It therefore supplies the REDWALL texture U.
    for column, ray_wall, projected_owner in (
        (57, 106, 77),
        (104, 77, 108),
        (146, 108, 69),
        (60, 105, 78),
        (104, 78, 104),
        (147, 104, 65),
        (105, 88, 125),
    ):
        assert geometric_intersections[0, column, ray_wall]
        assert not projected_intersections[0, column, ray_wall]
        assert not geometric_intersections[0, column, projected_owner]
        assert projected_intersections[0, column, projected_owner]
        assert projected_left_edges[0, column, projected_owner]
        assert (
            wall_distance[0, column, projected_owner]
            - wall_distance[0, column, ray_wall]
            > 1.0 / 16.0
        )
    for x, y, expected in (
        (57, 110, [79, 0, 0]),
        (104, 90, [115, 0, 0]),
        (104, 110, [91, 0, 0]),
        (146, 120, [79, 0, 0]),
        (59, 160, [79, 0, 0]),
        (103, 155, [115, 0, 0]),
        (143, 155, [79, 0, 0]),
        (60, 90, [91, 0, 0]),
        (147, 95, [107, 15, 15]),
        (105, 80, [115, 0, 0]),
    ):
        assert rgb[0, y, x].tolist() == expected


def test_native_projected_endpoint_owner_preserves_geometric_portal_path(
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
    engine.episode_time.fill_(1)

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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Portal 101 remains the geometric sector path while projected-only seg
    # 191 owns its raster column. Treating the owner as the path terminates
    # traversal and leaves a full-height sky/flat hole at column 21.
    assert geometric_intersections[0, 21, 101]
    assert not projected_intersections[0, 21, 101]
    assert not geometric_intersections[0, 21, 191]
    assert projected_intersections[0, 21, 191]
    assert projected_left_edges[0, 21, 191]
    assert wall_distance[0, 21, 191] - wall_distance[0, 21, 101] > 1.0 / 16.0
    assert rgb[0, 60, 21].tolist() == [79, 59, 43]
    assert rgb[0, 80, 21].tolist() == [103, 83, 63]
    assert rgb[0, 100, 21].tolist() == [119, 95, 75]
    assert rgb[0, 140, 21].tolist() == [71, 0, 0]


def test_native_projected_owner_resolves_short_opposite_endpoint_hit(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([789]))
    engine.x.fill_(262.9338836669922)
    engine.y.fill_(576.4520568847656)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(348.81591804996503))
    engine.episode_time.fill_(61)

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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    direct_frame = frame
    assert engine._native_direct_endpoint_neighbors is not None
    engine._native_direct_endpoint_neighbors = None
    generic_frame, generic_surface_depth, generic_scene_surface_depth = (
        engine._native_render_flats(
            engine._current_sector(),
            engine.view_z,
        )
    )
    generic_frame, *_generic_depths = engine._native_render_portal_walls(
        generic_frame,
        engine.view_z,
        generic_surface_depth,
        generic_scene_surface_depth,
    )
    assert torch.equal(direct_frame, generic_frame)
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # The column ray geometrically crosses tiny portal 68 near its end, but
    # FWallCoords excludes column 124 from that seg's half-open span. Portal
    # 191 owns the column through its other shared endpoint and must provide
    # the BFALL1 wallscan while portal 68 retains the subsector path.
    assert geometric_intersections[0, 124, 68]
    assert not projected_intersections[0, 124, 68]
    assert not geometric_intersections[0, 124, 191]
    assert projected_intersections[0, 124, 191]
    assert projected_left_edges[0, 124, 191]
    owner_depth_delta = wall_distance[0, 124, 191] - wall_distance[0, 124, 68]
    assert 0 < owner_depth_delta <= max(wall_distance[0, 124, 68] / 128, 4)
    assert rgb[0, 134:143, 124].tolist() == [
        [91, 0, 0],
        [103, 0, 0],
        [115, 0, 0],
        [71, 0, 0],
        [79, 0, 0],
        [71, 0, 0],
        [71, 0, 0],
        [71, 0, 0],
        [67, 0, 0],
    ]


def test_native_pit_boundary_separates_wallscan_owner_from_portal_path(
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
    engine.angle.fill_(math.radians(62.314453139508714))
    engine.episode_time.fill_(81)
    engine.item_available.zero_()
    engine.enemy_alive.zero_()

    (
        wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections(
        engine._native_wall_projection_geometry()
    )
    horizontal_offset_fixed, _vertical_step = engine._native_wall_texture_mapping()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Column 7 crosses tiny portal 86 outside its half-open projected span.
    # Projected portal 131 owns wallscan, while portal 86 still determines the
    # sector path behind it; coupling those choices creates a vertical seam.
    assert geometric_intersections[0, 7, 86]
    assert not projected_intersections[0, 7, 86]
    assert not geometric_intersections[0, 7, 131]
    assert projected_intersections[0, 7, 131]
    assert projected_left_edges[0, 7, 131]
    assert rgb[0, 124, 7].tolist() == [67, 0, 0]
    assert rgb[0, 125, 7].tolist() == [79, 0, 0]

    # Column 12 crosses wall 98 on its excluded right edge, but no projected
    # left edge owns that boundary column. It remains a traversal event only;
    # emitting a drawseg leaks BFALL1 over the visible floor.
    assert geometric_intersections[0, 12, 98]
    assert not projected_intersections[0, 12, 98]
    assert not torch.any(
        projected_left_edges[0, 12] & torch.isfinite(wall_distance[0, 12])
    )
    assert rgb[0, 124, 12].tolist() == [83, 63, 47]

    # PrepWallRoundFix intentionally preserves leading texture-coordinate
    # spill when a clipped drawseg begins at screen x == 0. Wall 192 reverses
    # this value to -17440 before applying its sidedef offset; clamping it
    # selects the adjacent BFALL1 column and produces [79, 0, 0] instead.
    wall_repeat_fixed = torch.round(
        engine.map.portal_wall_lengths[192] * (1 << 16)
    ).to(torch.int64)
    assert horizontal_offset_fixed[0, 0, 192] - wall_repeat_fixed == 17440
    assert rgb[0, 124, 0].tolist() == [67, 0, 0]
    assert rgb[0, 125, 0].tolist() == [67, 0, 0]


def test_native_wide_projected_owner_preserves_shared_boundary_traversal(
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
    engine.view_z.fill_(-23)
    engine.angle.fill_(math.radians(175.1440430095289))
    engine.episode_time.fill_(61)
    engine.item_available.zero_()
    engine.enemy_alive.zero_()

    wall_projection_geometry = engine._native_wall_projection_geometry()
    (
        _wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections(wall_projection_geometry)
    rgb = engine.render_native_frame(include_hud=False)

    # The rays cross walls 175 and 180 exactly on their excluded screen-right
    # edges. Projected-left neighbors 89 and 139 own those raster columns,
    # while the geometric segs retain the subsector traversal paths behind them.
    for column, ray_wall, projected_owner in (
        (36, 175, 89),
        (44, 180, 139),
    ):
        assert geometric_intersections[0, column, ray_wall]
        assert not projected_intersections[0, column, ray_wall]
        assert not geometric_intersections[0, column, projected_owner]
        assert projected_intersections[0, column, projected_owner]
        assert projected_left_edges[0, column, projected_owner]

    # Wall 181 is the next ray hit at the same map vertex, and its [8, 36)
    # raster span excludes column 36. It must update the traversed sector
    # without hiding line 200's two-row BFALL1 tier or the BRICK12 above it.
    assert geometric_intersections[0, 36, 181]
    assert not projected_intersections[0, 36, 181]
    assert rgb[0, 70:78, 36].tolist() == [
        [111, 87, 67],
        [71, 51, 35],
        [123, 99, 79],
        [123, 99, 79],
        [91, 0, 0],
        [103, 0, 0],
        [79, 0, 0],
        [91, 0, 0],
    ]
    assert rgb[0, 71:78, 44].tolist() == [
        [91, 0, 0],
        [79, 0, 0],
        [79, 0, 0],
        [79, 0, 0],
        [79, 0, 0],
        [79, 0, 0],
        [79, 0, 0],
    ]


def test_native_wide_projected_owner_keeps_far_wall_sector_path(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(877.9294128417969)
    engine.y.fill_(540.6578674316406)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(81)
    engine.item_available.zero_()
    engine.enemy_alive.zero_()

    wall_projection_geometry = engine._native_wall_projection_geometry()
    (
        _wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections(wall_projection_geometry)
    rgb = engine.render_native_frame(include_hud=False)

    # Wall 89 owns column 72's drawseg, but the ray crosses excluded wall 175
    # and then wall 181 at their shared endpoint. Following wall 89 as the
    # sector path instead would leave sky at row 41 instead of the far BRICK12.
    assert geometric_intersections[0, 72, 175]
    assert not projected_intersections[0, 72, 175]
    assert projected_intersections[0, 72, 89]
    assert projected_left_edges[0, 72, 89]
    assert geometric_intersections[0, 72, 181]
    assert not projected_intersections[0, 72, 181]
    assert rgb[0, 41, 72].tolist() == [123, 99, 79]
    assert rgb[0, 122, 72].tolist() == [79, 0, 0]


def test_native_projected_portal_bridge_draws_intermediate_sector_tier(
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
    engine.angle.fill_(math.radians(34.51904297678709))
    engine.episode_time.fill_(21)
    engine.item_available.zero_()
    engine.enemy_alive.zero_()

    wall_projection_geometry = engine._native_wall_projection_geometry()
    (
        _wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections(wall_projection_geometry)
    rgb = engine.render_native_frame(include_hud=False)

    # The ray crosses excluded wall 192 from sector 10 to 11, while wall 86
    # owns column 29 from sector 10 to 9. Doom stores projected bridge 194
    # between them, completing the -8 -> -4 -> 0 BFALL1 boundary chain.
    assert geometric_intersections[0, 29, 192]
    assert not projected_intersections[0, 29, 192]
    for projected_wall in (86, 194):
        assert not geometric_intersections[0, 29, projected_wall]
        assert projected_intersections[0, 29, projected_wall]
        assert projected_left_edges[0, 29, projected_wall]
    assert engine._native_projected_portal_bridge_mask[0, 192, 86]
    assert engine._native_projected_portal_bridge_indices[0, 192, 86] == 194
    assert rgb[0, 24:29, 29].tolist() == [
        [79, 0, 0],
        [79, 0, 0],
        [79, 0, 0],
        [103, 0, 0],
        [103, 0, 0],
    ]


def test_native_walls_reject_collapsed_screen_edge_span(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(877.4517669677734)
    engine.y.fill_(198.21546936035156)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(41)

    wall_distance, *_rest = engine._native_portal_intersections()
    screen_left, screen_right, _depth_left, _depth_right = (
        engine._native_wall_projection_geometry()
    )
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # FWallCoords clips solid 54 to [0, 0), so R_AddLine rejects it before
    # BSP clipping. The ray still meets its mathematical segment at column 0,
    # but must continue to visible wall 48 rather than painting BRICK12 there.
    assert screen_left[0, 54] == screen_right[0, 54] == 0
    assert torch.isinf(wall_distance[0, 0, 54])
    assert torch.isfinite(wall_distance[0, 0, 48])
    assert rgb[0, 36, 0].tolist() == [0, 0, 0]
    assert rgb[0, 38, 0].tolist() == [0, 0, 23]
    assert rgb[0, 51, 0].tolist() == [43, 35, 15]
    assert rgb[0, 113, 0].tolist() == [71, 51, 35]


def test_native_partially_clipped_solid_owns_left_frustum_column(
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
    engine.angle.fill_(math.radians(119.97070315293286))
    engine.episode_time.fill_(21)

    wall_projection_geometry = engine._native_wall_projection_geometry()
    screen_left, screen_right, _depth_left, _depth_right = wall_projection_geometry
    (
        wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        _projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections(wall_projection_geometry)
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Solid 6 enters from beyond the left frustum and ends at column 1, so
    # FWallCoords clips it to [0, 1). The surviving right endpoint must not be
    # mistaken for a new left bound: wall 6 closes column 0 before farther
    # solid 48 can leak through sector 11.
    assert screen_left[0, 6] == 0
    assert screen_right[0, 6] == 1
    assert geometric_intersections[0, 0, 6]
    assert projected_intersections[0, 0, 6]
    assert torch.isfinite(wall_distance[0, 0, 6])
    assert sprite_clip_wall[0, 58, 0] == 6
    assert rgb[0, 58, 0].tolist() == [23, 15, 7]
    assert rgb[0, 86, 0].tolist() == [39, 39, 39]
    assert rgb[0, 116, 0].tolist() == [31, 23, 11]


def test_native_deferred_masked_middle_texture_owns_shared_endpoint_column(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([456]))
    engine.x.fill_(838.550537109375)
    engine.y.fill_(385.9864501953125)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(165.67382816357394))
    engine.episode_time.fill_(61)

    (
        wall_distance,
        _wall_along,
        _geometric_intersections,
        _projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Doom records two-sided BIGBRIK1 wall 52 as a masked drawseg and paints
    # it after the opaque pass. At its projected left endpoint it therefore
    # overlays nearer solid wall 51 even though the mathematical column ray
    # reaches wall 51 first. The masked texture remains half-open at row 70.
    assert projected_left_edges[0, 69, 52]
    assert wall_distance[0, 69, 52] > wall_distance[0, 69, 51]
    torch.testing.assert_close(
        scene_depth[0, 38, 69],
        wall_distance[0, 69, 52],
    )
    assert rgb[0, 38, 69].tolist() == [63, 43, 27]
    assert rgb[0, 69, 69].tolist() == [43, 35, 15]
    assert rgb[0, 70, 69].tolist() == [31, 23, 11]


def test_native_same_column_portal_layers_retain_nested_wall(
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
    engine.angle.fill_(math.radians(205.52673344629042))
    engine.episode_time.fill_(81)

    wall_distance, *_rest = engine._native_portal_intersections()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Portals 210 and 209 form successive BSP layers in column 172, but their
    # float ray depths differ by less than the old arbitrary 0.001 cutoff.
    # Both must be crossed to reach BRICK12 wall 196 instead of terminating in
    # sector 12's sky.
    portal_depth_delta = wall_distance[0, 172, 209] - wall_distance[0, 172, 210]
    assert 0 < portal_depth_delta < 1e-3
    assert rgb[0, 48, 172].tolist() == [111, 87, 67]
    assert rgb[0, 90, 172].tolist() == [103, 83, 63]
    assert rgb[0, 106, 172].tolist() == [111, 87, 67]


def test_native_projected_portal_keeps_geometric_sector_path(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([789]))
    engine.x.fill_(262.93389892578125)
    engine.y.fill_(576.4520263671875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(348.81591804996503))
    engine.episode_time.fill_(61)

    (
        _wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        _projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Portal 206 owns column 50 through its projected half-open span, but the
    # mathematical ray crosses portal 213 at their shared vertex. Rasterizing
    # 206 must therefore retain 213's sector path so the farther solid wall is
    # reached instead of leaking the sky through one complete screen column.
    assert projected_intersections[0, 50, 206]
    assert not geometric_intersections[0, 50, 206]
    assert not projected_intersections[0, 50, 213]
    assert geometric_intersections[0, 50, 213]
    assert rgb[0, 40, 50].tolist() == [123, 99, 79]
    assert rgb[0, 60, 50].tolist() == [159, 135, 111]
    assert rgb[0, 100, 50].tolist() == [119, 95, 75]


def test_native_projected_portal_enters_non_touching_tied_sector_strip(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([789]))
    engine.x.fill_(262.93389892578125)
    engine.y.fill_(576.4520263671875)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(348.81591804996503))
    engine.episode_time.fill_(61)

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
    frame, scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Projected-only portal 149 and geometric portal 151 bound opposite sides
    # of sector 6 without sharing a map vertex. Both sector paths therefore
    # name portal 151 as their equal-depth continuation. Wallscan enters sector
    # 6 through 149, exits through 151, and reaches solid BIGBRIK1 wall 41.
    assert engine._native_opposing_portal_pairs[149, 151]
    assert not geometric_intersections[0, 135, 149]
    assert projected_intersections[0, 135, 149]
    assert projected_left_edges[0, 135, 149]
    assert geometric_intersections[0, 135, 151]
    assert projected_intersections[0, 135, 151]
    assert geometric_intersections[0, 135, 41]
    assert projected_intersections[0, 135, 41]
    assert wall_distance[0, 135, 149] < wall_distance[0, 135, 151]
    assert wall_distance[0, 135, 151] < wall_distance[0, 135, 41]
    torch.testing.assert_close(scene_depth[0, 97, 135], wall_distance[0, 135, 41])
    assert rgb[0, 96, 135].tolist() == [0, 0, 0]
    assert rgb[0, 97, 135].tolist() == [31, 23, 11]
    assert rgb[0, 103, 135].tolist() == [55, 35, 19]
    assert rgb[0, 108, 135].tolist() == [43, 35, 15]


def test_native_projected_portal_tie_retains_same_direction_boundary_chain(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(788.8523559570312)
    engine.y.fill_(381.226318359375)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(21)

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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Projected-only portal 190 and its tied continuation 73 separate the same
    # sectors but point along the same boundary chain. They do not form the
    # opposing sides of a strip, so entering sector 4 through 190 would turn
    # the farther brick wall into a full-height sky leak at column 21.
    assert not engine._native_opposing_portal_pairs[190, 73]
    assert not geometric_intersections[0, 21, 190]
    assert projected_intersections[0, 21, 190]
    assert projected_left_edges[0, 21, 190]
    assert geometric_intersections[0, 21, 73]
    assert projected_intersections[0, 21, 73]
    assert wall_distance[0, 21, 190] < wall_distance[0, 21, 73]
    first_direction = engine.map.portal_walls[190, 2:] - engine.map.portal_walls[190, :2]
    second_direction = engine.map.portal_walls[73, 2:] - engine.map.portal_walls[73, :2]
    assert torch.dot(first_direction, second_direction) > 0
    assert rgb[0, 40, 21].tolist() == [119, 95, 75]
    assert rgb[0, 60, 21].tolist() == [79, 59, 43]
    assert rgb[0, 100, 21].tolist() == [119, 95, 75]


def test_native_projected_portal_prefers_same_depth_geometric_path(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(1007.9770812988281)
    engine.y.fill_(453.31781005859375)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(61)

    (
        _wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        _projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections()
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Projected portal 207 owns column 109, but the ray crosses portal 203 at
    # the same shared depth. Portal 207's other sector also has a continuation,
    # yet following it reverses the nested-sector order and drops solid wall 56.
    # Raster ownership must not override the geometric subsector path.
    assert projected_intersections[0, 109, 207]
    assert not geometric_intersections[0, 109, 207]
    assert not projected_intersections[0, 109, 203]
    assert geometric_intersections[0, 109, 203]
    assert rgb[0, 24, 109].tolist() == [123, 99, 79]
    assert rgb[0, 40, 109].tolist() == [95, 75, 55]
    assert rgb[0, 80, 109].tolist() == [95, 75, 55]
    assert rgb[0, 100, 109].tolist() == [87, 67, 51]


def test_native_projected_solid_owns_shared_portal_endpoint(
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
    engine.angle.fill_(math.radians(26.993408209409893))
    engine.episode_time.fill_(21)

    (
        _wall_distance,
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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # The rays hit portals 208 and 173 just outside those segs' projected
    # ranges. Adjacent solid segs 169 and 195 own the shared columns through
    # their projected left edges, so they must clip the portals rather than
    # leaving two full-height sky leaks through the surrounding walls.
    for column, portal, solid in ((15, 208, 169), (28, 173, 195)):
        assert geometric_intersections[0, column, portal]
        assert not projected_intersections[0, column, portal]
        assert not geometric_intersections[0, column, solid]
        assert projected_intersections[0, column, solid]
        assert projected_left_edges[0, column, solid]
    assert rgb[0, 0, 15].tolist() == [83, 7, 7]
    assert rgb[0, 80, 15].tolist() == [79, 0, 0]
    assert rgb[0, 0, 28].tolist() == [103, 83, 63]
    assert rgb[0, 80, 28].tolist() == [95, 75, 55]


def test_native_projected_solid_retains_portal_plane_clips(
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
    engine.view_z.fill_(-23)
    engine.angle.fill_(math.radians(175.1440430095289))
    engine.episode_time.fill_(61)

    (
        _wall_distance,
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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Solid 145 owns portal 146's projected endpoint column, but it spans only
    # sector 5's short wall height. The geometric portal must still contribute
    # its ceiling clip and farther-sector path outside that solid span.
    assert geometric_intersections[0, 153, 146]
    assert not projected_intersections[0, 153, 146]
    assert not geometric_intersections[0, 153, 145]
    assert projected_intersections[0, 153, 145]
    assert projected_left_edges[0, 153, 145]
    assert rgb[0, 72, 153].tolist() == [0, 0, 0]
    assert rgb[0, 74, 153].tolist() == [0, 0, 23]
    assert rgb[0, 75, 153].tolist() == [31, 23, 11]
    assert rgb[0, 76, 153].tolist() == [31, 23, 11]


def test_native_projected_solid_chain_clips_endpoint_portal(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([789]))
    engine.x.fill_(100.23959350585938)
    engine.y.fill_(608.5520172119141)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(348.81591804996503))
    engine.episode_time.fill_(81)

    (
        _wall_distance,
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
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = engine._native_render_portal_walls(
        frame,
        engine.view_z,
        surface_depth,
        scene_surface_depth,
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Tiny collinear solids 169 and 170 extend beyond portal 208's endpoint,
    # but their map vertices collapse into the same projected column. Solid
    # 170's left edge must rewind traversal into sector 9 and clip the portal
    # even though it is not an immediate map-endpoint neighbor of wall 208.
    assert not geometric_intersections[0, 45, 208]
    assert projected_intersections[0, 45, 208]
    assert not geometric_intersections[0, 45, 170]
    assert projected_intersections[0, 45, 170]
    assert projected_left_edges[0, 45, 170]
    assert rgb[0, 43, 45].tolist() == [67, 0, 0]
    assert rgb[0, 60, 45].tolist() == [83, 7, 7]
    assert rgb[0, 80, 45].tolist() == [67, 0, 0]
    assert rgb[0, 112, 45].tolist() == [83, 7, 7]


def test_native_projected_solid_chain_clips_geometric_endpoint_portal(
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
    engine.angle.fill_(math.radians(69.12048341453087))
    engine.episode_time.fill_(101)

    wall_projection_geometry = engine._native_wall_projection_geometry()
    screen_left, screen_right, _depth_left, _depth_right = wall_projection_geometry
    (
        _wall_distance,
        _wall_along,
        geometric_intersections,
        projected_intersections,
        projected_left_edges,
        _wall_visibility,
    ) = engine._native_portal_intersections(wall_projection_geometry)
    frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    frame, _scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    rgb = engine.map.playpal[frame.to(torch.int64)]

    # Portal 208 ends its half-open span at column 146. Solid 169 shares its
    # map endpoint but collapses to [146, 146), while the next solid, 170,
    # starts [146, 147). Wall 170 therefore owns rasterization even though the
    # mathematical ray crosses portal 208 and the two do not share a vertex.
    assert geometric_intersections[0, 146, 208]
    assert not projected_intersections[0, 146, 208]
    assert not geometric_intersections[0, 146, 170]
    assert projected_intersections[0, 146, 170]
    assert projected_left_edges[0, 146, 170]
    assert screen_right[0, 208] == 146
    assert screen_left[0, 169] == screen_right[0, 169] == 146
    assert screen_left[0, 170] == 146
    bridge_indices = engine.map.portal_endpoint_solid_bridge_end_indices[208]
    bridge_mask = engine.map.portal_endpoint_solid_bridge_end_mask[208]
    assert 170 in bridge_indices[bridge_mask].tolist()
    assert not torch.any(
        torch.all(
            engine.map.portal_walls[208, 2:]
            == engine.map.portal_walls[170].reshape(2, 2),
            dim=1,
        )
    )
    assert rgb[0, 35, 146].tolist() == [67, 0, 0]
    assert rgb[0, 40, 146].tolist() == [107, 15, 15]
    assert rgb[0, 80, 146].tolist() == [67, 0, 0]
    assert rgb[0, 115, 146].tolist() == [91, 7, 7]


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


def test_native_weapon_preserves_fractional_bob_during_sampling(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.selected_weapon.fill_(6)
    engine.weapon_state_cooldown.zero_()
    engine.weapon_raise_cooldown.zero_()
    engine.pending_weapon.fill_(-1)
    engine.weapon_ready_tics.fill_(5)
    engine.episode_time.fill_(101)
    engine._player_bob_fixed.fill_(310906)

    frame_id, _flash_id, _flash_light = engine._native_weapon_frame_selection()
    assert frame_id.item() == 38  # PLSGA0

    black = engine._native_render_weapon(
        torch.zeros((1, 208, 320), dtype=torch.uint8)
    )
    white = engine._native_render_weapon(
        torch.full((1, 208, 320), 255, dtype=torch.uint8)
    )
    opaque = black == white
    coordinates = torch.nonzero(opaque[0])

    # R_DrawPSprite subtracts the 16.16 bob from texturemid before applying
    # its reciprocal 320x240 y scale. Shifting a ready-state raster by two
    # integer rows instead makes 30 extra texels opaque and samples the wrong
    # source row throughout the top of the plasma rifle.
    assert coordinates.min(dim=0).values.tolist() == [153, 127]
    assert coordinates.max(dim=0).values.tolist() == [207, 191]
    assert coordinates.shape[0] == 3001
    assert not opaque[0, 154, 142]
    assert opaque[0, 154, 147]
    assert black[0, 154, 147].item() == 108


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


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_item_animations_use_independent_level_spawn_tics(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        2,
        device=torch.device("cpu"),
    )
    seeds = torch.tensor([123, 456])
    engine.reset(torch.ones(2, dtype=torch.bool), seeds)
    first_tics = engine.item_animation_initial_tics.clone()
    animated = torch.isin(
        engine.map.item_types,
        torch.tensor([2014, 2015, 2018, 2019]),
    )

    assert torch.all((first_tics[:, animated] >= 1) & (first_tics[:, animated] <= 6))
    assert all(torch.unique(first_tics[lane, animated]).numel() > 1 for lane in range(2))
    assert not torch.equal(first_tics[0, animated], first_tics[1, animated])

    engine.reset(torch.ones(2, dtype=torch.bool), seeds)
    assert torch.equal(engine.item_animation_initial_tics, first_tics)
    engine.reset(torch.tensor([True, False]), torch.tensor([789, 999]))
    assert not torch.equal(engine.item_animation_initial_tics[0, animated], first_tics[0, animated])
    assert torch.equal(engine.item_animation_initial_tics[1], first_tics[1])

    green_armor = torch.nonzero(engine.map.item_types == 2018).flatten()[0]
    armor_bonus = torch.nonzero(engine.map.item_types == 2015).flatten()[0]
    engine.item_animation_initial_tics[0, green_armor] = 2
    engine.item_animation_initial_tics[0, armor_bonus] = 4
    engine.episode_time[0] = 41

    sprites, fullbright = engine._native_item_sprite_ids()

    # ViZDoom seed 123 reaches ARM1B0 with seven tics and BON2B0 with five
    # tics at this exact episode time. The initial remaining-tic values above
    # are the corresponding LevelSpawned trace.
    assert sprites[0, green_armor] == engine.map.raw_item_animation_sprite_ids[6]
    assert fullbright[0, green_armor]
    assert sprites[0, armor_bonus] == engine.map.raw_item_animation_sprite_ids[3]
    assert not fullbright[0, armor_bonus]


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

    # ViZDoom's status bar is painted from state captured at the start of each
    # game tic. A pickup on the first internal tic reaches a frame-skip-2 HUD;
    # one on the final internal tic remains one rendered frame behind.
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    before_cell_pickup = engine._native_render_hud()
    original_collect_items = engine._collect_items
    collect_calls = 0

    def collect_cell_on_first_tic() -> None:
        nonlocal collect_calls
        if collect_calls == 0:
            engine.ammo[0, 5] = 100
        collect_calls += 1

    engine._collect_items = collect_cell_on_first_tic
    engine.step(noop)
    assert engine.hud_ammo_counts.tolist() == [[50.0, 0.0, 0.0, 100.0]]
    assert not torch.equal(engine._native_render_hud(), before_cell_pickup)

    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    collect_calls = 0

    def collect_cell_on_second_tic() -> None:
        nonlocal collect_calls
        if collect_calls == 1:
            engine.ammo[0, 5] = 100
        collect_calls += 1

    engine._collect_items = collect_cell_on_second_tic
    engine.step(noop)
    engine._collect_items = original_collect_items
    assert engine.ammo[0, 5].item() == 100
    assert engine.hud_ammo_counts.tolist() == [[50.0, 0.0, 0.0, 0.0]]
    assert torch.equal(engine._native_render_hud(), before_cell_pickup)

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
    (
        pit_frame,
        _scene_depth,
        _sprite_clip_depth,
        _sprite_clip_wall,
    ) = engine._native_render_portal_walls(
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
        (
            portal_frame,
            scene_depth,
            sprite_clip_depth,
            sprite_clip_wall,
        ) = engine._native_render_portal_walls(
            flat_frame, view_z, surface_depth, scene_surface_depth
        )
        without_scene_depth = engine._native_render_sprites(
            portal_frame.clone(),
            wall_distance,
            view_z,
            torch.full_like(scene_depth, torch.inf),
        )
        with_portal_clip = engine._native_render_sprites(
            portal_frame.clone(),
            wall_distance,
            view_z,
            sprite_clip_depth,
            sprite_clip_wall,
        )

        assert torch.any(without_scene_depth != portal_frame)
        assert torch.equal(with_portal_clip, portal_frame)


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_drawseg_clips_preserve_item_bottom_rows(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.x.fill_(1091.2748260498047)
    engine.y.fill_(784.0078125)
    engine.z.zero_()
    engine.view_z.fill_(41.0)
    engine.view_height.copy_(engine.view_z - engine.z)
    engine.angle.fill_(math.radians(341.29028328258784))
    engine.episode_time.fill_(41)

    wall_distance = engine._native_raycast()
    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(), engine.view_z
    )
    (
        portal_frame,
        scene_depth,
        sprite_clip_depth,
        sprite_clip_wall,
    ) = engine._native_render_portal_walls(
        flat_frame, engine.view_z, surface_depth, scene_surface_depth
    )
    with_plane_depth = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        scene_depth,
    )
    with_drawseg_clips = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        sprite_clip_depth,
        sprite_clip_wall,
    )

    edge_pixels = ((142, 250), (152, 150), (159, 80), (149, 170))
    for y, x in edge_pixels:
        assert torch.isfinite(scene_depth[0, y, x])
        assert torch.isinf(sprite_clip_depth[0, y, x])
        assert with_plane_depth[0, y, x] == portal_frame[0, y, x]
        assert with_drawseg_clips[0, y, x] != portal_frame[0, y, x]

    rgb = engine.map.playpal[with_drawseg_clips.to(torch.int64)]
    assert rgb[0, 142, 250].tolist() == [19, 19, 19]
    assert rgb[0, 152, 150].tolist() == [19, 19, 19]
    assert rgb[0, 159, 80].tolist() == [95, 75, 55]
    assert rgb[0, 149, 170].tolist() == [19, 19, 19]
    # R_DrawVisSprite advances source rows with 0xffffffff / yscale.
    assert rgb[0, 205, 280].tolist() == [55, 63, 39]


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_drawseg_side_clips_item_behind_projected_corner(
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
    engine.view_z.fill_(41.0)
    engine.angle.fill_(0.9854868054389954)
    engine.episode_time.fill_(21)

    wall_distance = engine._native_raycast()
    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(), engine.view_z
    )
    (
        portal_frame,
        _scene_depth,
        sprite_clip_depth,
        sprite_clip_wall,
    ) = engine._native_render_portal_walls(
        flat_frame, engine.view_z, surface_depth, scene_surface_depth
    )
    without_drawseg_side = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        sprite_clip_depth,
    )
    with_drawseg_side = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        engine.view_z,
        sprite_clip_depth,
        sprite_clip_wall,
    )

    # Item 126 projects just around one-sided wall 26. Its view depth is
    # nearer than the wall's column intersection, but R_DrawSprite still clips
    # it because the item lies on the wall's obscured side.
    for y in range(112, 116):
        assert sprite_clip_wall[0, y, 62] == 26
        assert without_drawseg_side[0, y, 62] != portal_frame[0, y, 62]
        assert with_drawseg_side[0, y, 62] == portal_frame[0, y, 62]


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_fixed_wall_projection_clips_item_at_pit_step(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([246]))
    engine.x.fill_(567.9317626953125)
    engine.y.fill_(547.9797973632812)
    engine.z.fill_(-24.0)
    engine.view_z.fill_(-1.0313720703125)
    engine.view_height.copy_(engine.view_z - engine.z)
    engine.angle.fill_(math.radians(28.630371100416028))
    engine.episode_time.fill_(50)
    engine.player_dead.fill_(True)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.item_available.zero_()

    without_item = engine.render_native_frame(include_hud=False)
    assert engine.map.item_types[80].item() == 2046
    engine.item_available[:, 80] = True
    with_item = engine.render_native_frame(include_hud=False)

    # ViZDoom's OWallMost puts the near red pit wall on row 99. The distant
    # RocketBox remains visible immediately above it, but not through it.
    assert not torch.equal(with_item[:, 98, 153], without_item[:, 98, 153])
    assert torch.equal(with_item[:, 99, 153], without_item[:, 99, 153])


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_portal_clips_mark_foreground_floor_visplane(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(650.5361785888672)
    engine.y.fill_(351.4964141845703)
    engine.z.zero_()
    engine.view_z.fill_(41.0)
    engine.view_height.fill_(41.0)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(41)
    engine.player_dead.fill_(True)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.item_available.zero_()

    current_sector = engine._current_sector()
    approximate_indices, _surface_depth, _scene_depth = engine._native_render_flats(
        current_sector,
        engine.view_z,
    )
    approximate = engine.map.playpal[approximate_indices.to(torch.int64)]
    frame = engine.render_native_frame(include_hud=False)

    # The independent plane rays put this boundary pixel on the -8 pit floor.
    # R_RenderSegLoop instead marks the foreground sector 0 visplane through
    # its exact floorclip span, matching the raw ViZDoom RGB value.
    assert approximate[0, 183, 115].tolist() == [115, 0, 0]
    assert frame[0, 183, 115].tolist() == [95, 75, 55]


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_solid_wall_bottom_and_wallscan_tail_match_reference(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([789]))
    engine._x_fixed.fill_(29_837_736)
    engine._y_fixed.fill_(1_050_082)
    engine.x.fill_(29_837_736 / 65_536)
    engine.y.fill_(1_050_082 / 65_536)
    engine._angle_bam.fill_(4_161_536_001)
    engine.angle.fill_(4_161_536_001 * (2.0 * math.pi / float(1 << 32)))
    engine.z.zero_()
    engine.view_z.fill_(41.0)
    engine.view_height.fill_(41.0)
    engine.episode_time.fill_(101)

    flat_frame, surface_depth, scene_surface_depth = engine._native_render_flats(
        engine._current_sector(),
        engine.view_z,
    )
    portal_frame, scene_depth, _sprite_clip_depth, _sprite_clip_wall = (
        engine._native_render_portal_walls(
            flat_frame,
            engine.view_z,
            surface_depth,
            scene_surface_depth,
        )
    )
    wall17_distance = engine._native_portal_intersections(
        engine._native_wall_projection_geometry()
    )[0][0, :, 17]

    # ViZDoom's wallbottom is exclusive: wall 17 owns both final projected
    # rows in these columns, while the immediately following row is a plane.
    for y, x in ((127, 136), (128, 136), (163, 148), (164, 148)):
        assert scene_depth[0, y, x].item() == pytest.approx(
            wall17_distance[x].item(),
            abs=1e-3,
        )
    assert scene_depth[0, 129, 136].item() != pytest.approx(
        wall17_distance[136].item(),
        abs=1e-3,
    )
    assert scene_depth[0, 165, 148].item() != pytest.approx(
        wall17_distance[148].item(),
        abs=1e-3,
    )

    rgb = engine.map.playpal[portal_frame.to(torch.int64)]
    assert rgb[0, 127, 136].tolist() == [95, 75, 55]
    assert rgb[0, 128, 136].tolist() == [103, 83, 63]
    # wallscan's aligned four-column path reuses the first column's colormap
    # for uneven bottom tails. These raw ViZDoom samples cover three groups.
    expected_tail = {
        (129, 137): [103, 83, 63],
        (129, 138): [83, 63, 47],
        (130, 139): [95, 75, 55],
        (145, 143): [67, 47, 27],
        (153, 145): [119, 95, 75],
        (154, 147): [63, 43, 27],
    }
    for (y, x), expected in expected_tail.items():
        assert rgb[0, y, x].tolist() == expected


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_fixed_sprite_posts_do_not_start_one_row_early(
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
    engine.z.fill_(-28.0)
    engine.view_z.fill_(20.0)
    engine.view_height.copy_(engine.view_z - engine.z)
    engine.angle.fill_(math.radians(348.81591804996503))
    engine.episode_time.fill_(9)
    engine.player_dead.fill_(True)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.item_available.zero_()

    without_items = engine.render_native_frame(include_hud=False)
    assert engine.map.item_types[75].item() == 2046
    assert engine.map.item_types[51].item() == 2005
    engine.item_available[:, (75, 51)] = True
    with_items = engine.render_native_frame(include_hud=False)

    # R_DrawMaskedColumn begins these opaque posts on row 104. Sampling the
    # post through centery - 1 without its projected bound exposed row 103.
    assert torch.equal(with_items[:, 103, 11], without_items[:, 103, 11])
    assert torch.equal(with_items[:, 103, 125], without_items[:, 103, 125])
    assert not torch.equal(with_items[:, 104, 11], without_items[:, 104, 11])
    assert not torch.equal(with_items[:, 104, 125], without_items[:, 104, 125])


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
def test_native_drawseg_side_keeps_sprite_edges_past_nearest_ray(
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
    engine.angle.fill_(math.radians(154.96215823920494))
    engine.episode_time.fill_(61)
    engine.weapon_raise_cooldown.zero_()
    engine.player_dead.fill_(True)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.hitscan_puff_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.drop_spawned.zero_()
    engine.item_available.zero_()

    medikits = torch.tensor((119, 120, 121, 122))
    expected_spawns = torch.tensor(
        ((646, 1273), (679, 1272), (715, 1272), (746, 1272)),
        dtype=torch.float32,
    )
    assert torch.equal(engine.map.item_spawns[medikits, :2], expected_spawns)
    without_items = engine.render_native_frame(include_hud=False)
    engine.item_available[0, medikits] = True
    with_items = engine.render_native_frame(include_hud=False)

    actor_depth_fixed, _actor_side_fixed = engine._native_sprite_view_coordinates(
        engine.map.item_spawns[None, medikits, 0],
        engine.map.item_spawns[None, medikits, 1],
    )
    ray_distance = engine._native_raycast()
    edge_columns = torch.tensor((262, 274, 290, 306))
    # Each mathematical column ray reaches a nearby wall just before the
    # sprite center depth. Doom does not reject the sprite here: R_DrawSprite
    # resolves the owning drawseg endpoint and wall side per pixel.
    assert torch.all(
        ray_distance[0, edge_columns] * 4096 < actor_depth_fixed[0]
    )
    assert with_items[0, 113, 262].tolist() == [119, 119, 119]
    assert with_items[0, 114, 274].tolist() == [119, 119, 119]
    assert with_items[0, 115, 290].tolist() == [119, 119, 119]
    assert with_items[0, 116, 306].tolist() == [119, 119, 119]
    for y, x in ((113, 262), (114, 274), (115, 290), (116, 306)):
        assert not torch.equal(with_items[0, y, x], without_items[0, y, x])


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_blocking_drawseg_excludes_right_endpoint_from_sprite_clip(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(496.6570129394531)
    engine.y.fill_(318.4211883544922)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(math.radians(102.16735842222519))
    engine.episode_time.fill_(61)
    engine.weapon_raise_cooldown.zero_()
    engine.player_dead.fill_(True)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.hitscan_puff_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.drop_spawned.zero_()
    engine.item_available.zero_()

    item_index = 28
    assert engine.map.item_spawns[item_index].tolist() == [-228.0, 950.0, 0.0]
    sprite = engine.map.item_raw_visual_types[item_index].reshape(1, 1)
    sprite_left, sprite_right, _texture_step = (
        engine._native_sprite_horizontal_projection(
            engine.map.item_spawns[None, item_index, 0].reshape(1, 1),
            engine.map.item_spawns[None, item_index, 1].reshape(1, 1),
            sprite,
        )
    )
    assert (sprite_left.item(), sprite_right.item()) == (34, 45)
    _blocking_distance, blocking_wall = engine._native_blocking_raycast()
    wall_screen_left, wall_screen_right, _depth_left, _depth_right = (
        engine._native_wall_projection_geometry()
    )
    assert blocking_wall[0, 44].item() == 54
    assert wall_screen_left[0, 54].item() == 0
    assert wall_screen_right[0, 54].item() == 44

    without_item = engine.render_native_frame(include_hud=False)
    engine.item_available[0, item_index] = True
    with_item = engine.render_native_frame(include_hud=False)

    # The mathematical ray at column 44 still intersects wall 54, but Doom's
    # FWallCoords span is half-open and excludes that right endpoint. The
    # RocketBox therefore contributes its last projected sprite column there,
    # while the adjacent owned wall column remains occluded.
    assert torch.equal(with_item[0, 109:114, 43], without_item[0, 109:114, 43])
    assert with_item[0, 109, 44].tolist() == [111, 87, 67]
    assert with_item[0, 110:114, 44].tolist() == [[103, 83, 63]] * 4


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_blocking_drawseg_rejects_occluded_offscreen_item(
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
    engine.angle.fill_(math.radians(119.97070315293286))
    engine.episode_time.fill_(21)
    engine.weapon_raise_cooldown.zero_()
    engine.player_dead.fill_(True)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.hitscan_puff_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.drop_spawned.zero_()
    engine.item_available.zero_()

    item_index = 38
    assert engine.map.item_spawns[item_index].tolist() == [-99.0, 277.0, 0.0]
    sprite = engine.map.item_raw_visual_types[item_index].reshape(1, 1)
    sprite_left, sprite_right, _texture_step = (
        engine._native_sprite_horizontal_projection(
            engine.map.item_spawns[None, item_index, 0].reshape(1, 1),
            engine.map.item_spawns[None, item_index, 1].reshape(1, 1),
            sprite,
        )
    )
    assert sprite_left.item() <= 0 < sprite_right.item()
    _blocking_distance, blocking_wall = engine._native_blocking_raycast()
    assert blocking_wall[0, 0].item() == 6

    without_item = engine.render_native_frame(include_hud=False)
    engine.item_available[0, item_index] = True
    with_item = engine.render_native_frame(include_hud=False)
    # Both endpoints of blocking wall 6 are nearer than the clip box center,
    # so R_DrawSprite keeps the wall in front at the left screen edge.
    assert torch.equal(with_item[0, 111:116, 0], without_item[0, 111:116, 0])


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_sprites_keep_offcenter_spans_that_cross_view(
    pinned_deathmatch_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        pinned_deathmatch_scenario,
        1,
        device=torch.device("cpu"),
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([1337]))
    engine.x.fill_(1091.274826)
    engine.y.fill_(784.0078125)
    engine.z.zero_()
    engine.view_z.fill_(41.0)
    engine.angle.fill_(math.radians(341.290283))
    engine.episode_time.fill_(41)
    engine.player_dead.fill_(True)
    engine.enemy_alive.zero_()
    engine.enemy_death_tics.zero_()
    engine.projectile_alive.zero_()
    engine.enemy_projectile_alive.zero_()
    engine.projectile_impact_tics.zero_()
    engine.enemy_projectile_impact_tics.zero_()
    engine.teleport_fog_tics.zero_()
    engine.hitscan_puff_tics.zero_()
    engine.drop_type.fill_(-1)
    engine.drop_spawned.zero_()

    black = torch.zeros(
        (1, engine.native_view_height, engine.native_screen_width),
        dtype=torch.uint8,
    )
    white = torch.full_like(black, 255)
    wall_distance = torch.full((1, engine.native_screen_width), torch.inf)
    scene_depth = torch.full_like(black, torch.inf, dtype=torch.float32)

    # R_ProjectSprite clips the projected span, not the actor's center angle.
    # These two pickups have centers beyond opposite 45-degree view edges, but
    # their broad sprites still contribute columns at the edge of the screen.
    for spawn_x, spawn_y, expected_x_bounds in (
        (1126, 696, (317, 319)),
        (1248, 873, (0, 15)),
    ):
        item_index = torch.where(
            (engine.map.item_spawns[:, 0] == spawn_x)
            & (engine.map.item_spawns[:, 1] == spawn_y)
        )[0].item()
        engine.item_available.zero_()
        engine.item_available[0, item_index] = True
        over_black = engine._native_render_sprites(
            black,
            wall_distance,
            engine.view_z,
            scene_depth,
        )
        over_white = engine._native_render_sprites(
            white,
            wall_distance,
            engine.view_z,
            scene_depth,
        )
        _sprite_y, sprite_x = torch.where(over_black[0] == over_white[0])

        assert sprite_x.numel() > 0
        assert (sprite_x.min().item(), sprite_x.max().item()) == expected_x_bounds


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
    (
        portal_frame,
        _scene_depth,
        sprite_clip_depth,
        sprite_clip_wall,
    ) = engine._native_render_portal_walls(
        flat_frame, view_z, surface_depth, scene_surface_depth
    )
    with_dolls = engine._native_render_sprites(
        portal_frame.clone(),
        wall_distance,
        view_z,
        sprite_clip_depth,
        sprite_clip_wall,
    )

    assert torch.any(with_dolls != portal_frame)
