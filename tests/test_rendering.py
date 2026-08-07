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


def test_native_transparent_sprite_reveals_farther_actor(square_scenario) -> None:
    atlas = np.zeros((2, 3, 3), dtype=np.uint8)
    atlas[0] = 10
    atlas[1] = 20
    opaque = np.ones_like(atlas, dtype=np.bool_)
    opaque[0, 1, 1] = False
    enemy_ids = np.empty((6, 4, 8), dtype=np.int32)
    enemy_ids[0].fill(0)
    enemy_ids[1:].fill(1)
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
    engine.enemy_x[0, :2] = torch.tensor([64.0, 96.0])
    engine.enemy_y[0, :2] = 0
    engine.enemy_z[0, :2] = 0
    engine.enemy_type[0, :2] = torch.tensor([0, 1])
    engine.enemy_alive[0, :2] = True
    frame = torch.zeros((1, 208, 320), dtype=torch.uint8)

    rendered = engine._native_render_sprites(
        frame,
        torch.full((1, 320), torch.inf),
        engine.view_z,
        torch.full_like(frame, torch.inf, dtype=torch.float32),
    )

    assert rendered[0, 103, 160].item() == 20
    assert rendered[0, 103, 158].item() == 10


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_renderer_preserves_rgb_hud_and_enemy_animation() -> None:
    scenario = compile_deathmatch_scenario(SCENARIO, DOOM2)
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

    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    noop = torch.zeros((1, 20), dtype=torch.bool)
    for _ in range(8):
        engine.step(noop)
    assert engine.episode_time.item() == 17
    assert engine.mugshot_face_index.item() == 1
    engine.step(noop)
    assert engine.episode_time.item() == 19
    assert engine.mugshot_face_index.item() == 0

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
    engine.bonus_count.fill_(6)
    assert engine._native_mugshot_patch_index(0, 75) == 65
    engine.health.zero_()
    assert engine._native_mugshot_patch_index(0, 0) == 69
    engine.mugshot_grin.zero_()
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
    flat_frame, surface_depth = engine._native_render_flats(
        engine._current_sector(), engine.z + 41.0
    )
    pit_frame, _scene_depth = engine._native_render_portal_walls(
        flat_frame.clone(), engine.z + 41.0, surface_depth
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
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 10
    attack_frame = engine._native_enemy_sprite_ids()[:, 0].clone()

    engine.enemy_attack_phase[:, 0] = 2
    engine.enemy_cooldown[:, 0] = 16
    muzzle_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.enemy_cooldown[:, 0] = 8
    recovery_frame = engine._native_enemy_sprite_ids()[:, 0].clone()

    assert not torch.equal(first_walk_frame, second_walk_frame)
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
def test_native_pit_depth_occludes_map_items() -> None:
    scenario = compile_deathmatch_scenario(SCENARIO, DOOM2)
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
        flat_frame, surface_depth = engine._native_render_flats(
            engine._current_sector(), view_z
        )
        portal_frame, scene_depth = engine._native_render_portal_walls(
            flat_frame, view_z, surface_depth
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
def test_native_renderer_includes_voodoo_dolls() -> None:
    scenario = compile_deathmatch_scenario(SCENARIO, DOOM2)
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
    flat_frame, surface_depth = engine._native_render_flats(engine._current_sector(), view_z)
    portal_frame, scene_depth = engine._native_render_portal_walls(
        flat_frame, view_z, surface_depth
    )
    with_dolls = engine._native_render_sprites(
        portal_frame.clone(), wall_distance, view_z, scene_depth
    )

    assert torch.any(with_dolls != portal_frame)
