from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gradoom.engine import TorchDeathmatchEngine
from gradoom.scenario import compile_deathmatch_scenario

SCENARIO = Path("/Users/tsilva/repos/tsilva/ViZDoom-turbo/scenarios/deathmatch.wad")
DOOM2 = Path("/Users/tsilva/roms/vizdoom/doom2.wad")


@pytest.mark.skipif(not SCENARIO.is_file() or not DOOM2.is_file(), reason="operator WADs absent")
def test_native_renderer_preserves_rgb_hud_and_enemy_animation() -> None:
    scenario = compile_deathmatch_scenario(SCENARIO, DOOM2)
    engine = TorchDeathmatchEngine(scenario, 1, device=torch.device("cpu"))
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))

    initial = engine.render_native_frame(include_hud=True)

    assert initial.shape == (1, 240, 320, 3)
    assert initial.dtype == torch.uint8
    assert torch.any(initial[:, 200:232] != 0)
    assert torch.all(initial[:, 232:] == 0)
    assert not torch.equal(initial[..., 0], initial[..., 1])

    engine.enemy_type[:, 0] = 0
    engine.enemy_alive[:, 0] = True
    engine.enemy_x[:, 0] = engine.x + 64
    engine.enemy_y[:, 0] = engine.y
    engine.enemy_angle[:, 0] = 0
    engine.episode_time.fill_(4)
    first_walk_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.episode_time.fill_(8)
    second_walk_frame = engine._native_enemy_sprite_ids()[:, 0].clone()
    engine.enemy_attack_phase[:, 0] = 1
    attack_frame = engine._native_enemy_sprite_ids()[:, 0].clone()

    assert not torch.equal(first_walk_frame, second_walk_frame)
    assert not torch.equal(second_walk_frame, attack_frame)

    engine.enemy_health[:, 0] = 1
    engine._apply_enemy_damage(torch.ones_like(engine.enemy_health))

    assert not engine.enemy_alive[:, 0].any()
    assert engine.enemy_death_type[:, 0].tolist() == [0]
    assert engine.enemy_death_tics[:, 0].tolist() == [56]
    death_start = engine.render_native_frame(include_hud=False)
    for _ in range(8):
        engine._collect_drops()
    death_progressed = engine.render_native_frame(include_hud=False)
    assert not torch.equal(death_start, death_progressed)
