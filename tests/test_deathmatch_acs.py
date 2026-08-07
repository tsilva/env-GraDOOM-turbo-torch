from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch

from gradoom.engine import TorchDeathmatchEngine


def _engine(square_scenario) -> TorchDeathmatchEngine:
    engine = TorchDeathmatchEngine(
        square_scenario,
        2,
        device=torch.device("cpu"),
        frame_skip=2,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))
    engine.weapon_raise_cooldown.zero_()
    return engine


def _finish_pending_attack(engine: TorchDeathmatchEngine) -> torch.Tensor:
    reward = torch.zeros(engine.num_envs)
    noop = torch.zeros((engine.num_envs, 20), dtype=torch.bool)
    while torch.any(engine.pending_attack_weapon >= 0):
        reward += engine._player_attack(noop)
    return reward


def test_certified_enemy_actor_radii_match_reference(square_scenario) -> None:
    engine = _engine(square_scenario)

    assert engine._enemy_radius.tolist() == [20.0, 20.0, 16.0, 20.0, 30.0, 24.0]


def test_reset_uses_acs_teleport_and_delayed_spawn(square_scenario) -> None:
    engine = _engine(square_scenario)
    assert not torch.any(engine.enemy_alive)
    assert engine.next_spawn_check.tolist() == [106, 106]
    low_x, high_x, low_y, high_y = engine.map.spawn_bounds.tolist()
    assert torch.all((engine.x >= low_x) & (engine.x <= high_x))
    assert torch.all((engine.y >= low_y) & (engine.y <= high_y))
    assert torch.all((engine.angle >= 0) & (engine.angle < 2 * torch.pi))
    assert engine.ammo[:, [0, 1, 2, 3, 4, 5]].tolist() == [
        [0.0, 50.0, 0.0, 50.0, 0.0, 0.0],
        [0.0, 50.0, 0.0, 50.0, 0.0, 0.0],
    ]

    engine.episode_time.fill_(105)
    engine._spawn_tick()
    assert engine.next_spawn_check.tolist() == [106, 106]
    engine.episode_time.fill_(106)
    engine._spawn_tick()
    assert engine.next_spawn_check.tolist() == [116, 116]


def test_sequential_lane_seeds_cover_spawn_domain(square_scenario) -> None:
    lane_count = 256
    engine = TorchDeathmatchEngine(
        square_scenario,
        lane_count,
        device=torch.device("cpu"),
        frame_skip=2,
    )
    engine.reset(
        torch.ones(lane_count, dtype=torch.bool),
        torch.arange(lane_count, dtype=torch.int64),
    )
    low_x, high_x, low_y, high_y = engine.map.spawn_bounds
    normalized_x = (engine.x - low_x) / (high_x - low_x)
    normalized_y = (engine.y - low_y) / (high_y - low_y)

    assert 0.35 < float(normalized_x.mean()) < 0.65
    assert 0.35 < float(normalized_y.mean()) < 0.65
    assert float(normalized_x.std()) > 0.2
    assert float(normalized_y.std()) > 0.2


def test_spawn_check_attempts_each_acs_actor_class(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine._enemy_spawn_threshold.fill_(65536)
    engine.episode_time.fill_(106)
    engine._spawn_tick()
    assert torch.sum(engine.enemy_alive, dim=1).tolist() == [6, 6]
    assert torch.equal(
        torch.sort(engine.enemy_type[0, engine.enemy_alive[0]]).values,
        torch.arange(6),
    )
    assert torch.all(engine.enemy_target_slot[engine.enemy_alive] == -2)
    spawned_type = engine.enemy_type[engine.enemy_alive]
    assert torch.equal(
        engine.enemy_move_cooldown[engine.enemy_alive],
        engine._enemy_look_interval[spawned_type] - 2,
    )
    fog_alive = engine.teleport_fog_tics > 0
    assert torch.sum(fog_alive, dim=1).tolist() == [6, 6]
    assert torch.all(engine.teleport_fog_tics[fog_alive] == 71)
    assert torch.equal(engine.teleport_fog_x[:, :6], engine.enemy_x[:, :6])
    assert torch.equal(engine.teleport_fog_y[:, :6], engine.enemy_y[:, :6])
    assert torch.equal(engine.teleport_fog_z[:, :6], engine.enemy_z[:, :6])

    for _ in range(70):
        engine._collect_drops()
    assert torch.all(engine.teleport_fog_tics[fog_alive] == 1)
    engine._collect_drops()
    assert not torch.any(engine.teleport_fog_tics)


def test_unaware_monster_waits_for_front_facing_sight_check(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(-100)
    engine.y.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_angle[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -2
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_cooldown[:, 0] = 999

    engine._enemy_tick()

    assert engine.enemy_target_slot[:, 0].tolist() == [-2, -2]
    assert engine.enemy_x[:, 0].tolist() == [0.0, 0.0]
    assert engine.enemy_move_cooldown[:, 0].tolist() == [9, 9]

    engine.enemy_angle[:, 0] = math.pi
    for _ in range(9):
        engine._enemy_tick()
    assert engine.enemy_target_slot[:, 0].tolist() == [-2, -2]
    engine._enemy_tick()

    assert engine.enemy_target_slot[:, 0].tolist() == [-1, -1]
    # Fresh actors begin with DI_EAST. Doom avoids immediately reversing
    # direction, so a player directly behind them causes one eastward step.
    assert torch.all(engine.enemy_x[:, 0] > 0)


def test_player_weapon_noise_wakes_monster_outside_field_of_view(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(-100)
    engine.y.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_angle[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -2
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_cooldown[:, 0] = 999

    engine._execute_player_attack(
        torch.zeros(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    assert engine.enemy_heard_player[:, 0].tolist() == [True, True]

    engine._enemy_tick()

    assert engine.enemy_target_slot[:, 0].tolist() == [-1, -1]
    assert engine.enemy_heard_player[:, 0].tolist() == [False, False]
    assert torch.all(engine.enemy_x[:, 0] > 0)


def test_kill_reward_comes_from_spawned_actor_class(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(0)
    engine.y.fill_(0)
    engine.angle.fill_(0)
    engine.enemy_x[:, 0] = 48
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 1
    engine.enemy_alive[:, 0] = True
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    reward = engine._player_attack(buttons)
    reward += _finish_pending_attack(engine)

    assert reward.tolist() == [10.0, 10.0]
    assert engine.killcount.tolist() == [1, 1]
    assert not torch.any(engine.enemy_alive[:, 0])


def test_nonlethal_damage_enters_reference_pain_state(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_type[:, 0] = torch.tensor([0, 5])
    engine.enemy_health[:, 0] = torch.tensor([20.0, 500.0])
    engine.enemy_alive[:, 0] = True
    engine.enemy_attack_phase[:, 0] = 2
    engine.enemy_cooldown[:, 0] = 8
    engine._enemy_pain_chance.fill_(256)
    damage = torch.zeros_like(engine.enemy_health)
    damage[:, 0] = 1

    engine._apply_enemy_damage(damage)

    assert engine.enemy_pain_tics[:, 0].tolist() == [6, 4]
    assert engine.enemy_attack_phase[:, 0].tolist() == [0, 0]
    assert engine.enemy_cooldown[:, 0].tolist() == [0, 0]

    before_x = engine.enemy_x[:, 0].clone()
    engine._enemy_tick()
    assert engine.enemy_pain_tics[:, 0].tolist() == [5, 3]
    assert torch.equal(engine.enemy_x[:, 0], before_x)


def test_dying_monsters_remain_solid_until_no_block_frame(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_type[:, 0] = torch.tensor([0, 5])
    engine.enemy_health[:, 0] = 1
    engine.enemy_alive[:, 0] = True
    damage = torch.zeros_like(engine.enemy_health)
    damage[:, 0] = 1

    engine._apply_enemy_damage(damage)

    assert engine._enemy_solid_mask()[:, 0].tolist() == [True, True]
    assert engine.drop_delay[:, 0].tolist() == [10, 0]
    for _ in range(10):
        engine._collect_drops()
    assert engine._enemy_solid_mask()[:, 0].tolist() == [False, True]
    for _ in range(14):
        engine._collect_drops()
    assert engine._enemy_solid_mask()[:, 0].tolist() == [False, False]


def test_voodoo_doll_hits_damage_shared_player_health(square_scenario) -> None:
    engine = _engine(square_scenario)
    first_doll = engine.map.player_starts[0]
    engine.x.fill_(float(first_doll[0]) - 48)
    engine.y.fill_(float(first_doll[1]))
    engine.angle.fill_(0)
    engine.enemy_alive.fill_(False)
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    reward = engine._player_attack(buttons)
    reward += _finish_pending_attack(engine)

    assert reward.tolist() == [0.0, 0.0]
    assert torch.all(engine.health < 100)


def test_projectile_hit_damages_shared_health_through_voodoo_doll(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    first_doll = engine.map.player_starts[0]
    engine.x.fill_(float(first_doll[0]) - 64)
    engine.y.fill_(float(first_doll[1]))
    engine.z.fill_(float(engine._player_start_z[0]))
    engine.angle.zero_()
    engine.enemy_alive.zero_()

    engine._execute_player_attack(
        torch.full((2,), 7),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )
    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert engine.health.tolist() == [70.0, 85.0]
    assert not torch.any(engine.projectile_alive[:, 0])
    assert engine.projectile_impact_tics[:, 0].tolist() == [20, 20]
    # Voodoo dolls share health and armor, but not the camera body's momentum.
    assert torch.equal(engine.momentum_x, torch.zeros(2))
    assert torch.equal(engine.momentum_y, torch.zeros(2))


def test_rocket_radius_damage_reaches_voodoo_doll_without_moving_player(
    square_scenario,
) -> None:
    scenario = replace(
        square_scenario,
        blocking_segments=np.asarray(
            [(-28.0, -200.0, -28.0, 0.0)],
            dtype=np.float32,
        ),
    )
    engine = _engine(scenario)
    engine.enemy_alive.zero_()
    engine.x.fill_(200)
    engine.y.fill_(200)
    engine.z.zero_()
    engine.projectile_x[:, 0] = -40
    engine.projectile_y[:, 0] = -128
    engine.projectile_z[:, 0] = 32
    engine.projectile_velocity_x[:, 0] = 20
    engine.projectile_velocity_y[:, 0] = 0
    engine.projectile_velocity_z[:, 0] = 0
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert engine.health.tolist() == [44.0, 44.0]
    assert not torch.any(engine.projectile_alive[:, 0])
    assert engine.projectile_impact_tics[:, 0].tolist() == [18, 18]
    assert torch.equal(engine.momentum_x, torch.zeros(2))
    assert torch.equal(engine.momentum_y, torch.zeros(2))


def test_enemy_projectile_hits_voodoo_doll_without_moving_player(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    first_doll = engine.map.player_starts[0]
    engine.enemy_alive.zero_()
    engine.x.fill_(200)
    engine.y.fill_(200)
    engine.z.zero_()
    engine.enemy_projectile_x[:, 0] = float(first_doll[0]) - 42.0
    engine.enemy_projectile_y[:, 0] = float(first_doll[1])
    engine.enemy_projectile_z[:, 0] = float(engine._player_start_z[0]) + 32.0
    engine.enemy_projectile_velocity_x[:, 0] = 15
    engine.enemy_projectile_velocity_y[:, 0] = 0
    engine.enemy_projectile_velocity_z[:, 0] = 0
    engine.enemy_projectile_alive[:, 0] = True
    active = torch.ones(2, dtype=torch.bool)

    engine._enemy_projectile_tick(active)
    engine._enemy_projectile_tick(active)

    assert engine.health.tolist() == [52.0, 76.0]
    assert not torch.any(engine.enemy_projectile_alive[:, 0])
    assert engine.enemy_projectile_impact_tics[:, 0].tolist() == [18, 18]
    assert torch.equal(engine.momentum_x, torch.zeros(2))
    assert torch.equal(engine.momentum_y, torch.zeros(2))


def test_enemy_projectile_hits_monsters_without_player_kill_credit(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.fill_(200)
    engine.z.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = -100
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    engine.enemy_x[:, 1] = 0
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_type[:, 1] = torch.tensor([0, 5])
    engine.enemy_health[:, 1] = torch.tensor([20.0, 500.0])
    engine.enemy_alive[:, 1] = True
    engine.enemy_projectile_x[:, 0] = -42
    engine.enemy_projectile_y[:, 0] = 0
    engine.enemy_projectile_z[:, 0] = 32
    engine.enemy_projectile_velocity_x[:, 0] = 15
    engine.enemy_projectile_alive[:, 0] = True
    active = torch.ones(2, dtype=torch.bool)

    engine._enemy_projectile_tick(active)
    engine._enemy_projectile_tick(active)

    assert not torch.any(engine.enemy_projectile_alive[:, 0])
    assert engine.enemy_health[:, 1].tolist() == [0.0, 500.0]
    assert engine.enemy_alive[:, 1].tolist() == [False, True]
    assert engine.enemy_death_type[:, 1].tolist() == [0, -1]
    assert engine.drop_type[:, 1].tolist() == [2007, -1]
    # Hell Knight species absorb their own Baron balls without damage, and
    # an infighting kill never belongs to the controlled player's counters.
    assert engine.enemy_health[:, 0].tolist() == [500.0, 500.0]
    assert engine.killcount.tolist() == [0, 0]


def test_monster_damage_switches_targets_until_retaliation_threshold_expires(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.enemy_type[:, :3] = torch.tensor([0, 1, 4])
    engine.enemy_health[:, :3] = 150
    engine.enemy_alive[:, :3] = True
    engine._enemy_pain_chance.zero_()

    first_hit = torch.zeros(
        (engine.num_envs, engine.enemy_slots, engine.enemy_slots)
    )
    first_hit[:, 0, 2] = 1
    engine._apply_enemy_damage(
        torch.sum(first_hit, dim=1),
        pain_override=torch.zeros_like(engine.enemy_alive),
        credit_player=False,
        attacker_is_player=False,
        monster_damage_by_source=first_hit,
    )

    assert engine.enemy_target_slot[:, 2].tolist() == [0, 0]
    assert engine.enemy_target_threshold[:, 2].tolist() == [100, 100]

    competing_hit = torch.zeros_like(first_hit)
    competing_hit[:, 1, 2] = 1
    engine._apply_enemy_damage(
        torch.sum(competing_hit, dim=1),
        pain_override=torch.zeros_like(engine.enemy_alive),
        credit_player=False,
        attacker_is_player=False,
        monster_damage_by_source=competing_hit,
    )
    assert engine.enemy_target_slot[:, 2].tolist() == [0, 0]
    assert engine.enemy_target_threshold[:, 2].tolist() == [100, 100]

    engine.enemy_target_threshold[:, 2].zero_()
    engine._apply_enemy_damage(
        torch.sum(competing_hit, dim=1),
        pain_override=torch.zeros_like(engine.enemy_alive),
        credit_player=False,
        attacker_is_player=False,
        monster_damage_by_source=competing_hit,
    )
    assert engine.enemy_target_slot[:, 2].tolist() == [1, 1]
    assert engine.enemy_target_threshold[:, 2].tolist() == [100, 100]

    player_hit = torch.zeros_like(engine.enemy_health)
    player_hit[:, 2] = 1
    engine._apply_enemy_damage(
        player_hit,
        pain_override=torch.zeros_like(engine.enemy_alive),
    )
    assert engine.enemy_target_slot[:, 2].tolist() == [1, 1]
    engine.enemy_target_threshold[:, 2].zero_()
    engine._apply_enemy_damage(
        player_hit,
        pain_override=torch.zeros_like(engine.enemy_alive),
    )
    assert engine.enemy_target_slot[:, 2].tolist() == [-1, -1]
    assert engine.enemy_target_threshold[:, 2].tolist() == [100, 100]


def test_retaliating_monster_pursues_its_attacker_instead_of_player(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(-200)
    engine.y.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 4
    engine.enemy_health[:, 0] = 150
    engine.enemy_alive[:, 0] = True
    engine.enemy_pain_tics[:, 0] = 10
    engine.enemy_x[:, 1] = 0
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_type[:, 1] = 0
    engine.enemy_health[:, 1] = 20
    engine.enemy_alive[:, 1] = True
    engine.enemy_target_slot[:, 1] = 0
    engine.enemy_target_threshold[:, 1] = 100
    engine.enemy_cooldown[:, 1] = 999
    engine.enemy_move_cooldown[:, 1] = 0

    engine._enemy_tick()

    assert torch.all(engine.enemy_x[:, 1] > 0)
    assert engine.enemy_target_slot[:, 1].tolist() == [0, 0]
    assert engine.enemy_target_threshold[:, 1].tolist() == [99, 99]


def test_monster_chase_uses_doom_discrete_direction_and_gradual_turn(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(100)
    engine.y.fill_(-100)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_angle[:, 0] = math.radians(181.40625)
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -1
    engine.enemy_move_direction[:, 0] = 0
    engine.enemy_move_count[:, 0] = 0
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_cooldown[:, 0] = 999

    engine._enemy_tick()

    expected_step = 8 * 46341 / 65536
    assert torch.allclose(
        engine.enemy_x[:, 0],
        torch.full((2,), expected_step),
    )
    assert torch.allclose(
        engine.enemy_y[:, 0],
        torch.full((2,), -expected_step),
    )
    assert engine.enemy_move_direction[:, 0].tolist() == [7, 7]
    assert torch.allclose(
        torch.rad2deg(engine.enemy_angle[:, 0]),
        torch.full((2,), -135.0),
        atol=1e-4,
    )


def test_monster_hitscan_damages_and_angers_intervening_monster(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.fill_(100)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 4
    engine.enemy_health[:, 0] = 150
    engine.enemy_alive[:, 0] = True
    engine.enemy_pain_tics[:, 0] = 10
    engine.enemy_x[:, 1] = 100
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_type[:, 1] = 0
    engine.enemy_health[:, 1] = 20
    engine.enemy_alive[:, 1] = True
    engine.enemy_target_slot[:, 1] = 0
    engine.enemy_target_threshold[:, 1] = 100
    engine.enemy_attack_phase[:, 1] = 1
    engine.enemy_cooldown[:, 1] = 1
    engine._enemy_pain_chance.zero_()

    engine._enemy_tick()

    assert torch.all(engine.enemy_health[:, 0] < 150)
    assert engine.health.tolist() == [100.0, 100.0]
    assert engine.enemy_target_slot[:, 0].tolist() == [1, 1]
    assert engine.enemy_target_threshold[:, 0].tolist() == [100, 100]


def test_monster_melee_damages_target_monster_instead_of_player(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.fill_(100)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.enemy_pain_tics[:, 0] = 10
    engine.enemy_x[:, 1] = 32
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_type[:, 1] = 4
    engine.enemy_health[:, 1] = 150
    engine.enemy_alive[:, 1] = True
    engine.enemy_target_slot[:, 1] = 0
    engine.enemy_target_threshold[:, 1] = 100
    engine.enemy_attack_phase[:, 1] = 1
    engine.enemy_cooldown[:, 1] = 1
    engine._enemy_pain_chance.zero_()

    engine._enemy_tick()

    assert torch.all(engine.enemy_health[:, 0] < 100)
    assert engine.health.tolist() == [100.0, 100.0]
    assert engine.enemy_target_slot[:, 0].tolist() == [1, 1]


def test_hell_knight_aims_baron_ball_at_monster_target(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.fill_(100)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 4
    engine.enemy_health[:, 0] = 150
    engine.enemy_alive[:, 0] = True
    engine.enemy_pain_tics[:, 0] = 100
    engine.enemy_x[:, 1] = 100
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_type[:, 1] = 5
    engine.enemy_health[:, 1] = 500
    engine.enemy_alive[:, 1] = True
    engine.enemy_target_slot[:, 1] = 0
    engine.enemy_target_threshold[:, 1] = 100
    engine.enemy_attack_phase[:, 1] = 1
    engine.enemy_cooldown[:, 1] = 1
    engine._enemy_pain_chance.zero_()

    engine._enemy_tick()

    assert torch.all(engine.enemy_projectile_velocity_x[:, 1] < 0)
    assert torch.all(engine.enemy_projectile_velocity_y[:, 1] == 0)
    assert engine.health.tolist() == [100.0, 100.0]
    for _ in range(8):
        engine._enemy_projectile_tick(torch.ones(2, dtype=torch.bool))

    assert torch.all(engine.enemy_health[:, 0] < 150)
    assert engine.enemy_target_slot[:, 0].tolist() == [1, 1]
    assert engine.enemy_target_threshold[:, 0].tolist() == [100, 100]


def test_monster_reacquires_player_after_infighting_target_dies(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 1] = 100
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_type[:, 1] = 0
    engine.enemy_health[:, 1] = 20
    engine.enemy_alive[:, 1] = True
    engine.enemy_target_slot[:, 1] = 0
    engine.enemy_target_threshold[:, 1] = 50
    engine.enemy_cooldown[:, 1] = 999
    engine.enemy_move_cooldown[:, 1] = 0

    engine._enemy_tick()

    assert engine.enemy_target_slot[:, 1].tolist() == [-1, -1]
    assert engine.enemy_target_threshold[:, 1].tolist() == [0, 0]
    assert torch.all(engine.enemy_x[:, 1] > 100)


def test_enemy_projectile_only_passes_corpse_after_no_block_frame(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.fill_(200)
    engine.z.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_type.fill_(-1)
    engine.enemy_x[:, 1] = 0
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_death_type[:, 1] = 0
    engine.enemy_death_tics[:, 1] = 21
    engine.enemy_death_elapsed[:, 1] = torch.tensor([0, 10])
    engine.enemy_projectile_x[:, 0] = -42
    engine.enemy_projectile_y[:, 0] = 0
    engine.enemy_projectile_z[:, 0] = 32
    engine.enemy_projectile_velocity_x[:, 0] = 15
    engine.enemy_projectile_alive[:, 0] = True
    active = torch.ones(2, dtype=torch.bool)

    engine._enemy_projectile_tick(active)
    engine._enemy_projectile_tick(active)

    assert engine.enemy_projectile_alive[:, 0].tolist() == [False, True]
    assert engine.enemy_projectile_impact_tics[:, 0].tolist() == [18, 0]
    assert engine.enemy_projectile_x[:, 0].tolist() == [-27.0, -12.0]


def test_monster_hitscan_hits_intervening_voodoo_doll_without_player_thrust(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(-200)
    engine.y.fill_(-128)
    engine.z.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = -128
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 1

    engine._enemy_tick()

    # The first static Player 1 body at (-128, -128) intercepts both spread
    # bullets before they can reach the controlled body at (-200, -128).
    assert engine.health.tolist() == [91.0, 94.0]
    assert torch.equal(engine.momentum_x, torch.zeros(2))
    assert torch.equal(engine.momentum_y, torch.zeros(2))
    assert engine.enemy_attack_phase[:, 0].tolist() == [2, 2]


def test_reference_damage_and_pickup_flash_counters(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine._apply_player_damage(torch.tensor([10.0, 25.0]))

    assert engine.damage_count.tolist() == [10, 25]

    pickup_scenario = replace(
        square_scenario,
        item_spawns=np.asarray([(0, 0, 0)], dtype=np.float32),
        item_types=np.asarray([2014], dtype=np.int32),
    )
    pickup_engine = _engine(pickup_scenario)
    pickup_engine.x.zero_()
    pickup_engine.y.zero_()
    pickup_engine.z.zero_()
    pickup_engine._collect_map_items()

    assert pickup_engine.bonus_count.tolist() == [6, 6]


def test_player_damage_thrust_matches_vizdoom_fixed_point_oracle(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.momentum_x.zero_()
    engine.momentum_y.zero_()
    engine.armor[1] = 200
    engine.armor_save_fraction[1] = 0.5
    attacker_x = torch.full((2,), -58.794921875)
    attacker_y = torch.full((2,), 37.7698974609375)

    engine._apply_player_damage(
        torch.full((2,), 12.0),
        attacker_x,
        attacker_y,
    )

    assert engine.health.tolist() == [88.0, 94.0]
    assert engine.armor.tolist() == [0.0, 194.0]
    assert torch.equal(engine.momentum_x, torch.full((2,), 1.261688232421875))
    assert torch.equal(engine.momentum_y, torch.full((2,), -0.81024169921875))

    engine.reaction_time.zero_()
    engine._move_player(torch.zeros((2, 20), dtype=torch.bool))

    assert torch.equal(engine.x, torch.full((2,), 1.261688232421875))
    assert torch.equal(engine.y, torch.full((2,), -0.81024169921875))
    assert torch.equal(engine.momentum_x, torch.full((2,), 1.143402099609375))
    assert torch.equal(engine.momentum_y, torch.full((2,), -0.734283447265625))


def test_damage_thrust_uses_doom_integer_angle_lookup(square_scenario) -> None:
    engine = _engine(square_scenario)

    fine_angle = engine._doom_fine_angle(
        torch.full((2,), 137713, dtype=torch.int64),
        torch.full((2,), -640728, dtype=torch.int64),
    )

    # This matched rocket-blast vector is the boundary case where atan2
    # selects fine-angle bin 6420 instead of R_PointToAngle2's bin 6419.
    assert fine_angle.tolist() == [6419, 6419]


def test_simultaneous_hits_preserve_each_thrust_and_armor_rounding(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.momentum_x.zero_()
    engine.momentum_y.zero_()
    engine.armor.fill_(10)
    engine.armor_save_fraction.fill_(0.5)
    damage_by_source = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    attacker_x = torch.tensor([[-64.0, 0.0], [-64.0, 0.0]])
    attacker_y = torch.tensor([[0.0, -64.0], [0.0, -64.0]])
    thrust_x, thrust_y = engine._player_damage_thrust_components(
        damage_by_source,
        attacker_x,
        attacker_y,
    )

    engine._apply_player_damage(
        torch.sum(damage_by_source, dim=1),
        attacker_x[:, 0],
        attacker_y[:, 0],
        thrust_x_fixed=torch.sum(thrust_x, dim=1),
        thrust_y_fixed=torch.sum(thrust_y, dim=1),
        armor_absorb_request=torch.sum(
            torch.floor(damage_by_source * engine.armor_save_fraction[:, None]),
            dim=1,
        ),
    )

    assert engine.health.tolist() == [96.0, 96.0]
    assert engine.armor.tolist() == [8.0, 8.0]
    assert torch.equal(engine.momentum_x, torch.full((2,), 0.375))
    assert torch.equal(engine.momentum_y, torch.full((2,), 0.3749847412109375))


def test_pistol_and_chaingun_views_share_bullet_ammo(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.selected_weapon.fill_(2)
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)
    _finish_pending_attack(engine)

    assert engine.ammo[:, 1].tolist() == [49.0, 49.0]
    assert torch.equal(engine.ammo[:, 1], engine.ammo[:, 3])


def test_reference_teleport_lock_and_turn_rate(square_scenario) -> None:
    engine = _engine(square_scenario)
    initial = engine.angle.clone()
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 8] = True
    for _ in range(7):
        engine._move_player(buttons)
    assert torch.equal(engine.angle, initial)

    engine._move_player(buttons)

    expected = torch.remainder(initial + torch.deg2rad(torch.tensor(3.515625)), 2 * torch.pi)
    assert torch.allclose(engine.angle, expected)


def test_reference_forward_acceleration_and_right_strafe_basis(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.reaction_time.zero_()
    engine.angle.zero_()
    forward = torch.zeros((2, 20), dtype=torch.bool)
    forward[:, 6] = True
    before_x = engine.x.clone()

    engine._move_player(forward)

    assert torch.allclose(engine.x - before_x, torch.full((2,), 0.78125))
    assert torch.allclose(engine.momentum_x, torch.full((2,), 0.7080078125))
    before_y = engine.y.clone()
    engine.momentum_x.zero_()
    right = torch.zeros((2, 20), dtype=torch.bool)
    right[:, 3] = True
    engine._move_player(right)
    assert torch.allclose(engine.y - before_y, torch.full((2,), -0.75))
    assert torch.allclose(engine.momentum_y, torch.full((2,), -0.6796875))


def test_reference_air_control_and_air_friction(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.reaction_time.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.fill_(1.0)
    engine.angle.zero_()
    engine.momentum_x.zero_()
    engine.momentum_y.zero_()
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 1] = True
    buttons[:, 6] = True

    engine._move_player(buttons)

    expected = torch.full((2,), 400.0 / 65536.0)
    assert torch.equal(engine.x, expected)
    assert torch.equal(engine.momentum_x, expected)
    assert torch.equal(engine.y, torch.zeros(2))
    assert torch.equal(engine.momentum_y, torch.zeros(2))


def test_player_wall_collision_uses_doom_square_corner(square_scenario) -> None:
    scenario = replace(
        square_scenario,
        blocking_segments=np.asarray([(32.0, -64.0, 32.0, 0.0)], dtype=np.float32),
    )
    engine = _engine(scenario)

    assert torch.all(engine._points_collide(torch.full((2,), 16.1), torch.full((2,), 8.0)))
    assert not torch.any(
        engine._points_collide(torch.full((2,), 16.1), torch.full((2,), 16.0))
    )


def test_player_actor_collision_uses_doom_square_corner(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.enemy_x[:, 0] = 32.0
    engine.enemy_y[:, 0] = 32.0
    engine.enemy_z[:, 0] = 0.0
    engine.enemy_type[:, 0] = 0
    engine.enemy_alive[:, 0] = True

    assert torch.all(engine._player_collides(engine.x, engine.y))


def test_reference_gravity_trace_lands_on_lowered_floor(square_scenario) -> None:
    lowered = replace(
        square_scenario,
        sector_heights=np.asarray([(-64, 128)], dtype=np.float32),
    )
    engine = _engine(lowered)
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    active = torch.ones(2, dtype=torch.bool)
    z_trace = []
    velocity_trace = []
    view_z_trace = []

    for _ in range(12):
        engine._vertical_player_tick(active)
        z_trace.append(float(engine.z[0]))
        velocity_trace.append(float(engine.velocity_z[0]))
        view_z_trace.append(float(engine.view_z[0]))

    assert z_trace == [
        0.0,
        -1.0,
        -3.0,
        -6.0,
        -10.0,
        -15.0,
        -21.0,
        -28.0,
        -36.0,
        -45.0,
        -55.0,
        -64.0,
    ]
    assert velocity_trace == [
        -1.0,
        -2.0,
        -3.0,
        -4.0,
        -5.0,
        -6.0,
        -7.0,
        -8.0,
        -9.0,
        -10.0,
        -11.0,
        0.0,
    ]
    for _ in range(12):
        engine._vertical_player_tick(active)
        view_z_trace.append(float(engine.view_z[0]))
    assert view_z_trace == [
        41.0,
        41.0,
        40.0,
        38.0,
        35.0,
        31.0,
        26.0,
        20.0,
        13.0,
        5.0,
        -4.0,
        -14.0,
        -24.375,
        -25.5,
        -26.375,
        -27.0,
        -27.375,
        -27.5,
        -27.375,
        -27.0,
        -26.375,
        -25.5,
        -24.375,
        -23.0,
    ]


def test_reference_double_gravity_when_walking_off_ledge(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.z.zero_()
    engine.velocity_z.zero_()
    engine.previous_player_floor_z.zero_()
    engine.player_floor_z.fill_(-64.0)

    engine._vertical_player_tick(torch.ones(2, dtype=torch.bool))

    assert torch.equal(engine.z, torch.zeros(2))
    assert torch.equal(engine.velocity_z, torch.full((2,), -2.0))


def test_player_step_height_limit_is_twenty_four_units(square_scenario) -> None:
    allowed = _engine(
        replace(
            square_scenario,
            sector_heights=np.asarray([(24, 128)], dtype=np.float32),
        )
    )
    blocked = _engine(
        replace(
            square_scenario,
            sector_heights=np.asarray([(25, 128)], dtype=np.float32),
        )
    )
    for engine in (allowed, blocked):
        engine.x.zero_()
        engine.y.zero_()
        engine.z.zero_()

    assert not torch.any(allowed._player_collides(allowed.x, allowed.y))
    assert torch.all(blocked._player_collides(blocked.x, blocked.y))


def test_reset_places_player_on_the_local_sector_floor(square_scenario) -> None:
    lowered = replace(
        square_scenario,
        sector_heights=np.asarray([(-64, 128)], dtype=np.float32),
    )

    engine = _engine(lowered)

    assert engine.z.tolist() == [-64.0, -64.0]


def test_actor_collision_requires_overlapping_vertical_extents(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_alive[:, 0] = True
    engine.enemy_z[:, 0] = 56

    assert not torch.any(engine._player_collides(engine.x, engine.y))

    engine.enemy_z[:, 0] = 55

    assert torch.all(engine._player_collides(engine.x, engine.y))


def test_hitscan_autoaim_rejects_target_outside_vertical_window(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = 48
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 100
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)
    _finish_pending_attack(engine)

    assert engine.enemy_health[:, 0].tolist() == [20.0, 20.0]

    engine.attack_cooldown.zero_()
    engine.weapon_state_cooldown.zero_()
    engine.enemy_z[:, 0] = 0
    engine._player_attack(buttons)
    _finish_pending_attack(engine)

    assert torch.all(engine.enemy_health[:, 0] < 20)


def test_forward_trace_matches_vizdoom_fixed_point_oracle(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.angle.fill_(torch.deg2rad(torch.tensor(102.16735842222519)))
    engine.momentum_x.zero_()
    engine.momentum_y.zero_()
    engine.reaction_time.fill_(7)
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 6] = True

    for _ in range(10):
        engine._move_player(buttons)

    assert torch.equal(engine.x, torch.full((2,), 835.0191955566406))
    assert torch.equal(engine.y, torch.full((2,), 395.65065002441406))
    assert torch.equal(engine.momentum_x, torch.full((2,), -0.4057769775390625))
    assert torch.equal(engine.momentum_y, torch.full((2,), 1.8876495361328125))


def test_movement_camera_bob_matches_vizdoom_fixed_point_oracle(square_scenario) -> None:
    engine = TorchDeathmatchEngine(
        square_scenario,
        1,
        device=torch.device("cpu"),
        frame_skip=1,
    )
    engine.reset(torch.ones(1, dtype=torch.bool), torch.tensor([123]))
    engine.x.fill_(835.9440307617188)
    engine.y.fill_(391.3482971191406)
    engine.z.zero_()
    engine.view_z.fill_(41)
    engine.angle.fill_(torch.deg2rad(torch.tensor(102.16735842222519)))
    engine.momentum_x.zero_()
    engine.momentum_y.zero_()
    engine.reaction_time.fill_(7)
    buttons = torch.zeros((1, 20), dtype=torch.bool)
    buttons[:, 6] = True
    view_z_trace = []

    for _ in range(24):
        engine.step(buttons)
        view_z_trace.append(float(engine.view_z[0]))

    assert view_z_trace == [
        41.0,
        41.0,
        41.0,
        41.0,
        41.0,
        41.0,
        41.0,
        41.04481506347656,
        41.08551025390625,
        41.0,
        40.71632385253906,
        40.22947692871094,
        39.60401916503906,
        38.95379638671875,
        38.42231750488281,
        38.15003967285156,
        38.247222900390625,
        38.76953125,
        39.713623046875,
        41.0,
        42.49800109863281,
        44.03582763671875,
        45.41265869140625,
        46.44627380371094,
    ]


def test_wall_contact_uses_reference_slide_residual(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.fill_(230)
    engine.momentum_x.fill_(4)
    engine.momentum_y.fill_(20)
    engine.reaction_time.zero_()
    buttons = torch.zeros((2, 20), dtype=torch.bool)

    fraction = engine._axis_collision_fraction(engine.momentum_x, engine.momentum_y)
    engine._move_player(buttons)

    assert fraction.tolist() == [0.5, 0.5]
    assert torch.equal(engine.x, torch.full((2,), 4.0))
    assert torch.equal(engine.y, torch.full((2,), 240.0))
    assert torch.equal(engine.momentum_x, torch.full((2,), 3.625))
    assert torch.equal(engine.momentum_y, torch.zeros(2))


def test_axis_slide_contact_uses_reference_nearest_fixed_rounding(square_scenario) -> None:
    engine = _engine(square_scenario)
    fixed_unit = 1 << 16
    position_x = torch.zeros(2, dtype=torch.int64)
    position_y = torch.full(
        (2,),
        256 * fixed_unit - 16 * fixed_unit - 2,
        dtype=torch.int64,
    )
    move_x = torch.zeros(2, dtype=torch.int64)
    move_y = torch.full((2,), 3, dtype=torch.int64)

    fraction, horizontal, valid = engine._axis_slide_contact_fixed(
        position_x,
        position_y,
        move_x,
        move_y,
    )

    assert fraction.tolist() == [43691, 43691]
    assert torch.all(horizontal)
    assert torch.all(valid)


def test_player_cannot_move_through_solid_monster(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.momentum_x.fill_(20)
    engine.momentum_y.zero_()
    engine.reaction_time.zero_()
    engine.enemy_x[:, 0] = 40
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True

    engine._move_player(torch.zeros((2, 20), dtype=torch.bool))

    assert engine.x.tolist() == [0.0, 0.0]
    assert engine.momentum_x.tolist() == [0.0, 0.0]


def test_monster_chase_step_does_not_penetrate_player_radius(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 42
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine._enemy_x_fixed[:, 0] = 42 * 65536
    requested = torch.zeros_like(engine.enemy_alive)
    requested[:, 0] = True
    direction = torch.full_like(engine.enemy_type, 4)

    moved = engine._try_enemy_chase_step(
        requested,
        direction,
        engine.enemy_type.clamp_min(0),
    )

    assert not torch.any(moved[:, 0])
    assert engine.enemy_x[:, 0].tolist() == [42.0, 42.0]


def test_monster_wall_collision_uses_actor_specific_radius(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(400)
    engine.y.zero_()
    engine.enemy_x[:, 0] = 220
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[0, 0] = 4
    engine.enemy_type[1, 0] = 0
    engine.enemy_health[0, 0] = 150
    engine.enemy_health[1, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 999
    engine.enemy_move_cooldown[:, 0] = 0

    engine._enemy_tick()

    assert engine.enemy_x[:, 0].tolist() == [220.0, 228.0]


def test_moving_monsters_treat_other_monsters_as_solid(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_x[:, 1] = 60
    engine.enemy_y[:, :2] = 0
    engine.enemy_type[:, :2] = 0
    engine.enemy_health[:, :2] = 20
    engine.enemy_alive[:, :2] = True
    engine._enemy_x_fixed[:, 0] = 100 * 65536
    engine._enemy_x_fixed[:, 1] = 60 * 65536
    requested = torch.zeros_like(engine.enemy_alive)
    requested[:, 0] = True
    direction = torch.full_like(engine.enemy_type, 4)

    moved = engine._try_enemy_chase_step(
        requested,
        direction,
        engine.enemy_type.clamp_min(0),
    )

    assert not torch.any(moved[:, 0])
    assert engine.enemy_x[:, 0].tolist() == [100.0, 100.0]
    assert engine.enemy_x[:, 1].tolist() == [60.0, 60.0]


def test_solid_corpses_retain_actor_specific_collision_radius(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(-100)
    engine.y.fill_(-100)
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = -1
    engine.enemy_death_type[:, 0] = torch.tensor([4, 0])
    engine.enemy_death_tics[:, 0] = 10
    engine.enemy_death_elapsed[:, 0] = 0

    collision = engine._player_collides(
        torch.full((2,), 45.0),
        torch.zeros(2),
    )

    # Demon radius 30 plus player radius 16 overlaps at distance 45. The
    # zombieman corpse in the second lane retains radius 20 and does not.
    assert collision.tolist() == [True, False]


def test_zombieman_chase_uses_eight_unit_four_tic_cadence(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 100
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 999
    engine.enemy_move_cooldown[:, 0] = 0

    engine._enemy_tick()
    first_x = engine.enemy_x[:, 0].clone()
    first_y = engine.enemy_y[:, 0].clone()
    diagonal_stride = 8.0 / torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(first_x, torch.full((2,), 100.0 - diagonal_stride))
    assert torch.allclose(first_y, torch.full((2,), 100.0 - diagonal_stride))

    for _ in range(3):
        engine._enemy_tick()
    assert torch.equal(engine.enemy_x[:, 0], first_x)
    assert torch.equal(engine.enemy_y[:, 0], first_y)

    engine._enemy_tick()
    assert torch.all(engine.enemy_x[:, 0] < first_x)
    assert torch.all(engine.enemy_y[:, 0] < first_y)


def test_zombieman_damage_thrust_matches_vizdoom_fixed_point_oracle(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(2.794921875)
    engine.y.fill_(-37.7698974609375)
    engine.enemy_x[:, 0].zero_()
    engine.enemy_y[:, 0].zero_()
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True

    engine._apply_enemy_damage(
        torch.nn.functional.pad(torch.full((2, 1), 15.0), (0, engine.enemy_slots - 1)),
        engine.x[:, None],
        engine.y[:, None],
    )

    assert torch.equal(
        engine._enemy_momentum_x_fixed[:, 0],
        torch.full((2,), -8944, dtype=torch.int64),
    )
    assert torch.equal(
        engine._enemy_momentum_y_fixed[:, 0],
        torch.full((2,), 122546, dtype=torch.int64),
    )

    engine._move_enemy_thrust(torch.ones(2, dtype=torch.bool))

    assert torch.equal(
        engine._enemy_x_fixed[:, 0],
        torch.full((2,), -8944, dtype=torch.int64),
    )
    assert torch.equal(
        engine._enemy_y_fixed[:, 0],
        torch.full((2,), 122546, dtype=torch.int64),
    )
    assert torch.equal(
        engine._enemy_momentum_x_fixed[:, 0],
        torch.full((2,), -8106, dtype=torch.int64),
    )
    assert torch.equal(
        engine._enemy_momentum_y_fixed[:, 0],
        torch.full((2,), 111057, dtype=torch.int64),
    )


def test_reference_missile_distance_thresholds(square_scenario) -> None:
    engine = _engine(square_scenario)
    enemy_type = torch.tensor([[0, 0, 5, 5]])
    dx = torch.tensor([[128.0, 1_000.0, 128.0, 1_000.0]])
    dy = torch.zeros_like(dx)

    threshold = engine._enemy_missile_threshold(enemy_type, dx, dy)

    assert threshold.tolist() == [[0.0, 200.0, 64.0, 200.0]]


def test_monster_attacks_instead_of_moving_on_chase_tic(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0
    engine.enemy_move_cooldown[:, 0] = 0

    engine._enemy_tick()

    assert engine.enemy_x[:, 0].tolist() == [100.0, 100.0]
    assert engine.health.tolist() == [100.0, 100.0]
    assert engine.enemy_attack_phase[:, 0].tolist() == [1, 1]
    assert engine.enemy_just_attacked[:, 0].tolist() == [True, True]
    assert engine.enemy_cooldown[:, 0].tolist() == [10, 10]
    for _ in range(9):
        engine._enemy_tick()
    assert engine.health.tolist() == [100.0, 100.0]
    assert engine.enemy_x[:, 0].tolist() == [100.0, 100.0]

    engine._enemy_tick()

    assert torch.all(engine.health < 100)
    assert engine.enemy_attack_phase[:, 0].tolist() == [2, 2]
    assert engine.enemy_cooldown[:, 0].tolist() == [16, 16]


def test_monster_forces_new_chase_direction_after_ranged_attack(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine._enemy_x_fixed[:, 0] = 100 * 65536
    engine._enemy_y_fixed[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_attack_phase[:, 0] = 0
    engine.enemy_just_attacked[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_move_direction[:, 0] = 8
    engine.enemy_move_count[:, 0] = 15

    engine._enemy_tick()

    assert engine.enemy_x[:, 0].tolist() == [92.0, 92.0]
    assert engine.enemy_y[:, 0].tolist() == [0.0, 0.0]
    assert engine.enemy_move_direction[:, 0].tolist() == [4, 4]
    assert torch.all(engine.enemy_move_count[:, 0] <= 15)
    assert engine.enemy_attack_phase[:, 0].tolist() == [0, 0]
    assert engine.enemy_just_attacked[:, 0].tolist() == [False, False]


def test_monster_faces_target_at_prefire_and_hitscan_action(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_angle[:, 0] = math.pi / 2
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0

    engine._enemy_tick()

    assert torch.allclose(
        engine.enemy_angle[:, 0],
        torch.full((2,), math.pi),
    )

    engine.y.fill_(100)
    engine.enemy_cooldown[:, 0] = 1
    engine._enemy_tick()

    assert torch.allclose(
        engine.enemy_angle[:, 0],
        torch.full((2,), 3.0 * math.pi / 4.0),
    )


def test_chainsaw_marine_repeats_four_tic_attack_cycle(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 2
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0

    engine._enemy_tick()
    for _ in range(3):
        engine._enemy_tick()
    assert engine.health.tolist() == [100.0, 100.0]

    engine._enemy_tick()
    after_first_hit = engine.health.clone()
    assert torch.all(after_first_hit < 100)
    assert engine.enemy_cooldown[:, 0].tolist() == [4, 4]

    for _ in range(4):
        engine._enemy_tick()
    assert torch.all(engine.health < after_first_hit)


def test_chaingunner_uses_prefire_and_alternating_burst_gaps(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 3
    engine.enemy_health[:, 0] = 70
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0

    engine._enemy_tick()
    for _ in range(9):
        engine._enemy_tick()
    assert engine.health.tolist() == [100.0, 100.0]

    engine._enemy_tick()
    after_first_shot = engine.health.clone()
    assert torch.all(after_first_shot < 100)
    for _ in range(4):
        engine._enemy_tick()
    after_second_shot = engine.health.clone()
    assert torch.all(after_second_shot < after_first_shot)
    for _ in range(5):
        engine._enemy_tick()
    # The third action occurs on schedule but its spread bullet can miss.
    assert torch.all(engine.health <= after_second_shot)
    assert engine.enemy_attack_phase[:, 0].tolist() == [2, 2]
    assert engine.enemy_cooldown[:, 0].tolist() == [4, 4]


def test_monster_hitscan_uses_independent_reference_pellets(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = torch.tensor([0, 1])
    engine.enemy_alive[:, 0] = True
    engine.rng_state.copy_(torch.tensor([12345, 67890]))
    fires = torch.zeros_like(engine.enemy_alive)
    fires[:, 0] = True
    distance = torch.sqrt(
        (engine.x[:, None] - engine.enemy_x) ** 2
        + (engine.y[:, None] - engine.enemy_y) ** 2
    ).clamp_min(1e-4)

    damage, _actual_player_damage, _enemy_damage = engine._enemy_hitscan_damage(
        engine.enemy_type.clamp_min(0),
        fires,
        distance,
        torch.ones_like(engine.enemy_alive),
    )

    assert torch.count_nonzero(damage[0, 0]).item() == 1
    # The shotgun guy rolls three distinct pellets; the wide third pellet
    # misses the player's Doom-compatible diagonal at this distance.
    assert damage[1, 0].tolist() == [6.0, 3.0, 0.0]


def test_monster_spread_pellet_traces_its_own_blocking_linedef(
    square_scenario,
) -> None:
    off_axis_wall = np.asarray([(50.0, 1.0, 50.0, 6.0)], dtype=np.float32)
    walls = np.concatenate((square_scenario.wall_segments, off_axis_wall), axis=0)
    blocked_scenario = replace(
        square_scenario,
        wall_segments=walls,
        blocking_segments=walls.copy(),
        blocking_wall_indices=np.arange(5, dtype=np.int32),
        wall_texture_ids=np.zeros(5, dtype=np.int32),
        wall_texture_offsets=np.zeros((5, 2), dtype=np.float32),
        wall_side_texture_ids=np.concatenate(
            (
                np.zeros((5, 1, 1), dtype=np.int32),
                np.full((5, 1, 1), -1, dtype=np.int32),
            ),
            axis=1,
        ).repeat(3, axis=2),
        wall_side_texture_offsets=np.zeros((5, 2, 2), dtype=np.float32),
        wall_sectors=np.zeros((5, 2), dtype=np.int32),
        sector_edge_mask=np.ones((1, 5), dtype=np.bool_),
    )

    def shotgun_damage(scenario) -> torch.Tensor:
        engine = _engine(scenario)
        engine.x.zero_()
        engine.y.zero_()
        engine.z.zero_()
        engine.enemy_alive.zero_()
        engine.enemy_x[:, 0] = 100
        engine.enemy_y[:, 0] = 0
        engine.enemy_z[:, 0] = 0
        engine.enemy_type[:, 0] = 1
        engine.enemy_alive[:, 0] = True
        fires = torch.zeros_like(engine.enemy_alive)
        fires[:, 0] = True
        distance = torch.sqrt(
            (engine.x[:, None] - engine.enemy_x) ** 2
            + (engine.y[:, None] - engine.enemy_y) ** 2
        ).clamp_min(1e-4)
        visible = ~engine._sight_blocked(
            engine.enemy_x,
            engine.enemy_y,
            engine.enemy_z + 42.0,
            engine.x[:, None],
            engine.y[:, None],
            engine.z[:, None],
            torch.full_like(engine.z[:, None], 56.0),
        )
        assert torch.all(visible[:, 0])
        damage, _actual_player_damage, _enemy_damage = engine._enemy_hitscan_damage(
            engine.enemy_type.clamp_min(0),
            fires,
            distance,
            visible,
        )
        return damage[:, 0]

    clear_damage = shotgun_damage(square_scenario)
    blocked_damage = shotgun_damage(blocked_scenario)

    assert clear_damage.tolist() == [
        [9.0, 0.0, 15.0],
        [6.0, 3.0, 12.0],
    ]
    # The first pellet crosses the short off-axis wall in both lanes while
    # the center aim ray and the other player-bound pellets remain clear.
    assert blocked_damage.tolist() == [
        [0.0, 0.0, 15.0],
        [0.0, 3.0, 12.0],
    ]


def test_blocking_linedef_occludes_hitscan_and_monster_attacks(square_scenario) -> None:
    divider = np.asarray([(0, -256, 0, 256)], dtype=np.float32)
    walls = np.concatenate((square_scenario.wall_segments, divider), axis=0)
    divided = replace(
        square_scenario,
        wall_segments=walls,
        blocking_segments=walls.copy(),
        blocking_wall_indices=np.arange(5, dtype=np.int32),
        wall_texture_ids=np.zeros(5, dtype=np.int32),
        wall_texture_offsets=np.zeros((5, 2), dtype=np.float32),
        wall_side_texture_ids=np.concatenate(
            (
                np.zeros((5, 1, 1), dtype=np.int32),
                np.full((5, 1, 1), -1, dtype=np.int32),
            ),
            axis=1,
        ).repeat(3, axis=2),
        wall_side_texture_offsets=np.zeros((5, 2, 2), dtype=np.float32),
        wall_sectors=np.zeros((5, 2), dtype=np.int32),
        sector_edge_mask=np.ones((1, 5), dtype=np.bool_),
    )
    engine = _engine(divided)
    engine.x.fill_(-32)
    engine.y.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)
    _finish_pending_attack(engine)
    engine._enemy_tick()

    assert engine.enemy_health[:, 0].tolist() == [20.0, 20.0]
    assert engine.health.tolist() == [100.0, 100.0]


def test_two_sided_portal_does_not_occlude_hitscan_or_monster_sight(square_scenario) -> None:
    portal = np.asarray([(0, -256, 0, 256)], dtype=np.float32)
    walls = np.concatenate((square_scenario.wall_segments, portal), axis=0)
    scenario = replace(
        square_scenario,
        wall_segments=walls,
        wall_texture_ids=np.zeros(5, dtype=np.int32),
        wall_texture_offsets=np.zeros((5, 2), dtype=np.float32),
        wall_side_texture_ids=np.concatenate(
            (
                np.zeros((5, 1, 1), dtype=np.int32),
                np.full((5, 1, 1), -1, dtype=np.int32),
            ),
            axis=1,
        ).repeat(3, axis=2),
        wall_side_texture_offsets=np.zeros((5, 2, 2), dtype=np.float32),
        wall_sectors=np.zeros((5, 2), dtype=np.int32),
        sector_edge_mask=np.ones((1, 5), dtype=np.bool_),
    )
    engine = _engine(scenario)
    engine.x.fill_(-32)
    engine.y.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)
    _finish_pending_attack(engine)

    assert torch.all(engine.enemy_health[:, 0] < 20)
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_pain_tics[:, 0] = 0
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 1
    engine._enemy_tick()
    assert torch.all(engine.health < 100)


def test_height_transition_portal_clips_sight_from_deep_pit(square_scenario) -> None:
    portal = np.asarray([(0, -256, 0, 256)], dtype=np.float32)
    walls = np.concatenate((square_scenario.wall_segments, portal), axis=0)
    scenario = replace(
        square_scenario,
        wall_segments=walls,
        wall_texture_ids=np.zeros(5, dtype=np.int32),
        wall_texture_offsets=np.zeros((5, 2), dtype=np.float32),
        wall_side_texture_ids=np.concatenate(
            (
                np.zeros((5, 1, 1), dtype=np.int32),
                np.full((5, 1, 1), -1, dtype=np.int32),
            ),
            axis=1,
        ).repeat(3, axis=2),
        wall_side_texture_offsets=np.zeros((5, 2, 2), dtype=np.float32),
        wall_sectors=np.asarray(
            [[0, -1], [0, -1], [1, -1], [1, -1], [0, 1]],
            dtype=np.int32,
        ),
        sector_edge_mask=np.ones((2, 5), dtype=np.bool_),
        sector_heights=np.asarray([(-128, 128), (0, 128)], dtype=np.float32),
        sector_lights=np.asarray([192, 192], dtype=np.int16),
        sector_floor_texture_ids=np.zeros(2, dtype=np.int32),
        sector_ceiling_texture_ids=np.zeros(2, dtype=np.int32),
    )
    engine = _engine(scenario)
    engine.x.fill_(-64)
    engine.y.zero_()
    engine.z.copy_(torch.tensor([-128.0, -64.0]))
    engine.angle.zero_()
    engine.enemy_x[:, 0] = 64
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)
    _finish_pending_attack(engine)

    # At z=-128 the portal's floor clips the entire target sight cone. At
    # z=-64, the upper part of the zombieman remains visible over the rim.
    assert engine.enemy_health[0, 0].item() == 20
    assert engine.enemy_health[1, 0].item() < 20

    rocket_blocked = engine._rocket_splash_blocked(
        torch.full((2, 1), -64.0),
        torch.zeros((2, 1)),
        torch.tensor([[-128.0], [-32.0]]),
        torch.full((2, 1), 64.0),
        torch.zeros((2, 1)),
        torch.zeros((2, 1)),
        torch.full((2, 1), 56.0),
    )
    assert rocket_blocked[:, 0, 0].tolist() == [True, False]


def test_reference_monster_damage_distributions(square_scenario) -> None:
    engine = _engine(square_scenario)
    enemy_type = torch.arange(6).repeat(2, 1)
    attacks = torch.ones((2, 6), dtype=torch.bool)
    distance = torch.full((2, 6), 128.0)
    distance[:, 4] = 32
    padded_types = torch.zeros((2, engine.enemy_slots), dtype=torch.int64)
    padded_attacks = torch.zeros((2, engine.enemy_slots), dtype=torch.bool)
    padded_distance = torch.full((2, engine.enemy_slots), 128.0)
    padded_types[:, :6] = enemy_type
    padded_attacks[:, :6] = attacks
    padded_distance[:, :6] = distance

    damage = engine._enemy_damage_roll(
        padded_types,
        padded_attacks,
        padded_distance,
    )[:, :6]

    lower = torch.tensor((3, 9, 2, 3, 4, 8))
    upper = torch.tensor((15, 45, 20, 15, 40, 64))
    divisor = torch.tensor((3, 3, 2, 3, 4, 8))
    assert torch.all(damage >= lower)
    assert torch.all(damage <= upper)
    assert torch.all(torch.remainder(damage, divisor) == 0)


def test_hell_knight_ranged_attack_travels_before_damage(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 64
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0

    engine._enemy_tick()

    assert engine.health.tolist() == [100.0, 100.0]
    assert not torch.any(engine.enemy_projectile_alive)
    for _ in range(15):
        engine._enemy_tick()
    assert not torch.any(engine.enemy_projectile_alive)

    engine._enemy_tick()

    assert torch.sum(engine.enemy_projectile_alive, dim=1).tolist() == [1, 1]
    assert engine.enemy_projectile_x[:, 0].tolist() == [56.5, 56.5]
    active = torch.ones(2, dtype=torch.bool)
    for _ in range(8):
        engine._enemy_projectile_tick(active)

    assert torch.all(engine.health < 100)
    assert not torch.any(engine.enemy_projectile_alive)
    assert torch.all(engine.enemy_projectile_impact_tics[:, 0] > 0)
    assert torch.all(engine.enemy_projectile_impact_tics[:, 0] <= 18)


def test_hell_knight_projectile_quantizes_velocity_before_half_step(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 50
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    engine.x.copy_(torch.tensor([0.0, 160.0]))
    engine.y.copy_(torch.tensor([0.0, -10.0]))
    engine.z.copy_(torch.tensor([40.0, -24.0]))
    dx = engine.x[:, None] - engine.enemy_x
    dy = engine.y[:, None] - engine.enemy_y
    requested = torch.zeros_like(engine.enemy_alive)
    requested[:, 0] = True

    engine._spawn_enemy_projectiles(requested, dx, dy)

    velocity_fixed = torch.round(
        torch.stack(
            (
                engine.enemy_projectile_velocity_x[:, 0],
                engine.enemy_projectile_velocity_y[:, 0],
                engine.enemy_projectile_velocity_z[:, 0],
            ),
            dim=1,
        )
        * 65536
    ).to(torch.int64)
    position_fixed = torch.round(
        torch.stack(
            (
                engine.enemy_projectile_x[:, 0],
                engine.enemy_projectile_y[:, 0],
                engine.enemy_projectile_z[:, 0],
            ),
            dim=1,
        )
        * 65536
    ).to(torch.int64)

    assert velocity_fixed.tolist() == [
        [-827869, -413934, 331147],
        [668873, -668873, -267549],
    ]
    # P_CheckMissileSpawn advances each signed fixed-point component by >> 1;
    # notably, negative odd velocities round down rather than toward zero.
    assert position_fixed.tolist() == [
        [6139665, 3069833, 2262725],
        [6888036, 2942363, 1963377],
    ]


def test_hell_knight_melee_attack_fires_after_reference_prefire(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0

    engine._enemy_tick()

    assert engine.health.tolist() == [100.0, 100.0]
    for _ in range(15):
        engine._enemy_tick()
    assert engine.health.tolist() == [100.0, 100.0]

    engine._enemy_tick()

    assert torch.all(engine.health < 100)
    assert not torch.any(engine.enemy_projectile_alive)


def test_frame_skip_stops_after_fatal_internal_tic(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.health.fill_(1)
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 4
    engine.enemy_health[:, 0] = 150
    engine.enemy_alive[:, 0] = True
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 1

    _frames, _reward, terminated, truncated = engine.step(torch.zeros((2, 20), dtype=torch.bool))

    assert torch.all(terminated)
    assert not torch.any(truncated)
    assert engine.episode_time.tolist() == [2, 2]
    assert engine.enemy_attack_phase[:, 0].tolist() == [2, 2]
    assert engine.enemy_cooldown[:, 0].tolist() == [8, 8]
