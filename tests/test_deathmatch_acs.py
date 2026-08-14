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


def test_skill_one_halves_player_damage_above_one(square_scenario) -> None:
    engine = TorchDeathmatchEngine(
        square_scenario,
        2,
        device=torch.device("cpu"),
        doom_skill=1,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))

    engine._apply_player_damage(torch.tensor([5.0, 1.0]))

    assert engine.health.tolist() == [98.0, 99.0]


def test_skill_one_adjusts_simultaneous_damage_per_event(square_scenario) -> None:
    engine = TorchDeathmatchEngine(
        square_scenario,
        2,
        device=torch.device("cpu"),
        doom_skill=1,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))
    events = torch.tensor(((3.0, 3.0), (1.0, 5.0)))

    adjusted = engine._skill_adjust_player_damage(events)
    engine._apply_player_damage(
        adjusted.sum(dim=1),
        skill_adjusted=True,
    )

    assert adjusted.tolist() == [[1.0, 1.0], [1.0, 2.0]]
    assert engine.health.tolist() == [98.0, 97.0]


def test_wall_contact_damage_scale_only_applies_at_blocking_geometry(
    square_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        square_scenario,
        2,
        device=torch.device("cpu"),
        wall_contact_damage_scale=0.5,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))
    engine.x[:] = torch.tensor([240.0, 0.0])
    engine.y.zero_()

    scale = engine._wall_contact_enemy_damage_scale()
    assert scale is not None
    assert scale.tolist() == [0.5, 1.0]
    engine._apply_player_damage(torch.full((2,), 10.0), damage_scale=scale)

    assert engine.health.tolist() == [95.0, 90.0]


def test_player_combat_counters_match_vizdoom_game_variables(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive[:, 0] = True
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 10.0
    damage = torch.zeros_like(engine.enemy_health)
    damage[:, 0] = 5.0

    player_reward = engine._apply_enemy_damage(damage)
    infighting_reward = engine._apply_enemy_damage(damage, credit_player=False)

    assert engine.player_hitcount.tolist() == [1, 1]
    assert engine.player_damagecount.tolist() == [5.0, 5.0]
    assert torch.equal(player_reward, torch.zeros(2))
    assert torch.equal(infighting_reward, torch.ones(2))
    assert engine.killcount.tolist() == [1, 1]

    engine.health[0] = -1
    engine.step(torch.zeros((2, 20), dtype=torch.bool))
    assert engine.player_deathcount.tolist() == [1, 0]
    assert engine.signal_buffer[0, 1].item() == 0

    engine.reset(torch.tensor([True, False]), torch.tensor([789, 0]))
    assert engine.player_deathcount.tolist() == [0, 0]
    assert engine.player_hitcount.tolist() == [0, 1]
    assert engine.player_damagecount.tolist() == [0.0, 5.0]


def test_player_damage_taken_counters_match_post_armor_health_damage(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.armor.copy_(torch.tensor([20.0, 4.0]))
    engine.armor_save_fraction.fill_(0.5)

    engine._apply_player_damage(torch.tensor([10.0, 10.0]))
    engine._apply_player_damage(torch.tensor([0.0, 6.0]))

    assert engine.player_hits_taken.tolist() == [1, 2]
    assert engine.player_damage_taken.tolist() == [5.0, 12.0]
    engine._update_signal_buffer()
    signal_indices = {
        name: index for index, name in enumerate(engine.signals())
    }
    assert engine.signal_buffer[:, signal_indices["hits_taken"]].tolist() == [1.0, 2.0]
    assert engine.signal_buffer[:, signal_indices["damage_taken"]].tolist() == [5.0, 12.0]

    engine.reset(torch.tensor([True, False]), torch.tensor([789, 0]))
    assert engine.player_hits_taken.tolist() == [0, 2]
    assert engine.player_damage_taken.tolist() == [0.0, 12.0]


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
    spawned_angle_byte = (
        engine.enemy_angle[engine.enemy_alive] * (256.0 / (2.0 * math.pi))
    )
    assert torch.allclose(spawned_angle_byte, torch.round(spawned_angle_byte))
    assert torch.all(engine.enemy_animation_tics[engine.enemy_alive] == 1)
    assert torch.all(engine.enemy_cooldown[engine.enemy_alive] == 0)
    assert torch.all(engine.enemy_reaction_time[engine.enemy_alive] == 8)
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
    engine._x_fixed.fill_(-100 * 65536)
    engine._y_fixed.zero_()
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


def test_unaware_monster_uses_doom_approximate_distance_behind_it(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.copy_(torch.tensor((-45.0, -42.0)))
    engine.y.copy_(torch.tensor((-45.0, -42.0)))
    engine._x_fixed.copy_(torch.round(engine.x * 65536).to(torch.int64))
    engine._y_fixed.copy_(torch.round(engine.y * 65536).to(torch.int64))
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0].zero_()
    engine.enemy_y[:, 0].zero_()
    engine._enemy_x_fixed[:, 0].zero_()
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_z[:, 0].zero_()
    engine.enemy_angle[:, 0].zero_()
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -2
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_cooldown[:, 0] = 999

    engine._enemy_tick()

    # Both players are behind the east-facing monster and Euclidean-close.
    # Doom's P_AproxDistance is 67.5 at (-45,-45), outside MELEERANGE, but
    # 63 at (-42,-42), which is close enough to be noticed from behind.
    assert engine.enemy_target_slot[:, 0].tolist() == [-2, -1]


def test_player_weapon_noise_wakes_monster_outside_field_of_view(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(-100)
    engine.y.zero_()
    engine._x_fixed.fill_(-100 * 65536)
    engine._y_fixed.zero_()
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
    assert engine.player_hits_taken.tolist() == [0, 0]
    assert engine.player_damage_taken.tolist() == [0.0, 0.0]
    assert torch.all(engine.player_hitcount > 0)
    assert torch.all(engine.player_damagecount > 0)


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
    assert engine.player_hits_taken.tolist() == [0, 0]
    assert engine.player_damage_taken.tolist() == [0.0, 0.0]
    assert engine.player_hitcount.tolist() == [1, 1]
    assert engine.player_damagecount.tolist() == [30.0, 15.0]
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
    assert engine.player_hits_taken.tolist() == [0, 0]
    assert engine.player_damage_taken.tolist() == [0.0, 0.0]
    assert not torch.any(engine.enemy_projectile_alive[:, 0])
    assert engine.enemy_projectile_impact_tics[:, 0].tolist() == [18, 18]
    assert torch.equal(engine.momentum_x, torch.zeros(2))
    assert torch.equal(engine.momentum_y, torch.zeros(2))


def test_enemy_projectile_kill_counts_for_single_player_vizdoom(
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
    # Hell Knight species absorb their own Baron balls without damage. Doom's
    # single-player P_KillMobj path credits every other monster death to player 0.
    assert engine.enemy_health[:, 0].tolist() == [500.0, 500.0]
    assert engine.killcount.tolist() == [1, 0]
    assert engine.infighting_reward.tolist() == [1.0, 0.0]


def test_pooled_enemy_projectile_uses_recorded_owner_for_collision_and_retaliation(
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
    engine.enemy_type[:, 1] = torch.tensor([0, 4])
    engine.enemy_health[:, 1] = 150
    engine.enemy_alive[:, 1] = True
    # Projectile pool slot 1 belongs to monster slot 0. Treating its pool
    # index as its owner would incorrectly make it pass through enemy slot 1.
    engine.enemy_projectile_x[:, 1] = -42
    engine.enemy_projectile_y[:, 1] = 0
    engine.enemy_projectile_z[:, 1] = 32
    engine.enemy_projectile_velocity_x[:, 1] = 15
    engine.enemy_projectile_alive[:, 1] = True
    engine.enemy_projectile_source_slot[:, 1] = 0
    active = torch.ones(2, dtype=torch.bool)

    engine._enemy_projectile_tick(active)
    engine._enemy_projectile_tick(active)

    assert not torch.any(engine.enemy_projectile_alive[:, 1])
    assert torch.all(engine.enemy_health[:, 1] < 150)
    assert engine.enemy_target_slot[:, 1].tolist() == [0, 0]


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


def test_player_retaliation_threshold_decrements_on_chase_actions(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.zero_()
    engine._x_fixed.fill_(200 * 65536)
    engine._y_fixed.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0].zero_()
    engine.enemy_y[:, 0].zero_()
    engine._enemy_x_fixed[:, 0].zero_()
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -1
    engine.enemy_target_threshold[:, 0] = 100
    engine.enemy_cooldown[:, 0] = 999
    engine.enemy_move_cooldown[:, 0] = 0

    engine._enemy_tick()
    assert engine.enemy_target_threshold[:, 0].tolist() == [99, 99]

    engine._enemy_tick()
    assert engine.enemy_target_threshold[:, 0].tolist() == [99, 99]


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

    assert engine.enemy_projectile_source_slot[:, 0].tolist() == [1, 1]
    assert torch.all(engine.enemy_projectile_velocity_x[:, 0] < 0)
    assert torch.all(engine.enemy_projectile_velocity_y[:, 0] == 0)
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
    assert engine.enemy_x[:, 1].tolist() == [100.0, 100.0]


def test_monster_keeps_dead_target_until_chaingun_refire_exits(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_death_type[:, 0] = 0
    engine.enemy_death_tics[:, 0] = 20
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_x[:, 1] = 0
    engine.enemy_y[:, 1] = 0
    engine.enemy_z[:, 1] = 0
    engine.enemy_type[:, 1] = 3
    engine.enemy_health[:, 1] = 70
    engine.enemy_alive[:, 1] = True
    engine.enemy_target_slot[:, 1] = 0
    engine.enemy_target_threshold[:, 1] = 50
    engine.enemy_attack_phase[:, 1] = 2
    engine.enemy_cooldown[:, 1] = 1
    engine.enemy_move_cooldown[:, 1] = 0

    engine._enemy_tick()

    assert engine.enemy_target_slot[:, 1].tolist() == [0, 0]
    assert engine.enemy_attack_phase[:, 1].tolist() == [3, 3]

    engine.enemy_cooldown[:, 1] = 1
    engine._enemy_chaingun_refire_decision = lambda candidates: torch.zeros_like(candidates)
    engine._enemy_tick()

    assert engine.enemy_target_slot[:, 1].tolist() == [0, 0]
    assert engine.enemy_attack_phase[:, 1].tolist() == [0, 0]

    engine._enemy_tick()

    assert engine.enemy_target_slot[:, 1].tolist() == [-1, -1]
    assert engine.enemy_x[:, 1].tolist() == [0.0, 0.0]


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
    engine.enemy_projectile_z[:, 0] = 8
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
    engine._x_fixed.fill_(-200 * 65536)
    engine._y_fixed.fill_(-128 * 65536)
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
    assert engine.player_hits_taken.tolist() == [0, 0]
    assert engine.player_damage_taken.tolist() == [0.0, 0.0]
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
    assert engine.turn_held_tics.tolist() == [7, 7]

    engine._move_player(buttons)

    expected = torch.remainder(initial + torch.deg2rad(torch.tensor(3.515625)), 2 * torch.pi)
    assert torch.allclose(engine.angle, expected)
    assert engine.turn_held_tics.tolist() == [8, 8]


def test_reference_keyboard_turn_uses_five_tic_slow_start(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.reaction_time.zero_()
    engine.angle.zero_()
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 8] = True
    buttons[1, 1] = True
    actual_angle = []
    actual_bam = []

    for _ in range(8):
        engine._move_player(buttons)
        actual_angle.append(engine.angle.clone())
        actual_bam.append(engine._angle_bam.clone())

    slow_yaw = 320
    walk_yaw = 640
    run_yaw = 1280
    expected_yaw = torch.tensor(
        [
            [slow_yaw * min(tic, 5) + walk_yaw * max(tic - 5, 0) for tic in range(1, 9)],
            [slow_yaw * min(tic, 5) + run_yaw * max(tic - 5, 0) for tic in range(1, 9)],
        ],
        dtype=torch.int64,
    ).T
    expected_bam = expected_yaw << 16
    expected_angle = expected_bam.to(torch.float32) * (2.0 * math.pi / float(1 << 32))

    assert torch.equal(torch.stack(actual_bam), expected_bam)
    assert torch.equal(torch.stack(actual_angle), expected_angle)

    # Releasing both turn keys clears turnheld, so even a running turn starts
    # with the same five slow tics on its next press.
    engine._move_player(torch.zeros_like(buttons))
    before = engine._angle_bam.clone()
    engine._move_player(buttons)
    assert torch.equal(engine._angle_bam - before, torch.full((2,), slow_yaw << 16))


def test_strafe_modifier_converts_turn_keys_to_clamped_side_movement(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.reaction_time.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
    engine.angle.zero_()
    engine.momentum_x.zero_()
    engine.momentum_y.zero_()
    engine._momentum_x_fixed.zero_()
    engine._momentum_y_fixed.zero_()
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 2] = True
    buttons[:, 8] = True
    # Running left strafe from both the turn key and the dedicated move key is
    # clamped to Doom's MAXPLMOVE (50), rather than applying both 40-unit moves.
    buttons[1, 1] = True
    buttons[1, 4] = True

    engine._move_player(buttons)

    assert torch.equal(engine.angle, torch.zeros(2))
    assert engine.turn_held_tics.tolist() == [1, 1]
    assert torch.equal(engine.x, torch.zeros(2))
    assert torch.equal(engine.y, torch.tensor([0.75, 1.5625]))
    assert torch.equal(engine.momentum_x, torch.zeros(2))
    assert torch.equal(engine.momentum_y, torch.tensor([0.6796875, 1.416015625]))


def test_strafe_modifier_still_primes_keyboard_turn_acceleration(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.reaction_time.zero_()
    engine.angle.zero_()
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 2] = True
    buttons[:, 8] = True

    for _ in range(5):
        engine._move_player(buttons)
    assert torch.equal(engine.angle, torch.zeros(2))

    buttons[:, 2] = False
    engine._move_player(buttons)

    expected_bam = torch.full((2,), 640 << 16, dtype=torch.int64)
    assert torch.equal(engine._angle_bam, expected_bam)


def test_binary_delta_actions_contribute_reference_yaw_and_side_move(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.reaction_time.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
    engine.angle.zero_()
    engine.momentum_x.zero_()
    engine.momentum_y.zero_()
    engine._momentum_x_fixed.zero_()
    engine._momentum_y_fixed.zero_()
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 18] = True
    buttons[:, 19] = True
    buttons[1, 8] = True

    engine._move_player(buttons)

    # A binary custom action supplies value 1 to ViZDoom's delta axes. The
    # degree-to-yaw conversion floors to 182, while TURN_LEFT contributes the
    # first-tic keyboard yaw of 320 in the other direction.
    expected_bam = torch.tensor(
        [(-(182 << 16)) & ((1 << 32) - 1), (320 - 182) << 16],
        dtype=torch.int64,
    )
    assert torch.equal(engine._angle_bam, expected_bam)
    # MOVE_LEFT_RIGHT_DELTA=1 contributes one command unit, or 1/32 map unit
    # of side thrust at the newly updated angle before Doom's normal friction.
    sine = torch.sin(engine.angle)
    cosine = torch.cos(engine.angle)
    right_displacement = engine.x * sine - engine.y * cosine
    forward_displacement = engine.x * cosine + engine.y * sine
    right_momentum = engine.momentum_x * sine - engine.momentum_y * cosine
    assert torch.allclose(right_displacement, torch.full((2,), 0.03125), rtol=0, atol=2e-5)
    assert torch.allclose(forward_displacement, torch.zeros(2), rtol=0, atol=2e-5)
    assert torch.allclose(
        right_momentum,
        torch.full((2,), 0.0283203125),
        rtol=0,
        atol=2e-5,
    )


def test_binary_pitch_delta_matches_reference_during_teleport_lock_and_clamps(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 17] = True

    # ViZDoom applies LOOK_UP_DOWN_DELTA before the reaction-time early exit.
    # A binary value of one floors to 182 command units, or 0.999755859375°.
    engine.reaction_time.fill_(7)
    engine._move_player(buttons)

    expected_bam = torch.full((2,), -(182 << 16), dtype=torch.int64)
    expected_pitch = expected_bam.to(torch.float32) * (2.0 * math.pi / float(1 << 32))
    assert torch.equal(engine._pitch_bam, expected_bam)
    assert torch.equal(engine.pitch, expected_pitch)
    assert engine.reaction_time.tolist() == [6, 6]

    for _ in range(32):
        engine._move_player(buttons)

    minimum_pitch_bam = -32 * ((1 << 29) // 45)
    assert engine._pitch_bam.tolist() == [minimum_pitch_bam, minimum_pitch_bam]
    assert torch.allclose(
        torch.rad2deg(engine.pitch),
        torch.full((2,), -31.999998092651367),
        rtol=0,
        atol=2e-6,
    )

    engine.reset(torch.tensor([True, False]), torch.tensor([789, 456]))
    assert engine._pitch_bam.tolist() == [0, minimum_pitch_bam]
    assert engine.pitch[0] == 0


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


def test_reference_smooth_step_lowers_then_recovers_viewheight(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.z.fill_(-48.0)
    engine.player_floor_z.fill_(-24.0)
    engine.previous_player_floor_z.fill_(-48.0)
    engine.view_height.fill_(38.75)
    engine.delta_view_height.fill_(-0.75)
    engine._player_bob_fixed.zero_()
    engine.episode_time.fill_(14)
    active = torch.ones(2, dtype=torch.bool)

    engine._vertical_player_tick(active)

    # P_CalcHeight renders from the pre-step Z and recovered height for this
    # tic. P_ZMovement then subtracts the 24-unit step only from the state used
    # by subsequent tics and starts GetDeltaViewHeight's 1/8 recovery.
    assert engine.view_z.tolist() == [-10.0, -10.0]
    assert engine.z.tolist() == [-24.0, -24.0]
    assert engine.view_height.tolist() == [14.0, 14.0]
    assert engine.delta_view_height.tolist() == [3.375, 3.375]

    engine._vertical_player_tick(active)

    assert engine.view_z.tolist() == [-3.5, -3.5]
    assert engine.view_height.tolist() == [20.5, 20.5]
    assert engine.delta_view_height.tolist() == [3.625, 3.625]


def test_reference_viewheight_recovery_crosses_zero_by_one_fixed_unit(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.z.zero_()
    engine.player_floor_z.zero_()
    engine.view_height.fill_(37.5)
    engine.delta_view_height.fill_(-0.25)
    engine._player_bob_fixed.zero_()
    active = torch.ones(2, dtype=torch.bool)

    engine._vertical_player_tick(active)

    assert engine.view_height.tolist() == [37.25, 37.25]
    assert engine.delta_view_height.tolist() == [1.0 / 65536.0] * 2

    engine._vertical_player_tick(active)

    assert engine.view_height.tolist() == [37.25001525878906] * 2
    assert engine.delta_view_height.tolist() == [0.2500152587890625] * 2


def test_reference_soft_landing_does_not_squat_viewheight(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.z.fill_(-62.0)
    engine.player_floor_z.fill_(-64.0)
    engine.previous_player_floor_z.fill_(-64.0)
    engine.velocity_z.fill_(-6.0)
    engine.view_height.fill_(41.0)
    engine.delta_view_height.zero_()
    engine._player_bob_fixed.zero_()
    active = torch.ones(2, dtype=torch.bool)

    engine._vertical_player_tick(active)

    # P_ZMovement invokes PlayerLandedOnThing only below -8 units/tic.
    assert engine.view_z.tolist() == [-21.0, -21.0]
    assert engine.z.tolist() == [-64.0, -64.0]
    assert engine.velocity_z.tolist() == [0.0, 0.0]
    assert engine.view_height.tolist() == [41.0, 41.0]
    assert engine.delta_view_height.tolist() == [0.0, 0.0]

    engine._vertical_player_tick(active)

    assert engine.view_z.tolist() == [-23.0, -23.0]


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


def test_corner_slide_retries_with_unclipped_reference_motion(square_scenario) -> None:
    engine = _engine(square_scenario)
    fixed_unit = 1 << 16
    # Translate two certified-map oracle contacts onto the fixture's matching
    # bottom-right and top-right corners without changing their fixed geometry.
    engine._x_fixed[:] = torch.tensor(
        [65741189 - 768 * fixed_unit, 66018892 - 768 * fixed_unit]
    )
    engine._y_fixed[:] = torch.tensor(
        [1048970 - 256 * fixed_unit, 66058833 - 768 * fixed_unit]
    )
    engine._momentum_x_fixed[:] = torch.tensor([397773, 99394])
    engine._momentum_y_fixed[:] = torch.tensor([-12631, 46551])
    engine.x.copy_(engine._x_fixed.to(torch.float32) / fixed_unit)
    engine.y.copy_(engine._y_fixed.to(torch.float32) / fixed_unit)

    position_x, position_y, momentum_x, momentum_y, fallback, _floor, _ceiling = (
        engine._doom_axis_slide_move(torch.ones(2, dtype=torch.bool))
    )

    assert position_x.tolist() == [65741553 - 768 * fixed_unit, 66018988 - 768 * fixed_unit]
    assert position_y.tolist() == [1048958 - 256 * fixed_unit, 66058878 - 768 * fixed_unit]
    assert momentum_x.tolist() == [372958, 93181]
    assert momentum_y.tolist() == [0, 0]
    assert not torch.any(fallback)


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

    fraction, horizontal, valid, _contact_axis = engine._axis_slide_contact_fixed(
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


def test_player_retries_single_axis_step_when_blocked_by_monster(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.momentum_x.fill_(20)
    engine.momentum_y.fill_(20)
    engine.reaction_time.zero_()
    engine.enemy_x[:, 0] = 40
    engine.enemy_y[:, 0] = 30
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True

    engine._move_player(torch.zeros((2, 20), dtype=torch.bool))

    # P_XYMovement tries the Y-only substep first when another actor blocks
    # the combined move, then clears only the rejected X velocity component.
    assert engine.x.tolist() == [0.0, 0.0]
    assert engine.y.tolist() == [10.0, 10.0]
    assert engine.momentum_x.tolist() == [0.0, 0.0]
    assert engine.momentum_y.tolist() == [18.125, 18.125]


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

    assert engine._effective_enemy_height()[:, 0].tolist() == [14.0, 14.0]
    engine.z.fill_(15)
    vertical_collision = engine._player_collides(
        torch.zeros(2),
        torch.zeros(2),
    )
    # P_Die quarters both actors' 56-unit live height immediately, so a
    # player whose feet are at z=15 passes above these still-solid corpses.
    assert vertical_collision.tolist() == [False, False]


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
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
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


def test_spawn_reaction_time_counts_chase_actions_and_only_blocks_missiles(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = torch.tensor([100.0, 32.0])
    engine.enemy_y[:, 0].zero_()
    engine._enemy_x_fixed[:, 0] = torch.round(
        engine.enemy_x[:, 0] * 65536
    ).to(torch.int64)
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_type[:, 0] = torch.tensor([0, 4])
    engine.enemy_health[:, 0] = torch.tensor([20.0, 150.0])
    engine.enemy_alive[:, 0] = True
    engine.enemy_reaction_time[:, 0] = 8
    engine.enemy_cooldown[:, 0] = 0
    engine.enemy_move_cooldown[:, 0] = 0

    engine._enemy_tick()

    assert engine.enemy_reaction_time[:, 0].tolist() == [7, 7]
    assert engine.enemy_attack_phase[:, 0].tolist() == [0, 1]
    for _ in range(3):
        engine._enemy_tick()
    assert engine.enemy_reaction_time[0, 0].item() == 7
    engine._enemy_tick()
    assert engine.enemy_reaction_time[0, 0].item() == 6


def test_pain_wakes_spawned_monster_and_forces_next_eligible_retaliation(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(200)
    engine.y.zero_()
    engine._x_fixed.fill_(200 * 65536)
    engine._y_fixed.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0].zero_()
    engine.enemy_y[:, 0].zero_()
    engine._enemy_x_fixed[:, 0].zero_()
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_reaction_time[:, 0] = 8
    engine.enemy_move_count[:, 0] = 0
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_animation_tics[:, 0] = 7
    engine.rng_state.zero_()

    damage = torch.zeros_like(engine.enemy_health)
    damage[:, 0] = 1
    pain = torch.zeros_like(engine.enemy_alive)
    pain[:, 0] = True
    engine._apply_enemy_damage(damage, pain_override=pain)

    assert engine.enemy_reaction_time[:, 0].tolist() == [0, 0]
    assert engine.enemy_just_hit[:, 0].tolist() == [True, True]
    assert engine.enemy_pain_tics[:, 0].tolist() == [6, 6]
    assert engine.enemy_animation_tics[:, 0].tolist() == [0, 0]

    # Skip to the first A_Chase action after pain. At 200 units the seeded
    # missile roll is below P_CheckMissileRange's threshold, so only Doom's
    # MF_JUSTHIT path can select the attack.
    engine.enemy_pain_tics[:, 0] = 0
    engine._enemy_tick()

    assert engine.enemy_attack_phase[:, 0].tolist() == [1, 1]
    assert engine.enemy_just_attacked[:, 0].tolist() == [True, True]
    assert engine.enemy_just_hit[:, 0].tolist() == [False, False]


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
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
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
    engine._y_fixed.fill_(100 * 65536)
    engine.enemy_cooldown[:, 0] = 1
    engine._enemy_tick()

    assert torch.allclose(
        engine.enemy_angle[:, 0],
        torch.full((2,), 3.0 * math.pi / 4.0),
    )


def test_monster_face_target_uses_vizdoom_fixed_point_angle(square_scenario) -> None:
    engine = _engine(square_scenario)
    delta_x_fixed = 11_709_040
    delta_y_fixed = 4_041_447
    engine.x.fill_(delta_x_fixed / 65536.0)
    engine.y.fill_(delta_y_fixed / 65536.0)
    engine._x_fixed.fill_(delta_x_fixed)
    engine._y_fixed.fill_(delta_y_fixed)
    engine.enemy_x[:, 0] = 0
    engine.enemy_y[:, 0] = 0
    engine._enemy_x_fixed[:, 0] = 0
    engine._enemy_y_fixed[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -1
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 1

    bam_angle = engine._doom_bam_angle(
        torch.full((2,), delta_x_fixed, dtype=torch.int64),
        torch.full((2,), delta_y_fixed, dtype=torch.int64),
    )
    assert bam_angle.tolist() == [226_922_601, 226_922_601]

    engine._enemy_tick()

    # ViZDoom seed 123, tick 961, object 206 faces this exact fixed-point
    # delta at BAM angle 226922601 (fine angle 432). Floating atan2 selects
    # adjacent fine angle 433 and changes the bullet ray.
    expected = torch.full(
        (2,),
        226_922_601 * (2.0 * math.pi / float(1 << 32)),
    )
    assert torch.allclose(engine.enemy_angle[:, 0], expected, rtol=0, atol=1e-7)
    assert engine._fine_angle_index(engine.enemy_angle[:, 0]).tolist() == [432, 432]


def test_demon_and_knight_repeat_face_target_on_second_prefire_frame(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0].zero_()
    engine.enemy_y[:, 0].zero_()
    engine._enemy_x_fixed[:, 0].zero_()
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_angle[:, 0].zero_()
    engine.enemy_type[:, 0] = torch.tensor([4, 5])
    engine.enemy_health[:, 0] = torch.tensor([150.0, 500.0])
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -1
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 16
    engine.x.zero_()
    engine.y.fill_(100)
    engine._x_fixed.zero_()
    engine._y_fixed.fill_(100 * 65536)

    for _ in range(7):
        engine._enemy_tick()
    assert torch.equal(engine.enemy_angle[:, 0], torch.zeros(2))

    engine._enemy_tick()

    north_bam = engine._doom_bam_angle(
        torch.zeros(2, dtype=torch.int64),
        torch.full((2,), 100 * 65536, dtype=torch.int64),
    )
    north = north_bam.to(torch.float32) * (2.0 * math.pi / float(1 << 32))
    assert torch.allclose(engine.enemy_angle[:, 0], north, rtol=0, atol=1e-7)

    engine.x.fill_(-100)
    engine.y.zero_()
    engine._x_fixed.fill_(-100 * 65536)
    engine._y_fixed.zero_()
    for _ in range(7):
        engine._enemy_tick()
    assert torch.allclose(engine.enemy_angle[:, 0], north, rtol=0, atol=1e-7)

    engine._enemy_tick()

    west_bam = engine._doom_bam_angle(
        torch.full((2,), -100 * 65536, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
    )
    west = west_bam.to(torch.float32) * (2.0 * math.pi / float(1 << 32))
    # A_SargAttack faces again at the G action. A_BruisAttack does not, so the
    # Hell Knight retains the angle selected on its F prefire frame.
    assert torch.allclose(engine.enemy_angle[0, 0], west[0], rtol=0, atol=1e-7)
    assert torch.allclose(engine.enemy_angle[1, 0], north[1], rtol=0, atol=1e-7)


def test_chainsaw_marine_faces_target_and_applies_post_hit_turn(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0].zero_()
    engine.enemy_y[:, 0].zero_()
    engine._enemy_x_fixed[:, 0].zero_()
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_angle[:, 0].zero_()
    engine.enemy_type[:, 0] = 2
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -1
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 4
    engine.x.zero_()
    engine.y.fill_(32)
    engine._x_fixed.zero_()
    engine._y_fixed.fill_(32 * 65536)

    for _ in range(3):
        engine._enemy_tick()
    assert torch.equal(engine.enemy_angle[:, 0], torch.zeros(2))

    engine._enemy_tick()

    target_bam = engine._doom_bam_angle(
        torch.zeros(2, dtype=torch.int64),
        torch.full((2,), 32 * 65536, dtype=torch.int64),
    )
    expected_bam = (target_bam + ((1 << 30) // 20)) & ((1 << 32) - 1)
    expected = expected_bam.to(torch.float32) * (2.0 * math.pi / float(1 << 32))
    assert torch.allclose(engine.enemy_angle[:, 0], expected, rtol=0, atol=1e-7)
    assert torch.all(engine.health < 100)


def test_chainsaw_marine_repeats_four_tic_attack_cycle(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine._enemy_x_fixed[:, 0] = 32 * 65536
    engine._enemy_y_fixed[:, 0] = 0
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


def test_monster_melee_range_uses_target_radius_and_approximate_distance(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.copy_(torch.tensor([0.0, 200.0]))
    engine.y.zero_()
    engine._x_fixed.copy_(torch.round(engine.x * 65536).to(torch.int64))
    engine._y_fixed.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = torch.tensor([42.0, 70.0])
    engine.enemy_y[:, 0] = torch.tensor([42.0, 0.0])
    engine._enemy_x_fixed[:, 0] = torch.round(
        engine.enemy_x[:, 0] * 65536
    ).to(torch.int64)
    engine._enemy_y_fixed[:, 0] = torch.round(
        engine.enemy_y[:, 0] * 65536
    ).to(torch.int64)
    engine.enemy_type[:, 0] = 2
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = torch.tensor([-1, 1])
    engine.enemy_cooldown[:, 0] = 0
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_move_count[:, 0] = 1

    engine.enemy_x[1, 1] = 0
    engine.enemy_y[1, 1] = 0
    engine._enemy_x_fixed[1, 1] = 0
    engine._enemy_y_fixed[1, 1] = 0
    engine.enemy_type[1, 1] = 4
    engine.enemy_health[1, 1] = 150
    engine.enemy_alive[1, 1] = True
    engine.enemy_pain_tics[1, 1] = 99

    engine._enemy_tick()

    # The first target is only 59.4 Euclidean units away, but Doom's
    # AproxDistance is 63 and outside the player's 44+16 limit. A demon
    # target's 30-unit radius extends the second lane's limit to 74.
    assert engine.enemy_attack_phase[:, 0].tolist() == [0, 1]
    assert engine.enemy_x[0, 0].item() != 42.0
    assert engine.enemy_x[1, 0].item() == 70.0

    engine.enemy_cooldown[:, 0] = 1
    engine._enemy_tick()

    # CheckMeleeRange has no additional center-distance cap: the demon's
    # 30-unit radius keeps the target hittable at this 70-unit separation.
    assert engine.enemy_health[1, 1].item() < 150.0


def test_hell_knight_uses_projectile_when_close_target_is_above_melee_reach(
    square_scenario,
) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.z.fill_(65)
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 59
    engine.enemy_y[:, 0].zero_()
    engine.enemy_z[:, 0].zero_()
    engine._enemy_x_fixed[:, 0] = 59 * 65536
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    engine.enemy_cooldown[:, 0] = 0
    engine.enemy_move_cooldown[:, 0] = 0
    engine.enemy_move_count[:, 0] = 0

    engine._enemy_tick()

    # The horizontal distance is inside 44 + player radius, but the player's
    # feet are one unit above the knight's top. Doom therefore selects Missile.
    assert engine.enemy_attack_phase[:, 0].tolist() == [1, 1]
    assert engine.enemy_just_attacked[:, 0].tolist() == [True, True]
    engine.enemy_cooldown[:, 0] = 1

    engine._enemy_tick()

    assert torch.sum(engine.enemy_projectile_alive, dim=1).tolist() == [1, 1]
    assert torch.all(engine.enemy_projectile_velocity_z[:, 0] > 0)
    assert engine.health.tolist() == [100.0, 100.0]


def test_demon_melee_action_whiffs_after_target_retreats(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(100)
    engine.y.zero_()
    engine._x_fixed.fill_(100 * 65536)
    engine._y_fixed.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0].zero_()
    engine.enemy_y[:, 0].zero_()
    engine._enemy_x_fixed[:, 0].zero_()
    engine._enemy_y_fixed[:, 0].zero_()
    engine.enemy_type[:, 0] = 4
    engine.enemy_health[:, 0] = 150
    engine.enemy_alive[:, 0] = True
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 1

    engine._enemy_tick()

    assert engine.health.tolist() == [100.0, 100.0]
    assert engine.enemy_attack_phase[:, 0].tolist() == [2, 2]


def test_chaingunner_refire_state_does_not_fire_extra_bullet(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine._enemy_x_fixed[:, 0] = 32 * 65536
    engine._enemy_y_fixed[:, 0] = 0
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
    for _ in range(4):
        engine._enemy_tick()

    # ViZDoom's CPOS F 1 A_CPosRefire action only faces the target and decides
    # whether to loop. It neither fires nor consumes a four-tic attack frame.
    assert torch.equal(engine.health, after_second_shot)
    assert engine.enemy_attack_phase[:, 0].tolist() == [4, 4]
    assert engine.enemy_cooldown[:, 0].tolist() == [1, 1]

    engine._enemy_tick()

    assert torch.all(engine.health < after_second_shot)
    assert engine.enemy_attack_phase[:, 0].tolist() == [2, 2]
    assert engine.enemy_cooldown[:, 0].tolist() == [4, 4]


def test_chaingunner_refire_bypass_uses_reference_threshold(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.rng_state.copy_(torch.tensor([1, 2]))
    candidates = torch.zeros_like(engine.enemy_alive)
    candidates[:, 0] = True

    bypass = engine._enemy_chaingun_refire_decision(candidates)

    # The first xorshift draws have low bytes 33 and 66. A_CPosRefire only
    # bypasses its target/sight checks when the byte is strictly below 40.
    assert bypass[:, 0].tolist() == [True, False]
    assert not torch.any(bypass[:, 1:])


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
    engine._x_fixed.fill_(-32 * 65536)
    engine._y_fixed.zero_()
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


def test_monster_hitscan_autoaim_uses_portal_clipped_target_window(square_scenario) -> None:
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
    engine.z.fill_(-80)
    engine._x_fixed.fill_(-64 * 65536)
    engine._y_fixed.zero_()
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 64
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine._enemy_x_fixed[:, 0] = 64 * 65536
    engine._enemy_y_fixed[:, 0] = 0
    engine.enemy_type[:, 0] = 3
    engine.enemy_health[:, 0] = 70
    engine.enemy_alive[:, 0] = True
    engine.enemy_target_slot[:, 0] = -1
    engine.enemy_attack_phase[:, 0] = 1
    engine.enemy_cooldown[:, 0] = 1

    def centered_chaingun_pellet(
        _enemy_type: torch.Tensor,
        fires: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        damage = torch.zeros((*fires.shape, 3), dtype=torch.float32)
        damage[:, :, 0] = fires.to(torch.float32) * 3.0
        return damage, torch.zeros_like(damage, dtype=torch.int64)

    engine._enemy_hitscan_rolls = centered_chaingun_pellet
    engine._enemy_tick()

    # The raw midpoint ray strikes the high side of the portal floor. Doom
    # clips the target interval to the opening and aims at its visible portion.
    assert torch.all(engine.health < 100)


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
        padded_distance < 64.0,
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


def test_hell_knight_can_own_multiple_projectiles_in_flight(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    dx = engine.x[:, None] - engine.enemy_x
    dy = engine.y[:, None] - engine.enemy_y
    requested = torch.zeros_like(engine.enemy_alive)
    requested[:, 0] = True

    engine._spawn_enemy_projectiles(requested, dx, dy)
    engine._spawn_enemy_projectiles(requested, dx, dy)

    # P_SpawnMissile creates a new BaronBall actor for every attack; the
    # second shot must not be suppressed merely because its owner still has
    # a first shot in flight.
    assert engine.enemy_projectile_alive[:, :2].tolist() == [
        [True, True],
        [True, True],
    ]
    assert engine.enemy_projectile_source_slot[:, :2].tolist() == [
        [0, 0],
        [0, 0],
    ]
    assert torch.equal(
        engine.enemy_projectile_velocity_x[:, 0],
        engine.enemy_projectile_velocity_x[:, 1],
    )


def test_hell_knight_melee_attack_fires_after_reference_prefire(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine._x_fixed.zero_()
    engine._y_fixed.zero_()
    engine.enemy_x[:, 0] = 32
    engine.enemy_y[:, 0] = 0
    engine._enemy_x_fixed[:, 0] = 32 * 65536
    engine._enemy_y_fixed[:, 0] = 0
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
