from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from gradoom.engine import TorchDeathmatchEngine


def _item_scenario(square_scenario, *type_ids: int):
    return replace(
        square_scenario,
        item_spawns=np.asarray([(0.0, 0.0, 0.0)] * len(type_ids), dtype=np.float32),
        item_types=np.asarray(type_ids, dtype=np.int32),
    )


def _large_arena_scenario(square_scenario, half_extent: float = 4096.0):
    walls = np.asarray(
        [
            (-half_extent, -half_extent, half_extent, -half_extent),
            (half_extent, -half_extent, half_extent, half_extent),
            (half_extent, half_extent, -half_extent, half_extent),
            (-half_extent, half_extent, -half_extent, -half_extent),
        ],
        dtype=np.float32,
    )
    vertices = np.asarray(
        [
            (-half_extent, -half_extent),
            (half_extent, -half_extent),
            (half_extent, half_extent),
            (-half_extent, half_extent),
        ],
        dtype=np.float32,
    )
    return replace(
        square_scenario,
        vertices=vertices,
        wall_segments=walls,
        blocking_segments=walls.copy(),
    )


def _engine(scenario) -> TorchDeathmatchEngine:
    engine = TorchDeathmatchEngine(
        scenario,
        2,
        device=torch.device("cpu"),
        frame_skip=1,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))
    engine.weapon_raise_cooldown.zero_()
    engine.x.zero_()
    engine.y.zero_()
    return engine


def _advance_weapon_switch(engine: TorchDeathmatchEngine, tics: int = 16) -> None:
    active = torch.ones(engine.num_envs, dtype=torch.bool)
    for _ in range(tics):
        engine._weapon_switch_tick(active)


def _finish_pending_attack(engine: TorchDeathmatchEngine) -> torch.Tensor:
    reward = torch.zeros(engine.num_envs)
    noop = torch.zeros((engine.num_envs, 20), dtype=torch.bool)
    while torch.any(engine.pending_attack_weapon >= 0):
        reward += engine._player_attack(noop)
    return reward


def test_standard_health_stays_when_full_but_bonus_is_always_consumed(square_scenario) -> None:
    engine = _engine(_item_scenario(square_scenario, 2011, 2014))

    engine._collect_items()

    assert engine.health.tolist() == [101.0, 101.0]
    assert engine.item_available.tolist() == [[True, False], [True, False]]


def test_pickups_respect_vizdoom_vertical_reach_window(square_scenario) -> None:
    scenario = replace(
        square_scenario,
        item_spawns=np.asarray([(0.0, 0.0, 56.0), (0.0, 0.0, 57.0)], dtype=np.float32),
        item_types=np.asarray([2014, 2014], dtype=np.int32),
    )
    engine = _engine(scenario)

    engine._collect_items()

    assert engine.health.tolist() == [101.0, 101.0]
    assert engine.item_available.tolist() == [[False, True], [False, True]]

    below = _engine(_item_scenario(square_scenario, 2014))
    below.z.fill_(33)
    below._collect_items()
    assert torch.all(below.item_available)
    below.z.fill_(32)
    below._collect_items()
    assert not torch.any(below.item_available)


def test_ammo_pickup_consumes_only_boxes_needed_to_reach_capacity(square_scenario) -> None:
    engine = _engine(_item_scenario(square_scenario, 2048, 2048))
    engine.ammo[:, 1].fill_(175)
    engine.ammo[:, 3].fill_(175)

    engine._collect_items()

    assert engine.ammo[:, 1].tolist() == [200.0, 200.0]
    assert torch.equal(engine.ammo[:, 1], engine.ammo[:, 3])
    assert engine.item_available.tolist() == [[False, True], [False, True]]


def test_ammo_pickup_does_not_reduce_existing_overcap_amount(square_scenario) -> None:
    engine = _engine(_item_scenario(square_scenario, 2048))
    engine.ammo[:, 1].fill_(400)
    engine.ammo[:, 3].fill_(400)

    engine._collect_items()

    assert engine.ammo[:, 1].tolist() == [400.0, 400.0]
    assert torch.equal(engine.ammo[:, 1], engine.ammo[:, 3])
    assert torch.all(engine.item_available)


def test_green_and_blue_armor_use_reference_absorption_fractions(square_scenario) -> None:
    green = _engine(_item_scenario(square_scenario, 2018))
    green.armor.fill_(50)
    green.armor_save_fraction.fill_(0.5)
    green._collect_items()
    green._apply_player_damage(torch.full((2,), 30.0))

    assert green.armor.tolist() == [90.0, 90.0]
    assert green.health.tolist() == [80.0, 80.0]

    blue = _engine(_item_scenario(square_scenario, 2019))
    blue._collect_items()
    blue._apply_player_damage(torch.full((2,), 30.0))

    assert blue.armor.tolist() == [185.0, 185.0]
    assert blue.health.tolist() == [85.0, 85.0]


def test_armor_pickup_is_not_consumed_when_it_cannot_improve_armor(square_scenario) -> None:
    engine = _engine(_item_scenario(square_scenario, 2018, 2019))
    engine.armor.fill_(200)
    engine.armor_save_fraction.fill_(0.5)

    engine._collect_items()

    assert engine.item_available.tolist() == [[True, True], [True, True]]
    assert engine.armor.tolist() == [200.0, 200.0]


def test_weapon_slot_counts_preserve_chainsaw_and_shotgun_variants(square_scenario) -> None:
    engine = _engine(_item_scenario(square_scenario, 2005, 82))

    engine._collect_items()

    assert engine.weapons[:, 0].tolist() == [2.0, 2.0]
    assert engine.weapons[:, 2].tolist() == [1.0, 1.0]
    assert engine.super_shotgun_owned.tolist() == [True, True]
    assert engine.shotgun_owned.tolist() == [False, False]
    assert engine.ammo[:, 2].tolist() == [8.0, 8.0]
    assert engine.pending_weapon.tolist() == [4, 4]
    assert engine.mugshot_grin_tics.tolist() == [71, 71]
    _advance_weapon_switch(engine)
    assert engine._active_weapon().tolist() == [4, 4]

    touched = torch.ones((2, 1), dtype=torch.bool)
    engine._pickup_weapon(touched, code=3, ammo_amount=8.0, ammo_cap=50.0)
    assert engine.weapons[:, 2].tolist() == [2.0, 2.0]
    _advance_weapon_switch(engine)
    assert engine._active_weapon().tolist() == [3, 3]

    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 11] = True
    engine._select_weapons(buttons)
    _advance_weapon_switch(engine)
    assert engine._active_weapon().tolist() == [4, 4]
    engine._select_weapons(torch.zeros_like(buttons))
    engine._select_weapons(buttons)
    _advance_weapon_switch(engine)
    assert engine._active_weapon().tolist() == [3, 3]


def test_new_weapon_grin_matches_reference_state_duration(square_scenario) -> None:
    engine = _engine(_item_scenario(square_scenario, 2005))

    engine._collect_items()

    assert engine.mugshot_grin_tics.tolist() == [71, 71]
    noop = torch.zeros((2, 20), dtype=torch.bool)
    for _ in range(70):
        engine.step(noop)
    assert engine.mugshot_grin_tics.tolist() == [1, 1]
    engine.step(noop)
    assert engine.mugshot_grin_tics.tolist() == [0, 0]


def test_weapon_cycle_is_edge_triggered_across_frame_skip(square_scenario) -> None:
    engine = TorchDeathmatchEngine(
        square_scenario,
        2,
        device=torch.device("cpu"),
        frame_skip=2,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 15] = True

    engine._select_weapons(buttons)
    engine._select_weapons(buttons)

    assert engine.pending_weapon.tolist() == [0, 0]
    _advance_weapon_switch(engine)
    assert engine._active_weapon().tolist() == [0, 0]


def test_weapon_change_blocks_fire_for_reference_raise_window(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.shotgun_owned.fill_(True)
    engine.weapons[:, 2].fill_(1)
    engine.ammo[:, 2].fill_(10)
    select = torch.zeros((2, 20), dtype=torch.bool)
    select[:, 11] = True
    attack = torch.zeros_like(select)
    attack[:, 0] = True

    engine._select_weapons(select)

    assert engine._active_weapon().tolist() == [2, 2]
    assert engine.pending_weapon.tolist() == [3, 3]
    assert engine.weapon_lower_cooldown.tolist() == [16, 16]
    for _ in range(16):
        engine._weapon_switch_tick(torch.ones(2, dtype=torch.bool))
        engine._player_attack(attack)
    assert engine._active_weapon().tolist() == [3, 3]
    assert engine.weapon_raise_cooldown.tolist() == [16, 16]
    for _ in range(15):
        engine._weapon_switch_tick(torch.ones(2, dtype=torch.bool))
        engine._player_attack(attack)
    assert engine.ammo[:, 2].tolist() == [10.0, 10.0]

    engine._weapon_switch_tick(torch.ones(2, dtype=torch.bool))
    engine._player_attack(attack)
    _finish_pending_attack(engine)

    assert engine.ammo[:, 2].tolist() == [9.0, 9.0]


def test_existing_weapon_at_full_ammo_stays_in_world(square_scenario) -> None:
    engine = _engine(_item_scenario(square_scenario, 82))
    engine.super_shotgun_owned.fill_(True)
    engine.weapons[:, 2].fill_(1)
    engine.ammo[:, 2].fill_(50)

    engine._collect_items()

    assert torch.all(engine.item_available)
    assert engine.ammo[:, 2].tolist() == [50.0, 50.0]


def test_reference_weapon_refire_cadence_and_super_shotgun_ammo_cost(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.super_shotgun_owned.fill_(True)
    engine.weapons[:, 2].fill_(1)
    engine.ammo[:, 2].fill_(50)
    engine._set_active_weapon(torch.full((2,), 4), torch.ones(2, dtype=torch.bool))
    _advance_weapon_switch(engine)
    engine.weapon_raise_cooldown.zero_()
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True
    changes: list[int] = []
    previous = float(engine.ammo[0, 2])

    for tic in range(110):
        engine.attack_cooldown.sub_(1).clamp_min_(0)
        engine.weapon_state_cooldown.sub_(1).clamp_min_(0)
        engine._player_attack(buttons)
        current = float(engine.ammo[0, 2])
        if current != previous:
            changes.append(tic)
            previous = current

    assert changes[:3] == [3, 54, 105]
    assert engine.ammo[:, 2].tolist() == [44.0, 44.0]


def test_starting_pistol_cannot_fire_until_reference_raise_completes(square_scenario) -> None:
    engine = TorchDeathmatchEngine(
        square_scenario,
        2,
        device=torch.device("cpu"),
        frame_skip=1,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True
    ammo_change_times: list[int] = []
    previous = float(engine.ammo[0, 1])

    for _ in range(32):
        engine.step(buttons)
        current = float(engine.ammo[0, 1])
        if current != previous:
            ammo_change_times.append(int(engine.episode_time[0]))
            previous = current

    assert ammo_change_times == [19, 33]


def test_switching_during_initial_raise_preserves_reference_vertical_position(
    square_scenario,
) -> None:
    engine = TorchDeathmatchEngine(
        square_scenario,
        2,
        device=torch.device("cpu"),
        frame_skip=1,
    )
    engine.reset(torch.ones(2, dtype=torch.bool), torch.tensor([123, 456]))
    engine.chainsaw_owned.fill_(True)
    noop = torch.zeros((2, 20), dtype=torch.bool)
    previous_weapon = noop.clone()
    previous_weapon[:, 16] = True

    engine.step(noop)
    assert engine.episode_time.tolist() == [2, 2]
    engine.step(previous_weapon)

    assert engine.weapon_lower_cooldown.tolist() == [2, 2]
    assert engine._active_weapon().tolist() == [2, 2]
    engine.step(noop)
    assert engine._active_weapon().tolist() == [2, 2]
    engine.step(noop)
    assert engine.episode_time.tolist() == [5, 5]
    assert engine._active_weapon().tolist() == [1, 1]


def test_weapon_switch_waits_for_each_reference_fire_state(square_scenario) -> None:
    weapon_slots = (1, 1, 2, 3, 3, 4, 5, 6)
    variants = (False, True, False, False, True, False, False, False)
    expected_transitions = (37, 23, 34, 59, 77, 23, 35, 38)

    for weapon, (slot, variant, expected) in enumerate(
        zip(weapon_slots, variants, expected_transitions, strict=True)
    ):
        engine = _engine(square_scenario)
        engine.episode_time.fill_(50)
        engine.selected_weapon.fill_(slot)
        engine.selected_weapon_variant.fill_(variant)
        engine.weapons.fill_(1)
        engine.chainsaw_owned.fill_(True)
        engine.shotgun_owned.fill_(True)
        engine.super_shotgun_owned.fill_(True)
        engine.ammo.fill_(100)
        attack = torch.zeros((2, 20), dtype=torch.bool)
        attack[:, 0] = True
        select = torch.zeros_like(attack)
        select[:, 10 if weapon <= 1 else 9] = True

        engine.step(attack)
        transitions = 0
        while torch.all(engine._active_weapon() == weapon):
            engine.step(select)
            transitions += 1
            assert transitions <= 80

        assert transitions == expected


def test_chaingun_single_trigger_always_fires_two_rounds(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.episode_time.fill_(50)
    engine.selected_weapon.fill_(4)
    engine.weapons[:, 3].fill_(1)
    engine.ammo[:, 1].fill_(50)
    engine.ammo[:, 3].fill_(50)
    attack = torch.zeros((2, 20), dtype=torch.bool)
    attack[:, 0] = True
    noop = torch.zeros_like(attack)

    engine.step(attack)
    assert engine.ammo[:, 1].tolist() == [49.0, 49.0]
    for _ in range(4):
        engine.step(noop)

    assert engine.ammo[:, 1].tolist() == [48.0, 48.0]
    assert torch.equal(engine.ammo[:, 1], engine.ammo[:, 3])


def test_hitscan_rolls_reference_pellet_counts_and_spread(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.rng_state.copy_(torch.tensor([12345, 67890]))
    pistol_damage, pistol_horizontal, pistol_vertical = engine._hitscan_pellet_rolls(
        torch.full((2,), 2),
        torch.ones(2, dtype=torch.bool),
        torch.tensor([True, False]),
    )

    assert torch.count_nonzero(pistol_damage, dim=1).tolist() == [1, 1]
    assert pistol_horizontal[0, 0] == 0
    assert pistol_horizontal[1, 0] != 0
    assert not torch.any(pistol_vertical)

    engine.rng_state.copy_(torch.tensor([12345, 67890]))
    damage, horizontal, vertical = engine._hitscan_pellet_rolls(
        torch.tensor([3, 4]),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert torch.count_nonzero(damage, dim=1).tolist() == [7, 20]
    live_damage = damage[damage > 0]
    assert torch.all(
        (live_damage == 5) | (live_damage == 10) | (live_damage == 15)
    )
    assert torch.any(horizontal[0, :7] != 0)
    assert torch.any(horizontal[1] != 0)
    assert not torch.any(vertical[0])
    assert torch.any(vertical[1] != 0)


def test_delayed_pistol_preserves_first_shot_accuracy(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.selected_weapon.fill_(2)
    engine.attack_held_tics.copy_(torch.tensor([1, 8]))
    attack = torch.zeros((2, 20), dtype=torch.bool)
    attack[:, 0] = True

    engine._player_attack(attack)

    assert engine.pending_attack_weapon.tolist() == [2, 2]
    assert engine.pending_attack_accurate.tolist() == [True, False]


def test_melee_rolls_use_reference_damage_and_spread_scales(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.rng_state.fill_(12345)

    damage, spread = engine._melee_attack_rolls(
        torch.tensor([0, 1]),
        torch.ones(2, dtype=torch.bool),
    )

    assert damage.tolist() == [2.0, 2.0]
    assert torch.allclose(
        torch.rad2deg(spread),
        torch.tensor([0.32958984375, 0.16544117033481598]),
    )


def test_fist_uses_reference_range_and_snaps_to_hit_target(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = torch.tensor([60.0, 70.0])
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.rng_state.fill_(12345)

    engine._execute_player_attack(
        torch.zeros(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert engine.enemy_health[:, 0].tolist() == [98.0, 100.0]

    engine.enemy_x[:, 0] = 48
    engine.enemy_y[:, 0] = 8
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.rng_state.fill_(12345)
    engine._execute_player_attack(
        torch.zeros(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    expected = torch.atan2(torch.tensor(8.0), torch.tensor(48.0))
    assert torch.allclose(engine.angle, torch.full((2,), expected))


def test_chainsaw_hit_turns_and_forces_next_tic_pull(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.reaction_time.zero_()
    engine.enemy_x[:, 0] = 48
    engine.enemy_y[:, 0] = 8
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    engine.rng_state.fill_(12345)

    engine._execute_player_attack(
        torch.ones(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    target_angle = torch.atan2(torch.tensor(8.0), torch.tensor(48.0))
    expected_angle = target_angle - torch.deg2rad(torch.tensor(90.0 / 21.0))
    assert engine.enemy_health[:, 0].tolist() == [498.0, 498.0]
    assert torch.allclose(engine.angle, torch.full((2,), expected_angle))
    assert torch.all(engine.chainsaw_pull)

    contrary_input = torch.zeros((2, 20), dtype=torch.bool)
    contrary_input[:, 1] = True
    contrary_input[:, 3] = True
    contrary_input[:, 5] = True
    contrary_input[:, 7] = True
    engine._move_player(contrary_input)

    assert torch.allclose(engine.angle, torch.full((2,), expected_angle))
    assert engine._x_fixed.tolist() == [203959, 203959]
    assert engine._y_fixed.tolist() == [18353, 18353]
    assert engine._momentum_x_fixed.tolist() == [184837, 184837]
    assert engine._momentum_y_fixed.tolist() == [16632, 16632]
    assert not torch.any(engine.chainsaw_pull)


def test_shotgun_pellets_can_hit_distinct_angular_targets(square_scenario) -> None:
    engine = _engine(_large_arena_scenario(square_scenario))
    engine.enemy_alive.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    angles = torch.deg2rad(torch.tensor([2.48291015625, -2.39501953125]))
    engine.enemy_x[:, :2] = torch.cos(angles) * 512.0
    engine.enemy_y[:, :2] = torch.sin(angles) * 512.0
    engine.enemy_z[:, :2] = 0
    engine.enemy_type[:, :2] = 0
    engine.enemy_health[:, :2] = 200
    engine.enemy_alive[:, :2] = True
    engine.ammo[:, 2] = 10
    engine.rng_state.copy_(torch.tensor([12345, 67890]))

    engine._execute_player_attack(
        torch.full((2,), 3),
        torch.tensor([True, False]),
        torch.ones(2, dtype=torch.bool),
    )

    assert torch.all(engine.enemy_health[0, :2] < 200)
    assert engine.enemy_health[1, :2].tolist() == [200.0, 200.0]


def test_later_shotgun_pellets_pass_through_newly_killed_actor(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, :2] = torch.tensor([64.0, 128.0])
    engine.enemy_y[:, :2] = 0
    engine.enemy_z[:, :2] = 0
    engine.enemy_type[:, :2] = 0
    engine.enemy_health[:, :2] = torch.tensor([1.0, 200.0])
    engine.enemy_alive[:, :2] = True
    engine.ammo[:, 2] = 10
    engine.rng_state.copy_(torch.tensor([12345, 67890]))

    reward = engine._execute_player_attack(
        torch.full((2,), 3),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert not torch.any(engine.enemy_alive[:, 0])
    assert torch.all(engine.enemy_health[:, 1] < 200)
    assert reward.tolist() == [1.0, 1.0]


def test_hitscan_preserves_per_pellet_pain_result(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    damage = torch.zeros_like(engine.enemy_health)
    damage[:, 0] = 10
    pain_override = torch.zeros_like(engine.enemy_alive)
    pain_override[0, 0] = True
    rng_before = engine.rng_state.clone()

    engine._apply_enemy_damage(damage, pain_override=pain_override)

    assert engine.enemy_pain_tics[:, 0].tolist() == [4, 0]
    assert torch.equal(engine.rng_state, rng_before)


def test_bullet_autoaim_and_trace_use_distinct_reference_ranges(square_scenario) -> None:
    engine = _engine(_large_arena_scenario(square_scenario))
    engine.enemy_alive.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = torch.tensor([1000.0, 1500.0])
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 100
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.ammo[:, 1] = 10

    engine._execute_player_attack(
        torch.full((2,), 2),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert engine.enemy_health[0, 0] < 100
    assert engine.enemy_health[1, 0] == 100

    engine.enemy_x[:, 0] = 3000
    engine.enemy_z[:, 0] = 0
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.rng_state.copy_(torch.tensor([12345, 67890]))
    engine._execute_player_attack(
        torch.full((2,), 2),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert torch.all(engine.enemy_health[:, 0] < 100)


def test_shotgun_guy_drop_waits_for_death_state_and_gives_half_ammo(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.zero_()
    engine.y.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = 48
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 1
    engine.enemy_health[:, 0] = 1
    engine.enemy_alive[:, 0] = True
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)
    _finish_pending_attack(engine)

    assert engine.drop_type[:, 0].tolist() == [2001, 2001]
    assert engine.drop_delay[:, 0].tolist() == [10, 10]
    for _ in range(9):
        engine._collect_drops()
    assert not torch.any(engine.shotgun_owned)

    engine.x.fill_(48)
    engine._collect_drops()

    assert torch.all(engine.shotgun_owned)
    assert engine.ammo[:, 2].tolist() == [4.0, 4.0]
    assert engine.drop_type[:, 0].tolist() == [-1, -1]


def test_monster_drop_is_independently_tossed_and_gravity_driven(square_scenario) -> None:
    engine = _engine(_large_arena_scenario(square_scenario))
    engine.x.fill_(2000)
    engine.y.fill_(2000)
    engine.enemy_x[:, 0] = torch.tensor([10.0, 20.0])
    engine.enemy_y[:, 0] = torch.tensor([30.0, 40.0])
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 1
    engine.enemy_alive[:, 0] = True
    damage = torch.zeros_like(engine.enemy_health)
    damage[:, 0] = 1

    engine._apply_enemy_damage(damage)
    for _ in range(9):
        engine._collect_drops()
    assert not torch.any(engine.drop_spawned[:, 0])

    engine._collect_drops()

    fixed = 65536
    assert torch.all(engine.drop_spawned[:, 0])
    # P_Die quarters the 56-unit corpse to 14 units. P_DropItem starts at
    # half that height, then the new actor consumes its first toss tic.
    assert torch.equal(
        engine._drop_x_fixed[:, 0],
        engine._enemy_x_fixed[:, 0] + engine._drop_velocity_x_fixed[:, 0],
    )
    assert torch.equal(
        engine._drop_y_fixed[:, 0],
        engine._enemy_y_fixed[:, 0] + engine._drop_velocity_y_fixed[:, 0],
    )
    assert torch.equal(
        engine._drop_z_fixed[:, 0],
        7 * fixed + engine._drop_velocity_z_fixed[:, 0] + fixed,
    )

    previous_x = engine._drop_x_fixed[:, 0].clone()
    previous_y = engine._drop_y_fixed[:, 0].clone()
    previous_z = engine._drop_z_fixed[:, 0].clone()
    engine.enemy_x[:, 0] = 1000
    engine.enemy_y[:, 0] = 1000
    engine._drop_velocity_x_fixed[:, 0] = fixed // 2
    engine._drop_velocity_y_fixed[:, 0] = -fixed // 4
    engine._drop_velocity_z_fixed[:, 0] = 2 * fixed

    engine._collect_drops()

    assert torch.equal(engine._drop_x_fixed[:, 0], previous_x + fixed // 2)
    assert torch.equal(engine._drop_y_fixed[:, 0], previous_y - fixed // 4)
    assert torch.equal(engine._drop_z_fixed[:, 0], previous_z + 2 * fixed)
    assert torch.all(engine._drop_velocity_z_fixed[:, 0] == fixed)
    assert torch.all(engine.drop_x[:, 0] != engine.enemy_x[:, 0])

    # Grounded dropped actors move first and then receive Doom's 0xe800
    # friction, while pickup uses the independent item position.
    engine.drop_x[:, 0] = 0
    engine.drop_y[:, 0] = 0
    engine.drop_z[:, 0] = 0
    engine._drop_x_fixed[:, 0] = 0
    engine._drop_y_fixed[:, 0] = 0
    engine._drop_z_fixed[:, 0] = 0
    engine._drop_velocity_x_fixed[:, 0] = fixed
    engine._drop_velocity_y_fixed[:, 0] = 0
    engine._drop_velocity_z_fixed[:, 0] = 0
    engine._collect_drops()
    assert torch.all(engine._drop_x_fixed[:, 0] == fixed)
    assert torch.all(engine._drop_velocity_x_fixed[:, 0] == 0xE800)

    engine.x.fill_(1)
    engine.y.zero_()
    engine._drop_velocity_x_fixed[:, 0] = 0
    engine._collect_drops()
    assert engine.drop_type[:, 0].tolist() == [-1, -1]
    assert not torch.any(engine.drop_spawned[:, 0])


def test_policy_observation_contains_available_pickups(square_scenario) -> None:
    scenario = replace(
        square_scenario,
        item_spawns=np.asarray([(48.0, 0.0, 0.0)], dtype=np.float32),
        item_types=np.asarray([2012], dtype=np.int32),
    )
    engine = _engine(scenario)
    engine.angle.zero_()

    visible = engine.render_frame()
    engine.item_available.fill_(False)
    absent = engine.render_frame()

    assert not torch.equal(visible, absent)


def test_policy_observation_contains_selected_first_person_weapon(square_scenario) -> None:
    values = np.zeros((8, 84, 84), dtype=np.float32)
    alpha = np.zeros_like(values)
    values[0, 63:73, 31:53] = 48
    values[2, 63:73, 31:53] = 224
    alpha[:, 63:73, 31:53] = 1
    scenario = replace(
        square_scenario,
        weapon_screen_values=values,
        weapon_screen_alpha=alpha,
    )
    engine = _engine(scenario)
    engine.weapon_raise_cooldown.zero_()
    engine.selected_weapon.zero_()
    fist = engine.render_frame()
    engine.selected_weapon.fill_(2)
    pistol = engine.render_frame()

    assert not torch.equal(fist, pistol)
    assert torch.all(
        pistol[:, 63:73].to(torch.float32).mean(dim=(1, 2))
        > fist[:, 63:73].to(torch.float32).mean(dim=(1, 2))
    )


def test_first_person_weapon_resolves_shared_slot_variants(square_scenario) -> None:
    values = np.zeros((8, 84, 84), dtype=np.float32)
    alpha = np.zeros_like(values)
    for weapon in (0, 1, 3, 4):
        values[weapon, 63:73, 31:53] = 32 * (weapon + 1)
        alpha[weapon, 63:73, 31:53] = 1
    engine = _engine(
        replace(
            square_scenario,
            weapon_screen_values=values,
            weapon_screen_alpha=alpha,
        )
    )
    engine.weapon_raise_cooldown.zero_()

    engine.selected_weapon.fill_(1)
    engine.selected_weapon_variant.fill_(False)
    fist = engine.render_frame()
    engine.selected_weapon_variant.fill_(True)
    chainsaw = engine.render_frame()
    engine.selected_weapon.fill_(3)
    engine.selected_weapon_variant.fill_(False)
    shotgun = engine.render_frame()
    engine.selected_weapon_variant.fill_(True)
    super_shotgun = engine.render_frame()

    means = [
        frame[:, 63:73, 31:53].to(torch.float32).mean().item()
        for frame in (fist, chainsaw, shotgun, super_shotgun)
    ]
    assert means == sorted(means)


def test_empty_weapon_switches_to_best_owned_usable_weapon(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.selected_weapon.fill_(6)
    engine.weapons.fill_(1)
    engine.shotgun_owned.fill_(True)
    engine.super_shotgun_owned.fill_(True)
    engine.chainsaw_owned.fill_(True)
    engine.ammo.zero_()
    engine.ammo[:, 1].fill_(10)
    engine.ammo[:, 2].fill_(2)
    engine.ammo[:, 3].fill_(10)
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)

    assert engine.selected_weapon.tolist() == [6, 6]
    assert engine.pending_weapon.tolist() == [4, 4]
    assert engine.weapon_lower_cooldown.tolist() == [16, 16]
    _advance_weapon_switch(engine)
    assert engine.selected_weapon.tolist() == [3, 3]
    assert engine.selected_weapon_variant.tolist() == [True, True]
    assert engine.weapon_raise_cooldown.tolist() == [16, 16]
    assert engine.ammo[:, 2].tolist() == [2.0, 2.0]


def test_empty_weapon_falls_back_to_chainsaw_then_fist(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.selected_weapon.fill_(2)
    engine.ammo.zero_()
    engine.chainsaw_owned[0] = True
    engine.chainsaw_owned[1] = False
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)

    assert engine.pending_weapon.tolist() == [1, 0]
    _advance_weapon_switch(engine)
    assert engine.selected_weapon.tolist() == [1, 1]
    assert engine.selected_weapon_variant.tolist() == [True, False]
    assert engine.weapon_raise_cooldown.tolist() == [16, 16]


def test_rocket_uses_delayed_projectile_impact_and_splash(square_scenario) -> None:
    engine = _engine(
        replace(
            square_scenario,
            player_starts=square_scenario.player_starts[-1:],
        )
    )
    engine.angle.zero_()
    engine.weapons[:, 4].fill_(1)
    engine.ammo[:, 4].fill_(50)
    engine.selected_weapon.fill_(5)
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    immediate_reward = engine._player_attack(buttons)
    delayed_reward = _finish_pending_attack(engine)

    assert immediate_reward.tolist() == [0.0, 0.0]
    assert delayed_reward.tolist() == [0.0, 0.0]
    assert engine.enemy_health[:, 0].tolist() == [500.0, 500.0]
    assert torch.sum(engine.projectile_alive, dim=1).tolist() == [1, 1]
    assert torch.allclose(
        engine.projectile_x[:, 0],
        torch.full((2,), 9.993507385253906),
    )
    assert torch.allclose(
        engine.projectile_z[:, 0],
        torch.full((2,), 31.639732360839844),
    )
    active = torch.ones(2, dtype=torch.bool)
    for _ in range(3):
        engine._projectile_tick(active)

    assert torch.all(engine.enemy_health[:, 0] < 500)
    assert engine.health.tolist() == [16.0, 16.0]
    assert torch.equal(engine.momentum_x, torch.full((2,), -10.920028686523438))
    assert torch.equal(engine.momentum_y, torch.zeros(2))
    assert torch.allclose(
        engine.projectile_x[:, 0],
        torch.full((2,), 59.96104431152344),
    )
    assert not torch.any(engine.projectile_alive)
    assert engine.projectile_impact_type[:, 0].tolist() == [0, 0]
    assert engine.projectile_impact_tics[:, 0].tolist() == [18, 18]


def test_player_missile_uses_reference_side_probe_and_fine_angles(square_scenario) -> None:
    engine = _engine(_large_arena_scenario(square_scenario))
    engine.enemy_alive.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    target_angle = torch.deg2rad(torch.tensor([5.0, -5.0]))
    engine.enemy_x[:, 0] = torch.cos(target_angle) * 512.0
    engine.enemy_y[:, 0] = torch.sin(target_angle) * 512.0
    engine.enemy_z[:, 0] = 100
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True

    engine._execute_player_attack(
        torch.full((2,), 6),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    heading = torch.rad2deg(
        torch.atan2(
            engine.projectile_velocity_y[:, 0],
            engine.projectile_velocity_x[:, 0],
        )
    )
    # The negative lookup is intentionally asymmetric in ZDoom's generated
    # finesine table. Neither result points directly at the target's +/-5°.
    assert torch.allclose(
        heading,
        torch.tensor([5.624368667602539, -5.581620693206787]),
        atol=5e-4,
    )
    assert torch.all(engine.projectile_velocity_z[:, 0] > 0)
    speed = torch.sqrt(
        engine.projectile_velocity_x[:, 0] ** 2
        + engine.projectile_velocity_y[:, 0] ** 2
        + engine.projectile_velocity_z[:, 0] ** 2
    )
    assert torch.allclose(speed, torch.full((2,), 20.0), atol=2e-5)


def test_player_missile_autoaim_stops_at_reference_range(square_scenario) -> None:
    engine = _engine(_large_arena_scenario(square_scenario))
    engine.enemy_alive.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = torch.tensor([1000.0, 1500.0])
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 100
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True

    engine._execute_player_attack(
        torch.tensor([6, 7]),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert engine.projectile_velocity_z[0, 0] > 0
    assert engine.projectile_velocity_z[1, 0] == 0
    speed = torch.sqrt(
        engine.projectile_velocity_x[:, 0] ** 2
        + engine.projectile_velocity_y[:, 0] ** 2
        + engine.projectile_velocity_z[:, 0] ** 2
    )
    assert torch.allclose(speed, torch.tensor([20.0, 25.0]), atol=2e-5)
    assert engine.projectile_x[1, 0] == 12.5
    assert engine.projectile_z[1, 0] == 32.0


def test_player_missile_without_autoaim_uses_view_pitch(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.pitch.copy_(torch.deg2rad(torch.tensor([-10.0, 10.0])))

    engine._execute_player_attack(
        torch.tensor([6, 7]),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert engine.projectile_velocity_z[0, 0] > 0
    assert engine.projectile_velocity_z[1, 0] < 0


def test_melee_targeting_uses_view_pitch_and_actor_height(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.x.zero_()
    engine.y.zero_()
    engine.z.zero_()
    engine.angle.zero_()
    engine.enemy_x[:, 0] = 48
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 100
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 100
    engine.enemy_alive[:, 0] = True
    engine.pitch.copy_(torch.deg2rad(torch.tensor([0.0, -32.0])))

    engine._execute_player_attack(
        torch.zeros(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
    )

    assert engine.enemy_health[0, 0] == 100
    assert engine.enemy_health[1, 0] < 100


def test_rocket_radius_damage_uses_square_actor_bounds_and_height(square_scenario) -> None:
    engine = _engine(square_scenario)

    damage, points_fixed = engine._rocket_radius_damage(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([32.0, 100.0]),
        torch.tensor([100.0, 0.0]),
        torch.tensor([100.0, 0.0]),
        torch.tensor([0.0, 0.0]),
        torch.tensor([16.0, 16.0]),
        torch.tensor([56.0, 56.0]),
    )

    # Doom uses max(abs(dx), abs(dy)) and subtracts the target radius, so the
    # diagonal target takes the same 44 damage as an axial target at x=100.
    # Above the target, distance begins at the top of its actor box (z=56).
    assert damage.tolist() == [44.0, 84.0]
    assert points_fixed.tolist() == [44 * 65536, 84 * 65536]


def test_close_rocket_self_knockback_matches_vizdoom_oracle(square_scenario) -> None:
    scenario = replace(
        square_scenario,
        blocking_segments=np.asarray([(-128.0, 10.0, 128.0, 10.0)], dtype=np.float32),
        player_starts=square_scenario.player_starts[-1:],
    )
    engine = _engine(scenario)
    engine.enemy_alive.zero_()
    engine.x.fill_(2.1013336181640625)
    engine.y.fill_(-9.7767333984375)
    engine.z.zero_()
    engine.momentum_x.fill_(-0.4010772705078125)
    engine.momentum_y.zero_()
    engine.armor.fill_(200)
    engine.armor_save_fraction.fill_(0.5)
    engine.projectile_x[:, 0] = 0
    engine.projectile_y[:, 0] = 0
    engine.projectile_z[:, 0] = 32
    engine.projectile_velocity_x[:, 0] = 0
    engine.projectile_velocity_y[:, 0] = 20
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert engine.health.tolist() == [36.0, 36.0]
    assert engine.armor.tolist() == [136.0, 136.0]
    assert torch.equal(engine.momentum_x, torch.full((2,), 3.0825042724609375))
    assert torch.equal(engine.momentum_y, torch.full((2,), -16.268524169921875))
    assert torch.equal(engine.velocity_z, torch.zeros(2))

    engine.reaction_time.zero_()
    engine._move_player(torch.zeros((2, 20), dtype=torch.bool))

    assert torch.equal(engine.x, torch.full((2,), 5.183837890625))
    assert torch.equal(engine.y, torch.full((2,), -26.045257568359375))
    assert torch.equal(engine.momentum_x, torch.full((2,), 2.79351806640625))
    assert torch.equal(engine.momentum_y, torch.full((2,), -14.743362426757812))


def test_rocket_radius_thrust_can_launch_player_upward(square_scenario) -> None:
    scenario = replace(
        square_scenario,
        blocking_segments=np.asarray([(-128.0, 10.0, 128.0, 10.0)], dtype=np.float32),
    )
    engine = _engine(scenario)
    engine.enemy_alive.zero_()
    engine.health.fill_(1000)
    engine.x.fill_(2.1013336181640625)
    engine.y.fill_(-9.7767333984375)
    engine.z.zero_()
    engine.projectile_x[:, 0] = 0
    engine.projectile_y[:, 0] = 0
    engine.projectile_z[:, 0] = 0
    engine.projectile_velocity_x[:, 0] = 0
    engine.projectile_velocity_y[:, 0] = 20
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert torch.equal(engine.velocity_z, torch.full((2,), 14.33599853515625))


def test_rocket_radius_thrust_launches_enemy_with_reference_gravity(
    square_scenario,
) -> None:
    scenario = replace(
        square_scenario,
        blocking_segments=np.asarray([(-128.0, 10.0, 128.0, 10.0)], dtype=np.float32),
    )
    engine = _engine(scenario)
    engine.x.fill_(-200)
    engine.y.fill_(200)
    engine.health.fill_(1000)
    engine.enemy_x[:, 0] = 40
    engine.enemy_y[:, 0] = 0
    engine.enemy_z[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 1000
    engine.enemy_alive[:, 0] = True
    engine.projectile_x[:, 0] = 0
    engine.projectile_y[:, 0] = 0
    engine.projectile_z[:, 0] = 0
    engine.projectile_velocity_x[:, 0] = 0
    engine.projectile_velocity_y[:, 0] = 20
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert engine.enemy_health[:, 0].tolist() == [892.0, 892.0]
    # Radius thrust is 108 * 0.5 / mass 100, with the non-source vertical
    # factor 0.5 applied to the 28-unit center offset: 7.56 units/tic.
    assert torch.equal(
        engine._enemy_velocity_z_fixed[:, 0],
        torch.full((2,), 495452, dtype=torch.int64),
    )

    engine._move_enemy_thrust(torch.ones(2, dtype=torch.bool))

    assert torch.equal(
        engine._enemy_z_fixed[:, 0],
        torch.full((2,), 495452, dtype=torch.int64),
    )
    assert torch.equal(
        engine._enemy_velocity_z_fixed[:, 0],
        torch.full((2,), 495452 - 65536, dtype=torch.int64),
    )


def test_rocket_splash_respects_walls_and_pushes_only_visible_enemy(
    square_scenario,
) -> None:
    divider = np.asarray([(0.0, -256.0, 0.0, 256.0)], dtype=np.float32)
    walls = np.concatenate((square_scenario.wall_segments, divider), axis=0)
    scenario = replace(
        square_scenario,
        wall_segments=walls,
        blocking_segments=divider,
        blocking_wall_indices=np.asarray([4], dtype=np.int32),
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
    engine.x.fill_(-200)
    engine.y.fill_(200)
    engine.health.fill_(1000)
    engine.enemy_x[:, 0] = 30
    engine.enemy_x[:, 1] = -60
    engine.enemy_y[:, :2] = 0
    engine.enemy_z[:, :2] = 0
    engine.enemy_type[:, :2] = 5
    engine.enemy_health[:, :2] = 500
    engine.enemy_alive[:, :2] = True
    engine.projectile_x[:, 0] = -20
    engine.projectile_y[:, 0] = 0
    engine.projectile_z[:, 0] = 32
    engine.projectile_velocity_x[:, 0] = 20
    engine.projectile_velocity_y[:, 0] = 0
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    # The explosion remains at x=-13.333 when its next substep touches the
    # wall. The knight at x=30 is geometrically in range but has no sight line;
    # the knight on the explosion side takes 105 damage and both Doom thrusts.
    assert engine.enemy_health[:, :2].tolist() == [
        [500.0, 395.0],
        [500.0, 395.0],
    ]
    assert torch.equal(
        engine._enemy_momentum_x_fixed[:, :2],
        torch.tensor([[0, -89466], [0, -89466]], dtype=torch.int64),
    )
    assert torch.equal(
        engine._enemy_momentum_y_fixed[:, :2],
        torch.zeros((2, 2), dtype=torch.int64),
    )


def test_direct_rocket_kill_does_not_splash_already_dead_target(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.x.fill_(-200)
    engine.y.fill_(200)
    engine.health.fill_(1000)
    engine.enemy_x[:, 0] = 20
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 0
    engine.enemy_health[:, 0] = 20
    engine.enemy_alive[:, 0] = True
    engine.projectile_x[:, 0] = 0
    engine.projectile_y[:, 0] = 0
    engine.projectile_z[:, 0] = 32
    engine.projectile_velocity_x[:, 0] = 20
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert not torch.any(engine.enemy_alive[:, 0])
    assert engine.killcount.tolist() == [1, 1]
    # The seeded direct rolls are 120 and 60 damage. P_DamageMobj produces
    # exactly 15 and 7.5 units of thrust; radius damage never revisits a corpse.
    assert torch.equal(
        engine._enemy_momentum_x_fixed[:, 0],
        torch.tensor([15 * 65536, 15 * 65536 // 2], dtype=torch.int64),
    )
    assert torch.equal(
        engine._enemy_momentum_y_fixed[:, 0],
        torch.zeros(2, dtype=torch.int64),
    )


def test_plasma_uses_delayed_projectile_without_splash(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.angle.zero_()
    engine.weapons[:, 5].fill_(1)
    engine.ammo[:, 5].fill_(100)
    engine.selected_weapon.fill_(6)
    engine.enemy_x[:, 0] = 100
    engine.enemy_y[:, 0] = 0
    engine.enemy_type[:, 0] = 5
    engine.enemy_health[:, 0] = 500
    engine.enemy_alive[:, 0] = True
    buttons = torch.zeros((2, 20), dtype=torch.bool)
    buttons[:, 0] = True

    engine._player_attack(buttons)
    active = torch.ones(2, dtype=torch.bool)
    engine._projectile_tick(active)
    engine._projectile_tick(active)
    assert engine.enemy_health[:, 0].tolist() == [500.0, 500.0]
    engine._projectile_tick(active)

    assert torch.all(engine.enemy_health[:, 0] < 500)
    assert engine.health.tolist() == [100.0, 100.0]
    assert not torch.any(engine.projectile_alive)
    assert engine.projectile_impact_type[:, 0].tolist() == [1, 1]
    assert engine.projectile_impact_tics[:, 0].tolist() == [20, 20]


def test_missile_substeps_respect_actor_radius_when_grazing_walls(square_scenario) -> None:
    scenario = replace(
        square_scenario,
        blocking_segments=np.asarray([(10.0, -128.0, 10.0, 128.0)], dtype=np.float32),
    )
    engine = _engine(scenario)
    engine.enemy_alive.zero_()
    engine.projectile_x[:, 0] = 0
    engine.projectile_y[:, 0] = 0
    engine.projectile_z[:, 0] = 32
    engine.projectile_velocity_x[:, 0] = 0
    engine.projectile_velocity_y[:, 0] = 20
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert not torch.any(engine.projectile_alive[:, 0])
    assert engine.projectile_y[:, 0].tolist() == [0.0, 0.0]
    assert engine.projectile_impact_tics[:, 0].tolist() == [18, 18]


def test_missiles_clamp_explosion_origin_to_floor_and_ceiling(square_scenario) -> None:
    engine = _engine(square_scenario)
    engine.enemy_alive.zero_()
    engine.x.fill_(-200)
    engine.y.fill_(200)
    engine.health.fill_(1000)
    engine.projectile_x[:, 0] = 0
    engine.projectile_y[:, 0] = 0
    engine.projectile_z[:, 0] = torch.tensor([4.0, 120.0])
    engine.projectile_velocity_z[:, 0] = torch.tensor([-10.0, 10.0])
    engine.projectile_type[:, 0] = 0
    engine.projectile_alive[:, 0] = True

    engine._projectile_tick(torch.ones(2, dtype=torch.bool))

    assert engine.projectile_z[:, 0].tolist() == [0.0, 120.0]
    assert not torch.any(engine.projectile_alive[:, 0])

    engine.enemy_projectile_x[:, 0] = 0
    engine.enemy_projectile_y[:, 0] = 0
    engine.enemy_projectile_z[:, 0] = torch.tensor([4.0, 112.0])
    engine.enemy_projectile_velocity_z[:, 0] = torch.tensor([-10.0, 10.0])
    engine.enemy_projectile_alive[:, 0] = True

    engine._enemy_projectile_tick(torch.ones(2, dtype=torch.bool))

    assert engine.enemy_projectile_z[:, 0].tolist() == [0.0, 112.0]
    assert not torch.any(engine.enemy_projectile_alive[:, 0])


def test_enemy_missile_uses_reference_four_substeps(square_scenario) -> None:
    scenario = replace(
        square_scenario,
        blocking_segments=np.asarray([(5.0, -128.0, 5.0, 128.0)], dtype=np.float32),
    )
    engine = _engine(scenario)
    engine.x.fill_(200)
    engine.enemy_projectile_x[:, 0] = 0
    engine.enemy_projectile_y[:, 0] = 0
    engine.enemy_projectile_z[:, 0] = 32
    engine.enemy_projectile_velocity_x[:, 0] = 0
    engine.enemy_projectile_velocity_y[:, 0] = 15
    engine.enemy_projectile_alive[:, 0] = True

    engine._enemy_projectile_tick(torch.ones(2, dtype=torch.bool))

    assert not torch.any(engine.enemy_projectile_alive[:, 0])
    assert engine.enemy_projectile_y[:, 0].tolist() == [0.0, 0.0]
    assert engine.enemy_projectile_impact_tics[:, 0].tolist() == [18, 18]
