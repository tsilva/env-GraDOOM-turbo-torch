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
    engine = _engine(square_scenario)
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
    assert engine.projectile_x[:, 0].tolist() == [10.0, 10.0]
    active = torch.ones(2, dtype=torch.bool)
    for _ in range(3):
        engine._projectile_tick(active)

    assert torch.all(engine.enemy_health[:, 0] < 500)
    assert engine.health.tolist() == [36.0, 36.0]
    assert torch.allclose(engine.projectile_x[:, 0], torch.full((2,), 190.0 / 3.0))
    assert not torch.any(engine.projectile_alive)
    assert engine.projectile_impact_type[:, 0].tolist() == [0, 0]
    assert engine.projectile_impact_tics[:, 0].tolist() == [18, 18]


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
