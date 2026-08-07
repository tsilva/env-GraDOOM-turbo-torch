"""Device-resident vector execution model for the deathmatch fast path.

This is the correctness-first tensor implementation. It deliberately keeps the
state layout and API independent from Torch so individual operations can be
replaced by fused C++/CUDA kernels without changing the public contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .scenario import CompiledScenario

_UINT32_MASK = (1 << 32) - 1
_ENEMY_HEALTH = (20.0, 30.0, 100.0, 70.0, 150.0, 500.0)
_ENEMY_STRIDE = (8.0, 8.0, 8.0, 8.0, 10.0, 8.0)
_ENEMY_MOVE_INTERVAL = (4, 3, 4, 3, 2, 3)
_ENEMY_RADIUS = (20.0, 20.0, 20.0, 20.0, 30.0, 24.0)
_ENEMY_HEIGHT = (56.0, 56.0, 56.0, 56.0, 56.0, 64.0)
_ENEMY_ATTACK_RANGE = (2048.0, 2048.0, 64.0, 2048.0, 64.0, 2048.0)
_ENEMY_ATTACK_PREFIRE = (10, 10, 4, 10, 16, 16)
_ENEMY_ATTACK_RECOVERY = (16, 20, 4, 4, 8, 8)
_ENEMY_KILL_REWARD = (1.0, 3.0, 3.0, 4.0, 3.0, 10.0)
_ENEMY_SPAWN_THRESHOLD = (2621, 2621, 1310, 1310, 655, 655)
_ENEMY_SPAWN_DELAY = 105
_ENEMY_SPAWN_PERIOD = 10
_PLAYER_TELEPORT_LOCK_TICS = 7
_PLAYER_ACCELERATION = 0.78125
_PLAYER_FRICTION = 0.90625
_PLAYER_SLIDE_FRACTION = 31.0 / 32.0
_PLAYER_TURN_DEGREES = 3.515625
_WEAPON_LOWER_TICS = 16
_WEAPON_RAISE_TICS = 18
# Internal weapon order follows the DoomPlayer slot lists exactly:
# fist, chainsaw, pistol, shotgun, super shotgun, chaingun, rocket, plasma.
_WEAPON_SLOT = (1, 1, 2, 3, 3, 4, 5, 6)
_WEAPON_COOLDOWN = (22, 4, 14, 37, 51, 4, 20, 3)
_WEAPON_AMMO_SLOT = (-1, -1, 1, 2, 2, 1, 4, 5)
_WEAPON_AMMO_COST = (0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0)
# Ascending Doom Weapon.SelectionOrder, restricted to the certified profile.
_WEAPON_AUTO_SWITCH_ORDER = (7, 4, 5, 3, 2, 1, 6, 0)
_MONSTER_DROP_TYPE = (2007, 2001, -1, 2002, -1, -1)
_GREEN_ARMOR_SAVE = 21846.0 / 65536.0
_BLUE_ARMOR_SAVE = 0.5
_PLAYER_RADIUS = 16.0
_PLAYER_HEIGHT = 56.0
_PICKUP_RADIUS = 20.0
_PICKUP_REACH_BELOW = 32.0
_VIEW_HEIGHT = 41.0
_PROJECTION_FOCAL_LENGTH = 42.0
_PORTAL_LAYERS = 8
_HASH_GOLDEN_RATIO_SIGNED = -1640531527
_HASH_MURMUR_SIGNED = -2048144789
_PLAYER_PROJECTILE_SPEED = (20.0, 25.0)
_PLAYER_PROJECTILE_LIFETIME = 100
_ENEMY_PROJECTILE_SPEED = 15.0
_ENEMY_PROJECTILE_LIFETIME = 140
_ITEM_SPRITE_INDEX = {
    2011: 6,
    2012: 7,
    2014: 8,
    2015: 9,
    2018: 10,
    2019: 11,
    2007: 12,
    2048: 13,
    2049: 14,
    2046: 15,
    17: 16,
    2005: 17,
    2001: 18,
    82: 19,
    2002: 20,
    2003: 21,
    2004: 22,
}
DEVICE_SIGNAL_NAMES = (
    "killcount",
    "health",
    "armor",
    "selected_weapon",
    "selected_weapon_ammo",
    "weapon1",
    "weapon2",
    "weapon3",
    "weapon4",
    "weapon5",
    "weapon6",
    "ammo1",
    "ammo2",
    "ammo3",
    "ammo4",
    "ammo5",
    "ammo6",
    "episode_time",
    "episode_return",
    "player_dead",
    "pending_reset",
)


@dataclass(frozen=True)
class DeviceScenario:
    walls: torch.Tensor
    wall_lights: torch.Tensor
    wall_texture_ids: torch.Tensor
    wall_texture_offsets: torch.Tensor
    wall_lengths: torch.Tensor
    texture_atlas: torch.Tensor
    texture_widths: torch.Tensor
    texture_heights: torch.Tensor
    portal_walls: torch.Tensor
    portal_wall_sectors: torch.Tensor
    portal_wall_lights: torch.Tensor
    portal_texture_ids: torch.Tensor
    portal_texture_offsets: torch.Tensor
    portal_side_texture_ids: torch.Tensor
    portal_side_texture_offsets: torch.Tensor
    portal_wall_lengths: torch.Tensor
    sector_edges: torch.Tensor
    sector_edge_mask: torch.Tensor
    sector_heights: torch.Tensor
    sector_lights: torch.Tensor
    sector_floor_texture_ids: torch.Tensor
    sector_ceiling_texture_ids: torch.Tensor
    sprite_atlas: torch.Tensor
    sprite_opaque: torch.Tensor
    sprite_widths: torch.Tensor
    sprite_heights: torch.Tensor
    sprite_left_offsets: torch.Tensor
    sprite_top_offsets: torch.Tensor
    weapon_screen_values: torch.Tensor
    weapon_screen_alpha: torch.Tensor
    blocking_walls: torch.Tensor
    player_starts: torch.Tensor
    item_spawns: torch.Tensor
    item_types: torch.Tensor
    item_visual_types: torch.Tensor
    bounds: torch.Tensor
    spawn_bounds: torch.Tensor

    @classmethod
    def from_host(cls, scenario: CompiledScenario, device: torch.device) -> DeviceScenario:
        blocking_indices = scenario.blocking_wall_indices
        sector_indices = scenario.wall_sectors[blocking_indices, 0].clip(min=0)
        wall_lights = scenario.sector_lights[sector_indices].astype("float32")
        blocking_walls = scenario.blocking_segments
        wall_lengths = np.sqrt(
            np.square(blocking_walls[:, 2] - blocking_walls[:, 0])
            + np.square(blocking_walls[:, 3] - blocking_walls[:, 1])
        ).astype(np.float32)
        portal_sector_indices = scenario.wall_sectors[:, 0].clip(min=0)
        portal_wall_lights = scenario.sector_lights[portal_sector_indices].astype("float32")
        portal_wall_lengths = np.sqrt(
            np.square(scenario.wall_segments[:, 2] - scenario.wall_segments[:, 0])
            + np.square(scenario.wall_segments[:, 3] - scenario.wall_segments[:, 1])
        ).astype(np.float32)
        bounds = scenario.bounds
        if scenario.scenario_sha256 == (
            "1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d"
        ):
            spawn_bounds = (32.0, 992.0, 32.0, 992.0)
        else:
            inset = min(32.0, (bounds[1] - bounds[0]) / 4, (bounds[3] - bounds[2]) / 4)
            spawn_bounds = (
                bounds[0] + inset,
                bounds[1] - inset,
                bounds[2] + inset,
                bounds[3] - inset,
            )
        item_visual_types = np.full(len(scenario.item_types), -1, dtype=np.int64)
        for type_id, sprite_index in _ITEM_SPRITE_INDEX.items():
            item_visual_types[scenario.item_types == type_id] = sprite_index
        if np.any(item_visual_types < 0):
            unsupported = sorted(set(scenario.item_types[item_visual_types < 0].tolist()))
            raise ValueError(f"scenario contains unsupported item types: {unsupported}")
        return cls(
            walls=torch.as_tensor(scenario.blocking_segments, device=device),
            wall_lights=torch.as_tensor(wall_lights, device=device),
            wall_texture_ids=torch.as_tensor(
                scenario.wall_texture_ids[blocking_indices], device=device, dtype=torch.int64
            ),
            wall_texture_offsets=torch.as_tensor(
                scenario.wall_texture_offsets[blocking_indices], device=device
            ),
            wall_lengths=torch.as_tensor(wall_lengths, device=device),
            texture_atlas=torch.as_tensor(scenario.texture_atlas, device=device),
            texture_widths=torch.as_tensor(
                scenario.texture_widths, device=device, dtype=torch.int64
            ),
            texture_heights=torch.as_tensor(
                scenario.texture_heights, device=device, dtype=torch.int64
            ),
            portal_walls=torch.as_tensor(scenario.wall_segments, device=device),
            portal_wall_sectors=torch.as_tensor(
                scenario.wall_sectors, device=device, dtype=torch.int64
            ),
            portal_wall_lights=torch.as_tensor(portal_wall_lights, device=device),
            portal_texture_ids=torch.as_tensor(
                scenario.wall_texture_ids, device=device, dtype=torch.int64
            ),
            portal_texture_offsets=torch.as_tensor(scenario.wall_texture_offsets, device=device),
            portal_side_texture_ids=torch.as_tensor(
                scenario.wall_side_texture_ids,
                device=device,
                dtype=torch.int64,
            ),
            portal_side_texture_offsets=torch.as_tensor(
                scenario.wall_side_texture_offsets,
                device=device,
            ),
            portal_wall_lengths=torch.as_tensor(portal_wall_lengths, device=device),
            sector_edges=torch.as_tensor(scenario.wall_segments, device=device),
            sector_edge_mask=torch.as_tensor(scenario.sector_edge_mask, device=device),
            sector_heights=torch.as_tensor(scenario.sector_heights, device=device),
            sector_lights=torch.as_tensor(
                scenario.sector_lights, device=device, dtype=torch.float32
            ),
            sector_floor_texture_ids=torch.as_tensor(
                scenario.sector_floor_texture_ids, device=device, dtype=torch.int64
            ),
            sector_ceiling_texture_ids=torch.as_tensor(
                scenario.sector_ceiling_texture_ids, device=device, dtype=torch.int64
            ),
            sprite_atlas=torch.as_tensor(scenario.sprite_atlas, device=device),
            sprite_opaque=torch.as_tensor(scenario.sprite_opaque, device=device),
            sprite_widths=torch.as_tensor(scenario.sprite_widths, device=device, dtype=torch.int64),
            sprite_heights=torch.as_tensor(
                scenario.sprite_heights, device=device, dtype=torch.int64
            ),
            sprite_left_offsets=torch.as_tensor(
                scenario.sprite_left_offsets, device=device, dtype=torch.float32
            ),
            sprite_top_offsets=torch.as_tensor(
                scenario.sprite_top_offsets, device=device, dtype=torch.float32
            ),
            weapon_screen_values=torch.as_tensor(scenario.weapon_screen_values, device=device),
            weapon_screen_alpha=torch.as_tensor(scenario.weapon_screen_alpha, device=device),
            blocking_walls=torch.as_tensor(scenario.blocking_segments, device=device),
            player_starts=torch.as_tensor(scenario.player_starts, device=device),
            item_spawns=torch.as_tensor(scenario.item_spawns, device=device),
            item_types=torch.as_tensor(scenario.item_types, device=device, dtype=torch.int64),
            item_visual_types=torch.as_tensor(item_visual_types, device=device),
            bounds=torch.tensor(bounds, device=device),
            spawn_bounds=torch.tensor(spawn_bounds, device=device),
        )


class TorchDeathmatchEngine:
    """Batched Doom-like state machine whose mutable state never leaves its device."""

    observation_height = 84
    observation_width = 84
    enemy_slots = 64
    player_projectile_slots = 32

    def __init__(
        self,
        scenario: CompiledScenario,
        num_envs: int,
        *,
        device: torch.device,
        frame_skip: int = 2,
        frame_stack: int = 4,
        episode_timeout: int = 4200,
        mask_hud: bool = True,
        debug_checks: bool | None = None,
    ) -> None:
        self.device = device
        self.num_envs = num_envs
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.episode_timeout = episode_timeout
        self.mask_hud = mask_hud
        self.debug_checks = device.type == "cpu" if debug_checks is None else debug_checks
        self.map = DeviceScenario.from_host(scenario, device)
        n = num_envs
        self.rng_state = torch.ones(n, device=device, dtype=torch.int64)
        self.episode_time = torch.zeros(n, device=device, dtype=torch.int32)
        self.episode_return = torch.zeros(n, device=device)
        self.pending_reset = torch.ones(n, device=device, dtype=torch.bool)
        self.player_dead = torch.zeros(n, device=device, dtype=torch.bool)
        self.x = torch.zeros(n, device=device)
        self.y = torch.zeros(n, device=device)
        self.z = torch.zeros(n, device=device)
        self.angle = torch.zeros(n, device=device)
        self.momentum_x = torch.zeros(n, device=device)
        self.momentum_y = torch.zeros(n, device=device)
        self.velocity_z = torch.zeros(n, device=device)
        self.health = torch.full((n,), 100.0, device=device)
        self.armor = torch.zeros(n, device=device)
        self.armor_save_fraction = torch.zeros(n, device=device)
        self.killcount = torch.zeros(n, device=device, dtype=torch.int32)
        self.selected_weapon = torch.full((n,), 2, device=device, dtype=torch.int64)
        self.selected_weapon_variant = torch.zeros(n, device=device, dtype=torch.bool)
        self.weapons = torch.zeros((n, 6), device=device)
        self.chainsaw_owned = torch.zeros(n, device=device, dtype=torch.bool)
        self.shotgun_owned = torch.zeros(n, device=device, dtype=torch.bool)
        self.super_shotgun_owned = torch.zeros(n, device=device, dtype=torch.bool)
        self.ammo = torch.zeros((n, 6), device=device)
        self.attack_cooldown = torch.zeros(n, device=device, dtype=torch.int32)
        self.weapon_raise_cooldown = torch.zeros(n, device=device, dtype=torch.int32)
        self.pending_weapon = torch.full((n,), -1, device=device, dtype=torch.int64)
        self.weapon_lower_cooldown = torch.zeros(n, device=device, dtype=torch.int32)
        self.weapon_change_latched = torch.zeros(n, device=device, dtype=torch.bool)
        self.damage_flash = torch.zeros(n, device=device)
        self.reaction_time = torch.zeros(n, device=device, dtype=torch.int32)
        self.enemy_x = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_y = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_z = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_angle = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_type = torch.full((n, self.enemy_slots), -1, device=device, dtype=torch.int64)
        self.enemy_health = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_alive = torch.zeros((n, self.enemy_slots), device=device, dtype=torch.bool)
        self.enemy_cooldown = torch.zeros((n, self.enemy_slots), device=device, dtype=torch.int32)
        self.enemy_attack_phase = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_move_cooldown = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.drop_type = torch.full((n, self.enemy_slots), -1, device=device, dtype=torch.int64)
        self.drop_delay = torch.zeros((n, self.enemy_slots), device=device, dtype=torch.int32)
        self.projectile_x = torch.zeros((n, self.player_projectile_slots), device=device)
        self.projectile_y = torch.zeros((n, self.player_projectile_slots), device=device)
        self.projectile_z = torch.zeros((n, self.player_projectile_slots), device=device)
        self.projectile_velocity_x = torch.zeros((n, self.player_projectile_slots), device=device)
        self.projectile_velocity_y = torch.zeros((n, self.player_projectile_slots), device=device)
        self.projectile_velocity_z = torch.zeros((n, self.player_projectile_slots), device=device)
        self.projectile_type = torch.full(
            (n, self.player_projectile_slots), -1, device=device, dtype=torch.int64
        )
        self.projectile_age = torch.zeros(
            (n, self.player_projectile_slots), device=device, dtype=torch.int32
        )
        self.projectile_alive = torch.zeros(
            (n, self.player_projectile_slots), device=device, dtype=torch.bool
        )
        self.enemy_projectile_x = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_projectile_y = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_projectile_z = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_projectile_velocity_x = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_projectile_velocity_y = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_projectile_velocity_z = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_projectile_age = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_projectile_alive = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.bool
        )
        self.next_spawn_check = torch.zeros(n, device=device, dtype=torch.int32)
        self.item_available = torch.ones(
            (n, len(self.map.item_types)), device=device, dtype=torch.bool
        )
        self.frames = torch.zeros(
            (n, frame_stack, self.observation_height, self.observation_width),
            device=device,
            dtype=torch.uint8,
        )
        self.signal_buffer = torch.zeros(
            (n, len(DEVICE_SIGNAL_NAMES)), device=device, dtype=torch.float32
        )
        self._enemy_base_health = torch.tensor(_ENEMY_HEALTH, device=device)
        self._enemy_stride = torch.tensor(_ENEMY_STRIDE, device=device)
        self._enemy_move_interval = torch.tensor(
            _ENEMY_MOVE_INTERVAL, device=device, dtype=torch.int32
        )
        self._enemy_radius = torch.tensor(_ENEMY_RADIUS, device=device)
        self._enemy_height = torch.tensor(_ENEMY_HEIGHT, device=device)
        self._enemy_attack_range = torch.tensor(_ENEMY_ATTACK_RANGE, device=device)
        self._enemy_attack_prefire = torch.tensor(
            _ENEMY_ATTACK_PREFIRE, device=device, dtype=torch.int32
        )
        self._enemy_attack_recovery = torch.tensor(
            _ENEMY_ATTACK_RECOVERY, device=device, dtype=torch.int32
        )
        self._enemy_kill_reward = torch.tensor(_ENEMY_KILL_REWARD, device=device)
        self._enemy_spawn_threshold = torch.tensor(
            _ENEMY_SPAWN_THRESHOLD, device=device, dtype=torch.int64
        )
        self._weapon_slot = torch.tensor(_WEAPON_SLOT, device=device, dtype=torch.int64)
        self._weapon_cooldown = torch.tensor(_WEAPON_COOLDOWN, device=device, dtype=torch.int32)
        self._weapon_ammo_slot = torch.tensor(_WEAPON_AMMO_SLOT, device=device, dtype=torch.int64)
        self._weapon_ammo_cost = torch.tensor(_WEAPON_AMMO_COST, device=device)
        self._player_projectile_speed = torch.tensor(_PLAYER_PROJECTILE_SPEED, device=device)
        self._monster_drop_type = torch.tensor(_MONSTER_DROP_TYPE, device=device, dtype=torch.int64)
        self._slot_base_weapon = torch.tensor(
            (0, 0, 2, 3, 5, 6, 7), device=device, dtype=torch.int64
        )
        self._ray_offsets = torch.linspace(
            -math.pi / 4,
            math.pi / 4,
            self.observation_width,
            device=device,
        )
        self._pixel_x = torch.arange(self.observation_width, device=device).view(1, 1, -1)
        self._pixel_y = torch.arange(self.observation_height, device=device).view(1, -1, 1)
        player_start_sectors = self._sector_at(
            self.map.player_starts[:, 0], self.map.player_starts[:, 1]
        )
        self._player_start_z = self.map.sector_heights[player_start_sectors, 0]
        if len(self.map.item_spawns):
            item_sectors = self._sector_at(self.map.item_spawns[:, 0], self.map.item_spawns[:, 1])
            self._item_z = (
                self.map.sector_heights[item_sectors, 0] + self.map.item_spawns[:, 2]
            )
        else:
            self._item_z = torch.empty(0, device=device)

    def _random_u32(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        value = self.rng_state
        updated = torch.bitwise_xor(value, torch.bitwise_and(value << 13, _UINT32_MASK))
        updated = torch.bitwise_xor(updated, updated >> 17)
        updated = torch.bitwise_xor(updated, torch.bitwise_and(updated << 5, _UINT32_MASK))
        updated = torch.bitwise_and(updated, _UINT32_MASK)
        if mask is not None:
            updated = torch.where(mask, updated, value)
        self.rng_state.copy_(updated)
        return self.rng_state

    def _random_unit(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self._random_u32(mask).to(torch.float32) * (1.0 / 4294967296.0)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi

    def reset(self, mask: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise TypeError(
                "reset mask must be a device bool tensor with one value per environment"
            )
        safe_seeds = torch.bitwise_and(seeds.to(self.device, torch.int64), _UINT32_MASK)
        safe_seeds = torch.where(
            safe_seeds == 0,
            torch.full_like(safe_seeds, 0x6D2B79F5),
            safe_seeds,
        )
        self.rng_state.copy_(torch.where(mask, safe_seeds, self.rng_state))
        # Sequential lane seeds have strongly correlated first xorshift32 outputs.
        # Four masked diffusion rounds retain deterministic streams while preventing
        # the first spatial sample from collapsing toward the low edge of the map.
        for _ in range(4):
            self._random_u32(mask)
        self._reset_enemies(mask)
        spawn_x, spawn_y, spawn_angle, _ = self._random_spawn_positions(mask, avoid_player=False)
        self.x.copy_(torch.where(mask, spawn_x, self.x))
        self.y.copy_(torch.where(mask, spawn_y, self.y))
        self.angle.copy_(torch.where(mask, spawn_angle, self.angle))
        spawn_sector = self._sector_at(spawn_x, spawn_y)
        spawn_z = self.map.sector_heights[spawn_sector, 0]
        self.z.copy_(torch.where(mask, spawn_z, self.z))
        for tensor in (
            self.momentum_x,
            self.momentum_y,
            self.velocity_z,
            self.armor,
            self.armor_save_fraction,
            self.episode_return,
        ):
            tensor.masked_fill_(mask, 0)
        self.health.masked_fill_(mask, 100)
        self.killcount.masked_fill_(mask, 0)
        self.episode_time.masked_fill_(mask, 1)
        self.selected_weapon.masked_fill_(mask, 2)
        self.selected_weapon_variant.masked_fill_(mask, False)
        self.attack_cooldown.masked_fill_(mask, 0)
        self.weapon_raise_cooldown.masked_fill_(mask, 0)
        self.pending_weapon.masked_fill_(mask, -1)
        self.weapon_lower_cooldown.masked_fill_(mask, 0)
        self.weapon_change_latched.masked_fill_(mask, False)
        self.damage_flash.masked_fill_(mask, 0)
        self.reaction_time.masked_fill_(mask, _PLAYER_TELEPORT_LOCK_TICS)
        self.player_dead.masked_fill_(mask, False)
        self.pending_reset.masked_fill_(mask, False)
        self.weapons[mask] = 0
        self.weapons[mask, 0] = 1
        self.weapons[mask, 1] = 1
        self.chainsaw_owned.masked_fill_(mask, False)
        self.shotgun_owned.masked_fill_(mask, False)
        self.super_shotgun_owned.masked_fill_(mask, False)
        self.ammo[mask] = 0
        self.ammo[mask, 1] = 50
        self.ammo[mask, 3] = 50
        self.item_available[mask] = True
        frame = self.render_frame()
        self.frames[mask] = frame[mask, None].expand(-1, self.frame_stack, -1, -1)
        self._update_signal_buffer()
        return self.frames

    def _reset_enemies(self, mask: torch.Tensor) -> None:
        self.enemy_x[mask] = 0
        self.enemy_y[mask] = 0
        self.enemy_z[mask] = 0
        self.enemy_angle[mask] = 0
        self.enemy_type[mask] = -1
        self.enemy_health[mask] = 0
        self.enemy_alive[mask] = False
        self.enemy_cooldown[mask] = 0
        self.enemy_attack_phase[mask] = 0
        self.enemy_move_cooldown[mask] = 0
        self.drop_type[mask] = -1
        self.drop_delay[mask] = 0
        self.projectile_x[mask] = 0
        self.projectile_y[mask] = 0
        self.projectile_z[mask] = 0
        self.projectile_velocity_x[mask] = 0
        self.projectile_velocity_y[mask] = 0
        self.projectile_velocity_z[mask] = 0
        self.projectile_type[mask] = -1
        self.projectile_age[mask] = 0
        self.projectile_alive[mask] = False
        self.enemy_projectile_x[mask] = 0
        self.enemy_projectile_y[mask] = 0
        self.enemy_projectile_z[mask] = 0
        self.enemy_projectile_velocity_x[mask] = 0
        self.enemy_projectile_velocity_y[mask] = 0
        self.enemy_projectile_velocity_z[mask] = 0
        self.enemy_projectile_age[mask] = 0
        self.enemy_projectile_alive[mask] = False
        self.next_spawn_check.masked_fill_(mask, 1 + _ENEMY_SPAWN_DELAY)

    def _points_collide(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        radius: float | torch.Tensor = _PLAYER_RADIUS,
    ) -> torch.Tensor:
        walls = self.map.blocking_walls
        if not len(walls):
            return torch.zeros_like(x, dtype=torch.bool)
        point = torch.stack((x, y), dim=-1)[..., None, :]
        start = walls[:, :2]
        delta = walls[:, 2:] - start
        length_sq = torch.sum(delta * delta, dim=-1).clamp_min_(1e-6)
        along = torch.sum((point - start) * delta, dim=-1) / length_sq
        along = along.clamp(0, 1)
        closest = start + along[..., None] * delta
        distance_sq = torch.sum((point - closest) ** 2, dim=-1)
        collision_radius = torch.as_tensor(radius, device=self.device, dtype=x.dtype)
        return torch.any(distance_sq < collision_radius[..., None] ** 2, dim=-1)

    def _random_spawn_positions(
        self,
        mask: torch.Tensor,
        *,
        avoid_player: bool,
        candidate_count: int = 16,
        actor_radius: float | torch.Tensor = _PLAYER_RADIUS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low_x, high_x, low_y, high_y = self.map.spawn_bounds
        unit_x = torch.stack([self._random_unit(mask) for _ in range(candidate_count)], dim=1)
        unit_y = torch.stack([self._random_unit(mask) for _ in range(candidate_count)], dim=1)
        candidate_x = low_x + unit_x * (high_x - low_x)
        candidate_y = low_y + unit_y * (high_y - low_y)
        radius = torch.as_tensor(actor_radius, device=self.device, dtype=candidate_x.dtype)
        valid = ~self._points_collide(candidate_x, candidate_y, radius)

        if len(self.map.player_starts) > 1:
            dolls = self.map.player_starts[:-1, :2]
            doll_dx = candidate_x[..., None] - dolls[None, None, :, 0]
            doll_dy = candidate_y[..., None] - dolls[None, None, :, 1]
            valid &= torch.all(
                doll_dx * doll_dx + doll_dy * doll_dy >= (radius + _PLAYER_RADIUS) ** 2,
                dim=-1,
            )
        if avoid_player:
            player_dx = candidate_x - self.x[:, None]
            player_dy = candidate_y - self.y[:, None]
            valid &= player_dx * player_dx + player_dy * player_dy >= (radius + _PLAYER_RADIUS) ** 2
            enemy_dx = candidate_x[..., None] - self.enemy_x[:, None, :]
            enemy_dy = candidate_y[..., None] - self.enemy_y[:, None, :]
            enemy_radius = self._enemy_radius[self.enemy_type.clamp_min(0)]
            overlaps_enemy = self.enemy_alive[:, None, :] & (
                enemy_dx * enemy_dx + enemy_dy * enemy_dy < (radius + enemy_radius[:, None, :]) ** 2
            )
            valid &= ~torch.any(overlaps_enemy, dim=-1)

        has_valid = torch.any(valid, dim=1) & mask
        chosen = torch.argmax(valid.to(torch.int32), dim=1)
        row = torch.arange(self.num_envs, device=self.device)
        fallback = self.map.player_starts[-1]
        x = torch.where(has_valid, candidate_x[row, chosen], fallback[0])
        y = torch.where(has_valid, candidate_y[row, chosen], fallback[1])
        angle = self._random_unit(mask) * (2 * math.pi)
        return x, y, angle, has_valid

    def _collides(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._points_collide(x, y)

    @staticmethod
    def _vertical_overlap(
        first_z: torch.Tensor,
        first_height: float | torch.Tensor,
        second_z: torch.Tensor,
        second_height: float | torch.Tensor,
    ) -> torch.Tensor:
        return (first_z < second_z + second_height) & (second_z < first_z + first_height)

    def _player_collides(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        collision = self._collides(x, y)
        sector = self._sector_at(x, y)
        floor = self.map.sector_heights[sector, 0]
        ceiling = self.map.sector_heights[sector, 1]
        collision |= floor > self.z + 24.0
        collision |= ceiling - torch.maximum(self.z, floor) < 56.0
        enemy_type = self.enemy_type.clamp_min(0)
        enemy_radius = self._enemy_radius[enemy_type]
        enemy_dx = x[:, None] - self.enemy_x
        enemy_dy = y[:, None] - self.enemy_y
        enemy_vertical_overlap = self._vertical_overlap(
            self.z[:, None],
            _PLAYER_HEIGHT,
            self.enemy_z,
            self._enemy_height[enemy_type],
        )
        collision |= torch.any(
            self.enemy_alive
            & enemy_vertical_overlap
            & (enemy_dx * enemy_dx + enemy_dy * enemy_dy < (_PLAYER_RADIUS + enemy_radius) ** 2),
            dim=1,
        )
        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            dolls = self.map.player_starts[:-1, :2]
            doll_dx = x[:, None] - dolls[None, :, 0]
            doll_dy = y[:, None] - dolls[None, :, 1]
            doll_overlap = self._vertical_overlap(
                self.z[:, None],
                _PLAYER_HEIGHT,
                self._player_start_z[:-1][None, :],
                _PLAYER_HEIGHT,
            )
            collision |= torch.any(
                doll_overlap
                & (doll_dx * doll_dx + doll_dy * doll_dy < (2 * _PLAYER_RADIUS) ** 2),
                dim=1,
            )
        return collision

    def _enemy_collides(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        enemy_type: torch.Tensor,
        sector: torch.Tensor,
    ) -> torch.Tensor:
        radius = self._enemy_radius[enemy_type]
        collision = self._points_collide(x, y, radius)
        floor = self.map.sector_heights[sector, 0]
        ceiling = self.map.sector_heights[sector, 1]
        height = self._enemy_height[enemy_type]
        collision |= floor > self.enemy_z + 24.0
        collision |= ceiling - torch.maximum(self.enemy_z, floor) < height
        dx = x[:, :, None] - self.enemy_x[:, None, :]
        dy = y[:, :, None] - self.enemy_y[:, None, :]
        other_radius = self._enemy_radius[self.enemy_type.clamp_min(0)]
        vertical_overlap = self._vertical_overlap(
            self.enemy_z[:, :, None],
            height[:, :, None],
            self.enemy_z[:, None, :],
            self._enemy_height[self.enemy_type.clamp_min(0)][:, None, :],
        )
        not_self = ~torch.eye(
            self.enemy_slots,
            device=self.device,
            dtype=torch.bool,
        )[None, :, :]
        solid_enemy = self.enemy_alive[:, None, :] & not_self
        collision |= torch.any(
            solid_enemy
            & vertical_overlap
            & (dx * dx + dy * dy < (radius[:, :, None] + other_radius[:, None, :]) ** 2),
            dim=2,
        )
        player_dx = x - self.x[:, None]
        player_dy = y - self.y[:, None]
        player_overlap = self._vertical_overlap(
            self.enemy_z,
            height,
            self.z[:, None],
            _PLAYER_HEIGHT,
        )
        collision |= (
            ~self.player_dead[:, None]
            & player_overlap
            & (player_dx * player_dx + player_dy * player_dy < (radius + _PLAYER_RADIUS) ** 2)
        )
        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            dolls = self.map.player_starts[:-1, :2]
            doll_dx = x[:, :, None] - dolls[None, None, :, 0]
            doll_dy = y[:, :, None] - dolls[None, None, :, 1]
            doll_overlap = self._vertical_overlap(
                self.enemy_z[:, :, None],
                height[:, :, None],
                self._player_start_z[:-1][None, None, :],
                _PLAYER_HEIGHT,
            )
            collision |= torch.any(
                doll_overlap
                & (
                    doll_dx * doll_dx + doll_dy * doll_dy
                    < (radius[:, :, None] + _PLAYER_RADIUS) ** 2
                ),
                dim=2,
            )
        return collision

    def _axis_collision_fraction(
        self,
        move_x: torch.Tensor,
        move_y: torch.Tensor,
    ) -> torch.Tensor:
        """Return first swept contact with axis-aligned blocking lines."""

        walls = self.map.blocking_walls
        x1 = walls[:, 0]
        y1 = walls[:, 1]
        x2 = walls[:, 2]
        y2 = walls[:, 3]
        horizontal = (y1 == y2)[None, :]
        vertical = (x1 == x2)[None, :]
        safe_move_x = torch.where(move_x[:, None].abs() < 1e-6, 1.0, move_x[:, None])
        safe_move_y = torch.where(move_y[:, None].abs() < 1e-6, 1.0, move_y[:, None])

        target_y = y1[None, :] - torch.sign(move_y[:, None]) * _PLAYER_RADIUS
        horizontal_fraction = (target_y - self.y[:, None]) / safe_move_y
        horizontal_x = self.x[:, None] + move_x[:, None] * horizontal_fraction
        horizontal_valid = (
            horizontal
            & (move_y[:, None].abs() >= 1e-6)
            & (horizontal_fraction >= 0)
            & (horizontal_fraction <= 1)
            & (horizontal_x >= torch.minimum(x1, x2)[None, :] - _PLAYER_RADIUS)
            & (horizontal_x <= torch.maximum(x1, x2)[None, :] + _PLAYER_RADIUS)
        )

        target_x = x1[None, :] - torch.sign(move_x[:, None]) * _PLAYER_RADIUS
        vertical_fraction = (target_x - self.x[:, None]) / safe_move_x
        vertical_y = self.y[:, None] + move_y[:, None] * vertical_fraction
        vertical_valid = (
            vertical
            & (move_x[:, None].abs() >= 1e-6)
            & (vertical_fraction >= 0)
            & (vertical_fraction <= 1)
            & (vertical_y >= torch.minimum(y1, y2)[None, :] - _PLAYER_RADIUS)
            & (vertical_y <= torch.maximum(y1, y2)[None, :] + _PLAYER_RADIUS)
        )
        candidates = torch.cat(
            (
                torch.where(
                    horizontal_valid,
                    horizontal_fraction,
                    torch.full_like(horizontal_fraction, torch.inf),
                ),
                torch.where(
                    vertical_valid,
                    vertical_fraction,
                    torch.full_like(vertical_fraction, torch.inf),
                ),
            ),
            dim=1,
        )
        fraction = torch.min(candidates, dim=1).values
        return torch.where(
            torch.isfinite(fraction),
            fraction,
            torch.full_like(fraction, 1.0 / 32.0),
        ).clamp(0, 1)

    def _line_blocked(
        self,
        origin_x: torch.Tensor,
        origin_y: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
    ) -> torch.Tensor:
        direction_x = target_x - origin_x
        direction_y = target_y - origin_y
        start_x = self.map.walls[:, 0]
        start_y = self.map.walls[:, 1]
        segment_x = self.map.walls[:, 2] - start_x
        segment_y = self.map.walls[:, 3] - start_y
        offset_x = start_x - origin_x[..., None]
        offset_y = start_y - origin_y[..., None]
        denominator = direction_x[..., None] * segment_y - direction_y[..., None] * segment_x
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        along_ray = (offset_x * segment_y - offset_y * segment_x) / safe
        along_wall = (offset_x * direction_y[..., None] - offset_y * direction_x[..., None]) / safe
        intersects = (
            (denominator.abs() >= 1e-6)
            & (along_ray > 1e-4)
            & (along_ray < 1 - 1e-4)
            & (along_wall >= 0)
            & (along_wall <= 1)
        )
        return torch.any(intersects, dim=-1)

    def _spawn_enemy_type(self, enemy_type: int, requested: torch.Tensor) -> None:
        free = ~self.enemy_alive & (self.drop_type < 0) & ~self.enemy_projectile_alive
        has_free_slot = torch.any(free, dim=1)
        slot = torch.argmax(free.to(torch.int32), dim=1)
        spawn_mask = requested & has_free_slot
        x, y, angle, has_position = self._random_spawn_positions(
            spawn_mask,
            avoid_player=True,
            actor_radius=self._enemy_radius[enemy_type],
        )
        spawn = spawn_mask & has_position
        row = torch.arange(self.num_envs, device=self.device)
        old_x = self.enemy_x[row, slot]
        old_y = self.enemy_y[row, slot]
        old_z = self.enemy_z[row, slot]
        old_angle = self.enemy_angle[row, slot]
        old_type = self.enemy_type[row, slot]
        old_health = self.enemy_health[row, slot]
        old_cooldown = self.enemy_cooldown[row, slot]
        old_attack_phase = self.enemy_attack_phase[row, slot]
        old_move_cooldown = self.enemy_move_cooldown[row, slot]
        self.enemy_x[row, slot] = torch.where(spawn, x, old_x)
        self.enemy_y[row, slot] = torch.where(spawn, y, old_y)
        spawn_sector = self._sector_at(x, y)
        spawn_z = self.map.sector_heights[spawn_sector, 0]
        self.enemy_z[row, slot] = torch.where(spawn, spawn_z, old_z)
        self.enemy_angle[row, slot] = torch.where(spawn, angle, old_angle)
        self.enemy_type[row, slot] = torch.where(
            spawn, torch.full_like(old_type, enemy_type), old_type
        )
        self.enemy_health[row, slot] = torch.where(
            spawn, self._enemy_base_health[enemy_type], old_health
        )
        self.enemy_cooldown[row, slot] = torch.where(
            spawn, torch.full_like(old_cooldown, 18), old_cooldown
        )
        self.enemy_attack_phase[row, slot] = torch.where(
            spawn, torch.zeros_like(old_attack_phase), old_attack_phase
        )
        self.enemy_move_cooldown[row, slot] = torch.where(
            spawn,
            self._enemy_move_interval[enemy_type] - 1,
            old_move_cooldown,
        )
        self.enemy_alive[row, slot] |= spawn

    def _spawn_tick(self, active: torch.Tensor | None = None) -> None:
        if active is None:
            active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        check = (self.episode_time >= self.next_spawn_check) & active
        self.next_spawn_check.copy_(
            torch.where(
                check,
                self.next_spawn_check + _ENEMY_SPAWN_PERIOD,
                self.next_spawn_check,
            )
        )
        for enemy_type in range(len(_ENEMY_SPAWN_THRESHOLD)):
            roll = torch.remainder(self._random_u32(check), 65537)
            requested = check & (roll <= self._enemy_spawn_threshold[enemy_type])
            self._spawn_enemy_type(enemy_type, requested)

    def _apply_player_damage(self, incoming: torch.Tensor) -> None:
        incoming = torch.floor(incoming)
        absorbed = torch.floor(incoming * self.armor_save_fraction)
        absorbed = torch.minimum(self.armor, absorbed)
        self.armor.sub_(absorbed)
        self.armor_save_fraction.copy_(
            torch.where(
                self.armor > 0,
                self.armor_save_fraction,
                torch.zeros_like(self.armor_save_fraction),
            )
        )
        actual = incoming - absorbed
        self.health.sub_(actual)
        self.damage_flash.copy_(torch.maximum(self.damage_flash * 0.8, actual / 40.0))

    def _move_player(self, buttons: torch.Tensor) -> None:
        playing = ~self.player_dead & (self.episode_time < self.episode_timeout)
        active = (self.reaction_time <= 0) & playing
        self.reaction_time.sub_(1).clamp_min_(0)
        speed = torch.where(buttons[:, 1], 2.0, 1.0)
        turn = (buttons[:, 8].to(torch.float32) - buttons[:, 7].to(torch.float32)) * active.to(
            torch.float32
        )
        self.angle.add_(turn * speed * (_PLAYER_TURN_DEGREES * math.pi / 180.0))
        self.angle.remainder_(2 * math.pi)
        forward = (buttons[:, 6].to(torch.float32) - buttons[:, 5].to(torch.float32)) * active.to(
            torch.float32
        )
        side = (buttons[:, 3].to(torch.float32) - buttons[:, 4].to(torch.float32)) * active.to(
            torch.float32
        )
        acceleration = _PLAYER_ACCELERATION * speed
        cosine = torch.cos(self.angle)
        sine = torch.sin(self.angle)
        self.momentum_x.add_((forward * cosine + side * sine) * acceleration)
        self.momentum_y.add_((forward * sine - side * cosine) * acceleration)
        proposed_x = self.x + self.momentum_x
        proposed_y = self.y + self.momentum_y
        collision = self._player_collides(proposed_x, proposed_y)
        x_only_collision = self._player_collides(proposed_x, self.y)
        y_only_collision = self._player_collides(self.x, proposed_y)
        move_x = playing & (~collision | ~x_only_collision)
        move_y = playing & (~collision | ~y_only_collision)
        corner_collision = collision & move_x & move_y
        move_x &= ~corner_collision
        slide_x = collision & move_x & ~move_y
        slide_y = collision & move_y & ~move_x
        impact_fraction = self._axis_collision_fraction(self.momentum_x, self.momentum_y)
        approach_fraction = torch.clamp_min(impact_fraction - 1.0 / 32.0, 0)
        residual_fraction = 1.0 - impact_fraction
        x_position_fraction = torch.where(slide_x, _PLAYER_SLIDE_FRACTION, approach_fraction)
        y_position_fraction = torch.where(slide_y, _PLAYER_SLIDE_FRACTION, approach_fraction)
        x_position_fraction = torch.where(collision, x_position_fraction, 1.0)
        y_position_fraction = torch.where(collision, y_position_fraction, 1.0)
        self.x.copy_(
            torch.where(
                move_x | slide_y,
                self.x + self.momentum_x * x_position_fraction,
                self.x,
            )
        )
        self.y.copy_(
            torch.where(
                move_y | slide_x,
                self.y + self.momentum_y * y_position_fraction,
                self.y,
            )
        )
        self.momentum_x.copy_(
            torch.where(
                slide_x,
                self.momentum_x * residual_fraction,
                torch.where(move_x, self.momentum_x, torch.zeros_like(self.momentum_x)),
            )
        )
        self.momentum_y.copy_(
            torch.where(
                slide_y,
                self.momentum_y * residual_fraction,
                torch.where(move_y, self.momentum_y, torch.zeros_like(self.momentum_y)),
            )
        )
        self.momentum_x.mul_(_PLAYER_FRICTION)
        self.momentum_y.mul_(_PLAYER_FRICTION)

    def _vertical_player_tick(self, active: torch.Tensor) -> None:
        sector = self._current_sector()
        floor = self.map.sector_heights[sector, 0]
        proposed_z = self.z + self.velocity_z
        airborne = (self.z > floor) | (self.velocity_z < 0)
        next_velocity = torch.where(
            airborne,
            self.velocity_z - 1.0,
            torch.zeros_like(self.velocity_z),
        )
        landed = proposed_z <= floor
        next_z = torch.where(landed, floor, proposed_z)
        next_velocity = torch.where(landed, torch.zeros_like(next_velocity), next_velocity)
        self.z.copy_(torch.where(active, next_z, self.z))
        self.velocity_z.copy_(torch.where(active, next_velocity, self.velocity_z))

    def _active_weapon(self) -> torch.Tensor:
        weapon = self._slot_base_weapon[self.selected_weapon]
        alternate_slot = (self.selected_weapon == 1) | (self.selected_weapon == 3)
        return weapon + (alternate_slot & self.selected_weapon_variant).to(torch.int64)

    def _weapon_owned(self, weapon: torch.Tensor) -> torch.Tensor:
        owned = (weapon == 0) | (weapon == 2)
        owned |= (weapon == 1) & self.chainsaw_owned
        owned |= (weapon == 3) & self.shotgun_owned
        owned |= (weapon == 4) & self.super_shotgun_owned
        for code, slot in ((5, 3), (6, 4), (7, 5)):
            owned |= (weapon == code) & self.weapons[:, slot].bool()
        return owned

    def _weapon_ready(self, weapon: torch.Tensor) -> torch.Tensor:
        ammo_slot = self._weapon_ammo_slot[weapon]
        safe_slot = ammo_slot.clamp_min(0)
        ammo = self.ammo.gather(1, safe_slot[:, None]).squeeze(1)
        has_ammo = (ammo_slot < 0) | (ammo >= self._weapon_ammo_cost[weapon])
        return self._weapon_owned(weapon) & has_ammo

    def _best_ready_weapon(self) -> torch.Tensor:
        current = self._active_weapon()
        chosen = current
        found = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for code in _WEAPON_AUTO_SWITCH_ORDER:
            candidate = torch.full_like(current, code)
            usable = ~found & self._weapon_ready(candidate)
            chosen = torch.where(usable, candidate, chosen)
            found |= usable
        return chosen

    def _set_active_weapon(self, weapon: torch.Tensor, mask: torch.Tensor) -> None:
        current = self._active_weapon()
        target = torch.where(self.pending_weapon >= 0, self.pending_weapon, current)
        changed = mask & (weapon != target)
        current_vertical_tics = self.weapon_raise_cooldown.clamp(0, _WEAPON_LOWER_TICS)
        lower_tics = torch.clamp_min(_WEAPON_LOWER_TICS - current_vertical_tics, 0)
        self.pending_weapon.copy_(torch.where(changed, weapon, self.pending_weapon))
        self.weapon_lower_cooldown.copy_(
            torch.where(changed, lower_tics, self.weapon_lower_cooldown)
        )

    def _weapon_switch_tick(self, active: torch.Tensor) -> None:
        lowering = self.pending_weapon >= 0
        next_lower = torch.where(
            active & lowering,
            torch.clamp_min(self.weapon_lower_cooldown - 1, 0),
            self.weapon_lower_cooldown,
        )
        completed = active & lowering & (next_lower <= 0)
        safe_pending = self.pending_weapon.clamp_min(0)
        slot = self._weapon_slot[safe_pending]
        variant = (safe_pending == 1) | (safe_pending == 4)
        self.selected_weapon.copy_(torch.where(completed, slot, self.selected_weapon))
        self.selected_weapon_variant.copy_(
            torch.where(completed, variant, self.selected_weapon_variant)
        )
        self.pending_weapon.copy_(
            torch.where(completed, torch.full_like(self.pending_weapon, -1), self.pending_weapon)
        )
        self.weapon_lower_cooldown.copy_(
            torch.where(completed, torch.zeros_like(next_lower), next_lower)
        )
        next_raise = torch.where(
            active & ~lowering,
            torch.clamp_min(self.weapon_raise_cooldown - 1, 0),
            self.weapon_raise_cooldown,
        )
        self.weapon_raise_cooldown.copy_(
            torch.where(
                completed,
                torch.full_like(next_raise, _WEAPON_RAISE_TICS),
                next_raise,
            )
        )

    def _select_slot(self, slot: int, requested: torch.Tensor) -> None:
        current = self._active_weapon()
        if slot == 1:
            candidate = torch.where(
                current == 1,
                torch.zeros_like(current),
                torch.where(
                    self.chainsaw_owned, torch.ones_like(current), torch.zeros_like(current)
                ),
            )
            self._set_active_weapon(candidate, requested)
            return
        if slot == 3:
            shotgun = torch.full_like(current, 3)
            super_shotgun = torch.full_like(current, 4)
            shotgun_ready = self._weapon_ready(shotgun)
            super_ready = self._weapon_ready(super_shotgun)
            prefer_shotgun = current == 4
            first = torch.where(prefer_shotgun, shotgun, super_shotgun)
            second = torch.where(prefer_shotgun, super_shotgun, shotgun)
            first_ready = torch.where(prefer_shotgun, shotgun_ready, super_ready)
            second_ready = torch.where(prefer_shotgun, super_ready, shotgun_ready)
            candidate = torch.where(
                first_ready,
                first,
                torch.where(second_ready, second, current),
            )
            self._set_active_weapon(candidate, requested & (candidate != current))
            return
        code = {2: 2, 4: 5, 5: 6, 6: 7}[slot]
        candidate = torch.full_like(current, code)
        self._set_active_weapon(candidate, requested & self._weapon_ready(candidate))

    def _cycle_weapon(self, requested: torch.Tensor, direction: int) -> None:
        current = self._active_weapon()
        candidate = current
        found = torch.zeros_like(requested)
        for offset in range(1, 8):
            probe = torch.remainder(current + direction * offset, 8)
            usable = requested & ~found & self._weapon_ready(probe)
            candidate = torch.where(usable, probe, candidate)
            found |= usable
        self._set_active_weapon(candidate, requested & found)

    def _select_weapons(self, buttons: torch.Tensor) -> None:
        selection_down = torch.any(buttons[:, 9:17], dim=1)
        new_press = selection_down & ~self.weapon_change_latched
        for slot in range(1, 7):
            self._select_slot(slot, buttons[:, 8 + slot] & new_press)
        self._cycle_weapon(buttons[:, 15] & new_press, 1)
        self._cycle_weapon(buttons[:, 16] & new_press, -1)
        self.weapon_change_latched.copy_(selection_down)

    def _weapon_damage_roll(self, weapon: torch.Tensor, fires: torch.Tensor) -> torch.Tensor:
        damage = torch.zeros(self.num_envs, device=self.device)

        def add_dice(code: int, count: int, sides: int, multiplier: float) -> None:
            mask = fires & (weapon == code)
            subtotal = torch.zeros_like(damage)
            for _ in range(count):
                roll = torch.remainder(self._random_u32(mask), sides).to(torch.float32) + 1
                subtotal.add_(roll * multiplier)
            damage.add_(torch.where(mask, subtotal, torch.zeros_like(subtotal)))

        add_dice(0, 1, 10, 2.0)
        add_dice(1, 1, 10, 2.0)
        add_dice(2, 1, 3, 5.0)
        add_dice(3, 7, 3, 5.0)
        add_dice(4, 20, 3, 5.0)
        add_dice(5, 1, 3, 5.0)
        add_dice(6, 1, 8, 20.0)
        add_dice(7, 1, 8, 5.0)
        return damage

    def _apply_enemy_damage(self, damage: torch.Tensor) -> torch.Tensor:
        applied = torch.where(self.enemy_alive, damage, torch.zeros_like(damage))
        previous = self.enemy_health.clone()
        updated = torch.clamp_min(previous - applied, 0)
        self.enemy_health.copy_(torch.where(self.enemy_alive, updated, previous))
        killed = self.enemy_alive & (previous > 0) & (updated <= 0)
        killed_type = self.enemy_type.clamp_min(0)
        reward = torch.sum(
            torch.where(
                killed,
                self._enemy_kill_reward[killed_type],
                torch.zeros_like(applied),
            ),
            dim=1,
        )
        self.enemy_alive &= ~killed
        self.enemy_cooldown.masked_fill_(killed, 0)
        self.enemy_attack_phase.masked_fill_(killed, 0)
        drop = self._monster_drop_type[killed_type]
        self.drop_type.copy_(torch.where(killed, drop, self.drop_type))
        has_drop = killed & (drop >= 0)
        self.drop_delay.copy_(
            torch.where(has_drop, torch.full_like(self.drop_delay, 10), self.drop_delay)
        )
        self.enemy_type.copy_(
            torch.where(killed, torch.full_like(self.enemy_type, -1), self.enemy_type)
        )
        self.killcount.add_(torch.sum(killed.to(torch.int32), dim=1))
        return reward

    def _spawn_player_projectile(
        self,
        weapon: torch.Tensor,
        fires: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_height: torch.Tensor,
        has_target: torch.Tensor,
    ) -> None:
        requested = fires & ((weapon == 6) | (weapon == 7))
        free = ~self.projectile_alive
        has_slot = torch.any(free, dim=1)
        slot = torch.argmax(free.to(torch.int32), dim=1)
        spawn = requested & has_slot
        row = torch.arange(self.num_envs, device=self.device)
        projectile_type = (weapon - 6).clamp(0, 1)
        speed = self._player_projectile_speed[projectile_type]
        cosine = torch.cos(self.angle)
        sine = torch.sin(self.angle)
        spawn_z = self.z + 32.0
        aim_dx = torch.where(has_target, target_x - self.x, cosine)
        aim_dy = torch.where(has_target, target_y - self.y, sine)
        aim_dz = torch.where(
            has_target,
            target_z + target_height * 0.5 - spawn_z,
            torch.zeros_like(spawn_z),
        )
        aim_norm = torch.sqrt(aim_dx * aim_dx + aim_dy * aim_dy + aim_dz * aim_dz).clamp_min_(
            1e-4
        )
        self.projectile_x[row, slot] = torch.where(spawn, self.x, self.projectile_x[row, slot])
        self.projectile_y[row, slot] = torch.where(spawn, self.y, self.projectile_y[row, slot])
        self.projectile_z[row, slot] = torch.where(
            spawn,
            spawn_z,
            self.projectile_z[row, slot],
        )
        self.projectile_velocity_x[row, slot] = torch.where(
            spawn,
            aim_dx / aim_norm * speed,
            self.projectile_velocity_x[row, slot],
        )
        self.projectile_velocity_y[row, slot] = torch.where(
            spawn,
            aim_dy / aim_norm * speed,
            self.projectile_velocity_y[row, slot],
        )
        self.projectile_velocity_z[row, slot] = torch.where(
            spawn,
            aim_dz / aim_norm * speed,
            self.projectile_velocity_z[row, slot],
        )
        self.projectile_type[row, slot] = torch.where(
            spawn,
            projectile_type,
            self.projectile_type[row, slot],
        )
        self.projectile_age[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self.projectile_age[row, slot]),
            self.projectile_age[row, slot],
        )
        self.projectile_alive[row, slot] |= spawn

    def _projectile_tick(self, active: torch.Tensor) -> torch.Tensor:
        alive = self.projectile_alive & active[:, None]
        next_x = self.projectile_x + self.projectile_velocity_x
        next_y = self.projectile_y + self.projectile_velocity_y
        next_z = self.projectile_z + self.projectile_velocity_z
        wall_impact = alive & self._line_blocked(
            self.projectile_x,
            self.projectile_y,
            next_x,
            next_y,
        )
        dx = next_x[:, :, None] - self.enemy_x[:, None, :]
        dy = next_y[:, :, None] - self.enemy_y[:, None, :]
        enemy_distance = torch.sqrt(dx * dx + dy * dy)
        projectile_radius = torch.where(self.projectile_type == 0, 11.0, 13.0)
        enemy_type = self.enemy_type.clamp_min(0)
        enemy_overlap = self._vertical_overlap(
            next_z[:, :, None],
            8.0,
            self.enemy_z[:, None, :],
            self._enemy_height[enemy_type][:, None, :],
        )
        candidate = (
            alive[:, :, None]
            & self.enemy_alive[:, None, :]
            & enemy_overlap
            & (
                enemy_distance
                < projectile_radius[:, :, None] + self._enemy_radius[enemy_type][:, None, :]
            )
        )
        candidate_distance = torch.where(
            candidate,
            enemy_distance,
            torch.full_like(enemy_distance, torch.inf),
        )
        nearest_distance, nearest_enemy = torch.min(candidate_distance, dim=2)
        enemy_impact = torch.isfinite(nearest_distance)
        expired = self.projectile_age >= _PLAYER_PROJECTILE_LIFETIME
        sector = self._sector_at(next_x.reshape(-1), next_y.reshape(-1)).reshape_as(next_x)
        floor = self.map.sector_heights[sector, 0]
        ceiling = self.map.sector_heights[sector, 1]
        plane_impact = (next_z < floor) | (next_z + 8.0 > ceiling)
        impact = alive & (wall_impact | plane_impact | enemy_impact | expired)

        random_bits = self._random_u32(torch.any(impact, dim=1))[:, None]
        slot_bits = torch.arange(
            self.player_projectile_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, :]
        mixed = random_bits ^ (slot_bits * _HASH_GOLDEN_RATIO_SIGNED)
        mixed ^= mixed >> 16
        die = torch.remainder(mixed, 8).to(torch.float32) + 1
        direct_damage = torch.where(self.projectile_type == 0, die * 20.0, die * 5.0)
        direct_damage *= (impact & enemy_impact).to(torch.float32)
        damage_by_enemy = torch.zeros_like(self.enemy_health)
        damage_by_enemy.scatter_add_(1, nearest_enemy, direct_damage)

        rocket_impact = impact & (self.projectile_type == 0)
        splash = torch.clamp_min(128.0 - enemy_distance, 0)
        splash *= rocket_impact[:, :, None].to(torch.float32)
        damage_by_enemy.add_(torch.sum(splash, dim=1))
        player_distance = torch.sqrt(
            (next_x - self.x[:, None]) ** 2 + (next_y - self.y[:, None]) ** 2
        )
        self_damage = torch.sum(
            torch.clamp_min(128.0 - player_distance, 0) * rocket_impact.to(torch.float32),
            dim=1,
        )
        self._apply_player_damage(self_damage)
        reward = self._apply_enemy_damage(damage_by_enemy)

        self.projectile_x.copy_(torch.where(alive, next_x, self.projectile_x))
        self.projectile_y.copy_(torch.where(alive, next_y, self.projectile_y))
        self.projectile_z.copy_(torch.where(alive, next_z, self.projectile_z))
        self.projectile_age.add_(alive.to(torch.int32))
        self.projectile_alive &= ~impact
        self.projectile_type.masked_fill_(impact, -1)
        return reward

    def _player_attack(self, buttons: torch.Tensor) -> torch.Tensor:
        weapon = self._active_weapon()
        ammo_slot = self._weapon_ammo_slot[weapon]
        safe_ammo_slot = ammo_slot.clamp_min(0)
        ammo = self.ammo.gather(1, safe_ammo_slot[:, None]).squeeze(1)
        cost = self._weapon_ammo_cost[weapon]
        attempted_empty_fire = (
            buttons[:, 0]
            & (self.attack_cooldown <= 0)
            & (self.weapon_raise_cooldown <= 0)
            & ~self.player_dead
            & (self.episode_time < self.episode_timeout)
            & self._weapon_owned(weapon)
            & (ammo_slot >= 0)
            & (ammo < cost)
        )
        replacement = self._best_ready_weapon()
        self._set_active_weapon(replacement, attempted_empty_fire)
        fires = (
            buttons[:, 0]
            & (self.attack_cooldown <= 0)
            & (self.weapon_raise_cooldown <= 0)
            & (self.pending_weapon < 0)
            & ~self.player_dead
            & (self.episode_time < self.episode_timeout)
            & self._weapon_owned(weapon)
            & ((ammo_slot < 0) | (ammo >= cost))
        )
        new_ammo = torch.clamp_min(ammo - cost, 0)
        uses_ammo = fires & (ammo_slot >= 0)
        self.ammo.scatter_(
            1,
            safe_ammo_slot[:, None],
            torch.where(uses_ammo, new_ammo, ammo)[:, None],
        )
        uses_bullets = fires & ((weapon == 2) | (weapon == 5))
        shared_bullets = torch.where(uses_bullets, new_ammo, self.ammo[:, 1])
        self.ammo[:, 1].copy_(shared_bullets)
        self.ammo[:, 3].copy_(shared_bullets)
        self.attack_cooldown.copy_(
            torch.where(fires, self._weapon_cooldown[weapon], self.attack_cooldown)
        )
        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            dolls = self.map.player_starts[:-1, :2]
            doll_x = dolls[None, :, 0].expand(self.num_envs, -1)
            doll_y = dolls[None, :, 1].expand(self.num_envs, -1)
            target_x = torch.cat((self.enemy_x, doll_x), dim=1)
            target_y = torch.cat((self.enemy_y, doll_y), dim=1)
            target_z = torch.cat(
                (
                    self.enemy_z,
                    self._player_start_z[:-1][None, :].expand(self.num_envs, -1),
                ),
                dim=1,
            )
            target_height = torch.cat(
                (
                    self._enemy_height[self.enemy_type.clamp_min(0)],
                    torch.full_like(doll_x, _PLAYER_HEIGHT),
                ),
                dim=1,
            )
            target_alive = torch.cat(
                (
                    self.enemy_alive,
                    (~self.player_dead)[:, None].expand(-1, doll_count),
                ),
                dim=1,
            )
        else:
            target_x = self.enemy_x
            target_y = self.enemy_y
            target_z = self.enemy_z
            target_height = self._enemy_height[self.enemy_type.clamp_min(0)]
            target_alive = self.enemy_alive
        dx = target_x - self.x[:, None]
        dy = target_y - self.y[:, None]
        distance = torch.sqrt(dx * dx + dy * dy).clamp_min_(1e-4)
        bearing = self._wrap_angle(torch.atan2(dy, dx) - self.angle[:, None]).abs()
        melee = (weapon == 0) | (weapon == 1)
        max_range = torch.where(melee, 72.0, 2048.0)
        valid = target_alive & (bearing < (8.0 * math.pi / 180.0)) & (distance < max_range[:, None])
        shoot_z = self.z[:, None] + 36.0
        max_autoaim_slope = math.tan(35.0 * math.pi / 180.0)
        bottom_slope = (target_z - shoot_z) / distance
        top_slope = (target_z + target_height - shoot_z) / distance
        valid &= (top_slope >= -max_autoaim_slope) & (bottom_slope <= max_autoaim_slope)
        valid &= ~self._line_blocked(
            self.x[:, None],
            self.y[:, None],
            target_x,
            target_y,
        )
        target_distance = torch.where(valid, distance, torch.full_like(distance, torch.inf))
        target = torch.argmin(target_distance, dim=1)
        row = torch.arange(self.num_envs, device=self.device)
        selected_target_exists = torch.isfinite(target_distance[row, target])
        self._spawn_player_projectile(
            weapon,
            fires,
            target_x[row, target],
            target_y[row, target],
            target_z[row, target],
            target_height[row, target],
            selected_target_exists,
        )
        hitscan_fires = fires & (weapon <= 5)
        has_target = hitscan_fires & torch.isfinite(
            target_distance.gather(1, target[:, None]).squeeze(1)
        )
        damage = self._weapon_damage_roll(weapon, hitscan_fires)
        enemy_target = target.clamp_max(self.enemy_slots - 1)
        hits_enemy = has_target & (target < self.enemy_slots)
        hits_doll = has_target & (target >= self.enemy_slots)
        damage_by_enemy = torch.zeros_like(self.enemy_health)
        damage_by_enemy.scatter_add_(
            1,
            enemy_target[:, None],
            torch.where(hits_enemy, damage, torch.zeros_like(damage))[:, None],
        )
        self._apply_player_damage(torch.where(hits_doll, damage, torch.zeros_like(damage)))
        return self._apply_enemy_damage(damage_by_enemy)

    def _enemy_damage_roll(
        self,
        enemy_type: torch.Tensor,
        attacks: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        random_bits = self._random_u32(torch.any(attacks, dim=1))[:, None]
        slot = torch.arange(self.enemy_slots, device=self.device, dtype=torch.int64)[None, :]

        def die(draw: int, sides: int) -> torch.Tensor:
            mixed = random_bits ^ (slot * _HASH_GOLDEN_RATIO_SIGNED) ^ (draw * _HASH_MURMUR_SIGNED)
            mixed ^= mixed >> 16
            return torch.remainder(mixed, sides).to(torch.float32) + 1

        damage = torch.zeros_like(distance)
        damage = torch.where(enemy_type == 0, die(0, 5) * 3, damage)
        shotgun = (die(0, 5) + die(1, 5) + die(2, 5)) * 3
        damage = torch.where(enemy_type == 1, shotgun, damage)
        damage = torch.where(enemy_type == 2, die(0, 10) * 2, damage)
        damage = torch.where(enemy_type == 3, die(0, 5) * 3, damage)
        damage = torch.where(enemy_type == 4, die(0, 10) * 4, damage)
        knight_multiplier = torch.where(distance < 64, 10.0, 8.0)
        damage = torch.where(enemy_type == 5, die(0, 8) * knight_multiplier, damage)
        return torch.where(attacks, damage, torch.zeros_like(damage))

    @staticmethod
    def _enemy_missile_threshold(
        enemy_type: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> torch.Tensor:
        """Return Doom's 8-bit P_CheckMissileRange distance threshold."""
        abs_dx = dx.abs()
        abs_dy = dy.abs()
        approximate_distance = torch.maximum(abs_dx, abs_dy) + 0.5 * torch.minimum(abs_dx, abs_dy)
        has_no_melee_state = (enemy_type == 0) | (enemy_type == 1) | (enemy_type == 3)
        threshold = approximate_distance - 64.0
        threshold -= has_no_melee_state.to(threshold.dtype) * 128.0
        return torch.clamp(threshold, 0.0, 200.0)

    def _enemy_missile_decision(
        self,
        enemy_type: torch.Tensor,
        candidates: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> torch.Tensor:
        random_bits = self._random_u32(torch.any(candidates, dim=1))[:, None]
        slot = torch.arange(self.enemy_slots, device=self.device, dtype=torch.int64)[None, :]
        mixed = random_bits ^ (slot * _HASH_GOLDEN_RATIO_SIGNED)
        mixed ^= mixed >> 16
        roll = torch.remainder(mixed, 256).to(torch.float32)
        threshold = self._enemy_missile_threshold(enemy_type, dx, dy)
        return candidates & (roll >= threshold)

    def _spawn_enemy_projectiles(
        self,
        requested: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> None:
        spawn = requested & ~self.enemy_projectile_alive
        spawn_z = self.enemy_z + 32.0
        dz = self.z[:, None] - self.enemy_z
        aim_norm = torch.sqrt(dx * dx + dy * dy + dz * dz).clamp_min_(1e-4)
        self.enemy_projectile_x.copy_(torch.where(spawn, self.enemy_x, self.enemy_projectile_x))
        self.enemy_projectile_y.copy_(torch.where(spawn, self.enemy_y, self.enemy_projectile_y))
        self.enemy_projectile_z.copy_(torch.where(spawn, spawn_z, self.enemy_projectile_z))
        self.enemy_projectile_velocity_x.copy_(
            torch.where(
                spawn,
                dx / aim_norm * _ENEMY_PROJECTILE_SPEED,
                self.enemy_projectile_velocity_x,
            )
        )
        self.enemy_projectile_velocity_y.copy_(
            torch.where(
                spawn,
                dy / aim_norm * _ENEMY_PROJECTILE_SPEED,
                self.enemy_projectile_velocity_y,
            )
        )
        self.enemy_projectile_velocity_z.copy_(
            torch.where(
                spawn,
                dz / aim_norm * _ENEMY_PROJECTILE_SPEED,
                self.enemy_projectile_velocity_z,
            )
        )
        self.enemy_projectile_age.copy_(
            torch.where(
                spawn, torch.zeros_like(self.enemy_projectile_age), self.enemy_projectile_age
            )
        )
        self.enemy_projectile_alive |= spawn

    def _enemy_projectile_tick(self, active: torch.Tensor) -> None:
        alive = self.enemy_projectile_alive & active[:, None]
        next_x = self.enemy_projectile_x + self.enemy_projectile_velocity_x
        next_y = self.enemy_projectile_y + self.enemy_projectile_velocity_y
        next_z = self.enemy_projectile_z + self.enemy_projectile_velocity_z
        wall_impact = alive & self._line_blocked(
            self.enemy_projectile_x,
            self.enemy_projectile_y,
            next_x,
            next_y,
        )
        player_distance = torch.sqrt(
            (next_x - self.x[:, None]) ** 2 + (next_y - self.y[:, None]) ** 2
        )
        player_vertical_overlap = self._vertical_overlap(
            next_z,
            16.0,
            self.z[:, None],
            _PLAYER_HEIGHT,
        )
        player_impact = alive & player_vertical_overlap & (player_distance < 22.0)
        expired = self.enemy_projectile_age >= _ENEMY_PROJECTILE_LIFETIME
        sector = self._sector_at(next_x.reshape(-1), next_y.reshape(-1)).reshape_as(next_x)
        floor = self.map.sector_heights[sector, 0]
        ceiling = self.map.sector_heights[sector, 1]
        plane_impact = (next_z < floor) | (next_z + 16.0 > ceiling)
        impact = alive & (wall_impact | plane_impact | player_impact | expired)
        random_bits = self._random_u32(torch.any(impact, dim=1))[:, None]
        slot_bits = torch.arange(
            self.enemy_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, :]
        mixed = random_bits ^ (slot_bits * _HASH_GOLDEN_RATIO_SIGNED)
        mixed ^= mixed >> 16
        damage = (torch.remainder(mixed, 8).to(torch.float32) + 1) * 8.0
        incoming = torch.sum(
            torch.where(player_impact, damage, torch.zeros_like(damage)),
            dim=1,
        )
        self._apply_player_damage(incoming)
        self.enemy_projectile_x.copy_(torch.where(alive, next_x, self.enemy_projectile_x))
        self.enemy_projectile_y.copy_(torch.where(alive, next_y, self.enemy_projectile_y))
        self.enemy_projectile_z.copy_(torch.where(alive, next_z, self.enemy_projectile_z))
        self.enemy_projectile_age.add_(alive.to(torch.int32))
        self.enemy_projectile_alive &= ~impact

    def _enemy_tick(self, active: torch.Tensor | None = None) -> None:
        if active is None:
            active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        alive = self.enemy_alive & active[:, None]
        enemy_type = self.enemy_type.clamp_min(0)
        dx = self.x[:, None] - self.enemy_x
        dy = self.y[:, None] - self.enemy_y
        distance = torch.sqrt(dx * dx + dy * dy).clamp_min_(1e-4)
        visible = ~self._line_blocked(
            self.enemy_x,
            self.enemy_y,
            self.x[:, None],
            self.y[:, None],
        )
        shoot_z = self.enemy_z + 36.0
        player_bottom_slope = (self.z[:, None] - shoot_z) / distance
        player_top_slope = (self.z[:, None] + _PLAYER_HEIGHT - shoot_z) / distance
        max_autoaim_slope = math.tan(35.0 * math.pi / 180.0)
        visible &= (player_top_slope >= -max_autoaim_slope) & (
            player_bottom_slope <= max_autoaim_slope
        )
        melee_vertical_overlap = self._vertical_overlap(
            self.enemy_z,
            self._enemy_height[enemy_type],
            self.z[:, None],
            _PLAYER_HEIGHT,
        )

        attack_phase = self.enemy_attack_phase
        phase_due = alive & (attack_phase > 0) & (self.enemy_cooldown <= 1)
        phase_one_due = phase_due & (attack_phase == 1)
        phase_two_due = phase_due & (attack_phase == 2)
        phase_three_due = phase_due & (attack_phase == 3)
        finite_type = (enemy_type == 0) | (enemy_type == 1) | (enemy_type == 4) | (enemy_type == 5)
        chainsaw_target = visible & melee_vertical_overlap & (distance < 64.0)
        chaingun_target = visible & (distance < self._enemy_attack_range[enemy_type])
        finite_fire = phase_one_due & finite_type
        chainsaw_fire = phase_one_due & (enemy_type == 2) & chainsaw_target
        chaingun_fire = (enemy_type == 3) & (
            phase_one_due | phase_two_due | (phase_three_due & chaingun_target)
        )
        fire_event = finite_fire | chainsaw_fire | chaingun_fire
        knight_projectile = fire_event & (enemy_type == 5) & (distance >= 64.0)
        self._spawn_enemy_projectiles(knight_projectile, dx, dy)
        direct_attack = fire_event & ~knight_projectile & visible
        direct_attack &= ~((enemy_type == 2) | (enemy_type == 4) | (enemy_type == 5)) | (
            melee_vertical_overlap | (distance >= 64.0)
        )
        direct_attack &= distance < self._enemy_attack_range[enemy_type]
        incoming = torch.sum(
            self._enemy_damage_roll(enemy_type, direct_attack, distance),
            dim=1,
        )
        self._apply_player_damage(incoming)

        next_cooldown = torch.where(
            active[:, None],
            torch.clamp_min(self.enemy_cooldown - 1, 0),
            self.enemy_cooldown,
        )
        next_phase = attack_phase.clone()
        finite_prefire_done = phase_one_due & finite_type
        next_phase = torch.where(
            finite_prefire_done,
            torch.full_like(next_phase, 2),
            next_phase,
        )
        next_cooldown = torch.where(
            finite_prefire_done,
            self._enemy_attack_recovery[enemy_type],
            next_cooldown,
        )
        finite_recovery_done = phase_two_due & finite_type
        next_phase = torch.where(
            finite_recovery_done,
            torch.zeros_like(next_phase),
            next_phase,
        )
        next_cooldown = torch.where(
            finite_recovery_done,
            torch.zeros_like(next_cooldown),
            next_cooldown,
        )

        chainsaw_done = phase_one_due & (enemy_type == 2)
        next_phase = torch.where(
            chainsaw_done,
            chainsaw_target.to(next_phase.dtype),
            next_phase,
        )
        next_cooldown = torch.where(
            chainsaw_done,
            torch.where(
                chainsaw_target,
                self._enemy_attack_recovery[enemy_type],
                torch.zeros_like(next_cooldown),
            ),
            next_cooldown,
        )

        chaingun_first = phase_one_due & (enemy_type == 3)
        next_phase = torch.where(
            chaingun_first,
            torch.full_like(next_phase, 2),
            next_phase,
        )
        next_cooldown = torch.where(
            chaingun_first,
            self._enemy_attack_recovery[enemy_type],
            next_cooldown,
        )
        chaingun_second = phase_two_due & (enemy_type == 3)
        next_phase = torch.where(
            chaingun_second,
            torch.full_like(next_phase, 3),
            next_phase,
        )
        next_cooldown = torch.where(
            chaingun_second,
            torch.full_like(next_cooldown, 5),
            next_cooldown,
        )
        chaingun_refire = phase_three_due & (enemy_type == 3)
        next_phase = torch.where(
            chaingun_refire,
            torch.where(
                chaingun_target,
                torch.full_like(next_phase, 2),
                torch.zeros_like(next_phase),
            ),
            next_phase,
        )
        next_cooldown = torch.where(
            chaingun_refire,
            torch.where(
                chaingun_target,
                self._enemy_attack_recovery[enemy_type],
                torch.zeros_like(next_cooldown),
            ),
            next_cooldown,
        )

        move_ready = alive & (next_phase == 0) & (self.enemy_move_cooldown <= 0)
        attack_ready = move_ready & (next_cooldown <= 0)
        melee_type = (enemy_type == 2) | (enemy_type == 4) | (enemy_type == 5)
        melee_attack = (
            attack_ready
            & visible
            & melee_vertical_overlap
            & melee_type
            & (distance < 64.0)
        )
        ranged_type = (enemy_type == 0) | (enemy_type == 1) | (enemy_type == 3) | (enemy_type == 5)
        ranged_candidate = (
            attack_ready
            & visible
            & ranged_type
            & (distance < self._enemy_attack_range[enemy_type])
            & ~((enemy_type == 5) & (distance < 64.0))
        )
        ranged_attack = self._enemy_missile_decision(
            enemy_type,
            ranged_candidate,
            dx,
            dy,
        )
        can_attack = melee_attack | ranged_attack
        stride = self._enemy_stride[enemy_type]
        stop_distance = _PLAYER_RADIUS + self._enemy_radius[enemy_type]
        travel = torch.minimum(stride, torch.clamp_min(distance - stop_distance, 0))
        wants_x = dx.abs() > 10.0
        wants_y = dy.abs() > 10.0
        fallback_x = (~wants_x & ~wants_y) & (dx.abs() >= dy.abs())
        fallback_y = (~wants_x & ~wants_y) & ~fallback_x
        direction_x = torch.sign(dx) * (wants_x | fallback_x).to(torch.float32)
        direction_y = torch.sign(dy) * (wants_y | fallback_y).to(torch.float32)
        direction_norm = torch.sqrt(
            direction_x * direction_x + direction_y * direction_y
        ).clamp_min_(1)
        direction_x /= direction_norm
        direction_y /= direction_norm
        moving = move_ready & ~can_attack & (travel > 0)
        proposed_x = self.enemy_x + torch.where(moving, direction_x * travel, 0)
        proposed_y = self.enemy_y + torch.where(moving, direction_y * travel, 0)
        proposed_sector = self._sector_at(
            proposed_x.reshape(-1), proposed_y.reshape(-1)
        ).reshape_as(proposed_x)
        collision = self._enemy_collides(
            proposed_x,
            proposed_y,
            enemy_type,
            proposed_sector,
        )
        self.enemy_x.copy_(torch.where(collision, self.enemy_x, proposed_x))
        self.enemy_y.copy_(torch.where(collision, self.enemy_y, proposed_y))
        proposed_z = self.map.sector_heights[proposed_sector, 0]
        self.enemy_z.copy_(torch.where(moving & ~collision, proposed_z, self.enemy_z))
        self.enemy_angle.copy_(
            torch.where(
                moving & ~collision,
                torch.atan2(direction_y, direction_x),
                self.enemy_angle,
            )
        )
        decremented_move = torch.clamp_min(self.enemy_move_cooldown - 1, 0)
        self.enemy_move_cooldown.copy_(
            torch.where(active[:, None], decremented_move, self.enemy_move_cooldown)
        )
        self.enemy_move_cooldown.copy_(
            torch.where(
                move_ready,
                self._enemy_move_interval[enemy_type] - 1,
                self.enemy_move_cooldown,
            )
        )
        next_phase = torch.where(
            can_attack,
            torch.ones_like(next_phase),
            next_phase,
        )
        next_cooldown = torch.where(
            can_attack,
            self._enemy_attack_prefire[enemy_type],
            next_cooldown,
        )
        self.enemy_attack_phase.copy_(next_phase)
        self.enemy_cooldown.copy_(next_cooldown)

    def _touching(
        self,
        item_x: torch.Tensor,
        item_y: torch.Tensor,
        item_z: torch.Tensor,
    ) -> torch.Tensor:
        distance = _PLAYER_RADIUS + _PICKUP_RADIUS
        vertical_reach = (
            (item_z - self.z[:, None] <= _PLAYER_HEIGHT)
            & (item_z - self.z[:, None] >= -_PICKUP_REACH_BELOW)
        )
        return (
            (torch.abs(item_x - self.x[:, None]) < distance)
            & (torch.abs(item_y - self.y[:, None]) < distance)
            & vertical_reach
            & ~self.player_dead[:, None]
            & (self.episode_time < self.episode_timeout)[:, None]
        )

    @staticmethod
    def _successful_fixed_gain_pickups(
        touched: torch.Tensor,
        current: torch.Tensor,
        amount: float,
        cap: float,
    ) -> torch.Tensor:
        rank = torch.cumsum(touched.to(torch.int32), dim=1)
        needed = torch.ceil(torch.clamp_min(cap - current, 0) / amount).to(torch.int32)
        return touched & (rank <= needed[:, None])

    def _add_ammo(self, slot: int, gain: torch.Tensor, cap: float) -> None:
        updated = torch.minimum(self.ammo[:, slot] + gain, torch.full_like(gain, cap))
        self.ammo[:, slot].copy_(updated)
        if slot == 1:
            self.ammo[:, 3].copy_(updated)

    def _owns_weapon_code(self, code: int) -> torch.Tensor:
        if code == 1:
            return self.chainsaw_owned
        if code == 3:
            return self.shotgun_owned
        if code == 4:
            return self.super_shotgun_owned
        return self.weapons[:, {5: 3, 6: 4, 7: 5}[code]].bool()

    def _grant_weapon_code(self, code: int, acquired: torch.Tensor) -> None:
        if code == 1:
            self.chainsaw_owned |= acquired
            self.weapons[:, 0].copy_(1 + self.chainsaw_owned.to(torch.float32))
        elif code == 3:
            self.shotgun_owned |= acquired
            self.weapons[:, 2].copy_(
                self.shotgun_owned.to(torch.float32) + self.super_shotgun_owned.to(torch.float32)
            )
        elif code == 4:
            self.super_shotgun_owned |= acquired
            self.weapons[:, 2].copy_(
                self.shotgun_owned.to(torch.float32) + self.super_shotgun_owned.to(torch.float32)
            )
        else:
            self.weapons[:, {5: 3, 6: 4, 7: 5}[code]].copy_(
                torch.where(
                    acquired,
                    torch.ones_like(self.weapons[:, 0]),
                    self.weapons[:, {5: 3, 6: 4, 7: 5}[code]],
                )
            )

    def _pickup_weapon(
        self,
        touched: torch.Tensor,
        *,
        code: int,
        ammo_amount: float = 0,
        ammo_cap: float = 0,
    ) -> torch.Tensor:
        previously_owned = self._owns_weapon_code(code).clone()
        ammo_slot = _WEAPON_AMMO_SLOT[code]
        can_receive_ammo = (
            torch.zeros_like(previously_owned)
            if ammo_slot < 0
            else self.ammo[:, ammo_slot] < ammo_cap
        )
        can_pick_up = ~previously_owned | can_receive_ammo
        successful = touched & can_pick_up[:, None]
        acquired = torch.any(successful, dim=1)
        newly_owned = acquired & ~previously_owned
        self._grant_weapon_code(code, acquired)
        if ammo_slot >= 0:
            count = torch.sum(successful, dim=1).to(torch.float32)
            self._add_ammo(ammo_slot, count * ammo_amount, ammo_cap)
        weapon = torch.full((self.num_envs,), code, device=self.device, dtype=torch.int64)
        self._set_active_weapon(weapon, newly_owned)
        return successful

    def _collect_map_items(self) -> None:
        if not self.item_available.numel():
            return
        touched = self.item_available & self._touching(
            self.map.item_spawns[None, :, 0],
            self.map.item_spawns[None, :, 1],
            self._item_z[None, :],
        )
        types = self.map.item_types[None, :]
        consumed = torch.zeros_like(touched)

        standard_health = touched & ((types == 2011) | (types == 2012))
        health_gain = torch.where(
            types == 2011,
            torch.full_like(touched, 10, dtype=torch.float32),
            torch.where(
                types == 2012,
                torch.full_like(touched, 25, dtype=torch.float32),
                torch.zeros_like(touched, dtype=torch.float32),
            ),
        )
        prior_health_gain = torch.cumsum(health_gain * standard_health, dim=1) - health_gain
        health_success = standard_health & (self.health[:, None] + prior_health_gain < 100)
        total_health = torch.sum(health_gain * health_success, dim=1)
        self.health.copy_(
            torch.minimum(self.health + total_health, torch.full_like(self.health, 100))
        )
        consumed |= health_success

        health_bonus = touched & (types == 2014)
        bonus_gain = torch.sum(health_bonus, dim=1).to(torch.float32)
        self.health.copy_(
            torch.minimum(self.health + bonus_gain, torch.full_like(self.health, 200))
        )
        consumed |= health_bonus

        armor_bonus = touched & (types == 2015)
        had_armor = self.armor > 0
        armor_gain = torch.sum(armor_bonus, dim=1).to(torch.float32)
        got_armor_bonus = torch.any(armor_bonus, dim=1)
        self.armor.copy_(torch.minimum(self.armor + armor_gain, torch.full_like(self.armor, 200)))
        self.armor_save_fraction.copy_(
            torch.where(
                got_armor_bonus & ~had_armor,
                torch.full_like(self.armor_save_fraction, _GREEN_ARMOR_SAVE),
                self.armor_save_fraction,
            )
        )
        consumed |= armor_bonus

        for type_id, amount, save_fraction in (
            (2018, 100.0, _GREEN_ARMOR_SAVE),
            (2019, 200.0, _BLUE_ARMOR_SAVE),
        ):
            armor_touch = touched & (types == type_id)
            successful = armor_touch & (self.armor < amount)[:, None]
            acquired = torch.any(successful, dim=1)
            self.armor.copy_(torch.where(acquired, torch.full_like(self.armor, amount), self.armor))
            self.armor_save_fraction.copy_(
                torch.where(
                    acquired,
                    torch.full_like(self.armor_save_fraction, save_fraction),
                    self.armor_save_fraction,
                )
            )
            consumed |= successful

        for type_id, slot, amount, cap in (
            (2007, 1, 10.0, 200.0),
            (2048, 1, 50.0, 200.0),
            (2049, 2, 20.0, 50.0),
            (2046, 4, 5.0, 50.0),
            (17, 5, 100.0, 300.0),
        ):
            ammo_touch = touched & (types == type_id)
            successful = self._successful_fixed_gain_pickups(
                ammo_touch, self.ammo[:, slot], amount, cap
            )
            count = torch.sum(successful, dim=1).to(torch.float32)
            self._add_ammo(slot, count * amount, cap)
            consumed |= successful

        for type_id, code, ammo_amount, ammo_cap in (
            (2005, 1, 0.0, 0.0),
            (2001, 3, 8.0, 50.0),
            (82, 4, 8.0, 50.0),
            (2002, 5, 20.0, 200.0),
            (2003, 6, 2.0, 50.0),
            (2004, 7, 40.0, 300.0),
        ):
            consumed |= self._pickup_weapon(
                touched & (types == type_id),
                code=code,
                ammo_amount=ammo_amount,
                ammo_cap=ammo_cap,
            )
        self.item_available &= ~consumed

    def _collect_drops(self) -> None:
        self.drop_delay.sub_(1).clamp_min_(0)
        available = (self.drop_type >= 0) & (self.drop_delay <= 0)
        touched = available & self._touching(self.enemy_x, self.enemy_y, self.enemy_z)
        consumed = torch.zeros_like(touched)
        clip = touched & (self.drop_type == 2007)
        clip_success = self._successful_fixed_gain_pickups(clip, self.ammo[:, 1], 5.0, 200.0)
        self._add_ammo(1, torch.sum(clip_success, dim=1).to(torch.float32) * 5.0, 200.0)
        consumed |= clip_success
        consumed |= self._pickup_weapon(
            touched & (self.drop_type == 2001),
            code=3,
            ammo_amount=4.0,
            ammo_cap=50.0,
        )
        consumed |= self._pickup_weapon(
            touched & (self.drop_type == 2002),
            code=5,
            ammo_amount=10.0,
            ammo_cap=200.0,
        )
        self.drop_type.masked_fill_(consumed, -1)

    def _collect_items(self) -> None:
        self._collect_map_items()
        self._collect_drops()

    def step(
        self, buttons: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.debug_checks and torch.any(self.pending_reset):
            lanes = torch.nonzero(self.pending_reset).flatten().to("cpu").tolist()
            raise RuntimeError(f"terminal lanes must be reset before step: {lanes}")
        reward = torch.zeros(self.num_envs, device=self.device)
        for _ in range(self.frame_skip):
            self.player_dead |= self.health <= 0
            active = ~self.player_dead & (self.episode_time < self.episode_timeout)
            active_buttons = buttons & active[:, None]
            decremented_attack = torch.clamp_min(self.attack_cooldown - 1, 0)
            self.attack_cooldown.copy_(
                torch.where(active, decremented_attack, self.attack_cooldown)
            )
            self._weapon_switch_tick(active)
            self._select_weapons(active_buttons)
            self._move_player(active_buttons)
            self._vertical_player_tick(active)
            reward.add_(self._player_attack(active_buttons))
            reward.add_(self._projectile_tick(active))
            self.player_dead |= self.health <= 0
            self._enemy_tick(active & ~self.player_dead)
            self._enemy_projectile_tick(active & ~self.player_dead)
            self.player_dead |= self.health <= 0
            self._collect_items()
            self.episode_time.add_(active.to(torch.int32))
            self._spawn_tick(active & ~self.player_dead)
        self.episode_return.add_(reward)
        self.player_dead.copy_(self.health <= 0)
        terminated = self.player_dead.clone()
        truncated = (self.episode_time >= self.episode_timeout) & ~terminated
        self.pending_reset.copy_(terminated | truncated)
        frame = self.render_frame()
        self.frames = torch.roll(self.frames, shifts=-1, dims=1)
        self.frames[:, -1].copy_(frame)
        self._update_signal_buffer()
        return self.frames, reward, terminated, truncated

    def _raycast(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ray_angles = self.angle[:, None] + self._ray_offsets[None, :]
        direction = torch.stack((torch.cos(ray_angles), torch.sin(ray_angles)), dim=-1)
        origin = torch.stack((self.x, self.y), dim=-1)[:, None, None, :]
        start = self.map.walls[None, None, :, :2]
        segment = self.map.walls[None, None, :, 2:] - start
        ray = direction[:, :, None, :]
        offset = start - origin
        denominator = ray[..., 0] * segment[..., 1] - ray[..., 1] * segment[..., 0]
        safe = torch.where(denominator.abs() < 1e-6, torch.ones_like(denominator), denominator)
        distance = (offset[..., 0] * segment[..., 1] - offset[..., 1] * segment[..., 0]) / safe
        along = (offset[..., 0] * ray[..., 1] - offset[..., 1] * ray[..., 0]) / safe
        valid = (denominator.abs() >= 1e-6) & (distance > 0) & (along >= 0) & (along <= 1)
        distance = torch.where(valid, distance, torch.full_like(distance, torch.inf))
        nearest_distance, nearest_wall = torch.min(distance, dim=2)
        nearest_along = along.gather(2, nearest_wall[:, :, None]).squeeze(2).clamp(0, 1)
        corrected = nearest_distance * torch.cos(self._ray_offsets)[None, :]
        return corrected.clamp(1, 4096), nearest_wall, nearest_along

    def _sector_at(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        edges = self.map.sector_edges
        x1 = edges[:, 0]
        y1 = edges[:, 1]
        x2 = edges[:, 2]
        y2 = edges[:, 3]
        point_x = x[:, None]
        point_y = y[:, None]
        crosses_y = (y1 > point_y) != (y2 > point_y)
        safe_dy = torch.where((y2 - y1).abs() < 1e-6, torch.ones_like(y1), y2 - y1)
        crossing_x = x1 + (point_y - y1) * (x2 - x1) / safe_dy
        ray_crossing = crosses_y & (point_x < crossing_x)
        parity = torch.remainder(
            torch.sum(
                ray_crossing[:, None, :] & self.map.sector_edge_mask[None, :, :],
                dim=2,
            ),
            2,
        ).bool()
        return torch.argmax(parity.to(torch.int64), dim=1)

    def _current_sector(self) -> torch.Tensor:
        return self._sector_at(self.x, self.y)

    def _render_flats(
        self,
        sector: torch.Tensor,
        view_z: torch.Tensor,
        center: float,
    ) -> torch.Tensor:
        ray_angles = self.angle[:, None] + self._ray_offsets[None, :]
        cosine_correction = torch.cos(self._ray_offsets)[None, :]
        pixel_delta = self._pixel_y.to(torch.float32) - center
        floor_height = view_z - self.map.sector_heights[sector, 0]
        floor_depth = (
            floor_height[:, None, None] * _PROJECTION_FOCAL_LENGTH / pixel_delta.clamp_min(0.25)
        )
        ceiling_height = self.map.sector_heights[sector, 1] - view_z
        ceiling_depth = (
            ceiling_height[:, None, None]
            * _PROJECTION_FOCAL_LENGTH
            / (-pixel_delta).clamp_min(0.25)
        )
        perpendicular_depth = torch.where(pixel_delta > 0, floor_depth, ceiling_depth)
        ray_distance = perpendicular_depth / cosine_correction[:, None, :]
        world_x = self.x[:, None, None] + torch.cos(ray_angles)[:, None, :] * ray_distance
        world_y = self.y[:, None, None] + torch.sin(ray_angles)[:, None, :] * ray_distance
        floor_texture = self.map.sector_floor_texture_ids[sector]
        ceiling_texture = self.map.sector_ceiling_texture_ids[sector]
        texture_id = torch.where(
            pixel_delta > 0,
            floor_texture[:, None, None],
            ceiling_texture[:, None, None],
        )
        texture_width = self.map.texture_widths[texture_id]
        texture_height = self.map.texture_heights[texture_id]
        texture_u = torch.remainder(torch.floor(world_x).to(torch.int64), texture_width)
        texture_v = torch.remainder(torch.floor(-world_y).to(torch.int64), texture_height)
        value = self.map.texture_atlas[texture_id, texture_v, texture_u].to(torch.float32)
        light = self.map.sector_lights[sector][:, None, None]
        attenuation = light.clamp(24, 220)
        return (value * attenuation / 210.0).clamp(0, 255)

    def _portal_intersections(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ray_angles = self.angle[:, None] + self._ray_offsets[None, :]
        direction = torch.stack((torch.cos(ray_angles), torch.sin(ray_angles)), dim=-1)
        origin = torch.stack((self.x, self.y), dim=-1)[:, None, None, :]
        start = self.map.portal_walls[None, None, :, :2]
        segment = self.map.portal_walls[None, None, :, 2:] - start
        ray = direction[:, :, None, :]
        offset = start - origin
        denominator = ray[..., 0] * segment[..., 1] - ray[..., 1] * segment[..., 0]
        safe = torch.where(denominator.abs() < 1e-6, torch.ones_like(denominator), denominator)
        distance = (offset[..., 0] * segment[..., 1] - offset[..., 1] * segment[..., 0]) / safe
        along = (offset[..., 0] * ray[..., 1] - offset[..., 1] * ray[..., 0]) / safe
        valid = (denominator.abs() >= 1e-6) & (distance > 0) & (along >= 0) & (along <= 1)
        distance = torch.where(valid, distance, torch.full_like(distance, torch.inf))
        distance = distance * torch.cos(self._ray_offsets)[None, :, None]
        layer_count = min(_PORTAL_LAYERS, self.map.portal_walls.shape[0])
        nearest_distance, nearest_wall = torch.topk(
            distance,
            layer_count,
            dim=2,
            largest=False,
            sorted=True,
        )
        nearest_along = along.gather(2, nearest_wall).clamp(0, 1)
        return nearest_distance.clamp(1, 4096), nearest_wall, nearest_along

    def _render_portal_walls(
        self,
        frame: torch.Tensor,
        view_z: torch.Tensor,
        center: float,
    ) -> torch.Tensor:
        distances, wall_indices, wall_along = self._portal_intersections()
        filled = torch.zeros_like(frame, dtype=torch.bool)
        pixel_y = self._pixel_y.to(torch.float32)
        for layer in range(distances.shape[2]):
            distance = distances[:, :, layer]
            wall_index = wall_indices[:, :, layer]
            along = wall_along[:, :, layer]
            sectors = self.map.portal_wall_sectors[wall_index]
            front = sectors[..., 0].clamp_min(0)
            back_raw = sectors[..., 1]
            back = back_raw.clamp_min(0)
            front_floor = self.map.sector_heights[front, 0]
            front_ceiling = self.map.sector_heights[front, 1]
            back_floor = self.map.sector_heights[back, 0]
            back_ceiling = self.map.sector_heights[back, 1]
            one_sided = back_raw < 0
            lower_low = torch.minimum(front_floor, back_floor)
            lower_high = torch.maximum(front_floor, back_floor)
            upper_low = torch.minimum(front_ceiling, back_ceiling)
            upper_high = torch.maximum(front_ceiling, back_ceiling)

            def project(
                world_z: torch.Tensor,
                layer_distance: torch.Tensor = distance,
            ) -> torch.Tensor:
                return center - (
                    (world_z - view_z[:, None]) * _PROJECTION_FOCAL_LENGTH / layer_distance
                )

            one_top = project(front_ceiling)
            one_bottom = project(front_floor)
            lower_top = project(lower_high)
            lower_bottom = project(lower_low)
            upper_top = project(upper_high)
            upper_bottom = project(upper_low)
            wall = self.map.portal_walls[wall_index]
            segment_x = wall[..., 2] - wall[..., 0]
            segment_y = wall[..., 3] - wall[..., 1]
            camera_cross = segment_x * (self.y[:, None] - wall[..., 1]) - segment_y * (
                self.x[:, None] - wall[..., 0]
            )
            side_index = (camera_cross < 0).to(torch.int64)
            from_front = side_index == 0
            view_floor = torch.where(from_front, front_floor, back_floor)
            other_floor = torch.where(from_front, back_floor, front_floor)
            view_ceiling = torch.where(from_front, front_ceiling, back_ceiling)
            other_ceiling = torch.where(from_front, back_ceiling, front_ceiling)
            one_span = (
                (one_sided & from_front)[:, None, :]
                & (pixel_y >= one_top[:, None, :])
                & (pixel_y <= one_bottom[:, None, :])
            )
            lower_span = (
                (~one_sided & (view_floor < other_floor))[:, None, :]
                & (pixel_y >= lower_top[:, None, :])
                & (pixel_y <= lower_bottom[:, None, :])
            )
            upper_span = (
                (~one_sided & (view_ceiling > other_ceiling))[:, None, :]
                & (pixel_y >= upper_top[:, None, :])
                & (pixel_y <= upper_bottom[:, None, :])
            )
            side_textures = self.map.portal_side_texture_ids[wall_index, side_index]
            texture_id = torch.where(
                one_span,
                side_textures[..., 0][:, None, :],
                torch.where(
                    lower_span,
                    side_textures[..., 1][:, None, :],
                    side_textures[..., 2][:, None, :],
                ),
            )
            has_texture = texture_id >= 0
            span = (
                (one_span | lower_span | upper_span)
                & has_texture
                & torch.isfinite(distance)[:, None, :]
                & ~filled
            )
            safe_texture_id = texture_id.clamp_min(0)
            texture_width = self.map.texture_widths[safe_texture_id]
            texture_height = self.map.texture_heights[safe_texture_id]
            texture_offset = self.map.portal_side_texture_offsets[wall_index, side_index]
            texture_u = torch.remainder(
                torch.floor(
                    along * self.map.portal_wall_lengths[wall_index] + texture_offset[..., 0]
                ).to(torch.int64)[:, None, :],
                texture_width,
            )
            world_z = view_z[:, None, None] + (
                (center - pixel_y) * distance[:, None, :] / _PROJECTION_FOCAL_LENGTH
            )
            texture_v = torch.remainder(
                torch.floor(-world_z + texture_offset[:, None, :, 1]).to(torch.int64),
                texture_height,
            )
            texture_u = texture_u.expand(-1, self.observation_height, -1)
            texture_index = safe_texture_id
            texture = self.map.texture_atlas[texture_index, texture_v, texture_u].to(torch.float32)
            view_sector = torch.where(from_front, front, back)
            light = self.map.sector_lights[view_sector]
            attenuation = light.clamp(24, 220)
            wall_value = (texture * attenuation[:, None, :] / 210.0).clamp(0, 255)
            frame = torch.where(span, wall_value, frame)
            filled |= span
        return frame

    def _render_weapon(self, frame: torch.Tensor) -> torch.Tensor:
        weapon = self._active_weapon().clamp(0, 7)
        value = self.map.weapon_screen_values[weapon]
        alpha = self.map.weapon_screen_alpha[weapon]
        lower_vertical_tics = torch.clamp(
            _WEAPON_LOWER_TICS - self.weapon_lower_cooldown,
            0,
            _WEAPON_LOWER_TICS,
        )
        vertical_tics = torch.where(
            self.pending_weapon >= 0,
            lower_vertical_tics,
            self.weapon_raise_cooldown,
        )
        raise_pixels = torch.round(
            vertical_tics.to(torch.float32) * (6.0 * self.observation_height / 200.0)
        ).to(torch.int64)
        source_y = self._pixel_y.to(torch.int64) - raise_pixels[:, None, None]
        valid = (source_y >= 0) & (source_y < self.observation_height)
        source_y = source_y.clamp(0, self.observation_height - 1).expand(
            -1,
            -1,
            self.observation_width,
        )
        value = value.gather(1, source_y)
        alpha = alpha.gather(1, source_y)
        visible = (valid & ~self.player_dead[:, None, None]).to(torch.float32)
        value *= visible
        alpha *= visible
        return value + frame * (1.0 - alpha)

    def render_frame(self) -> torch.Tensor:
        distance, _wall_index, _wall_along = self._raycast()
        center = self.observation_height * 0.48
        sector = self._current_sector()
        view_z = self.z + _VIEW_HEIGHT
        frame = self._render_flats(sector, view_z, center)
        frame = self._render_portal_walls(frame, view_z, center)

        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            dolls = self.map.player_starts[:-1, :2]
            actor_x = torch.cat((self.enemy_x, dolls[None, :, 0].expand(self.num_envs, -1)), dim=1)
            actor_y = torch.cat((self.enemy_y, dolls[None, :, 1].expand(self.num_envs, -1)), dim=1)
            actor_z = torch.cat(
                (
                    self.enemy_z,
                    self._player_start_z[:-1][None, :].expand(self.num_envs, -1),
                ),
                dim=1,
            )
            actor_alive = torch.cat(
                (
                    self.enemy_alive,
                    (~self.player_dead)[:, None].expand(-1, doll_count),
                ),
                dim=1,
            )
            actor_type = torch.cat(
                (
                    self.enemy_type,
                    torch.full(
                        (self.num_envs, doll_count),
                        2,
                        device=self.device,
                        dtype=torch.int64,
                    ),
                ),
                dim=1,
            )
        else:
            actor_x = self.enemy_x
            actor_y = self.enemy_y
            actor_z = self.enemy_z
            actor_alive = self.enemy_alive
            actor_type = self.enemy_type
        actor_x = torch.cat((actor_x, self.projectile_x), dim=1)
        actor_y = torch.cat((actor_y, self.projectile_y), dim=1)
        actor_z = torch.cat((actor_z, self.projectile_z), dim=1)
        actor_alive = torch.cat((actor_alive, self.projectile_alive), dim=1)
        actor_type = torch.cat((actor_type, self.projectile_type.clamp_min(0) + 23), dim=1)
        actor_x = torch.cat((actor_x, self.enemy_projectile_x), dim=1)
        actor_y = torch.cat((actor_y, self.enemy_projectile_y), dim=1)
        actor_z = torch.cat((actor_z, self.enemy_projectile_z), dim=1)
        actor_alive = torch.cat((actor_alive, self.enemy_projectile_alive), dim=1)
        actor_type = torch.cat(
            (
                actor_type,
                torch.full_like(self.enemy_projectile_age, 25, dtype=torch.int64),
            ),
            dim=1,
        )

        map_item_x = self.map.item_spawns[None, :, 0].expand(self.num_envs, -1)
        map_item_y = self.map.item_spawns[None, :, 1].expand(self.num_envs, -1)
        map_item_z = self._item_z[None, :].expand(self.num_envs, -1)
        map_item_type = self.map.item_visual_types[None, :].expand(self.num_envs, -1)
        drop_visible = (self.drop_type >= 0) & (self.drop_delay <= 0)
        drop_visual_type = torch.full_like(self.drop_type, 18)
        drop_visual_type = torch.where(self.drop_type == 2007, 12, drop_visual_type)
        drop_visual_type = torch.where(self.drop_type == 2002, 20, drop_visual_type)
        actor_x = torch.cat((actor_x, map_item_x, self.enemy_x), dim=1)
        actor_y = torch.cat((actor_y, map_item_y, self.enemy_y), dim=1)
        actor_z = torch.cat((actor_z, map_item_z, self.enemy_z), dim=1)
        actor_alive = torch.cat((actor_alive, self.item_available, drop_visible), dim=1)
        actor_type = torch.cat((actor_type, map_item_type, drop_visual_type), dim=1)
        dx = actor_x - self.x[:, None]
        dy = actor_y - self.y[:, None]
        actor_distance = torch.sqrt(dx * dx + dy * dy).clamp_min_(1)
        relative = self._wrap_angle(torch.atan2(dy, dx) - self.angle[:, None])
        screen_center = (relative / (math.pi / 2) + 0.5) * self.observation_width
        safe_actor_type = actor_type.clamp_min(0)
        projection_scale = (self.observation_width * 0.5) / actor_distance
        sprite_width = self.map.sprite_widths[safe_actor_type].to(torch.float32)
        sprite_height = self.map.sprite_heights[safe_actor_type].to(torch.float32)
        sprite_left = (
            screen_center - self.map.sprite_left_offsets[safe_actor_type] * projection_scale
        )
        sprite_top = (
            center
            + (view_z[:, None] - actor_z) * projection_scale
            - self.map.sprite_top_offsets[safe_actor_type] * projection_scale
        )
        sprite_right = sprite_left + sprite_width * projection_scale
        column_inside = (self._pixel_x >= sprite_left[:, :, None]) & (
            self._pixel_x < sprite_right[:, :, None]
        )
        candidate = (
            column_inside
            & actor_alive[:, :, None]
            & (relative[:, :, None].abs() < math.pi / 4)
            & (actor_distance[:, :, None] < distance[:, None, :])
        )
        candidate_distance = torch.where(
            candidate,
            actor_distance[:, :, None],
            torch.full_like(actor_distance[:, :, None], torch.inf),
        )
        nearest_distance, nearest_actor = torch.min(candidate_distance, dim=1)
        selected_type = safe_actor_type.gather(1, nearest_actor)
        selected_scale = projection_scale.gather(1, nearest_actor)
        selected_left = sprite_left.gather(1, nearest_actor)
        selected_top = sprite_top.gather(1, nearest_actor)
        selected_width = sprite_width.gather(1, nearest_actor).to(torch.int64)
        selected_height = sprite_height.gather(1, nearest_actor).to(torch.int64)
        sprite_u = torch.floor((self._pixel_x[:, 0, :] - selected_left) / selected_scale).to(
            torch.int64
        )
        sprite_v = torch.floor(
            (self._pixel_y - selected_top[:, None, :]) / selected_scale[:, None, :]
        ).to(torch.int64)
        inside_sprite = (
            torch.isfinite(nearest_distance)[:, None, :]
            & (sprite_u[:, None, :] >= 0)
            & (sprite_u[:, None, :] < selected_width[:, None, :])
            & (sprite_v >= 0)
            & (sprite_v < selected_height[:, None, :])
        )
        sprite_u = sprite_u.clamp_min(0)[:, None, :].expand(-1, self.observation_height, -1)
        sprite_v = sprite_v.clamp_min(0)
        sprite_u = torch.minimum(
            sprite_u,
            (selected_width - 1)[:, None, :],
        )
        sprite_v = torch.minimum(
            sprite_v,
            (selected_height - 1)[:, None, :],
        )
        sprite_type = selected_type[:, None, :].expand(-1, self.observation_height, -1)
        sprite_opaque = self.map.sprite_opaque[sprite_type, sprite_v, sprite_u]
        sprite_value = self.map.sprite_atlas[sprite_type, sprite_v, sprite_u].to(torch.float32)
        frame = torch.where(inside_sprite & sprite_opaque, sprite_value, frame)
        frame = self._render_weapon(frame)
        flash = self.damage_flash.clamp(0, 1)[:, None, None]
        frame = frame * (1 - flash * 0.35) + 255 * flash * 0.35
        if self.mask_hud:
            frame[:, -11:, :] = 0
        return frame.clamp(0, 255).to(torch.uint8)

    def _update_signal_buffer(self) -> None:
        weapon_index = (self.selected_weapon - 1)[:, None]
        selected_ammo = self.ammo.gather(1, weapon_index).squeeze(1)
        self.signal_buffer[:, 0].copy_(self.killcount)
        self.signal_buffer[:, 1].copy_(self.health)
        self.signal_buffer[:, 2].copy_(self.armor)
        self.signal_buffer[:, 3].copy_(self.selected_weapon)
        self.signal_buffer[:, 4].copy_(selected_ammo)
        self.signal_buffer[:, 5:11].copy_(self.weapons)
        self.signal_buffer[:, 11:17].copy_(self.ammo)
        self.signal_buffer[:, 17].copy_(self.episode_time)
        self.signal_buffer[:, 18].copy_(self.episode_return)
        self.signal_buffer[:, 19].copy_(self.player_dead)
        self.signal_buffer[:, 20].copy_(self.pending_reset)

    def signals(self) -> dict[str, torch.Tensor]:
        return {
            name: self.signal_buffer[:, index].to(torch.float64)
            for index, name in enumerate(DEVICE_SIGNAL_NAMES)
        }
