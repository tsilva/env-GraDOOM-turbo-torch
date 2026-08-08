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
_FIXED_UNIT = 1 << 16
_FINE_ANGLES = 8192
_FINE_ANGLE_SCALE = _FINE_ANGLES / (2.0 * math.pi)
_ANGLE_45 = 1 << 29
_ANGLE_90 = 1 << 30
_ANGLE_180 = 1 << 31
_ANGLE_270 = 3 << 30
_ANGLE_TO_FINE_SHIFT = 19
_SLOPE_RANGE = 2048
_ENEMY_HEALTH = (20.0, 30.0, 100.0, 70.0, 150.0, 500.0)
_ENEMY_STRIDE = (8.0, 8.0, 8.0, 8.0, 10.0, 8.0)
_ENEMY_MOVE_INTERVAL = (4, 3, 4, 3, 2, 3)
_ENEMY_WALK_FRAME_TICS = (8, 6, 4, 6, 4, 6)
_ENEMY_LOOK_INTERVAL = (10, 10, 8, 10, 10, 10)
_ENEMY_IDLE_FRAME_TICS = (10, 10, 8, 10, 10, 10)
_ENEMY_RADIUS = (20.0, 20.0, 16.0, 20.0, 30.0, 24.0)
_ENEMY_HEIGHT = (56.0, 56.0, 56.0, 56.0, 56.0, 64.0)
_ENEMY_MASS = (100.0, 100.0, 100.0, 100.0, 400.0, 1000.0)
_ENEMY_ATTACK_RANGE = (2048.0, 2048.0, 64.0, 2048.0, 64.0, 2048.0)
_ENEMY_ATTACK_PREFIRE = (10, 10, 4, 10, 16, 16)
_ENEMY_ATTACK_RECOVERY = (16, 20, 4, 4, 8, 8)
_ENEMY_PAIN_CHANCE = (200, 170, 160, 170, 180, 50)
_ENEMY_PAIN_TICS = (6, 6, 8, 6, 4, 4)
_ENEMY_NO_BLOCK_DELAY = (10, 10, 20, 10, 20, 24)
_ENEMY_XDEATH_NO_BLOCK_DELAY = (10, 10, 10, 10, 20, 24)
_ENEMY_HAS_XDEATH = (True, True, True, True, False, False)
_ENEMY_KILL_REWARD = (1.0, 3.0, 3.0, 4.0, 3.0, 10.0)
_ENEMY_SPAWN_THRESHOLD = (2621, 2621, 1310, 1310, 655, 655)
_ENEMY_SPAWN_DELAY = 105
_ENEMY_SPAWN_PERIOD = 10
_ENEMY_RETALIATION_THRESHOLD = 100
_ENEMY_SPAWN_REACTION_TIME = 8
_ENEMY_CHASE_X_SPEED_FIXED = (_FIXED_UNIT, 46341, 0, -46341, -_FIXED_UNIT, -46341, 0, 46341)
_ENEMY_CHASE_Y_SPEED_FIXED = (0, 46341, _FIXED_UNIT, 46341, 0, -46341, -_FIXED_UNIT, -46341)
_ENEMY_OPPOSITE_DIRECTION = (4, 5, 6, 7, 0, 1, 2, 3, 8)
_ENEMY_DIAGONAL_DIRECTION = (3, 1, 5, 7)
_TELEPORT_FOG_TOTAL_TICS = 72
_TELEPORT_FOG_INITIAL_TICS = _TELEPORT_FOG_TOTAL_TICS - 1
_PLAYER_TELEPORT_LOCK_TICS = 7
_PLAYER_FORWARD_ACCELERATION_FIXED = 25 << 11
_PLAYER_RUN_FORWARD_ACCELERATION_FIXED = 50 << 11
_PLAYER_SIDE_ACCELERATION_FIXED = 24 << 11
_PLAYER_RUN_SIDE_ACCELERATION_FIXED = 40 << 11
_PLAYER_MAX_INPUT_ACCELERATION_FIXED = 50 << 11
_CHAINSAW_PULL_ACCELERATION_FIXED = 100 << 11
_PLAYER_FRICTION_FIXED = 0xE800
_ACTOR_STOP_SPEED_FIXED = _FIXED_UNIT // 16
_PLAYER_AIR_CONTROL_FIXED = 0x0100
_PLAYER_AIR_FRICTION_FIXED = _FIXED_UNIT
_PLAYER_SLOW_TURN_TICS = 6
_PLAYER_SLOW_TURN_YAW = 320
_PLAYER_WALK_TURN_YAW = 640
_PLAYER_RUN_TURN_YAW = 1280
_PLAYER_BINARY_DELTA_TURN_YAW = 182
_PLAYER_BINARY_DELTA_PITCH = 182 << 16
_PLAYER_MIN_PITCH_BAM = -32 * (_ANGLE_45 // 45)
_PLAYER_MAX_PITCH_BAM = 56 * (_ANGLE_45 // 45)
_PLAYER_BINARY_DELTA_SIDE_ACCELERATION_FIXED = 1 << 11
_PLAYER_MOVE_BOB_FIXED = _FIXED_UNIT // 4
_PLAYER_MAX_BOB_FIXED = 16 * _FIXED_UNIT
_PLAYER_VIEW_BOB_PERIOD_TICS = 20
_PLAYER_DAMAGE_THRUST_PER_POINT_FIXED = _FIXED_UNIT // 8
_PLAYER_MAX_DAMAGE_THRUST_FIXED = 32 * _FIXED_UNIT
_PLAYER_RADIUS_THRUST_DENOMINATOR_FIXED = 200 * _FIXED_UNIT
_PLAYER_SELF_RADIUS_VERTICAL_THRUST_DENOMINATOR_FIXED = 1000 * _FIXED_UNIT
_ROCKET_SPLASH_DAMAGE = 128.0
_ROCKET_WALL_GRID_CELL = 64.0
_ROCKET_MAX_TARGET_CENTER_OFFSET = _ROCKET_SPLASH_DAMAGE + max(_ENEMY_RADIUS)
_WEAPON_LOWER_TICS = 16
_WEAPON_RAISE_TICS = 16
_WEAPON_SPAWN_RAISE_TICS = 14
_WEAPON_VERTICAL_STEP_PIXELS = 7.2
# Internal weapon order follows the DoomPlayer slot lists exactly:
# fist, chainsaw, pistol, shotgun, super shotgun, chaingun, rocket, plasma.
_WEAPON_SLOT = (1, 1, 2, 3, 3, 4, 5, 6)
_WEAPON_COOLDOWN = (17, 8, 14, 37, 51, 8, 20, 3)
_WEAPON_ACTION_DELAY = (4, 0, 4, 3, 3, 0, 8, 0)
# Remaining fire-state tics immediately after the trigger transition. These
# include recovery frames after A_ReFire, unlike _WEAPON_COOLDOWN, which is
# the interval until the next legal refire action.
_WEAPON_READY_DURATION = (21, 7, 18, 43, 61, 7, 19, 22)
_WEAPON_AMMO_SLOT = (-1, -1, 1, 2, 2, 1, 4, 5)
_WEAPON_AMMO_COST = (0.0, 0.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0)
_HITSCAN_PELLET_COUNTS = (0, 0, 1, 7, 20, 1, 0, 0)
_HITSCAN_MAX_PELLETS = 20
_BULLET_AUTOAIM_RANGE = 1024.0
_PLAYER_HITSCAN_RANGE = 8192.0
_BULLET_AUTOAIM_OFFSET = 2.0 * math.pi / 64.0
_BULLET_AUTOAIM_MAX_SLOPE = math.tan(35.0 * math.pi / 180.0)
_BAM_TO_RADIANS = 2.0 * math.pi / float(1 << 32)
_NATIVE_FOCAL_X_FIXED = 160 * _FIXED_UNIT
# R_ExecuteSetViewSize derives 320x240's y aspect as integer 16.16.
_NATIVE_Y_ASPECT_FIXED = (320 * _FIXED_UNIT * 240) // (200 * 320)
_NATIVE_FOCAL_Y_FIXED = 160 * _NATIVE_Y_ASPECT_FIXED
_FIST_RANGE = 64.0
_CHAINSAW_RANGE = 65.0
_CHAINSAW_SPREAD_RADIANS = 2.8125 * math.pi / 180.0
_CHAINSAW_TURN_STEP = (90.0 / 20.0) * math.pi / 180.0
_CHAINSAW_TURN_OFFSET = (90.0 / 21.0) * math.pi / 180.0
# Ascending Doom Weapon.SelectionOrder, restricted to the certified profile.
_WEAPON_AUTO_SWITCH_ORDER = (7, 4, 5, 3, 2, 1, 6, 0)
_MONSTER_DROP_TYPE = (2007, 2001, -1, 2002, -1, -1)
_GREEN_ARMOR_SAVE = 21846.0 / 65536.0
_BLUE_ARMOR_SAVE = 0.5
_PLAYER_RADIUS = 16.0
_PLAYER_HEIGHT = 56.0
_PICKUP_RADIUS = 20.0
_PICKUP_REACH_BELOW = 32.0
_DROP_HEIGHT = 16.0
_DROP_GRAVITY_FIXED = _FIXED_UNIT
_VIEW_HEIGHT = 41.0
_MUGSHOT_STATE_TICS = 35
_MUGSHOT_RAMPAGE_DELAY = 70
_MUGSHOT_NORMAL_FRAME_TICS = 18
_MUGSHOT_GRIN_TICS = 71
_PROJECTION_FOCAL_LENGTH = 42.0
_PORTAL_LAYERS = 8
_HASH_GOLDEN_RATIO_SIGNED = -1640531527
_HASH_MURMUR_SIGNED = -2048144789
_PLAYER_PROJECTILE_SPEED = (20.0, 25.0)
_ENEMY_PROJECTILE_SPEED = 15.0
_DAMAGE_TO_ALPHA = (
    0,
    8,
    16,
    23,
    30,
    36,
    42,
    47,
    53,
    58,
    62,
    67,
    71,
    75,
    79,
    83,
    87,
    90,
    94,
    97,
    100,
    103,
    107,
    109,
    112,
    115,
    118,
    120,
    123,
    125,
    128,
    130,
    133,
    135,
    137,
    139,
    141,
    143,
    145,
    147,
    149,
    151,
    153,
    155,
    157,
    159,
    160,
    162,
    164,
    165,
    167,
    169,
    170,
    172,
    173,
    175,
    176,
    178,
    179,
    181,
    182,
    183,
    185,
    186,
    187,
    189,
    190,
    191,
    192,
    194,
    195,
    196,
    197,
    198,
    200,
    201,
    202,
    203,
    204,
    205,
    206,
    207,
    209,
    210,
    211,
    212,
    213,
    214,
    215,
    216,
    217,
    218,
    219,
    220,
    221,
    221,
    222,
    223,
    224,
    225,
    226,
    227,
    228,
    229,
    229,
    230,
    231,
    232,
    233,
    234,
    235,
    235,
    236,
    237,
)


def _build_fine_sine_fixed() -> np.ndarray:
    """Reproduce ZDoom's R_InitTables 16.16 finesine table exactly."""
    quarter = _FINE_ANGLES // 4
    table = np.empty(_FINE_ANGLES, dtype=np.int64)
    phase = np.arange(quarter, dtype=np.float64) * (2.0 * math.pi / _FINE_ANGLES)
    first_quarter = np.trunc(np.sin(phase) * _FIXED_UNIT).astype(np.int64)
    table[:quarter] = first_quarter
    table[quarter : 2 * quarter] = first_quarter[::-1]
    table[2 * quarter :] = -table[: 2 * quarter]
    table[quarter] = _FIXED_UNIT
    table[3 * quarter] = -_FIXED_UNIT
    return table


def _build_fine_tangent_fixed() -> np.ndarray:
    """Reproduce ZDoom's R_InitTables 16.16 finetangent table exactly."""
    half = _FINE_ANGLES // 2
    phase = (
        np.arange(half, dtype=np.float64) - _FINE_ANGLES // 4
    ) * (2.0 * math.pi / _FINE_ANGLES)
    phase[0] += math.pi / _FINE_ANGLES
    return np.trunc(np.tan(phase) * _FIXED_UNIT + 0.5).astype(np.int64)


def _build_tangent_to_angle() -> np.ndarray:
    """Reproduce the unsigned angle table used by Doom's R_PointToAngle2."""
    slope = np.arange(_SLOPE_RANGE + 1, dtype=np.float64) / _SLOPE_RANGE
    fraction = np.arctan(slope) / (2.0 * math.pi)
    return np.trunc(((1 << 32) - 1) * fraction).astype(np.int64)


def _build_rocket_wall_grid(
    scenario: CompiledScenario,
) -> tuple[float, float, int, int, np.ndarray, np.ndarray]:
    """Index every wall that can cross a nonzero rocket-splash trace."""
    minimum_x, maximum_x, minimum_y, maximum_y = scenario.bounds
    grid_minimum_x = math.floor(minimum_x / _ROCKET_WALL_GRID_CELL) * _ROCKET_WALL_GRID_CELL
    grid_minimum_y = math.floor(minimum_y / _ROCKET_WALL_GRID_CELL) * _ROCKET_WALL_GRID_CELL
    grid_width = max(
        math.ceil((maximum_x - grid_minimum_x) / _ROCKET_WALL_GRID_CELL),
        1,
    )
    grid_height = max(
        math.ceil((maximum_y - grid_minimum_y) / _ROCKET_WALL_GRID_CELL),
        1,
    )
    walls = scenario.wall_segments
    candidates: list[np.ndarray] = []
    for grid_y in range(grid_height):
        cell_minimum_y = grid_minimum_y + grid_y * _ROCKET_WALL_GRID_CELL
        cell_maximum_y = cell_minimum_y + _ROCKET_WALL_GRID_CELL
        for grid_x in range(grid_width):
            cell_minimum_x = grid_minimum_x + grid_x * _ROCKET_WALL_GRID_CELL
            cell_maximum_x = cell_minimum_x + _ROCKET_WALL_GRID_CELL
            if len(walls):
                overlaps = (
                    (np.maximum(walls[:, 0], walls[:, 2])
                    >= cell_minimum_x - _ROCKET_MAX_TARGET_CENTER_OFFSET)
                    & (np.minimum(walls[:, 0], walls[:, 2])
                    <= cell_maximum_x + _ROCKET_MAX_TARGET_CENTER_OFFSET)
                    & (np.maximum(walls[:, 1], walls[:, 3])
                    >= cell_minimum_y - _ROCKET_MAX_TARGET_CENTER_OFFSET)
                    & (np.minimum(walls[:, 1], walls[:, 3])
                    <= cell_maximum_y + _ROCKET_MAX_TARGET_CENTER_OFFSET)
                )
                candidates.append(np.flatnonzero(overlaps).astype(np.int64))
            else:
                candidates.append(np.empty(0, dtype=np.int64))
    candidate_width = max(max((len(value) for value in candidates), default=0), 1)
    wall_indices = np.zeros((len(candidates), candidate_width), dtype=np.int64)
    wall_valid = np.zeros((len(candidates), candidate_width), dtype=np.bool_)
    for cell_index, values in enumerate(candidates):
        wall_indices[cell_index, : len(values)] = values
        wall_valid[cell_index, : len(values)] = True
    return (
        grid_minimum_x,
        grid_minimum_y,
        grid_width,
        grid_height,
        wall_indices,
        wall_valid,
    )


_FINE_SINE_FIXED = _build_fine_sine_fixed()
_FINE_TANGENT_FIXED = _build_fine_tangent_fixed()
_TANGENT_TO_ANGLE = _build_tangent_to_angle()
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
    texture_animation_ids: torch.Tensor
    texture_animation_counts: torch.Tensor
    portal_walls: torch.Tensor
    portal_wall_sectors: torch.Tensor
    portal_wall_blocks_sight: torch.Tensor
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
    item_raw_visual_types: torch.Tensor
    playpal: torch.Tensor
    colormap: torch.Tensor
    texture_index_atlas: torch.Tensor
    raw_sprite_atlas: torch.Tensor
    raw_sprite_opaque: torch.Tensor
    raw_sprite_widths: torch.Tensor
    raw_sprite_heights: torch.Tensor
    raw_sprite_left_offsets: torch.Tensor
    raw_sprite_top_offsets: torch.Tensor
    enemy_walk_sprite_ids: torch.Tensor
    enemy_attack_sprite_ids: torch.Tensor
    enemy_death_sprite_ids: torch.Tensor
    enemy_death_frame_counts: torch.Tensor
    enemy_death_frame_durations: torch.Tensor
    enemy_death_total_tics: torch.Tensor
    enemy_xdeath_sprite_ids: torch.Tensor
    enemy_xdeath_frame_counts: torch.Tensor
    enemy_xdeath_frame_durations: torch.Tensor
    enemy_xdeath_total_tics: torch.Tensor
    enemy_pain_sprite_ids: torch.Tensor
    raw_projectile_flight_sprite_ids: torch.Tensor
    raw_projectile_explosion_sprite_ids: torch.Tensor
    raw_teleport_fog_sprite_ids: torch.Tensor
    projectile_explosion_frame_counts: torch.Tensor
    projectile_explosion_frame_durations: torch.Tensor
    projectile_explosion_total_tics: torch.Tensor
    projectile_additive_luts: torch.Tensor
    raw_static_sprite_ids: torch.Tensor
    raw_item_animation_sprite_ids: torch.Tensor
    native_weapon_screen_values: torch.Tensor
    native_weapon_screen_alpha: torch.Tensor
    native_weapon_frame_values: torch.Tensor
    native_weapon_frame_alpha: torch.Tensor
    native_weapon_frame_ids: torch.Tensor
    native_weapon_flash_ids: torch.Tensor
    native_weapon_flash_lights: torch.Tensor
    hud_patch_atlas: torch.Tensor
    hud_patch_opaque: torch.Tensor
    hud_patch_widths: torch.Tensor
    hud_patch_heights: torch.Tensor
    hud_patch_left_offsets: torch.Tensor
    hud_patch_top_offsets: torch.Tensor
    bounds: torch.Tensor
    spawn_bounds: torch.Tensor

    @classmethod
    def from_host(cls, scenario: CompiledScenario, device: torch.device) -> DeviceScenario:
        blocking_indices = scenario.blocking_wall_indices
        wall_blocks_sight = np.zeros(len(scenario.wall_segments), dtype=np.bool_)
        wall_blocks_sight[blocking_indices] = True
        sector_indices = scenario.wall_sectors[blocking_indices, 0].clip(min=0)
        wall_lights = scenario.sector_lights[sector_indices].astype("float32")
        blocking_walls = scenario.blocking_segments
        wall_lengths = np.sqrt(
            np.square(blocking_walls[:, 2] - blocking_walls[:, 0])
            + np.square(blocking_walls[:, 3] - blocking_walls[:, 1])
        ).astype(np.float32)
        portal_sector_indices = scenario.wall_sectors[:, 0].clip(min=0)
        portal_wall_lights = scenario.sector_lights[portal_sector_indices].astype("float32")
        # P_FinishLoadingLineDef stores sidedef TexelLength as sqrt(dx²+dy²)
        # rounded to the nearest integer before any wall-column mapping.
        portal_wall_lengths = np.floor(
            np.sqrt(
                np.square(scenario.wall_segments[:, 2] - scenario.wall_segments[:, 0])
                + np.square(scenario.wall_segments[:, 3] - scenario.wall_segments[:, 1])
            )
            + 0.5
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
        texture_index_atlas = (
            scenario.texture_atlas
            if scenario.texture_index_atlas is None
            else scenario.texture_index_atlas
        )
        colormap = (
            np.broadcast_to(np.arange(256, dtype=np.uint8), (34, 256)).copy()
            if scenario.colormap is None
            else scenario.colormap
        )
        texture_animation_ids = (
            np.arange(len(scenario.texture_atlas), dtype=np.int32)[:, None]
            if scenario.texture_animation_ids is None
            else scenario.texture_animation_ids
        )
        texture_animation_counts = (
            np.ones(len(scenario.texture_atlas), dtype=np.int32)
            if scenario.texture_animation_counts is None
            else scenario.texture_animation_counts
        )
        raw_sprite_atlas = (
            scenario.sprite_atlas
            if scenario.raw_sprite_atlas is None
            else scenario.raw_sprite_atlas
        )
        raw_sprite_opaque = (
            scenario.sprite_opaque
            if scenario.raw_sprite_opaque is None
            else scenario.raw_sprite_opaque
        )
        raw_sprite_widths = (
            scenario.sprite_widths
            if scenario.raw_sprite_widths is None
            else scenario.raw_sprite_widths
        )
        raw_sprite_heights = (
            scenario.sprite_heights
            if scenario.raw_sprite_heights is None
            else scenario.raw_sprite_heights
        )
        raw_sprite_left_offsets = (
            scenario.sprite_left_offsets
            if scenario.raw_sprite_left_offsets is None
            else scenario.raw_sprite_left_offsets
        )
        raw_sprite_top_offsets = (
            scenario.sprite_top_offsets
            if scenario.raw_sprite_top_offsets is None
            else scenario.raw_sprite_top_offsets
        )
        fallback_enemy_ids = np.empty((6, 4, 8), dtype=np.int32)
        for enemy_type in range(6):
            fallback_enemy_ids[enemy_type].fill(min(enemy_type, len(raw_sprite_atlas) - 1))
        enemy_walk_sprite_ids = (
            fallback_enemy_ids
            if scenario.enemy_walk_sprite_ids is None
            else scenario.enemy_walk_sprite_ids
        )
        enemy_attack_sprite_ids = (
            fallback_enemy_ids
            if scenario.enemy_attack_sprite_ids is None
            else scenario.enemy_attack_sprite_ids
        )
        enemy_death_sprite_ids = (
            fallback_enemy_ids[:, :1, 0]
            if scenario.enemy_death_sprite_ids is None
            else scenario.enemy_death_sprite_ids
        )
        enemy_death_frame_counts = (
            np.ones(6, dtype=np.int32)
            if scenario.enemy_death_frame_counts is None
            else scenario.enemy_death_frame_counts
        )
        enemy_death_frame_durations = (
            np.ones_like(enemy_death_sprite_ids, dtype=np.int32)
            if scenario.enemy_death_frame_durations is None
            else scenario.enemy_death_frame_durations
        )
        enemy_death_total_tics = (
            enemy_death_frame_durations.sum(axis=1, dtype=np.int32)
            if scenario.enemy_death_total_tics is None
            else scenario.enemy_death_total_tics
        )
        enemy_xdeath_sprite_ids = (
            enemy_death_sprite_ids
            if scenario.enemy_xdeath_sprite_ids is None
            else scenario.enemy_xdeath_sprite_ids
        )
        enemy_xdeath_frame_counts = (
            enemy_death_frame_counts
            if scenario.enemy_xdeath_frame_counts is None
            else scenario.enemy_xdeath_frame_counts
        )
        enemy_xdeath_frame_durations = (
            enemy_death_frame_durations
            if scenario.enemy_xdeath_frame_durations is None
            else scenario.enemy_xdeath_frame_durations
        )
        enemy_xdeath_total_tics = (
            enemy_xdeath_frame_durations.sum(axis=1, dtype=np.int32)
            if scenario.enemy_xdeath_total_tics is None
            else scenario.enemy_xdeath_total_tics
        )
        enemy_pain_sprite_ids = (
            fallback_enemy_ids[:, 0]
            if scenario.enemy_pain_sprite_ids is None
            else scenario.enemy_pain_sprite_ids
        )
        raw_projectile_flight_sprite_ids = (
            np.zeros((3, 2, 8), dtype=np.int32)
            if scenario.raw_projectile_flight_sprite_ids is None
            else scenario.raw_projectile_flight_sprite_ids
        )
        raw_projectile_explosion_sprite_ids = (
            np.zeros((3, 5), dtype=np.int32)
            if scenario.raw_projectile_explosion_sprite_ids is None
            else scenario.raw_projectile_explosion_sprite_ids
        )
        raw_teleport_fog_sprite_ids = (
            np.zeros(12, dtype=np.int32)
            if scenario.raw_teleport_fog_sprite_ids is None
            else scenario.raw_teleport_fog_sprite_ids
        )
        projectile_explosion_frame_counts = (
            np.asarray((3, 5, 3), dtype=np.int32)
            if scenario.projectile_explosion_frame_counts is None
            else scenario.projectile_explosion_frame_counts
        )
        projectile_explosion_frame_durations = (
            np.asarray(
                (
                    (8, 6, 4, 0, 0),
                    (4, 4, 4, 4, 4),
                    (6, 6, 6, 0, 0),
                ),
                dtype=np.int32,
            )
            if scenario.projectile_explosion_frame_durations is None
            else scenario.projectile_explosion_frame_durations
        )
        projectile_explosion_total_tics = (
            projectile_explosion_frame_durations.sum(axis=1, dtype=np.int32)
            if scenario.projectile_explosion_total_tics is None
            else scenario.projectile_explosion_total_tics
        )
        projectile_additive_luts = (
            np.broadcast_to(
                np.arange(256, dtype=np.uint8)[None, None, :],
                (2, 256, 256),
            ).copy()
            if scenario.projectile_additive_luts is None
            else scenario.projectile_additive_luts
        )
        if scenario.raw_static_sprite_ids is None:
            last_sprite = max(len(raw_sprite_atlas) - 1, 0)
            raw_static_sprite_ids = np.asarray(
                [min(index, last_sprite) for index in range(6, 26)],
                dtype=np.int32,
            )
        else:
            raw_static_sprite_ids = scenario.raw_static_sprite_ids
        raw_item_animation_sprite_ids = (
            np.zeros(8, dtype=np.int32)
            if scenario.raw_item_animation_sprite_ids is None
            else scenario.raw_item_animation_sprite_ids
        )
        item_raw_visual_types = np.full(len(scenario.item_types), -1, dtype=np.int64)
        for type_id, sprite_index in _ITEM_SPRITE_INDEX.items():
            item_raw_visual_types[scenario.item_types == type_id] = raw_static_sprite_ids[
                sprite_index - 6
            ]
        native_weapon_screen_values = (
            np.zeros((8, 208, 320), dtype=np.uint8)
            if scenario.native_weapon_screen_values is None
            else scenario.native_weapon_screen_values
        )
        native_weapon_screen_alpha = (
            np.zeros((8, 208, 320), dtype=np.bool_)
            if scenario.native_weapon_screen_alpha is None
            else scenario.native_weapon_screen_alpha
        )
        native_weapon_frame_values = (
            native_weapon_screen_values
            if scenario.native_weapon_frame_values is None
            else scenario.native_weapon_frame_values
        )
        native_weapon_frame_alpha = (
            native_weapon_screen_alpha
            if scenario.native_weapon_frame_alpha is None
            else scenario.native_weapon_frame_alpha
        )
        if scenario.native_weapon_frame_ids is None:
            native_weapon_frame_ids = np.broadcast_to(
                np.arange(8, dtype=np.int32)[:, None, None],
                (8, 2, 52),
            ).copy()
        else:
            native_weapon_frame_ids = scenario.native_weapon_frame_ids
        native_weapon_flash_ids = (
            np.full((8, 2, 52), -1, dtype=np.int32)
            if scenario.native_weapon_flash_ids is None
            else scenario.native_weapon_flash_ids
        )
        native_weapon_flash_lights = (
            np.zeros((8, 2, 52), dtype=np.int32)
            if scenario.native_weapon_flash_lights is None
            else scenario.native_weapon_flash_lights
        )
        hud_patch_atlas = (
            np.zeros((70, 32, 320), dtype=np.uint8)
            if scenario.hud_patch_atlas is None
            else scenario.hud_patch_atlas
        )
        hud_patch_opaque = (
            np.zeros_like(hud_patch_atlas, dtype=np.bool_)
            if scenario.hud_patch_opaque is None
            else scenario.hud_patch_opaque
        )
        hud_patch_widths = (
            np.zeros(len(hud_patch_atlas), dtype=np.int32)
            if scenario.hud_patch_widths is None
            else scenario.hud_patch_widths
        )
        hud_patch_heights = (
            np.zeros(len(hud_patch_atlas), dtype=np.int32)
            if scenario.hud_patch_heights is None
            else scenario.hud_patch_heights
        )
        hud_patch_left_offsets = (
            np.zeros(len(hud_patch_atlas), dtype=np.int32)
            if scenario.hud_patch_left_offsets is None
            else scenario.hud_patch_left_offsets
        )
        hud_patch_top_offsets = (
            np.zeros(len(hud_patch_atlas), dtype=np.int32)
            if scenario.hud_patch_top_offsets is None
            else scenario.hud_patch_top_offsets
        )
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
            texture_animation_ids=torch.as_tensor(
                texture_animation_ids, device=device, dtype=torch.int64
            ),
            texture_animation_counts=torch.as_tensor(
                texture_animation_counts, device=device, dtype=torch.int64
            ),
            portal_walls=torch.as_tensor(scenario.wall_segments, device=device),
            portal_wall_sectors=torch.as_tensor(
                scenario.wall_sectors, device=device, dtype=torch.int64
            ),
            portal_wall_blocks_sight=torch.as_tensor(
                wall_blocks_sight,
                device=device,
                dtype=torch.bool,
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
            item_raw_visual_types=torch.as_tensor(
                item_raw_visual_types, device=device, dtype=torch.int64
            ),
            playpal=torch.as_tensor(scenario.playpal, device=device),
            colormap=torch.as_tensor(colormap, device=device),
            texture_index_atlas=torch.as_tensor(texture_index_atlas, device=device),
            raw_sprite_atlas=torch.as_tensor(raw_sprite_atlas, device=device),
            raw_sprite_opaque=torch.as_tensor(raw_sprite_opaque, device=device),
            raw_sprite_widths=torch.as_tensor(raw_sprite_widths, device=device, dtype=torch.int64),
            raw_sprite_heights=torch.as_tensor(
                raw_sprite_heights, device=device, dtype=torch.int64
            ),
            raw_sprite_left_offsets=torch.as_tensor(
                raw_sprite_left_offsets, device=device, dtype=torch.float32
            ),
            raw_sprite_top_offsets=torch.as_tensor(
                raw_sprite_top_offsets, device=device, dtype=torch.float32
            ),
            enemy_walk_sprite_ids=torch.as_tensor(
                enemy_walk_sprite_ids, device=device, dtype=torch.int64
            ),
            enemy_attack_sprite_ids=torch.as_tensor(
                enemy_attack_sprite_ids, device=device, dtype=torch.int64
            ),
            enemy_death_sprite_ids=torch.as_tensor(
                enemy_death_sprite_ids, device=device, dtype=torch.int64
            ),
            enemy_death_frame_counts=torch.as_tensor(
                enemy_death_frame_counts, device=device, dtype=torch.int64
            ),
            enemy_death_frame_durations=torch.as_tensor(
                enemy_death_frame_durations, device=device, dtype=torch.int64
            ),
            enemy_death_total_tics=torch.as_tensor(
                enemy_death_total_tics, device=device, dtype=torch.int64
            ),
            enemy_xdeath_sprite_ids=torch.as_tensor(
                enemy_xdeath_sprite_ids, device=device, dtype=torch.int64
            ),
            enemy_xdeath_frame_counts=torch.as_tensor(
                enemy_xdeath_frame_counts, device=device, dtype=torch.int64
            ),
            enemy_xdeath_frame_durations=torch.as_tensor(
                enemy_xdeath_frame_durations, device=device, dtype=torch.int64
            ),
            enemy_xdeath_total_tics=torch.as_tensor(
                enemy_xdeath_total_tics, device=device, dtype=torch.int64
            ),
            enemy_pain_sprite_ids=torch.as_tensor(
                enemy_pain_sprite_ids, device=device, dtype=torch.int64
            ),
            raw_projectile_flight_sprite_ids=torch.as_tensor(
                raw_projectile_flight_sprite_ids, device=device, dtype=torch.int64
            ),
            raw_projectile_explosion_sprite_ids=torch.as_tensor(
                raw_projectile_explosion_sprite_ids, device=device, dtype=torch.int64
            ),
            raw_teleport_fog_sprite_ids=torch.as_tensor(
                raw_teleport_fog_sprite_ids, device=device, dtype=torch.int64
            ),
            projectile_explosion_frame_counts=torch.as_tensor(
                projectile_explosion_frame_counts, device=device, dtype=torch.int64
            ),
            projectile_explosion_frame_durations=torch.as_tensor(
                projectile_explosion_frame_durations, device=device, dtype=torch.int64
            ),
            projectile_explosion_total_tics=torch.as_tensor(
                projectile_explosion_total_tics, device=device, dtype=torch.int64
            ),
            projectile_additive_luts=torch.as_tensor(
                projectile_additive_luts, device=device, dtype=torch.uint8
            ),
            raw_static_sprite_ids=torch.as_tensor(
                raw_static_sprite_ids, device=device, dtype=torch.int64
            ),
            raw_item_animation_sprite_ids=torch.as_tensor(
                raw_item_animation_sprite_ids, device=device, dtype=torch.int64
            ),
            native_weapon_screen_values=torch.as_tensor(native_weapon_screen_values, device=device),
            native_weapon_screen_alpha=torch.as_tensor(native_weapon_screen_alpha, device=device),
            native_weapon_frame_values=torch.as_tensor(native_weapon_frame_values, device=device),
            native_weapon_frame_alpha=torch.as_tensor(native_weapon_frame_alpha, device=device),
            native_weapon_frame_ids=torch.as_tensor(
                native_weapon_frame_ids, device=device, dtype=torch.int64
            ),
            native_weapon_flash_ids=torch.as_tensor(
                native_weapon_flash_ids, device=device, dtype=torch.int64
            ),
            native_weapon_flash_lights=torch.as_tensor(
                native_weapon_flash_lights, device=device, dtype=torch.int64
            ),
            hud_patch_atlas=torch.as_tensor(hud_patch_atlas, device=device),
            hud_patch_opaque=torch.as_tensor(hud_patch_opaque, device=device),
            hud_patch_widths=torch.as_tensor(hud_patch_widths, device=device, dtype=torch.int64),
            hud_patch_heights=torch.as_tensor(hud_patch_heights, device=device, dtype=torch.int64),
            hud_patch_left_offsets=torch.as_tensor(
                hud_patch_left_offsets, device=device, dtype=torch.int64
            ),
            hud_patch_top_offsets=torch.as_tensor(
                hud_patch_top_offsets, device=device, dtype=torch.int64
            ),
            bounds=torch.tensor(bounds, device=device),
            spawn_bounds=torch.tensor(spawn_bounds, device=device),
        )


class TorchDeathmatchEngine:
    """Batched Doom-like state machine whose mutable state never leaves its device."""

    observation_height = 84
    observation_width = 84
    native_view_height = 208
    native_screen_height = 240
    native_screen_width = 320
    native_vertical_aspect = 1.2
    enemy_slots = 64
    enemy_projectile_slots = 64
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
        (
            self._rocket_wall_grid_minimum_x,
            self._rocket_wall_grid_minimum_y,
            self._rocket_wall_grid_width,
            self._rocket_wall_grid_height,
            rocket_wall_indices,
            rocket_wall_valid,
        ) = _build_rocket_wall_grid(scenario)
        self._rocket_wall_indices = torch.as_tensor(
            rocket_wall_indices,
            device=device,
            dtype=torch.int64,
        )
        self._rocket_wall_valid = torch.as_tensor(
            rocket_wall_valid,
            device=device,
            dtype=torch.bool,
        )
        n = num_envs
        self.rng_state = torch.ones(n, device=device, dtype=torch.int64)
        self.enemy_chase_rng_state = torch.ones(n, device=device, dtype=torch.int64)
        self.episode_time = torch.zeros(n, device=device, dtype=torch.int32)
        self.episode_return = torch.zeros(n, device=device)
        self.pending_reset = torch.ones(n, device=device, dtype=torch.bool)
        self.player_dead = torch.zeros(n, device=device, dtype=torch.bool)
        self.x = torch.zeros(n, device=device)
        self.y = torch.zeros(n, device=device)
        self.z = torch.zeros(n, device=device)
        self.player_floor_z = torch.zeros(n, device=device)
        self.previous_player_floor_z = torch.zeros(n, device=device)
        self.player_ceiling_z = torch.zeros(n, device=device)
        self.view_z = torch.zeros(n, device=device)
        self.view_height = torch.full((n,), _VIEW_HEIGHT, device=device)
        self.delta_view_height = torch.zeros(n, device=device)
        self.angle = torch.zeros(n, device=device)
        # Doom applies keyboard yaw in binary angle measurement (BAM) units.
        # Keep the exact retained angle while exposing radians through the
        # established public tensor API.
        self._angle_bam = torch.zeros(n, device=device, dtype=torch.int64)
        # Actor pitch is a signed BAM angle. ViZDoom exposes it in degrees,
        # while the render and weapon paths consume the retained integer.
        self._pitch_bam = torch.zeros(n, device=device, dtype=torch.int64)
        self.pitch = torch.zeros(n, device=device)
        self.turn_held_tics = torch.zeros(n, device=device, dtype=torch.int32)
        self.momentum_x = torch.zeros(n, device=device)
        self.momentum_y = torch.zeros(n, device=device)
        # Doom keeps actor position and momentum in signed 16.16 fixed point.
        # Public float tensors retain the established API; these tensors preserve
        # the low bits that float32 cannot represent at map-scale coordinates.
        self._x_fixed = torch.zeros(n, device=device, dtype=torch.int64)
        self._y_fixed = torch.zeros(n, device=device, dtype=torch.int64)
        self._momentum_x_fixed = torch.zeros(n, device=device, dtype=torch.int64)
        self._momentum_y_fixed = torch.zeros(n, device=device, dtype=torch.int64)
        self._player_bob_fixed = torch.zeros(n, device=device, dtype=torch.int64)
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
        # SBARINFO caches the ready weapon's ammo during Draw and consumes that
        # cache on the next status-bar Tick.  The large current-ammo number is
        # therefore one rendered observation behind inventory/game variables.
        self.hud_ready_ammo = torch.zeros(n, device=device)
        self.attack_cooldown = torch.zeros(n, device=device, dtype=torch.int32)
        self.weapon_state_cooldown = torch.zeros(n, device=device, dtype=torch.int32)
        self.pending_attack_weapon = torch.full((n,), -1, device=device, dtype=torch.int64)
        self.pending_attack_delay = torch.zeros(n, device=device, dtype=torch.int32)
        self.pending_attack_accurate = torch.zeros(n, device=device, dtype=torch.bool)
        self.weapon_fire_count = torch.zeros(n, device=device, dtype=torch.int32)
        self.weapon_ready_tics = torch.zeros(n, device=device, dtype=torch.int32)
        self.weapon_raise_cooldown = torch.zeros(n, device=device, dtype=torch.int32)
        self.pending_weapon = torch.full((n,), -1, device=device, dtype=torch.int64)
        self.weapon_lower_cooldown = torch.zeros(n, device=device, dtype=torch.int32)
        self.weapon_change_latched = torch.zeros(n, device=device, dtype=torch.bool)
        self.damage_count = torch.zeros(n, device=device, dtype=torch.int32)
        self.bonus_count = torch.zeros(n, device=device, dtype=torch.int32)
        self.mugshot_pain_tics = torch.zeros(n, device=device, dtype=torch.int32)
        self.mugshot_pain_direction = torch.ones(n, device=device, dtype=torch.int64)
        self.mugshot_ouch = torch.zeros(n, device=device, dtype=torch.bool)
        self.mugshot_grin = torch.zeros(n, device=device, dtype=torch.bool)
        self.mugshot_grin_tics = torch.zeros(n, device=device, dtype=torch.int32)
        self.mugshot_face_index = torch.ones(n, device=device, dtype=torch.int64)
        self.mugshot_face_tics = torch.full(
            (n,), _MUGSHOT_NORMAL_FRAME_TICS, device=device, dtype=torch.int32
        )
        self.mugshot_rng_state = torch.ones(n, device=device, dtype=torch.int64)
        self.attack_held_tics = torch.zeros(n, device=device, dtype=torch.int32)
        self.chainsaw_pull = torch.zeros(n, device=device, dtype=torch.bool)
        self.reaction_time = torch.zeros(n, device=device, dtype=torch.int32)
        self.enemy_x = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_y = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_z = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_angle = torch.zeros((n, self.enemy_slots), device=device)
        self._enemy_x_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._enemy_y_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._enemy_z_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._enemy_floor_z_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._enemy_ceiling_z_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._enemy_opening_initialized = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.bool
        )
        self._enemy_momentum_x_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._enemy_momentum_y_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._enemy_velocity_z_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self.enemy_type = torch.full((n, self.enemy_slots), -1, device=device, dtype=torch.int64)
        self.enemy_health = torch.zeros((n, self.enemy_slots), device=device)
        self.enemy_alive = torch.zeros((n, self.enemy_slots), device=device, dtype=torch.bool)
        self.enemy_cooldown = torch.zeros((n, self.enemy_slots), device=device, dtype=torch.int32)
        self.enemy_attack_phase = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_just_attacked = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.bool
        )
        self.enemy_just_hit = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.bool
        )
        self.enemy_reaction_time = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        # -1 denotes the controlled player; non-negative values are monster
        # slots. Doom's target pointer changes when a shootable attacker hurts
        # a monster and its retaliation threshold permits a switch.
        self.enemy_target_slot = torch.full(
            (n, self.enemy_slots), -1, device=device, dtype=torch.int64
        )
        self.enemy_target_threshold = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_heard_player = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.bool
        )
        self.enemy_move_direction = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self.enemy_move_count = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_move_cooldown = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_animation_tics = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_pain_tics = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.enemy_death_type = torch.full(
            (n, self.enemy_slots), -1, device=device, dtype=torch.int64
        )
        self.enemy_death_extreme = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.bool
        )
        self.enemy_death_tics = torch.zeros((n, self.enemy_slots), device=device, dtype=torch.int32)
        self.enemy_death_elapsed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.teleport_fog_x = torch.zeros((n, self.enemy_slots), device=device)
        self.teleport_fog_y = torch.zeros((n, self.enemy_slots), device=device)
        self.teleport_fog_z = torch.zeros((n, self.enemy_slots), device=device)
        self.teleport_fog_tics = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int32
        )
        self.drop_type = torch.full((n, self.enemy_slots), -1, device=device, dtype=torch.int64)
        self.drop_delay = torch.zeros((n, self.enemy_slots), device=device, dtype=torch.int32)
        self.drop_spawned = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.bool
        )
        self.drop_x = torch.zeros((n, self.enemy_slots), device=device)
        self.drop_y = torch.zeros((n, self.enemy_slots), device=device)
        self.drop_z = torch.zeros((n, self.enemy_slots), device=device)
        self._drop_x_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._drop_y_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._drop_z_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._drop_velocity_x_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._drop_velocity_y_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
        self._drop_velocity_z_fixed = torch.zeros(
            (n, self.enemy_slots), device=device, dtype=torch.int64
        )
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
        self.projectile_impact_type = torch.full(
            (n, self.player_projectile_slots), -1, device=device, dtype=torch.int64
        )
        self.projectile_impact_tics = torch.zeros(
            (n, self.player_projectile_slots), device=device, dtype=torch.int32
        )
        self.enemy_projectile_x = torch.zeros((n, self.enemy_projectile_slots), device=device)
        self.enemy_projectile_y = torch.zeros((n, self.enemy_projectile_slots), device=device)
        self.enemy_projectile_z = torch.zeros((n, self.enemy_projectile_slots), device=device)
        self.enemy_projectile_velocity_x = torch.zeros(
            (n, self.enemy_projectile_slots), device=device
        )
        self.enemy_projectile_velocity_y = torch.zeros(
            (n, self.enemy_projectile_slots), device=device
        )
        self.enemy_projectile_velocity_z = torch.zeros(
            (n, self.enemy_projectile_slots), device=device
        )
        self.enemy_projectile_age = torch.zeros(
            (n, self.enemy_projectile_slots), device=device, dtype=torch.int32
        )
        self.enemy_projectile_alive = torch.zeros(
            (n, self.enemy_projectile_slots), device=device, dtype=torch.bool
        )
        self.enemy_projectile_source_slot = torch.full(
            (n, self.enemy_projectile_slots), -1, device=device, dtype=torch.int64
        )
        self.enemy_projectile_impact_tics = torch.zeros(
            (n, self.enemy_projectile_slots), device=device, dtype=torch.int32
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
        self._enemy_walk_frame_tics = torch.tensor(
            _ENEMY_WALK_FRAME_TICS, device=device, dtype=torch.int32
        )
        self._enemy_look_interval = torch.tensor(
            _ENEMY_LOOK_INTERVAL, device=device, dtype=torch.int32
        )
        self._enemy_idle_frame_tics = torch.tensor(
            _ENEMY_IDLE_FRAME_TICS, device=device, dtype=torch.int32
        )
        self._enemy_radius = torch.tensor(_ENEMY_RADIUS, device=device)
        self._enemy_height = torch.tensor(_ENEMY_HEIGHT, device=device)
        self._enemy_mass = torch.tensor(_ENEMY_MASS, device=device)
        self._enemy_attack_range = torch.tensor(_ENEMY_ATTACK_RANGE, device=device)
        self._enemy_attack_prefire = torch.tensor(
            _ENEMY_ATTACK_PREFIRE, device=device, dtype=torch.int32
        )
        self._enemy_attack_recovery = torch.tensor(
            _ENEMY_ATTACK_RECOVERY, device=device, dtype=torch.int32
        )
        self._enemy_pain_chance = torch.tensor(
            _ENEMY_PAIN_CHANCE, device=device, dtype=torch.int64
        )
        self._enemy_pain_duration = torch.tensor(
            _ENEMY_PAIN_TICS, device=device, dtype=torch.int32
        )
        self._enemy_no_block_delay = torch.tensor(
            _ENEMY_NO_BLOCK_DELAY, device=device, dtype=torch.int32
        )
        self._enemy_xdeath_no_block_delay = torch.tensor(
            _ENEMY_XDEATH_NO_BLOCK_DELAY, device=device, dtype=torch.int32
        )
        self._enemy_has_xdeath = torch.tensor(
            _ENEMY_HAS_XDEATH, device=device, dtype=torch.bool
        )
        self._enemy_kill_reward = torch.tensor(_ENEMY_KILL_REWARD, device=device)
        self._enemy_spawn_threshold = torch.tensor(
            _ENEMY_SPAWN_THRESHOLD, device=device, dtype=torch.int64
        )
        chase_x = torch.tensor(_ENEMY_CHASE_X_SPEED_FIXED, device=device, dtype=torch.int64)
        chase_y = torch.tensor(_ENEMY_CHASE_Y_SPEED_FIXED, device=device, dtype=torch.int64)
        stride_fixed = torch.tensor(_ENEMY_STRIDE, device=device, dtype=torch.int64)
        self._enemy_chase_step_x_fixed = stride_fixed[:, None] * chase_x[None, :]
        self._enemy_chase_step_y_fixed = stride_fixed[:, None] * chase_y[None, :]
        self._enemy_opposite_direction = torch.tensor(
            _ENEMY_OPPOSITE_DIRECTION, device=device, dtype=torch.int64
        )
        self._enemy_diagonal_direction = torch.tensor(
            _ENEMY_DIAGONAL_DIRECTION, device=device, dtype=torch.int64
        )
        self._weapon_slot = torch.tensor(_WEAPON_SLOT, device=device, dtype=torch.int64)
        self._weapon_cooldown = torch.tensor(_WEAPON_COOLDOWN, device=device, dtype=torch.int32)
        self._weapon_ready_duration = torch.tensor(
            _WEAPON_READY_DURATION,
            device=device,
            dtype=torch.int32,
        )
        self._weapon_action_delay = torch.tensor(
            _WEAPON_ACTION_DELAY, device=device, dtype=torch.int32
        )
        self._weapon_ammo_slot = torch.tensor(_WEAPON_AMMO_SLOT, device=device, dtype=torch.int64)
        self._weapon_ammo_cost = torch.tensor(_WEAPON_AMMO_COST, device=device)
        self._hitscan_pellet_counts = torch.tensor(
            _HITSCAN_PELLET_COUNTS,
            device=device,
            dtype=torch.int64,
        )
        self._player_projectile_speed = torch.tensor(_PLAYER_PROJECTILE_SPEED, device=device)
        self._monster_drop_type = torch.tensor(_MONSTER_DROP_TYPE, device=device, dtype=torch.int64)
        self._damage_to_alpha = torch.tensor(
            _DAMAGE_TO_ALPHA,
            device=device,
            dtype=torch.float32,
        )
        self._fine_sine_fixed = torch.as_tensor(
            _FINE_SINE_FIXED,
            device=device,
            dtype=torch.int64,
        )
        self._fine_tangent_fixed = torch.as_tensor(
            _FINE_TANGENT_FIXED,
            device=device,
            dtype=torch.int64,
        )
        self._tangent_to_angle = torch.as_tensor(
            _TANGENT_TO_ANGLE,
            device=device,
            dtype=torch.int64,
        )
        self._blocking_walls_fixed = torch.round(
            self.map.blocking_walls * _FIXED_UNIT
        ).to(torch.int64)
        self._slot_base_weapon = torch.tensor(
            (0, 0, 2, 3, 5, 6, 7), device=device, dtype=torch.int64
        )
        self._ray_offsets = torch.linspace(
            math.pi / 4,
            -math.pi / 4,
            self.observation_width,
            device=device,
        )
        self._pixel_x = torch.arange(self.observation_width, device=device).view(1, 1, -1)
        self._pixel_y = torch.arange(self.observation_height, device=device).view(1, -1, 1)
        native_columns = (
            torch.arange(self.native_screen_width, device=device, dtype=torch.float32)
            - self.native_screen_width / 2.0
        )
        self._native_ray_offsets = -torch.atan(native_columns / (self.native_screen_width / 2.0))
        self._native_flat_columns = (
            torch.arange(self.native_screen_width, device=device, dtype=torch.int64)
            - (self.native_screen_width // 2 - 1)
        )
        self._native_pixel_x = torch.arange(self.native_screen_width, device=device).view(1, 1, -1)
        self._native_pixel_y = torch.arange(self.native_view_height, device=device).view(1, -1, 1)
        player_start_sectors = self._sector_at(
            self.map.player_starts[:, 0], self.map.player_starts[:, 1]
        )
        self._player_start_z = self.map.sector_heights[player_start_sectors, 0]
        if len(self.map.item_spawns):
            item_sectors = self._sector_at(self.map.item_spawns[:, 0], self.map.item_spawns[:, 1])
            self._item_z = self.map.sector_heights[item_sectors, 0] + self.map.item_spawns[:, 2]
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
    def _public_or_retained_fixed(
        public: torch.Tensor,
        retained: torch.Tensor,
    ) -> torch.Tensor:
        """Honor mutable public coordinates without discarding retained low bits."""

        visible = retained.to(torch.float32) / _FIXED_UNIT
        return torch.where(
            public != visible,
            torch.round(public * _FIXED_UNIT).to(torch.int64),
            retained,
        )

    def _enemy_chase_random(self, mask: torch.Tensor) -> torch.Tensor:
        """Advance Doom's logically separate chase-direction random stream."""

        value = self.enemy_chase_rng_state
        updated = torch.bitwise_xor(value, torch.bitwise_and(value << 13, _UINT32_MASK))
        updated = torch.bitwise_xor(updated, updated >> 17)
        updated = torch.bitwise_xor(updated, torch.bitwise_and(updated << 5, _UINT32_MASK))
        updated = torch.bitwise_and(updated, _UINT32_MASK)
        self.enemy_chase_rng_state.copy_(torch.where(mask, updated, value))
        slot = torch.arange(self.enemy_slots, device=self.device, dtype=torch.int64)[None, :]
        mixed = self.enemy_chase_rng_state[:, None] ^ (slot * _HASH_GOLDEN_RATIO_SIGNED)
        mixed ^= mixed >> 16
        return torch.bitwise_and(mixed, 255)

    @staticmethod
    def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi

    @staticmethod
    def _fine_angle_index(angle: torch.Tensor) -> torch.Tensor:
        """Quantize radians to the lookup-table index used by Doom traces."""
        return torch.floor(
            torch.remainder(angle, 2.0 * math.pi) * _FINE_ANGLE_SCALE
        ).to(torch.int64) & (_FINE_ANGLES - 1)

    def _fine_direction(
        self,
        angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._fine_direction_from_index(self._fine_angle_index(angle))

    def _fine_direction_from_index(
        self,
        fine_angle: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fine_angle = fine_angle & (_FINE_ANGLES - 1)
        sine = self._fine_sine_fixed[fine_angle].to(torch.float32) / _FIXED_UNIT
        cosine = self._fine_sine_fixed[
            (fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ].to(torch.float32) / _FIXED_UNIT
        return cosine, sine

    def _doom_bam_angle(
        self,
        delta_x_fixed: torch.Tensor,
        delta_y_fixed: torch.Tensor,
    ) -> torch.Tensor:
        """Return R_PointToAngle2's unsigned 32-bit BAM result on device."""

        def slope_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
            use_lookup = denominator >= 512
            divisor = torch.where(
                use_lookup,
                denominator >> 8,
                torch.ones_like(denominator),
            )
            index = torch.div(
                numerator << 3,
                divisor,
                rounding_mode="trunc",
            ).clamp_max(_SLOPE_RANGE)
            return torch.where(
                use_lookup,
                self._tangent_to_angle[index],
                torch.full_like(index, _ANGLE_45 - 1),
            )

        x_positive = delta_x_fixed >= 0
        y_positive = delta_y_fixed >= 0
        absolute_x = delta_x_fixed.abs()
        absolute_y = delta_y_fixed.abs()
        x_dominant = absolute_x > absolute_y
        shallow = slope_div(absolute_y, absolute_x)
        steep = slope_div(absolute_x, absolute_y)

        first_quadrant = torch.where(
            x_dominant,
            shallow,
            _ANGLE_90 - 1 - steep,
        )
        fourth_quadrant = torch.where(
            x_dominant,
            -shallow,
            _ANGLE_270 + steep,
        )
        second_quadrant = torch.where(
            x_dominant,
            _ANGLE_180 - 1 - shallow,
            _ANGLE_90 + steep,
        )
        third_quadrant = torch.where(
            x_dominant,
            _ANGLE_180 + shallow,
            _ANGLE_270 - 1 - steep,
        )
        angle = torch.where(
            x_positive,
            torch.where(y_positive, first_quadrant, fourth_quadrant),
            torch.where(y_positive, second_quadrant, third_quadrant),
        )
        angle = torch.where(
            (delta_x_fixed == 0) & (delta_y_fixed == 0),
            torch.zeros_like(angle),
            angle,
        )
        return angle & _UINT32_MASK

    def _doom_fine_angle(
        self,
        delta_x_fixed: torch.Tensor,
        delta_y_fixed: torch.Tensor,
    ) -> torch.Tensor:
        """Return R_PointToAngle2's 13-bit fine-angle result on device."""

        return self._doom_bam_angle(
            delta_x_fixed,
            delta_y_fixed,
        ) >> _ANGLE_TO_FINE_SHIFT

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
        chase_seeds = torch.bitwise_and(safe_seeds ^ 0xA511E9B3, _UINT32_MASK)
        chase_seeds = torch.where(
            chase_seeds == 0,
            torch.full_like(chase_seeds, 0x9E3779B9),
            chase_seeds,
        )
        self.enemy_chase_rng_state.copy_(
            torch.where(mask, chase_seeds, self.enemy_chase_rng_state)
        )
        self.mugshot_rng_state.copy_(
            torch.where(mask, safe_seeds, self.mugshot_rng_state)
        )
        # Sequential lane seeds have strongly correlated first xorshift32 outputs.
        # Four masked diffusion rounds retain deterministic streams while preventing
        # the first spatial sample from collapsing toward the low edge of the map.
        for _ in range(4):
            self._random_u32(mask)
        self._reset_enemies(mask)
        spawn_x, spawn_y, spawn_angle, _ = self._random_spawn_positions(mask, avoid_player=False)
        self.x.copy_(torch.where(mask, spawn_x, self.x))
        self.y.copy_(torch.where(mask, spawn_y, self.y))
        spawn_angle_bam = torch.bitwise_and(
            torch.round(torch.remainder(spawn_angle, 2.0 * math.pi) / _BAM_TO_RADIANS).to(
                torch.int64
            ),
            _UINT32_MASK,
        )
        self._angle_bam.copy_(torch.where(mask, spawn_angle_bam, self._angle_bam))
        self.angle.copy_(
            torch.where(
                mask,
                self._angle_bam.to(torch.float32) * _BAM_TO_RADIANS,
                self.angle,
            )
        )
        self._pitch_bam.masked_fill_(mask, 0)
        self.pitch.masked_fill_(mask, 0)
        spawn_x_fixed = torch.round(spawn_x * _FIXED_UNIT).to(torch.int64)
        spawn_y_fixed = torch.round(spawn_y * _FIXED_UNIT).to(torch.int64)
        self._x_fixed.copy_(torch.where(mask, spawn_x_fixed, self._x_fixed))
        self._y_fixed.copy_(torch.where(mask, spawn_y_fixed, self._y_fixed))
        self.x.copy_(self._x_fixed.to(torch.float32) / _FIXED_UNIT)
        self.y.copy_(self._y_fixed.to(torch.float32) / _FIXED_UNIT)
        spawn_floor, spawn_ceiling = self._player_opening_at(self.x, self.y)
        self.player_floor_z.copy_(
            torch.where(mask, spawn_floor, self.player_floor_z)
        )
        self.previous_player_floor_z.copy_(
            torch.where(mask, spawn_floor, self.previous_player_floor_z)
        )
        self.player_ceiling_z.copy_(
            torch.where(mask, spawn_ceiling, self.player_ceiling_z)
        )
        spawn_sector = self._sector_at(spawn_x, spawn_y)
        spawn_z = self.map.sector_heights[spawn_sector, 0]
        self.z.copy_(torch.where(mask, spawn_z, self.z))
        self.view_z.copy_(torch.where(mask, spawn_z + _VIEW_HEIGHT, self.view_z))
        self.view_height.masked_fill_(mask, _VIEW_HEIGHT)
        self.delta_view_height.masked_fill_(mask, 0)
        for tensor in (
            self.momentum_x,
            self.momentum_y,
            self.velocity_z,
            self.armor,
            self.armor_save_fraction,
            self.episode_return,
        ):
            tensor.masked_fill_(mask, 0)
        self._momentum_x_fixed.masked_fill_(mask, 0)
        self._momentum_y_fixed.masked_fill_(mask, 0)
        self._player_bob_fixed.masked_fill_(mask, 0)
        self.health.masked_fill_(mask, 100)
        self.killcount.masked_fill_(mask, 0)
        self.episode_time.masked_fill_(mask, 1)
        self.selected_weapon.masked_fill_(mask, 2)
        self.selected_weapon_variant.masked_fill_(mask, False)
        self.attack_cooldown.masked_fill_(mask, 0)
        self.weapon_state_cooldown.masked_fill_(mask, 0)
        self.pending_attack_weapon.masked_fill_(mask, -1)
        self.pending_attack_delay.masked_fill_(mask, 0)
        self.pending_attack_accurate.masked_fill_(mask, False)
        self.weapon_fire_count.masked_fill_(mask, 0)
        self.weapon_ready_tics.masked_fill_(mask, 0)
        self.weapon_raise_cooldown.masked_fill_(mask, _WEAPON_SPAWN_RAISE_TICS)
        self.pending_weapon.masked_fill_(mask, -1)
        self.weapon_lower_cooldown.masked_fill_(mask, 0)
        self.weapon_change_latched.masked_fill_(mask, False)
        self.damage_count.masked_fill_(mask, 0)
        self.bonus_count.masked_fill_(mask, 0)
        self.mugshot_pain_tics.masked_fill_(mask, 0)
        self.mugshot_pain_direction.masked_fill_(mask, 1)
        self.mugshot_ouch.masked_fill_(mask, False)
        self.mugshot_grin.masked_fill_(mask, False)
        self.mugshot_grin_tics.masked_fill_(mask, 0)
        self.mugshot_face_index.masked_fill_(mask, 1)
        self.mugshot_face_tics.masked_fill_(mask, _MUGSHOT_NORMAL_FRAME_TICS)
        self.attack_held_tics.masked_fill_(mask, 0)
        self.turn_held_tics.masked_fill_(mask, 0)
        self.chainsaw_pull.masked_fill_(mask, False)
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
        self.hud_ready_ammo.masked_fill_(mask, 50)
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
        self._enemy_x_fixed[mask] = 0
        self._enemy_y_fixed[mask] = 0
        self._enemy_z_fixed[mask] = 0
        self._enemy_floor_z_fixed[mask] = 0
        self._enemy_ceiling_z_fixed[mask] = 0
        self._enemy_opening_initialized[mask] = False
        self._enemy_momentum_x_fixed[mask] = 0
        self._enemy_momentum_y_fixed[mask] = 0
        self._enemy_velocity_z_fixed[mask] = 0
        self.enemy_type[mask] = -1
        self.enemy_health[mask] = 0
        self.enemy_alive[mask] = False
        self.enemy_cooldown[mask] = 0
        self.enemy_attack_phase[mask] = 0
        self.enemy_just_attacked[mask] = False
        self.enemy_just_hit[mask] = False
        self.enemy_reaction_time[mask] = 0
        self.enemy_target_slot[mask] = -1
        self.enemy_target_threshold[mask] = 0
        self.enemy_heard_player[mask] = False
        self.enemy_move_direction[mask] = 0
        self.enemy_move_count[mask] = 0
        self.enemy_move_cooldown[mask] = 0
        self.enemy_animation_tics[mask] = 0
        self.enemy_pain_tics[mask] = 0
        self.enemy_death_type[mask] = -1
        self.enemy_death_extreme[mask] = False
        self.enemy_death_tics[mask] = 0
        self.enemy_death_elapsed[mask] = 0
        self.teleport_fog_x[mask] = 0
        self.teleport_fog_y[mask] = 0
        self.teleport_fog_z[mask] = 0
        self.teleport_fog_tics[mask] = 0
        self.drop_type[mask] = -1
        self.drop_delay[mask] = 0
        self.drop_spawned[mask] = False
        self.drop_x[mask] = 0
        self.drop_y[mask] = 0
        self.drop_z[mask] = 0
        self._drop_x_fixed[mask] = 0
        self._drop_y_fixed[mask] = 0
        self._drop_z_fixed[mask] = 0
        self._drop_velocity_x_fixed[mask] = 0
        self._drop_velocity_y_fixed[mask] = 0
        self._drop_velocity_z_fixed[mask] = 0
        self.projectile_x[mask] = 0
        self.projectile_y[mask] = 0
        self.projectile_z[mask] = 0
        self.projectile_velocity_x[mask] = 0
        self.projectile_velocity_y[mask] = 0
        self.projectile_velocity_z[mask] = 0
        self.projectile_type[mask] = -1
        self.projectile_age[mask] = 0
        self.projectile_alive[mask] = False
        self.projectile_impact_type[mask] = -1
        self.projectile_impact_tics[mask] = 0
        self.enemy_projectile_x[mask] = 0
        self.enemy_projectile_y[mask] = 0
        self.enemy_projectile_z[mask] = 0
        self.enemy_projectile_velocity_x[mask] = 0
        self.enemy_projectile_velocity_y[mask] = 0
        self.enemy_projectile_velocity_z[mask] = 0
        self.enemy_projectile_age[mask] = 0
        self.enemy_projectile_alive[mask] = False
        self.enemy_projectile_source_slot[mask] = -1
        self.enemy_projectile_impact_tics[mask] = 0
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
        collision_radius = torch.as_tensor(radius, device=self.device, dtype=x.dtype)
        left = x[..., None] - collision_radius[..., None]
        right = x[..., None] + collision_radius[..., None]
        bottom = y[..., None] - collision_radius[..., None]
        top = y[..., None] + collision_radius[..., None]
        x1 = walls[:, 0]
        y1 = walls[:, 1]
        x2 = walls[:, 2]
        y2 = walls[:, 3]
        bounds_overlap = (
            (right > torch.minimum(x1, x2))
            & (left < torch.maximum(x1, x2))
            & (top > torch.minimum(y1, y2))
            & (bottom < torch.maximum(y1, y2))
        )
        delta_x = x2 - x1
        delta_y = y2 - y1
        side_bottom_left = delta_x * (bottom - y1) - delta_y * (left - x1)
        side_bottom_right = delta_x * (bottom - y1) - delta_y * (right - x1)
        side_top_left = delta_x * (top - y1) - delta_y * (left - x1)
        side_top_right = delta_x * (top - y1) - delta_y * (right - x1)
        minimum_side = torch.minimum(
            torch.minimum(side_bottom_left, side_bottom_right),
            torch.minimum(side_top_left, side_top_right),
        )
        maximum_side = torch.maximum(
            torch.maximum(side_bottom_left, side_bottom_right),
            torch.maximum(side_top_left, side_top_right),
        )
        crosses_line = (minimum_side <= 0) & (maximum_side >= 0)
        return torch.any(bounds_overlap & crosses_line, dim=-1)

    def _actor_opening_bounds(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        radius: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Doom's floor, ceiling, and dropoff floor under an actor box."""

        center_sector = self._sector_at(
            x.reshape(-1),
            y.reshape(-1),
        ).reshape_as(x)
        floor = self.map.sector_heights[center_sector, 0]
        ceiling = self.map.sector_heights[center_sector, 1]
        dropoff = floor
        walls = self.map.portal_walls
        if not len(walls):
            return floor, ceiling, dropoff

        actor_radius = torch.broadcast_to(
            torch.as_tensor(radius, device=self.device, dtype=x.dtype),
            x.shape,
        )
        left = x[..., None] - actor_radius[..., None]
        right = x[..., None] + actor_radius[..., None]
        bottom = y[..., None] - actor_radius[..., None]
        top = y[..., None] + actor_radius[..., None]
        x1 = walls[:, 0]
        y1 = walls[:, 1]
        x2 = walls[:, 2]
        y2 = walls[:, 3]
        bounds_overlap = (
            (right > torch.minimum(x1, x2))
            & (left < torch.maximum(x1, x2))
            & (top > torch.minimum(y1, y2))
            & (bottom < torch.maximum(y1, y2))
        )
        delta_x = x2 - x1
        delta_y = y2 - y1
        side_bottom_left = delta_x * (bottom - y1) - delta_y * (left - x1)
        side_bottom_right = delta_x * (bottom - y1) - delta_y * (right - x1)
        side_top_left = delta_x * (top - y1) - delta_y * (left - x1)
        side_top_right = delta_x * (top - y1) - delta_y * (right - x1)
        minimum_side = torch.minimum(
            torch.minimum(side_bottom_left, side_bottom_right),
            torch.minimum(side_top_left, side_top_right),
        )
        maximum_side = torch.maximum(
            torch.maximum(side_bottom_left, side_bottom_right),
            torch.maximum(side_top_left, side_top_right),
        )
        touches_line = bounds_overlap & (minimum_side <= 0) & (maximum_side >= 0)

        wall_sectors = self.map.portal_wall_sectors
        valid_sector = wall_sectors >= 0
        safe_sectors = wall_sectors.clamp_min(0)
        wall_floors = self.map.sector_heights[safe_sectors, 0]
        wall_ceilings = self.map.sector_heights[safe_sectors, 1]
        leading = (1,) * x.ndim
        touched_side = touches_line[..., :, None] & valid_sector.view(
            *leading,
            *valid_sector.shape,
        )
        touched_floors = torch.where(
            touched_side,
            wall_floors.view(*leading, *wall_floors.shape),
            torch.full_like(
                wall_floors.view(*leading, *wall_floors.shape),
                -torch.inf,
            ),
        )
        touched_ceilings = torch.where(
            touched_side,
            wall_ceilings.view(*leading, *wall_ceilings.shape),
            torch.full_like(
                wall_ceilings.view(*leading, *wall_ceilings.shape),
                torch.inf,
            ),
        )
        touched_dropoffs = torch.where(
            touched_side,
            wall_floors.view(*leading, *wall_floors.shape),
            torch.full_like(
                wall_floors.view(*leading, *wall_floors.shape),
                torch.inf,
            ),
        )
        floor = torch.maximum(floor, torch.amax(touched_floors, dim=(-2, -1)))
        ceiling = torch.minimum(ceiling, torch.amin(touched_ceilings, dim=(-2, -1)))
        dropoff = torch.minimum(
            dropoff,
            torch.amin(touched_dropoffs, dim=(-2, -1)),
        )
        return floor, ceiling, dropoff

    def _actor_opening_at(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        radius: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        floor, ceiling, _dropoff = self._actor_opening_bounds(x, y, radius)
        return floor, ceiling

    def _player_opening_at(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._actor_opening_at(x, y, _PLAYER_RADIUS)

    def _random_spawn_positions(
        self,
        mask: torch.Tensor,
        *,
        avoid_player: bool,
        candidate_count: int = 16,
        actor_radius: float | torch.Tensor = _PLAYER_RADIUS,
        actor_z: float | torch.Tensor | None = None,
        actor_height: float | torch.Tensor = _PLAYER_HEIGHT,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low_x, high_x, low_y, high_y = self.map.spawn_bounds
        unit_x = torch.stack([self._random_unit(mask) for _ in range(candidate_count)], dim=1)
        unit_y = torch.stack([self._random_unit(mask) for _ in range(candidate_count)], dim=1)
        candidate_x = low_x + unit_x * (high_x - low_x)
        candidate_y = low_y + unit_y * (high_y - low_y)
        radius = torch.as_tensor(actor_radius, device=self.device, dtype=candidate_x.dtype)
        valid = ~self._points_collide(candidate_x, candidate_y, radius)
        candidate_z: torch.Tensor | None = None
        candidate_height: torch.Tensor | None = None
        if actor_z is not None:
            candidate_z = torch.broadcast_to(
                torch.as_tensor(actor_z, device=self.device, dtype=candidate_x.dtype),
                (self.num_envs,),
            )[:, None]
            candidate_height = torch.broadcast_to(
                torch.as_tensor(
                    actor_height,
                    device=self.device,
                    dtype=candidate_x.dtype,
                ),
                (self.num_envs,),
            )[:, None]
            floor, ceiling = self._actor_opening_at(candidate_x, candidate_y, radius)
            valid &= (candidate_z >= floor) & (
                candidate_z + candidate_height <= ceiling
            )

        if len(self.map.player_starts) > 1:
            dolls = self.map.player_starts[:-1, :2]
            doll_dx = candidate_x[..., None] - dolls[None, None, :, 0]
            doll_dy = candidate_y[..., None] - dolls[None, None, :, 1]
            overlaps_doll = (
                (doll_dx.abs() < radius + _PLAYER_RADIUS)
                & (doll_dy.abs() < radius + _PLAYER_RADIUS)
            )
            if candidate_z is not None and candidate_height is not None:
                overlaps_doll &= self._vertical_overlap(
                    candidate_z[..., None],
                    candidate_height[..., None],
                    self._player_start_z[:-1][None, None, :],
                    _PLAYER_HEIGHT,
                )
            valid &= ~torch.any(overlaps_doll, dim=-1)
        if avoid_player:
            player_dx = candidate_x - self.x[:, None]
            player_dy = candidate_y - self.y[:, None]
            overlaps_player = (
                (player_dx.abs() < radius + _PLAYER_RADIUS)
                & (player_dy.abs() < radius + _PLAYER_RADIUS)
            )
            if candidate_z is not None and candidate_height is not None:
                overlaps_player &= self._vertical_overlap(
                    candidate_z,
                    candidate_height,
                    self.z[:, None],
                    _PLAYER_HEIGHT,
                )
            valid &= ~overlaps_player
            enemy_dx = candidate_x[..., None] - self.enemy_x[:, None, :]
            enemy_dy = candidate_y[..., None] - self.enemy_y[:, None, :]
            enemy_radius = self._enemy_radius[self._effective_enemy_type()]
            overlaps_enemy = self._enemy_solid_mask()[:, None, :] & (
                (enemy_dx.abs() < radius + enemy_radius[:, None, :])
                & (enemy_dy.abs() < radius + enemy_radius[:, None, :])
            )
            if candidate_z is not None and candidate_height is not None:
                overlaps_enemy &= self._vertical_overlap(
                    candidate_z[..., None],
                    candidate_height[..., None],
                    self.enemy_z[:, None, :],
                    self._effective_enemy_height()[:, None, :],
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

    def _enemy_solid_mask(self) -> torch.Tensor:
        death_type = self.enemy_death_type.clamp(0, 5)
        no_block_delay = torch.where(
            self.enemy_death_extreme,
            self._enemy_xdeath_no_block_delay[death_type],
            self._enemy_no_block_delay[death_type],
        )
        dying_solid = (
            (self.enemy_death_type >= 0)
            & (self.enemy_death_tics > 0)
            & (self.enemy_death_elapsed < no_block_delay)
        )
        return self.enemy_alive | dying_solid

    def _effective_enemy_type(self) -> torch.Tensor:
        return torch.where(
            self.enemy_type >= 0,
            self.enemy_type,
            self.enemy_death_type,
        ).clamp_min(0)

    def _effective_enemy_height(self) -> torch.Tensor:
        enemy_type = self._effective_enemy_type()
        live_height = self._enemy_height[enemy_type]
        # P_Die changes the actor height before entering either death state.
        return torch.where(self.enemy_alive, live_height, live_height * 0.25)

    def _player_collides(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        floor: torch.Tensor | None = None,
        ceiling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        collision = self._collides(x, y)
        if floor is None or ceiling is None:
            floor, ceiling = self._player_opening_at(x, y)
        collision |= floor > self.z + 24.0
        collision |= ceiling - torch.maximum(self.z, floor) < 56.0
        enemy_type = self._effective_enemy_type()
        enemy_radius = self._enemy_radius[enemy_type]
        enemy_dx = x[:, None] - self.enemy_x
        enemy_dy = y[:, None] - self.enemy_y
        enemy_vertical_overlap = self._vertical_overlap(
            self.z[:, None],
            _PLAYER_HEIGHT,
            self.enemy_z,
            self._effective_enemy_height(),
        )
        collision |= torch.any(
            self._enemy_solid_mask()
            & enemy_vertical_overlap
            & (enemy_dx.abs() < _PLAYER_RADIUS + enemy_radius)
            & (enemy_dy.abs() < _PLAYER_RADIUS + enemy_radius),
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
                & (doll_dx.abs() < 2 * _PLAYER_RADIUS)
                & (doll_dy.abs() < 2 * _PLAYER_RADIUS),
                dim=1,
            )
        return collision

    def _enemy_collides(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        enemy_type: torch.Tensor,
        *,
        allow_dropoff: bool = False,
        actor_height: torch.Tensor | None = None,
    ) -> torch.Tensor:
        radius = self._enemy_radius[enemy_type]
        collision = self._points_collide(x, y, radius)
        floor, ceiling, dropoff = self._actor_opening_bounds(x, y, radius)
        height = self._enemy_height[enemy_type] if actor_height is None else actor_height
        collision |= floor > self.enemy_z + 24.0
        if not allow_dropoff:
            collision |= dropoff < self.enemy_z - 24.0
        collision |= ceiling - torch.maximum(self.enemy_z, floor) < height
        dx = x[:, :, None] - self.enemy_x[:, None, :]
        dy = y[:, :, None] - self.enemy_y[:, None, :]
        other_type = self._effective_enemy_type()
        other_radius = self._enemy_radius[other_type]
        vertical_overlap = self._vertical_overlap(
            self.enemy_z[:, :, None],
            height[:, :, None],
            self.enemy_z[:, None, :],
            self._effective_enemy_height()[:, None, :],
        )
        not_self = ~torch.eye(
            self.enemy_slots,
            device=self.device,
            dtype=torch.bool,
        )[None, :, :]
        solid_enemy = self._enemy_solid_mask()[:, None, :] & not_self
        collision |= torch.any(
            solid_enemy
            & vertical_overlap
            & (dx.abs() < radius[:, :, None] + other_radius[:, None, :])
            & (dy.abs() < radius[:, :, None] + other_radius[:, None, :]),
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
            & (player_dx.abs() < radius + _PLAYER_RADIUS)
            & (player_dy.abs() < radius + _PLAYER_RADIUS)
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
                & (doll_dx.abs() < radius[:, :, None] + _PLAYER_RADIUS)
                & (doll_dy.abs() < radius[:, :, None] + _PLAYER_RADIUS),
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

    @staticmethod
    def _trunc_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Signed integer division with the C/C++ truncation used by ZDoom."""

        return torch.div(numerator, denominator, rounding_mode="trunc")

    def _axis_slide_contact_fixed(
        self,
        position_x: torch.Tensor,
        position_y: torch.Tensor,
        move_x: torch.Tensor,
        move_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find Doom's leading-box contact fraction for axis-aligned walls."""

        walls = self._blocking_walls_fixed
        x1 = walls[:, 0][None, :]
        y1 = walls[:, 1][None, :]
        x2 = walls[:, 2][None, :]
        y2 = walls[:, 3][None, :]
        horizontal = y1 == y2
        vertical = x1 == x2
        radius = int(_PLAYER_RADIUS * _FIXED_UNIT)
        sentinel = torch.full(
            (self.num_envs, len(walls)),
            _FIXED_UNIT + 1,
            device=self.device,
            dtype=torch.int64,
        )

        safe_move_y = torch.where(move_y == 0, torch.ones_like(move_y), move_y)
        horizontal_target = y1 - torch.sign(move_y[:, None]) * radius
        horizontal_fraction = torch.round(
            (horizontal_target - position_y[:, None]).to(torch.float64)
            * _FIXED_UNIT
            / safe_move_y[:, None].to(torch.float64)
        ).to(torch.int64)
        horizontal_x = position_x[:, None] + (
            move_x[:, None] * horizontal_fraction >> 16
        )
        horizontal_valid = (
            horizontal
            & (move_y[:, None] != 0)
            & (horizontal_fraction >= 0)
            & (horizontal_fraction <= _FIXED_UNIT)
            & (horizontal_x >= torch.minimum(x1, x2) - radius)
            & (horizontal_x <= torch.maximum(x1, x2) + radius)
        )
        horizontal_candidates = torch.where(
            horizontal_valid,
            horizontal_fraction,
            sentinel,
        )

        safe_move_x = torch.where(move_x == 0, torch.ones_like(move_x), move_x)
        vertical_target = x1 - torch.sign(move_x[:, None]) * radius
        vertical_fraction = torch.round(
            (vertical_target - position_x[:, None]).to(torch.float64)
            * _FIXED_UNIT
            / safe_move_x[:, None].to(torch.float64)
        ).to(torch.int64)
        vertical_y = position_y[:, None] + (
            move_y[:, None] * vertical_fraction >> 16
        )
        vertical_valid = (
            vertical
            & (move_x[:, None] != 0)
            & (vertical_fraction >= 0)
            & (vertical_fraction <= _FIXED_UNIT)
            & (vertical_y >= torch.minimum(y1, y2) - radius)
            & (vertical_y <= torch.maximum(y1, y2) + radius)
        )
        vertical_candidates = torch.where(
            vertical_valid,
            vertical_fraction,
            sentinel,
        )

        horizontal_best = torch.min(horizontal_candidates, dim=1).values
        vertical_best = torch.min(vertical_candidates, dim=1).values
        best = torch.minimum(horizontal_best, vertical_best)
        return best, horizontal_best <= vertical_best, best <= _FIXED_UNIT

    def _doom_axis_slide_move(
        self,
        playing: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Integrate ZDoom's common one-contact axis-wall slide in one pass."""

        start_x = self._x_fixed
        start_y = self._y_fixed
        move_x = self._momentum_x_fixed
        move_y = self._momentum_y_fixed
        dominant_speed = torch.maximum(move_x.abs(), move_y.abs())
        max_step = int((_PLAYER_RADIUS - 1.0) * _FIXED_UNIT)
        steps = torch.where(
            dominant_speed > max_step,
            1 + torch.div(dominant_speed, max_step, rounding_mode="floor"),
            torch.ones_like(dominant_speed),
        ).clamp_max(3)
        proposed_x = start_x + move_x
        proposed_y = start_y + move_y
        proposed_x_float = proposed_x.to(torch.float32) / _FIXED_UNIT
        proposed_y_float = proposed_y.to(torch.float32) / _FIXED_UNIT
        proposed_floor, proposed_ceiling = self._player_opening_at(
            proposed_x_float,
            proposed_y_float,
        )
        blocked = playing & self._player_collides(
            proposed_x_float,
            proposed_y_float,
            proposed_floor,
            proposed_ceiling,
        )
        wall_blocked = blocked & self._points_collide(proposed_x_float, proposed_y_float)

        full_fraction, _, full_contact = self._axis_slide_contact_fixed(
            start_x,
            start_y,
            move_x,
            move_y,
        )
        collision_step = torch.div(
            full_fraction * steps,
            _FIXED_UNIT,
            rounding_mode="floor",
        ) + 1
        collision_step = torch.minimum(collision_step, steps)
        prior_x = start_x + self._trunc_divide(
            move_x * (collision_step - 1),
            steps,
        )
        prior_y = start_y + self._trunc_divide(
            move_y * (collision_step - 1),
            steps,
        )
        one_step_x = self._trunc_divide(move_x, steps)
        one_step_y = self._trunc_divide(move_y, steps)
        fraction, hit_horizontal, step_contact = self._axis_slide_contact_fixed(
            prior_x,
            prior_y,
            one_step_x,
            one_step_y,
        )
        slide = wall_blocked & full_contact & step_contact
        approach_fraction = torch.clamp_min(fraction - (_FIXED_UNIT // 32), 0)
        approach_x = one_step_x * approach_fraction >> 16
        approach_y = one_step_y * approach_fraction >> 16
        remainder = (_FIXED_UNIT - fraction).clamp(0, _FIXED_UNIT)
        slide_x = one_step_x * remainder >> 16
        slide_y = one_step_y * remainder >> 16
        slide_x = torch.where(hit_horizontal, slide_x, torch.zeros_like(slide_x))
        slide_y = torch.where(hit_horizontal, torch.zeros_like(slide_y), slide_y)
        remaining_moves = 1 + steps - collision_step
        slide_target_x = prior_x + approach_x + slide_x * remaining_moves
        slide_target_y = prior_y + approach_y + slide_y * remaining_moves
        corner_blocked = slide & self._points_collide(
            slide_target_x.to(torch.float32) / _FIXED_UNIT,
            slide_target_y.to(torch.float32) / _FIXED_UNIT,
        )
        accepted_slide = slide & ~corner_blocked
        stalled_slide = slide & corner_blocked
        position_x = torch.where(
            accepted_slide,
            slide_target_x,
            torch.where(stalled_slide, prior_x + approach_x, proposed_x),
        )
        position_y = torch.where(
            accepted_slide,
            slide_target_y,
            torch.where(stalled_slide, prior_y + approach_y, proposed_y),
        )
        clipped_x = slide_x * steps
        clipped_y = slide_y * steps
        result_move_x = torch.where(slide, clipped_x, move_x)
        result_move_y = torch.where(slide, clipped_y, move_y)
        position_x = torch.where(playing, position_x, start_x)
        position_y = torch.where(playing, position_y, start_y)
        fallback = blocked & ~slide
        result_floor = torch.where(blocked, self.player_floor_z, proposed_floor)
        result_ceiling = torch.where(blocked, self.player_ceiling_z, proposed_ceiling)
        return (
            position_x,
            position_y,
            result_move_x,
            result_move_y,
            fallback,
            result_floor,
            result_ceiling,
        )

    def _sight_opening(
        self,
        origin_x: torch.Tensor,
        origin_y: torch.Tensor,
        sight_z: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_height: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return solid blockage and the portal-clipped target Z interval."""
        direction_x = target_x - origin_x
        direction_y = target_y - origin_y
        walls = self.map.portal_walls
        start_x = walls[:, 0]
        start_y = walls[:, 1]
        segment_x = walls[:, 2] - start_x
        segment_y = walls[:, 3] - start_y
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

        wall_sectors = self.map.portal_wall_sectors
        valid_portal = torch.all(wall_sectors >= 0, dim=1)
        safe_sectors = wall_sectors.clamp_min(0)
        opening_bottom = torch.amax(
            self.map.sector_heights[safe_sectors, 0],
            dim=1,
        )
        opening_top = torch.amin(
            self.map.sector_heights[safe_sectors, 1],
            dim=1,
        )
        solid = intersects & (self.map.portal_wall_blocks_sight | ~valid_portal)
        portal = intersects & ~self.map.portal_wall_blocks_sight & valid_portal
        safe_fraction = torch.where(
            portal,
            along_ray,
            torch.ones_like(along_ray),
        )
        bottom_clip = torch.where(
            portal,
            (opening_bottom - sight_z[..., None]) / safe_fraction,
            torch.full_like(along_ray, -torch.inf),
        )
        top_clip = torch.where(
            portal,
            (opening_top - sight_z[..., None]) / safe_fraction,
            torch.full_like(along_ray, torch.inf),
        )
        bottom_slope = torch.maximum(
            target_z - sight_z,
            torch.amax(bottom_clip, dim=-1),
        )
        top_slope = torch.minimum(
            target_z + target_height - sight_z,
            torch.amin(top_clip, dim=-1),
        )
        return torch.any(solid, dim=-1), bottom_slope, top_slope

    def _sight_blocked(
        self,
        origin_x: torch.Tensor,
        origin_y: torch.Tensor,
        sight_z: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_height: torch.Tensor,
    ) -> torch.Tensor:
        """Reproduce Doom sight-cone clipping across simple sector portals."""
        solid, bottom_slope, top_slope = self._sight_opening(
            origin_x,
            origin_y,
            sight_z,
            target_x,
            target_y,
            target_z,
            target_height,
        )
        return solid | (top_slope <= bottom_slope)

    def _player_ray_actor_distance(
        self,
        ray_angle: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_radius: torch.Tensor,
    ) -> torch.Tensor:
        """Return Doom-compatible XY trace intercepts for player rays."""
        cosine, sine = self._fine_direction(ray_angle)
        cosine = cosine[..., None]
        sine = sine[..., None]
        radius = target_radius[:, None, :]
        target_x = target_x[:, None, :]
        target_y = target_y[:, None, :]

        # PT_COMPATIBLE intersects one actor-box diagonal. Doom selects its
        # slope from the trace signs, rather than tracing a circle or near box
        # edge. This notably puts a horizontal actor intercept at its center.
        same_sign = (cosine >= 0) == (sine >= 0)
        diagonal_x = target_x - radius
        diagonal_y = target_y + torch.where(same_sign, radius, -radius)
        diagonal_dx = radius * 2.0
        diagonal_dy = torch.where(same_sign, -radius * 2.0, radius * 2.0)
        offset_x = diagonal_x - self.x[:, None, None]
        offset_y = diagonal_y - self.y[:, None, None]
        denominator = cosine * diagonal_dy - sine * diagonal_dx
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        along_ray = (offset_x * diagonal_dy - offset_y * diagonal_dx) / safe
        along_diagonal = (offset_x * sine - offset_y * cosine) / safe
        intersects = (
            (denominator.abs() >= 1e-6)
            & (along_ray >= 0)
            & (along_diagonal >= 0)
            & (along_diagonal <= 1)
        )
        return torch.where(
            intersects,
            along_ray,
            torch.full_like(along_ray, torch.inf),
        )

    def _player_ray_wall_distance(self, ray_angle: torch.Tensor) -> torch.Tensor:
        """Return horizontal distances to every linedef crossed by each ray."""
        cosine, sine = self._fine_direction(ray_angle)
        cosine = cosine[..., None]
        sine = sine[..., None]
        walls = self.map.portal_walls
        start_x = walls[:, 0]
        start_y = walls[:, 1]
        segment_x = walls[:, 2] - start_x
        segment_y = walls[:, 3] - start_y
        offset_x = start_x - self.x[:, None, None]
        offset_y = start_y - self.y[:, None, None]
        denominator = cosine * segment_y - sine * segment_x
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        along_ray = (offset_x * segment_y - offset_y * segment_x) / safe
        along_wall = (offset_x * sine - offset_y * cosine) / safe
        intersects = (
            (denominator.abs() >= 1e-6) & (along_ray > 1e-4) & (along_wall >= 0) & (along_wall <= 1)
        )
        return torch.where(
            intersects,
            along_ray,
            torch.full_like(along_ray, torch.inf),
        )

    def _rocket_splash_blocked(
        self,
        origin_x: torch.Tensor,
        origin_y: torch.Tensor,
        origin_z: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_height: torch.Tensor,
    ) -> torch.Tensor:
        """Apply P_CheckSight to every in-range rocket/target pair."""
        grid_x = torch.floor(
            (origin_x - self._rocket_wall_grid_minimum_x) / _ROCKET_WALL_GRID_CELL
        ).to(torch.int64)
        grid_y = torch.floor(
            (origin_y - self._rocket_wall_grid_minimum_y) / _ROCKET_WALL_GRID_CELL
        ).to(torch.int64)
        grid_x.clamp_(0, self._rocket_wall_grid_width - 1)
        grid_y.clamp_(0, self._rocket_wall_grid_height - 1)
        grid_index = grid_y * self._rocket_wall_grid_width + grid_x
        wall_indices = self._rocket_wall_indices[grid_index]
        wall_valid = self._rocket_wall_valid[grid_index]
        walls = self.map.portal_walls[wall_indices]

        # P_RadiusAttack asks whether the damaged actor can see the bomb spot.
        # The trace therefore starts at three quarters of the actor height and
        # clips a cone against the rocket's eight-unit actor box.
        direction_x = origin_x[:, :, None] - target_x[:, None, :]
        direction_y = origin_y[:, :, None] - target_y[:, None, :]
        start_x = walls[..., 0]
        start_y = walls[..., 1]
        segment_x = walls[..., 2] - start_x
        segment_y = walls[..., 3] - start_y
        offset_x = start_x[..., None, :] - target_x[:, None, :, None]
        offset_y = start_y[..., None, :] - target_y[:, None, :, None]
        denominator = (
            direction_x[..., None] * segment_y[..., None, :]
            - direction_y[..., None] * segment_x[..., None, :]
        )
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        along_ray = (
            offset_x * segment_y[..., None, :]
            - offset_y * segment_x[..., None, :]
        ) / safe
        along_wall = (
            offset_x * direction_y[..., None]
            - offset_y * direction_x[..., None]
        ) / safe
        intersects = (
            wall_valid[..., None, :]
            & (denominator.abs() >= 1e-6)
            & (along_ray > 1e-4)
            & (along_ray < 1 - 1e-4)
            & (along_wall >= 0)
            & (along_wall <= 1)
        )

        wall_sectors = self.map.portal_wall_sectors[wall_indices]
        valid_portal = torch.all(wall_sectors >= 0, dim=-1)
        safe_sectors = wall_sectors.clamp_min(0)
        opening_bottom = torch.amax(
            self.map.sector_heights[safe_sectors, 0],
            dim=-1,
        )
        opening_top = torch.amin(
            self.map.sector_heights[safe_sectors, 1],
            dim=-1,
        )
        blocks_sight = self.map.portal_wall_blocks_sight[wall_indices]
        solid = intersects & (
            blocks_sight[..., None, :] | ~valid_portal[..., None, :]
        )
        portal = (
            intersects
            & ~blocks_sight[..., None, :]
            & valid_portal[..., None, :]
        )
        safe_fraction = torch.where(
            portal,
            along_ray,
            torch.ones_like(along_ray),
        )
        sight_z = target_z + target_height * 0.75
        bottom_clip = torch.where(
            portal,
            (
                opening_bottom[..., None, :]
                - sight_z[:, None, :, None]
            )
            / safe_fraction,
            torch.full_like(along_ray, -torch.inf),
        )
        top_clip = torch.where(
            portal,
            (
                opening_top[..., None, :]
                - sight_z[:, None, :, None]
            )
            / safe_fraction,
            torch.full_like(along_ray, torch.inf),
        )
        bottom_slope = torch.maximum(
            origin_z[:, :, None] - sight_z[:, None, :],
            torch.amax(bottom_clip, dim=-1),
        )
        top_slope = torch.minimum(
            origin_z[:, :, None] + 8.0 - sight_z[:, None, :],
            torch.amin(top_clip, dim=-1),
        )
        return torch.any(solid, dim=-1) | (top_slope <= bottom_slope)

    def _spawn_enemy_type(self, enemy_type: int, requested: torch.Tensor) -> None:
        free = (
            ~self.enemy_alive
            & (self.enemy_death_tics <= 0)
            & (self.drop_type < 0)
        )
        has_free_slot = torch.any(free, dim=1)
        slot = torch.argmax(free.to(torch.int32), dim=1)
        spawn_mask = requested & has_free_slot
        x, y, angle, has_position = self._random_spawn_positions(
            spawn_mask,
            avoid_player=True,
            actor_radius=self._enemy_radius[enemy_type],
            actor_z=0.0,
            actor_height=self._enemy_height[enemy_type],
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
        old_just_attacked = self.enemy_just_attacked[row, slot]
        old_just_hit = self.enemy_just_hit[row, slot]
        old_reaction_time = self.enemy_reaction_time[row, slot]
        old_target_slot = self.enemy_target_slot[row, slot]
        old_target_threshold = self.enemy_target_threshold[row, slot]
        old_heard_player = self.enemy_heard_player[row, slot]
        old_move_direction = self.enemy_move_direction[row, slot]
        old_move_count = self.enemy_move_count[row, slot]
        old_move_cooldown = self.enemy_move_cooldown[row, slot]
        old_animation_tics = self.enemy_animation_tics[row, slot]
        # ACS Spawn accepts its angle as an 8-bit turn. Runtime object traces
        # therefore expose exact 360/256-degree increments.
        spawn_angle = torch.floor(angle * (256.0 / (2.0 * math.pi))) * (
            2.0 * math.pi / 256.0
        )
        self.enemy_x[row, slot] = torch.where(spawn, x, old_x)
        self.enemy_y[row, slot] = torch.where(spawn, y, old_y)
        self._enemy_x_fixed[row, slot] = torch.where(
            spawn,
            torch.round(x * _FIXED_UNIT).to(torch.int64),
            self._enemy_x_fixed[row, slot],
        )
        self._enemy_y_fixed[row, slot] = torch.where(
            spawn,
            torch.round(y * _FIXED_UNIT).to(torch.int64),
            self._enemy_y_fixed[row, slot],
        )
        spawn_sector = self._sector_at(x, y)
        spawn_floor = self.map.sector_heights[spawn_sector, 0]
        spawn_ceiling = self.map.sector_heights[spawn_sector, 1]
        # deathmatch.acs passes absolute z=0 to Spawn. StaticSpawn retains the
        # center subsector's floor until the actor successfully moves in XY.
        spawn_z = torch.zeros_like(spawn_floor)
        self.enemy_z[row, slot] = torch.where(spawn, spawn_z, old_z)
        self._enemy_z_fixed[row, slot] = torch.where(
            spawn,
            torch.round(spawn_z * _FIXED_UNIT).to(torch.int64),
            self._enemy_z_fixed[row, slot],
        )
        self._enemy_floor_z_fixed[row, slot] = torch.where(
            spawn,
            torch.round(spawn_floor * _FIXED_UNIT).to(torch.int64),
            self._enemy_floor_z_fixed[row, slot],
        )
        self._enemy_ceiling_z_fixed[row, slot] = torch.where(
            spawn,
            torch.round(spawn_ceiling * _FIXED_UNIT).to(torch.int64),
            self._enemy_ceiling_z_fixed[row, slot],
        )
        self._enemy_opening_initialized[row, slot] |= spawn
        self.enemy_angle[row, slot] = torch.where(spawn, spawn_angle, old_angle)
        self.enemy_type[row, slot] = torch.where(
            spawn, torch.full_like(old_type, enemy_type), old_type
        )
        self.enemy_health[row, slot] = torch.where(
            spawn, self._enemy_base_health[enemy_type], old_health
        )
        self.enemy_cooldown[row, slot] = torch.where(
            spawn, torch.zeros_like(old_cooldown), old_cooldown
        )
        self.enemy_attack_phase[row, slot] = torch.where(
            spawn, torch.zeros_like(old_attack_phase), old_attack_phase
        )
        self.enemy_just_attacked[row, slot] = torch.where(
            spawn, torch.zeros_like(old_just_attacked), old_just_attacked
        )
        self.enemy_just_hit[row, slot] = torch.where(
            spawn, torch.zeros_like(old_just_hit), old_just_hit
        )
        self.enemy_reaction_time[row, slot] = torch.where(
            spawn,
            torch.full_like(old_reaction_time, _ENEMY_SPAWN_REACTION_TIME),
            old_reaction_time,
        )
        self.enemy_target_slot[row, slot] = torch.where(
            spawn,
            torch.full_like(old_target_slot, -2),
            old_target_slot,
        )
        self.enemy_target_threshold[row, slot] = torch.where(
            spawn,
            torch.zeros_like(old_target_threshold),
            old_target_threshold,
        )
        self.enemy_heard_player[row, slot] = torch.where(
            spawn,
            torch.zeros_like(old_heard_player),
            old_heard_player,
        )
        self.enemy_move_direction[row, slot] = torch.where(
            spawn,
            torch.zeros_like(old_move_direction),
            old_move_direction,
        )
        self.enemy_move_count[row, slot] = torch.where(
            spawn,
            torch.zeros_like(old_move_count),
            old_move_count,
        )
        self.enemy_move_cooldown[row, slot] = torch.where(
            spawn,
            self._enemy_look_interval[enemy_type] - 2,
            old_move_cooldown,
        )
        self.enemy_animation_tics[row, slot] = torch.where(
            spawn,
            torch.ones_like(old_animation_tics),
            old_animation_tics,
        )
        self._enemy_momentum_x_fixed[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self._enemy_momentum_x_fixed[row, slot]),
            self._enemy_momentum_x_fixed[row, slot],
        )
        self._enemy_momentum_y_fixed[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self._enemy_momentum_y_fixed[row, slot]),
            self._enemy_momentum_y_fixed[row, slot],
        )
        self._enemy_velocity_z_fixed[row, slot] = torch.where(
            spawn,
            torch.where(
                spawn_z > spawn_floor,
                torch.full_like(self._enemy_velocity_z_fixed[row, slot], -_FIXED_UNIT),
                torch.zeros_like(self._enemy_velocity_z_fixed[row, slot]),
            ),
            self._enemy_velocity_z_fixed[row, slot],
        )
        self.enemy_death_type[row, slot] = torch.where(
            spawn,
            torch.full_like(self.enemy_death_type[row, slot], -1),
            self.enemy_death_type[row, slot],
        )
        self.enemy_death_extreme[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self.enemy_death_extreme[row, slot]),
            self.enemy_death_extreme[row, slot],
        )
        self.enemy_death_tics[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self.enemy_death_tics[row, slot]),
            self.enemy_death_tics[row, slot],
        )
        self.enemy_death_elapsed[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self.enemy_death_elapsed[row, slot]),
            self.enemy_death_elapsed[row, slot],
        )
        self.drop_spawned[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self.drop_spawned[row, slot]),
            self.drop_spawned[row, slot],
        )
        self._drop_velocity_x_fixed[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self._drop_velocity_x_fixed[row, slot]),
            self._drop_velocity_x_fixed[row, slot],
        )
        self._drop_velocity_y_fixed[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self._drop_velocity_y_fixed[row, slot]),
            self._drop_velocity_y_fixed[row, slot],
        )
        self._drop_velocity_z_fixed[row, slot] = torch.where(
            spawn,
            torch.zeros_like(self._drop_velocity_z_fixed[row, slot]),
            self._drop_velocity_z_fixed[row, slot],
        )
        self.enemy_alive[row, slot] |= spawn

        # ACS Thing_Spawn creates an independent, non-solid TeleportFog at the
        # successful monster position. The new thinker has already consumed
        # one tic when the resulting ViZDoom observation becomes visible.
        fog_free = self.teleport_fog_tics <= 0
        fog_slot = torch.argmax(fog_free.to(torch.int32), dim=1)
        fog_spawn = spawn & torch.any(fog_free, dim=1)
        old_fog_x = self.teleport_fog_x[row, fog_slot]
        old_fog_y = self.teleport_fog_y[row, fog_slot]
        old_fog_z = self.teleport_fog_z[row, fog_slot]
        old_fog_tics = self.teleport_fog_tics[row, fog_slot]
        self.teleport_fog_x[row, fog_slot] = torch.where(fog_spawn, x, old_fog_x)
        self.teleport_fog_y[row, fog_slot] = torch.where(fog_spawn, y, old_fog_y)
        self.teleport_fog_z[row, fog_slot] = torch.where(fog_spawn, spawn_z, old_fog_z)
        self.teleport_fog_tics[row, fog_slot] = torch.where(
            fog_spawn,
            torch.full_like(old_fog_tics, _TELEPORT_FOG_INITIAL_TICS),
            old_fog_tics,
        )

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

    def _add_player_thrust_fixed(
        self,
        thrust_x_fixed: torch.Tensor,
        thrust_y_fixed: torch.Tensor,
    ) -> None:
        # Tests and advanced callers can alter the public tensors directly.
        # Retain invisible low bits whenever the visible mirrors still match.
        visible_momentum_x = self._momentum_x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_momentum_y = self._momentum_y_fixed.to(torch.float32) / _FIXED_UNIT
        self._momentum_x_fixed.copy_(
            torch.where(
                self.momentum_x != visible_momentum_x,
                torch.round(self.momentum_x * _FIXED_UNIT).to(torch.int64),
                self._momentum_x_fixed,
            )
        )
        self._momentum_y_fixed.copy_(
            torch.where(
                self.momentum_y != visible_momentum_y,
                torch.round(self.momentum_y * _FIXED_UNIT).to(torch.int64),
                self._momentum_y_fixed,
            )
        )
        self._momentum_x_fixed.add_(thrust_x_fixed)
        self._momentum_y_fixed.add_(thrust_y_fixed)
        self.momentum_x.copy_(self._momentum_x_fixed.to(torch.float32) / _FIXED_UNIT)
        self.momentum_y.copy_(self._momentum_y_fixed.to(torch.float32) / _FIXED_UNIT)

    def _player_damage_thrust_components(
        self,
        incoming: torch.Tensor,
        attacker_x: torch.Tensor,
        attacker_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        player_shape = (self.num_envs,) + (1,) * (incoming.ndim - 1)
        player_x = self.x.reshape(player_shape)
        player_y = self.y.reshape(player_shape)
        fine_angle = self._doom_fine_angle(
            torch.round((player_x - attacker_x) * _FIXED_UNIT).to(torch.int64),
            torch.round((player_y - attacker_y) * _FIXED_UNIT).to(torch.int64),
        )
        sine_fixed = self._fine_sine_fixed[fine_angle]
        cosine_fixed = self._fine_sine_fixed[
            (fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ]
        thrust_fixed = (
            torch.floor(incoming).to(torch.int64)
            * _PLAYER_DAMAGE_THRUST_PER_POINT_FIXED
        ).clamp(0, _PLAYER_MAX_DAMAGE_THRUST_FIXED)
        return (
            thrust_fixed * cosine_fixed >> 16,
            thrust_fixed * sine_fixed >> 16,
        )

    def _apply_player_damage(
        self,
        incoming: torch.Tensor,
        attacker_x: torch.Tensor | None = None,
        attacker_y: torch.Tensor | None = None,
        *,
        thrust_x_fixed: torch.Tensor | None = None,
        thrust_y_fixed: torch.Tensor | None = None,
        armor_absorb_request: torch.Tensor | None = None,
    ) -> None:
        incoming = torch.floor(incoming)
        if attacker_x is not None and attacker_y is not None:
            # P_DamageMobj applies thrust before armor absorption. DoomPlayer's
            # mass and Doom's default monster kickback are both 100, reducing
            # the reference formula to one eighth of a map unit per damage
            # point, capped at 32 units/tic.
            attacker_bearing = torch.atan2(attacker_y - self.y, attacker_x - self.x)
            if thrust_x_fixed is None or thrust_y_fixed is None:
                thrust_x_fixed, thrust_y_fixed = self._player_damage_thrust_components(
                    incoming,
                    attacker_x,
                    attacker_y,
                )
            self._add_player_thrust_fixed(
                thrust_x_fixed,
                thrust_y_fixed,
            )

        absorbed = (
            torch.floor(incoming * self.armor_save_fraction)
            if armor_absorb_request is None
            else armor_absorb_request
        )
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
        self.damage_count.add_(actual.to(torch.int32)).clamp_max_(100)
        damaged = actual > 0
        if attacker_x is None or attacker_y is None:
            direction = torch.ones_like(self.mugshot_pain_direction)
        else:
            relative = self._wrap_angle(
                attacker_bearing - self.angle
            )
            direction = torch.where(
                relative > math.pi / 4,
                torch.full_like(self.mugshot_pain_direction, 2),
                torch.where(
                    relative < -math.pi / 4,
                    torch.zeros_like(self.mugshot_pain_direction),
                    torch.ones_like(self.mugshot_pain_direction),
                ),
            )
        self.mugshot_pain_direction.copy_(
            torch.where(damaged, direction, self.mugshot_pain_direction)
        )
        self.mugshot_ouch |= damaged & (actual > 20)
        self.mugshot_pain_tics.copy_(
            torch.where(
                damaged,
                torch.full_like(self.mugshot_pain_tics, _MUGSHOT_STATE_TICS),
                self.mugshot_pain_tics,
            )
        )

    def _move_player(self, buttons: torch.Tensor) -> None:
        # Tests and advanced callers may directly set the public state tensors.
        # Resynchronize only lanes whose visible value no longer represents the
        # retained fixed-point value, preserving otherwise invisible low bits.
        visible_x = self._x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_y = self._y_fixed.to(torch.float32) / _FIXED_UNIT
        visible_momentum_x = self._momentum_x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_momentum_y = self._momentum_y_fixed.to(torch.float32) / _FIXED_UNIT
        visible_angle = self._angle_bam.to(torch.float32) * _BAM_TO_RADIANS
        visible_pitch = self._pitch_bam.to(torch.float32) * _BAM_TO_RADIANS
        position_resynchronized = (self.x != visible_x) | (self.y != visible_y)
        self._x_fixed.copy_(
            torch.where(
                self.x != visible_x,
                torch.round(self.x * _FIXED_UNIT).to(torch.int64),
                self._x_fixed,
            )
        )
        self._y_fixed.copy_(
            torch.where(
                self.y != visible_y,
                torch.round(self.y * _FIXED_UNIT).to(torch.int64),
                self._y_fixed,
            )
        )
        self._momentum_x_fixed.copy_(
            torch.where(
                self.momentum_x != visible_momentum_x,
                torch.round(self.momentum_x * _FIXED_UNIT).to(torch.int64),
                self._momentum_x_fixed,
            )
        )
        self._momentum_y_fixed.copy_(
            torch.where(
                self.momentum_y != visible_momentum_y,
                torch.round(self.momentum_y * _FIXED_UNIT).to(torch.int64),
                self._momentum_y_fixed,
            )
        )
        public_angle_bam = torch.bitwise_and(
            torch.round(torch.remainder(self.angle, 2.0 * math.pi) / _BAM_TO_RADIANS).to(
                torch.int64
            ),
            _UINT32_MASK,
        )
        self._angle_bam.copy_(
            torch.where(self.angle != visible_angle, public_angle_bam, self._angle_bam)
        )
        public_pitch_bam = torch.round(self.pitch / _BAM_TO_RADIANS).to(torch.int64)
        self._pitch_bam.copy_(
            torch.where(self.pitch != visible_pitch, public_pitch_bam, self._pitch_bam)
        )
        if self.debug_checks and torch.any(position_resynchronized):
            current_floor, current_ceiling = self._player_opening_at(self.x, self.y)
            self.player_floor_z.copy_(current_floor)
            self.player_ceiling_z.copy_(current_ceiling)

        playing = ~self.player_dead & (self.episode_time < self.episode_timeout)
        # P_PlayerThink applies vertical look before checking reactiontime, so
        # the delta axis remains live during the seven-tic teleport freeze.
        pitch_delta = (
            buttons[:, 17].to(torch.int64)
            * _PLAYER_BINARY_DELTA_PITCH
            * playing.to(torch.int64)
        )
        self._pitch_bam.copy_(
            torch.clamp(
                self._pitch_bam - pitch_delta,
                _PLAYER_MIN_PITCH_BAM,
                _PLAYER_MAX_PITCH_BAM,
            )
        )
        self.pitch.copy_(self._pitch_bam.to(torch.float32) * _BAM_TO_RADIANS)
        active = (self.reaction_time <= 0) & playing
        pull_requested = self.chainsaw_pull & playing
        pull_active = pull_requested & active
        self.chainsaw_pull &= ~pull_requested
        self.reaction_time.sub_(1).clamp_min_(0)
        current_floor = self.player_floor_z
        self.previous_player_floor_z.copy_(current_floor)
        on_ground = self.z <= current_floor
        turning = buttons[:, 7] | buttons[:, 8]
        self.turn_held_tics.copy_(
            torch.where(
                turning,
                self.turn_held_tics + 1,
                torch.zeros_like(self.turn_held_tics),
            )
        )
        turn_yaw = torch.where(
            self.turn_held_tics < _PLAYER_SLOW_TURN_TICS,
            torch.full_like(self._angle_bam, _PLAYER_SLOW_TURN_YAW),
            torch.where(
                buttons[:, 1],
                torch.full_like(self._angle_bam, _PLAYER_RUN_TURN_YAW),
                torch.full_like(self._angle_bam, _PLAYER_WALK_TURN_YAW),
            ),
        )
        turn_direction = buttons[:, 8].to(torch.int64) - buttons[:, 7].to(torch.int64)
        keyboard_turn_active = active & ~pull_requested & ~buttons[:, 2]
        delta_turn_active = active & ~pull_requested
        turn_delta_bam = (turn_direction * turn_yaw << 16) * keyboard_turn_active.to(
            torch.int64
        )
        turn_delta_bam -= (
            buttons[:, 18].to(torch.int64) * _PLAYER_BINARY_DELTA_TURN_YAW << 16
        ) * delta_turn_active.to(torch.int64)
        self._angle_bam.copy_(
            torch.bitwise_and(self._angle_bam + turn_delta_bam, _UINT32_MASK)
        )
        self.angle.copy_(self._angle_bam.to(torch.float32) * _BAM_TO_RADIANS)
        forward = (buttons[:, 6].to(torch.float32) - buttons[:, 5].to(torch.float32)) * active.to(
            torch.float32
        )
        side_direction = buttons[:, 3].to(torch.int64) - buttons[:, 4].to(torch.int64)
        side_direction += buttons[:, 2].to(torch.int64) * (
            buttons[:, 7].to(torch.int64) - buttons[:, 8].to(torch.int64)
        )
        side_direction *= active.to(torch.int64)
        forward = torch.where(pull_active, torch.ones_like(forward), forward)
        side_direction = torch.where(
            pull_requested,
            torch.zeros_like(side_direction),
            side_direction,
        )
        forward_acceleration_fixed = torch.where(
            buttons[:, 1],
            torch.full_like(self._momentum_x_fixed, _PLAYER_RUN_FORWARD_ACCELERATION_FIXED),
            torch.full_like(self._momentum_x_fixed, _PLAYER_FORWARD_ACCELERATION_FIXED),
        )
        forward_acceleration_fixed = torch.where(
            pull_active,
            torch.full_like(
                forward_acceleration_fixed,
                _CHAINSAW_PULL_ACCELERATION_FIXED,
            ),
            forward_acceleration_fixed,
        )
        side_input_acceleration_fixed = torch.where(
            buttons[:, 1],
            torch.full_like(self._momentum_x_fixed, _PLAYER_RUN_SIDE_ACCELERATION_FIXED),
            torch.full_like(self._momentum_x_fixed, _PLAYER_SIDE_ACCELERATION_FIXED),
        )
        forward_acceleration_fixed = torch.where(
            on_ground,
            forward_acceleration_fixed,
            forward_acceleration_fixed * _PLAYER_AIR_CONTROL_FIXED >> 16,
        )
        side_move_fixed = side_direction * side_input_acceleration_fixed
        side_move_fixed += (
            buttons[:, 19].to(torch.int64)
            * _PLAYER_BINARY_DELTA_SIDE_ACCELERATION_FIXED
            * active.to(torch.int64)
        )
        side_move_fixed = torch.where(
            pull_requested,
            torch.zeros_like(side_move_fixed),
            side_move_fixed,
        )
        side_move_fixed = torch.clamp(
            side_move_fixed,
            -_PLAYER_MAX_INPUT_ACCELERATION_FIXED,
            _PLAYER_MAX_INPUT_ACCELERATION_FIXED,
        )
        side_move_fixed = torch.where(
            on_ground,
            side_move_fixed,
            side_move_fixed * _PLAYER_AIR_CONTROL_FIXED >> 16,
        )
        fine_angle = self._angle_bam >> _ANGLE_TO_FINE_SHIFT
        sine_fixed = self._fine_sine_fixed[fine_angle]
        cosine_fixed = self._fine_sine_fixed[(fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)]
        forward_move_fixed = forward.to(torch.int64) * forward_acceleration_fixed
        self._momentum_x_fixed.add_(
            (forward_move_fixed * cosine_fixed >> 16)
            + (side_move_fixed * sine_fixed >> 16)
        )
        self._momentum_y_fixed.add_(
            (forward_move_fixed * sine_fixed >> 16)
            + (side_move_fixed * -cosine_fixed >> 16)
        )
        # P_CalcHeight observes the thrust-adjusted actor velocity before the
        # actor thinker moves and applies friction.  Preserve that fixed-point
        # magnitude for both camera and psprite bobbing.
        motion_squared_fixed = (
            self._momentum_x_fixed * self._momentum_x_fixed
            + self._momentum_y_fixed * self._momentum_y_fixed
        ) >> 16
        self._player_bob_fixed.copy_(
            ((motion_squared_fixed * _PLAYER_MOVE_BOB_FIXED) >> 16).clamp(
                0,
                _PLAYER_MAX_BOB_FIXED,
            )
        )
        (
            doom_position_x_fixed,
            doom_position_y_fixed,
            doom_momentum_x_fixed,
            doom_momentum_y_fixed,
            doom_slide_fallback,
            doom_floor,
            doom_ceiling,
        ) = self._doom_axis_slide_move(playing)
        self._x_fixed.copy_(
            torch.where(
                doom_slide_fallback,
                self._x_fixed,
                doom_position_x_fixed,
            )
        )
        self._y_fixed.copy_(
            torch.where(
                doom_slide_fallback,
                self._y_fixed,
                doom_position_y_fixed,
            )
        )
        next_momentum_x_fixed = torch.where(
            doom_slide_fallback,
            torch.zeros_like(doom_momentum_x_fixed),
            doom_momentum_x_fixed,
        )
        next_momentum_y_fixed = torch.where(
            doom_slide_fallback,
            torch.zeros_like(doom_momentum_y_fixed),
            doom_momentum_y_fixed,
        )
        self.player_floor_z.copy_(doom_floor)
        self.player_ceiling_z.copy_(doom_ceiling)
        friction_fixed = torch.where(
            self.z <= doom_floor,
            torch.full_like(next_momentum_x_fixed, _PLAYER_FRICTION_FIXED),
            torch.full_like(next_momentum_x_fixed, _PLAYER_AIR_FRICTION_FIXED),
        )
        self._momentum_x_fixed.copy_(next_momentum_x_fixed * friction_fixed >> 16)
        self._momentum_y_fixed.copy_(next_momentum_y_fixed * friction_fixed >> 16)
        self.x.copy_(self._x_fixed.to(torch.float32) / _FIXED_UNIT)
        self.y.copy_(self._y_fixed.to(torch.float32) / _FIXED_UNIT)
        self.momentum_x.copy_(self._momentum_x_fixed.to(torch.float32) / _FIXED_UNIT)
        self.momentum_y.copy_(self._momentum_y_fixed.to(torch.float32) / _FIXED_UNIT)

    def _vertical_player_tick(self, active: torch.Tensor) -> None:
        next_view_height = self.view_height + self.delta_view_height
        above_default = next_view_height > _VIEW_HEIGHT
        below_half = next_view_height < _VIEW_HEIGHT / 2.0
        next_view_height = torch.where(
            above_default,
            torch.full_like(next_view_height, _VIEW_HEIGHT),
            torch.where(
                below_half,
                torch.full_like(next_view_height, _VIEW_HEIGHT / 2.0),
                next_view_height,
            ),
        )
        next_delta_view_height = torch.where(
            above_default,
            torch.zeros_like(self.delta_view_height),
            self.delta_view_height,
        )
        next_delta_view_height = torch.where(
            below_half & (next_delta_view_height <= 0),
            torch.full_like(next_delta_view_height, 1.0 / 65536.0),
            next_delta_view_height,
        )
        moving_view = next_delta_view_height != 0
        next_delta_view_height = torch.where(
            moving_view,
            next_delta_view_height + 0.25,
            next_delta_view_height,
        )
        bob_angle = torch.div(
            self.episode_time.to(torch.int64) * _FINE_ANGLES,
            _PLAYER_VIEW_BOB_PERIOD_TICS,
            rounding_mode="trunc",
        ) & (_FINE_ANGLES - 1)
        view_bob_fixed = (
            (self._player_bob_fixed >> 1) * self._fine_sine_fixed[bob_angle]
        ) >> 16
        next_view_z = (
            self.z
            + next_view_height
            + view_bob_fixed.to(torch.float32) / _FIXED_UNIT
        )
        next_view_z = torch.minimum(next_view_z, self.player_ceiling_z - 4.0)
        next_view_z = torch.maximum(next_view_z, self.player_floor_z + 4.0)
        self.view_height.copy_(torch.where(active, next_view_height, self.view_height))
        self.view_z.copy_(torch.where(active, next_view_z, self.view_z))

        floor = self.player_floor_z
        proposed_z = self.z + self.velocity_z
        airborne = (self.z > floor) | (self.velocity_z < 0)
        walked_off_ledge = (
            (self.velocity_z == 0)
            & (self.previous_player_floor_z > floor)
            & (proposed_z == self.previous_player_floor_z)
        )
        gravity_step = torch.where(
            walked_off_ledge,
            torch.full_like(self.velocity_z, 2.0),
            torch.ones_like(self.velocity_z),
        )
        next_velocity = torch.where(
            airborne,
            self.velocity_z - gravity_step,
            torch.zeros_like(self.velocity_z),
        )
        landed = proposed_z <= floor
        next_z = torch.where(landed, floor, proposed_z)
        landed_from_air = landed & airborne
        next_delta_view_height = torch.where(
            landed_from_air,
            self.velocity_z / 8.0,
            next_delta_view_height,
        )
        next_velocity = torch.where(landed, torch.zeros_like(next_velocity), next_velocity)
        self.z.copy_(torch.where(active, next_z, self.z))
        self.velocity_z.copy_(torch.where(active, next_velocity, self.velocity_z))
        self.delta_view_height.copy_(
            torch.where(active, next_delta_view_height, self.delta_view_height)
        )

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
        initial_pistol_raise = (
            (self.episode_time <= _WEAPON_SPAWN_RAISE_TICS + 1)
            & (current == 2)
            & (self.weapon_fire_count == 0)
            & (
                (self.weapon_raise_cooldown > 0)
                | (self.episode_time == _WEAPON_SPAWN_RAISE_TICS + 1)
            )
        )
        lower_tics = torch.where(
            initial_pistol_raise,
            self.episode_time.clamp(0, _WEAPON_LOWER_TICS),
            lower_tics,
        )
        lower_tics = lower_tics + self.weapon_state_cooldown
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

    def _melee_attack_rolls(
        self,
        weapon: torch.Tensor,
        fires: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll Doom's fist/chainsaw damage and triangular XY spread."""
        melee = fires & (weapon <= 1)
        chainsaw = melee & (weapon == 1)
        damage_roll = torch.remainder(self._random_u32(melee), 10).to(torch.float32) + 1.0
        damage = torch.where(melee, damage_roll * 2.0, 0.0)
        first_horizontal = torch.bitwise_and(self._random_u32(melee), 255).to(torch.float32)
        second_horizontal = torch.bitwise_and(self._random_u32(melee), 255).to(torch.float32)
        random2 = first_horizontal - second_horizontal
        fist_spread = random2 * float(1 << 18) * _BAM_TO_RADIANS
        chainsaw_spread = random2 * (_CHAINSAW_SPREAD_RADIANS / 255.0)

        # A_Saw evaluates Random2 for its zero default vertical spread too.
        # Retain the reference stream consumption even though the result is 0.
        self._random_u32(chainsaw)
        self._random_u32(chainsaw)
        return damage, torch.where(chainsaw, chainsaw_spread, fist_spread)

    def _hitscan_pellet_rolls(
        self,
        weapon: torch.Tensor,
        fires: torch.Tensor,
        accurate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll Doom's per-pellet damage and horizontal/vertical spread."""
        pellet_count = self._hitscan_pellet_counts[weapon]
        pistol_or_chaingun = (weapon == 2) | (weapon == 5)
        shotgun = weapon == 3
        super_shotgun = weapon == 4
        damage_pellets: list[torch.Tensor] = []
        horizontal_offsets: list[torch.Tensor] = []
        vertical_offsets: list[torch.Tensor] = []

        for pellet_index in range(_HITSCAN_MAX_PELLETS):
            active = fires & (pellet_count > pellet_index)
            damage_roll = torch.remainder(self._random_u32(active), 3).to(torch.float32)
            damage_pellets.append(torch.where(active, (damage_roll + 1.0) * 5.0, 0.0))

            spread = active & (shotgun | super_shotgun | (pistol_or_chaingun & ~accurate))
            first_horizontal = torch.bitwise_and(
                self._random_u32(spread),
                255,
            ).to(torch.float32)
            second_horizontal = torch.bitwise_and(
                self._random_u32(spread),
                255,
            ).to(torch.float32)
            horizontal_random2 = first_horizontal - second_horizontal
            horizontal_bam_scale = torch.where(
                super_shotgun,
                torch.full_like(horizontal_random2, float(1 << 19)),
                torch.full_like(horizontal_random2, float(1 << 18)),
            )
            horizontal_offsets.append(
                torch.where(
                    spread,
                    horizontal_random2 * horizontal_bam_scale * _BAM_TO_RADIANS,
                    0.0,
                )
            )

            first_vertical = torch.bitwise_and(
                self._random_u32(active & super_shotgun),
                255,
            ).to(torch.float32)
            second_vertical = torch.bitwise_and(
                self._random_u32(active & super_shotgun),
                255,
            ).to(torch.float32)
            vertical_offsets.append(
                torch.where(
                    active & super_shotgun,
                    (first_vertical - second_vertical) * 332063.0 * _BAM_TO_RADIANS,
                    0.0,
                )
            )

        return (
            torch.stack(damage_pellets, dim=1),
            torch.stack(horizontal_offsets, dim=1),
            torch.stack(vertical_offsets, dim=1),
        )

    def _enemy_damage_thrust_components(
        self,
        damage: torch.Tensor,
        attacker_x: torch.Tensor,
        attacker_y: torch.Tensor,
        kickback: torch.Tensor | float = 100.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_dimensions = damage.ndim - 2
        enemy_shape = (
            self.num_envs,
            *((1,) * source_dimensions),
            self.enemy_slots,
        )
        enemy_type = self.enemy_type.clamp_min(0).reshape(enemy_shape)
        enemy_x = self.enemy_x.reshape(enemy_shape)
        enemy_y = self.enemy_y.reshape(enemy_shape)
        fine_angle = self._doom_fine_angle(
            torch.round((enemy_x - attacker_x) * _FIXED_UNIT).to(torch.int64),
            torch.round((enemy_y - attacker_y) * _FIXED_UNIT).to(torch.int64),
        )
        sine_fixed = self._fine_sine_fixed[fine_angle]
        cosine_fixed = self._fine_sine_fixed[
            (fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ]
        kickback_tensor = torch.as_tensor(
            kickback,
            device=self.device,
            dtype=damage.dtype,
        )
        thrust_fixed = torch.round(
            damage
            * (0.125 * _FIXED_UNIT)
            * kickback_tensor
            / self._enemy_mass[enemy_type]
        ).to(torch.int64)
        thrust_fixed.clamp_(0, _PLAYER_MAX_DAMAGE_THRUST_FIXED)
        return (
            thrust_fixed * cosine_fixed >> 16,
            thrust_fixed * sine_fixed >> 16,
        )

    def _retarget_enemies_from_player_damage(self, damage: torch.Tensor) -> None:
        """Apply P_DamageMobj's target/threshold update for player damage."""
        hurt = self.enemy_alive & (damage > 0)
        current_is_player = self.enemy_target_slot < 0
        switches = hurt & (current_is_player | (self.enemy_target_threshold <= 0))
        self.enemy_target_slot.masked_fill_(switches, -1)
        self.enemy_heard_player.masked_fill_(switches, False)
        self.enemy_target_threshold.copy_(
            torch.where(
                switches,
                torch.full_like(
                    self.enemy_target_threshold,
                    _ENEMY_RETALIATION_THRESHOLD,
                ),
                self.enemy_target_threshold,
            )
        )

    def _retarget_enemies_from_monster_damage(
        self,
        damage_by_source: torch.Tensor,
    ) -> None:
        """Apply Doom retaliation for [lane, source slot, target slot] damage."""
        not_self = ~torch.eye(
            self.enemy_slots,
            device=self.device,
            dtype=torch.bool,
        )[None, :, :]
        valid = (
            (damage_by_source > 0)
            & self.enemy_alive[:, :, None]
            & self.enemy_alive[:, None, :]
            & not_self
        )
        current = self.enemy_target_slot
        current_is_monster = current >= 0
        safe_current = current.clamp(0, self.enemy_slots - 1)
        hit_by_current = current_is_monster & valid.gather(
            1,
            safe_current[:, None, :],
        ).squeeze(1)
        has_attacker = torch.any(valid, dim=1)
        candidate = torch.argmax(valid.to(torch.int32), dim=1)
        switches = (
            has_attacker
            & ~hit_by_current
            & (self.enemy_target_threshold <= 0)
        )
        refreshes = hit_by_current | switches
        self.enemy_target_slot.copy_(
            torch.where(switches, candidate, self.enemy_target_slot)
        )
        self.enemy_heard_player.masked_fill_(refreshes, False)
        self.enemy_target_threshold.copy_(
            torch.where(
                refreshes,
                torch.full_like(
                    self.enemy_target_threshold,
                    _ENEMY_RETALIATION_THRESHOLD,
                ),
                self.enemy_target_threshold,
            )
        )

    def _apply_enemy_damage(
        self,
        damage: torch.Tensor,
        attacker_x: torch.Tensor | None = None,
        attacker_y: torch.Tensor | None = None,
        *,
        kickback: torch.Tensor | float = 100.0,
        thrust_x_fixed: torch.Tensor | None = None,
        thrust_y_fixed: torch.Tensor | None = None,
        pain_override: torch.Tensor | None = None,
        credit_player: bool = True,
        attacker_is_player: bool = True,
        monster_damage_by_source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        applied = torch.where(self.enemy_alive, damage, torch.zeros_like(damage))
        if monster_damage_by_source is not None:
            self._retarget_enemies_from_monster_damage(
                monster_damage_by_source,
            )
        elif attacker_is_player:
            self._retarget_enemies_from_player_damage(applied)
        if (
            (thrust_x_fixed is None or thrust_y_fixed is None)
            and attacker_x is not None
            and attacker_y is not None
        ):
            thrust_x_fixed, thrust_y_fixed = self._enemy_damage_thrust_components(
                applied,
                attacker_x,
                attacker_y,
                kickback,
            )
        if thrust_x_fixed is not None and thrust_y_fixed is not None:
            self._enemy_momentum_x_fixed.add_(
                torch.where(
                    self.enemy_alive,
                    thrust_x_fixed,
                    torch.zeros_like(thrust_x_fixed),
                )
            )
            self._enemy_momentum_y_fixed.add_(
                torch.where(
                    self.enemy_alive,
                    thrust_y_fixed,
                    torch.zeros_like(thrust_y_fixed),
                )
            )
        previous = self.enemy_health.clone()
        raw_updated = previous - applied
        updated = torch.clamp_min(raw_updated, 0)
        self.enemy_health.copy_(torch.where(self.enemy_alive, updated, previous))
        killed = self.enemy_alive & (previous > 0) & (updated <= 0)
        killed_type = self.enemy_type.clamp_min(0)
        extreme_death = (
            killed
            & self._enemy_has_xdeath[killed_type]
            & (raw_updated < -self._enemy_base_health[killed_type])
        )
        hurt = self.enemy_alive & (applied > 0) & ~killed
        # P_DamageMobj wakes the target regardless of whether the hit enters
        # its Pain state. Explicitly spawned monsters must not retain their
        # initial missile-attack delay after taking damage.
        self.enemy_reaction_time.masked_fill_(hurt, 0)
        if pain_override is None:
            random_bits = self._random_u32(torch.any(hurt, dim=1))[:, None]
            slot = torch.arange(self.enemy_slots, device=self.device, dtype=torch.int64)[None, :]
            mixed = random_bits ^ (slot * _HASH_GOLDEN_RATIO_SIGNED)
            mixed ^= mixed >> 16
            pain_roll = torch.remainder(mixed, 256)
            pain = hurt & (pain_roll < self._enemy_pain_chance[killed_type])
        else:
            pain = hurt & pain_override
        self.enemy_pain_tics.copy_(
            torch.where(pain, self._enemy_pain_duration[killed_type], self.enemy_pain_tics)
        )
        # A successful pain reaction sets MF_JUSTHIT. P_CheckMissileRange
        # consumes it to force the next eligible retaliation shot.
        self.enemy_just_hit |= pain
        self.enemy_attack_phase.masked_fill_(pain, 0)
        self.enemy_cooldown.masked_fill_(pain, 0)
        self.enemy_move_cooldown.masked_fill_(pain, 0)
        self.enemy_animation_tics.masked_fill_(pain, 0)
        death_duration = torch.where(
            extreme_death,
            self.map.enemy_xdeath_total_tics[killed_type],
            self.map.enemy_death_total_tics[killed_type],
        )
        self.enemy_death_type.copy_(torch.where(killed, killed_type, self.enemy_death_type))
        self.enemy_death_extreme.copy_(
            torch.where(killed, extreme_death, self.enemy_death_extreme)
        )
        self.enemy_death_tics.copy_(
            torch.where(killed, death_duration.to(torch.int32), self.enemy_death_tics)
        )
        self.enemy_death_elapsed.masked_fill_(killed, 0)
        if credit_player:
            reward = torch.sum(
                torch.where(
                    killed,
                    self._enemy_kill_reward[killed_type],
                    torch.zeros_like(applied),
                ),
                dim=1,
            )
        else:
            reward = torch.zeros(self.num_envs, device=self.device)
        self.enemy_alive &= ~killed
        self.enemy_pain_tics.masked_fill_(killed, 0)
        self.enemy_cooldown.masked_fill_(killed, 0)
        self.enemy_attack_phase.masked_fill_(killed, 0)
        self.enemy_just_attacked.masked_fill_(killed, False)
        self.enemy_just_hit.masked_fill_(killed, False)
        self.enemy_reaction_time.masked_fill_(killed, 0)
        self.enemy_target_slot.masked_fill_(killed, -1)
        self.enemy_target_threshold.masked_fill_(killed, 0)
        self.enemy_heard_player.masked_fill_(killed, False)
        self.enemy_move_direction.masked_fill_(killed, 0)
        self.enemy_move_count.masked_fill_(killed, 0)
        drop = self._monster_drop_type[killed_type]
        self.drop_type.copy_(torch.where(killed, drop, self.drop_type))
        self.drop_spawned.masked_fill_(killed, False)
        self._drop_velocity_x_fixed.masked_fill_(killed, 0)
        self._drop_velocity_y_fixed.masked_fill_(killed, 0)
        self._drop_velocity_z_fixed.masked_fill_(killed, 0)
        has_drop = killed & (drop >= 0)
        self.drop_delay.copy_(
            torch.where(
                has_drop,
                torch.where(
                    extreme_death,
                    self._enemy_xdeath_no_block_delay[killed_type],
                    self._enemy_no_block_delay[killed_type],
                ),
                self.drop_delay,
            )
        )
        self.enemy_type.copy_(
            torch.where(killed, torch.full_like(self.enemy_type, -1), self.enemy_type)
        )
        if credit_player:
            self.killcount.add_(torch.sum(killed.to(torch.int32), dim=1))
        return reward

    def _spawn_player_projectile(
        self,
        weapon: torch.Tensor,
        fires: torch.Tensor,
        aim_angle: torch.Tensor,
        aim_pitch: torch.Tensor,
    ) -> None:
        requested = fires & ((weapon == 6) | (weapon == 7))
        free = ~self.projectile_alive & (self.projectile_impact_tics <= 0)
        has_slot = torch.any(free, dim=1)
        slot = torch.argmax(free.to(torch.int32), dim=1)
        spawn = requested & has_slot
        row = torch.arange(self.num_envs, device=self.device)
        projectile_type = (weapon - 6).clamp(0, 1)
        speed = self._player_projectile_speed[projectile_type]
        spawn_z = self.z + 32.0

        fine_angle = self._fine_angle_index(aim_angle)
        fine_pitch = self._fine_angle_index(aim_pitch)
        sine_angle_fixed = self._fine_sine_fixed[fine_angle]
        cosine_angle_fixed = self._fine_sine_fixed[
            (fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ]
        sine_pitch_fixed = self._fine_sine_fixed[fine_pitch]
        cosine_pitch_fixed = self._fine_sine_fixed[
            (fine_pitch + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ]
        aim_x_fixed = cosine_pitch_fixed * cosine_angle_fixed >> 16
        aim_y_fixed = cosine_pitch_fixed * sine_angle_fixed >> 16
        aim_z_fixed = -sine_pitch_fixed
        aim_norm = torch.sqrt(
            aim_x_fixed.to(torch.float32) * aim_x_fixed.to(torch.float32)
            + aim_y_fixed.to(torch.float32) * aim_y_fixed.to(torch.float32)
            + aim_z_fixed.to(torch.float32) * aim_z_fixed.to(torch.float32)
        ).clamp_min_(1.0)
        velocity_x = torch.trunc(
            aim_x_fixed.to(torch.float32) / aim_norm * speed * _FIXED_UNIT
        ) / _FIXED_UNIT
        velocity_y = torch.trunc(
            aim_y_fixed.to(torch.float32) / aim_norm * speed * _FIXED_UNIT
        ) / _FIXED_UNIT
        velocity_z = torch.trunc(
            aim_z_fixed.to(torch.float32) / aim_norm * speed * _FIXED_UNIT
        ) / _FIXED_UNIT
        self.projectile_x[row, slot] = torch.where(
            spawn,
            self.x + velocity_x * 0.5,
            self.projectile_x[row, slot],
        )
        self.projectile_y[row, slot] = torch.where(
            spawn,
            self.y + velocity_y * 0.5,
            self.projectile_y[row, slot],
        )
        self.projectile_z[row, slot] = torch.where(
            spawn,
            spawn_z + velocity_z * 0.5,
            self.projectile_z[row, slot],
        )
        self.projectile_velocity_x[row, slot] = torch.where(
            spawn,
            velocity_x,
            self.projectile_velocity_x[row, slot],
        )
        self.projectile_velocity_y[row, slot] = torch.where(
            spawn,
            velocity_y,
            self.projectile_velocity_y[row, slot],
        )
        self.projectile_velocity_z[row, slot] = torch.where(
            spawn,
            velocity_z,
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

    @staticmethod
    def _rocket_radius_damage(
        bomb_x: torch.Tensor,
        bomb_y: torch.Tensor,
        bomb_z: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_radius: torch.Tensor,
        target_height: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return Doom radius damage and 16.16 pre-truncation points."""
        delta_x = (bomb_x - target_x).abs()
        delta_y = (bomb_y - target_y).abs()
        horizontal = torch.maximum(delta_x, delta_y)
        horizontal_from_box = torch.clamp_min(horizontal - target_radius, 0)
        target_top = target_z + target_height
        inside_target_height = (bomb_z >= target_z) & (bomb_z < target_top)
        vertical_distance = torch.where(
            bomb_z > target_z,
            bomb_z - target_top,
            target_z - bomb_z,
        ).clamp_min(0)
        outside_distance = torch.where(
            horizontal <= target_radius,
            vertical_distance,
            torch.sqrt(
                horizontal_from_box * horizontal_from_box
                + vertical_distance * vertical_distance
            ),
        )
        distance = torch.where(
            inside_target_height,
            horizontal_from_box,
            outside_distance,
        )
        points = torch.clamp(
            _ROCKET_SPLASH_DAMAGE - distance,
            0,
            _ROCKET_SPLASH_DAMAGE,
        )
        return (
            torch.floor(points),
            torch.round(points * _FIXED_UNIT).to(torch.int64),
        )

    def _projectile_tick(self, active: torch.Tensor) -> torch.Tensor:
        self.projectile_impact_tics.copy_(
            torch.where(
                active[:, None],
                torch.clamp_min(self.projectile_impact_tics - 1, 0),
                self.projectile_impact_tics,
            )
        )
        self.projectile_impact_type.masked_fill_(self.projectile_impact_tics <= 0, -1)
        alive = self.projectile_alive & active[:, None]
        projectile_radius = torch.where(self.projectile_type == 0, 11.0, 13.0)
        projectile_height = torch.full_like(projectile_radius, 8.0)
        dominant_speed = torch.maximum(
            self.projectile_velocity_x.abs(),
            self.projectile_velocity_y.abs(),
        )
        max_step = projectile_radius - 1.0
        movement_steps = torch.where(
            dominant_speed > max_step,
            1 + torch.floor(dominant_speed / max_step).to(torch.int32),
            torch.ones_like(self.projectile_age),
        )

        start_x = self.projectile_x.clone()
        start_y = self.projectile_y.clone()
        current_x = start_x.clone()
        current_y = start_y.clone()
        current_z = self.projectile_z.clone()
        moving = alive.clone()
        impact = torch.zeros_like(alive)
        enemy_impact = torch.zeros_like(alive)
        doll_impact = torch.zeros_like(alive)
        nearest_enemy = torch.zeros_like(self.projectile_age, dtype=torch.int64)
        enemy_type = self.enemy_type.clamp_min(0)
        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            doll_x = self.map.player_starts[:-1, 0]
            doll_y = self.map.player_starts[:-1, 1]
            doll_z = self._player_start_z[:-1]
        # Rocket and plasma definitions require at most three P_XYMovement
        # subdivisions. Keeping this bound static avoids a device sync per tic.
        for step in range(1, 4):
            enabled = moving & (movement_steps >= step)
            fraction = step / movement_steps.clamp_min(1).to(torch.float32)
            candidate_x = start_x + self.projectile_velocity_x * fraction
            candidate_y = start_y + self.projectile_velocity_y * fraction
            wall_impact = enabled & self._points_collide(
                candidate_x,
                candidate_y,
                projectile_radius,
            )
            sector = self._sector_at(candidate_x.reshape(-1), candidate_y.reshape(-1)).reshape_as(
                candidate_x
            )
            floor = self.map.sector_heights[sector, 0]
            ceiling = self.map.sector_heights[sector, 1]
            opening_impact = enabled & (
                (current_z < floor) | (current_z + projectile_height > ceiling)
            )
            dx = candidate_x[:, :, None] - self.enemy_x[:, None, :]
            dy = candidate_y[:, :, None] - self.enemy_y[:, None, :]
            enemy_distance = torch.sqrt(dx * dx + dy * dy)
            enemy_overlap = self._vertical_overlap(
                current_z[:, :, None],
                projectile_height[:, :, None],
                self.enemy_z[:, None, :],
                self._enemy_height[enemy_type][:, None, :],
            )
            candidate = (
                enabled[:, :, None]
                & self.enemy_alive[:, None, :]
                & enemy_overlap
                & (
                    dx.abs()
                    < projectile_radius[:, :, None]
                    + self._enemy_radius[enemy_type][:, None, :]
                )
                & (
                    dy.abs()
                    < projectile_radius[:, :, None]
                    + self._enemy_radius[enemy_type][:, None, :]
                )
            )
            candidate_distance = torch.where(
                candidate,
                enemy_distance,
                torch.full_like(enemy_distance, torch.inf),
            )
            nearest_distance, step_nearest_enemy = torch.min(candidate_distance, dim=2)
            step_enemy_impact = torch.isfinite(nearest_distance)
            if doll_count:
                doll_dx = candidate_x[:, :, None] - doll_x[None, None, :]
                doll_dy = candidate_y[:, :, None] - doll_y[None, None, :]
                doll_distance = torch.sqrt(doll_dx * doll_dx + doll_dy * doll_dy)
                doll_overlap = self._vertical_overlap(
                    current_z[:, :, None],
                    projectile_height[:, :, None],
                    doll_z[None, None, :],
                    _PLAYER_HEIGHT,
                )
                doll_candidate = (
                    enabled[:, :, None]
                    & ~self.player_dead[:, None, None]
                    & doll_overlap
                    & (
                        doll_dx.abs()
                        < projectile_radius[:, :, None] + _PLAYER_RADIUS
                    )
                    & (
                        doll_dy.abs()
                        < projectile_radius[:, :, None] + _PLAYER_RADIUS
                    )
                )
                nearest_doll_distance, _ = torch.min(
                    torch.where(
                        doll_candidate,
                        doll_distance,
                        torch.full_like(doll_distance, torch.inf),
                    ),
                    dim=2,
                )
                step_doll_impact = torch.isfinite(nearest_doll_distance) & (
                    nearest_doll_distance < nearest_distance
                )
                step_enemy_impact &= ~step_doll_impact
            else:
                step_doll_impact = torch.zeros_like(step_enemy_impact)
            step_actor_impact = step_enemy_impact | step_doll_impact
            step_impact = enabled & (wall_impact | opening_impact | step_actor_impact)
            successful = enabled & ~step_impact
            current_x.copy_(torch.where(successful, candidate_x, current_x))
            current_y.copy_(torch.where(successful, candidate_y, current_y))
            nearest_enemy.copy_(
                torch.where(step_impact & step_enemy_impact, step_nearest_enemy, nearest_enemy)
            )
            enemy_impact |= step_impact & step_enemy_impact
            doll_impact |= step_impact & step_doll_impact
            impact |= step_impact
            moving &= ~step_impact

        next_z = current_z + self.projectile_velocity_z
        sector = self._sector_at(current_x.reshape(-1), current_y.reshape(-1)).reshape_as(current_x)
        floor = self.map.sector_heights[sector, 0]
        ceiling = self.map.sector_heights[sector, 1]
        plane_impact = moving & (
            (next_z < floor) | (next_z + projectile_height > ceiling)
        )
        clipped_next_z = torch.where(
            next_z < floor,
            floor,
            torch.where(
                next_z + projectile_height > ceiling,
                ceiling - projectile_height,
                next_z,
            ),
        )
        current_z.copy_(torch.where(moving, clipped_next_z, current_z))
        impact |= plane_impact

        random_bits = self._random_u32(torch.any(impact, dim=1))[:, None]
        slot_bits = torch.arange(
            self.player_projectile_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, :]
        mixed = random_bits ^ (slot_bits * _HASH_GOLDEN_RATIO_SIGNED)
        mixed ^= mixed >> 16
        die = torch.remainder(mixed, 8).to(torch.float32) + 1
        rolled_direct_damage = torch.where(
            self.projectile_type == 0,
            die * 20.0,
            die * 5.0,
        )
        direct_damage = rolled_direct_damage * (impact & enemy_impact).to(torch.float32)
        direct_doll_damage = rolled_direct_damage * (impact & doll_impact).to(torch.float32)
        direct_damage_by_enemy = torch.zeros(
            (
                self.num_envs,
                self.player_projectile_slots,
                self.enemy_slots,
            ),
            device=self.device,
            dtype=direct_damage.dtype,
        )
        direct_damage_by_enemy.scatter_add_(
            2,
            nearest_enemy[:, :, None],
            direct_damage[:, :, None],
        )

        rocket_impact = impact & (self.projectile_type == 0)
        splash_damage, enemy_splash_points_fixed = self._rocket_radius_damage(
            current_x[:, :, None],
            current_y[:, :, None],
            current_z[:, :, None],
            self.enemy_x[:, None, :],
            self.enemy_y[:, None, :],
            self.enemy_z[:, None, :],
            self._enemy_radius[enemy_type][:, None, :],
            self._enemy_height[enemy_type][:, None, :],
        )
        visible_to_enemy = ~self._rocket_splash_blocked(
            current_x,
            current_y,
            current_z,
            self.enemy_x,
            self.enemy_y,
            self.enemy_z,
            self._enemy_height[enemy_type],
        )
        killed_by_direct_impact = (
            (direct_damage_by_enemy > 0)
            & (direct_damage_by_enemy >= self.enemy_health[:, None, :])
        )
        enemy_splash = (
            rocket_impact[:, :, None]
            & self.enemy_alive[:, None, :]
            & visible_to_enemy
            & ~killed_by_direct_impact
        )
        splash_damage *= enemy_splash.to(torch.float32)
        enemy_splash_points_fixed *= enemy_splash.to(torch.int64)
        damage_by_enemy = torch.sum(
            direct_damage_by_enemy + splash_damage,
            dim=1,
        )

        direct_enemy_thrust_x, direct_enemy_thrust_y = (
            self._enemy_damage_thrust_components(
                direct_damage_by_enemy,
                current_x[:, :, None],
                current_y[:, :, None],
            )
        )
        splash_enemy_thrust_x, splash_enemy_thrust_y = (
            self._enemy_damage_thrust_components(
                splash_damage,
                current_x[:, :, None],
                current_y[:, :, None],
            )
        )
        enemy_fine_angle = self._doom_fine_angle(
            torch.round(
                (self.enemy_x[:, None, :] - current_x[:, :, None]) * _FIXED_UNIT
            ).to(torch.int64),
            torch.round(
                (self.enemy_y[:, None, :] - current_y[:, :, None]) * _FIXED_UNIT
            ).to(torch.int64),
        )
        enemy_sine_fixed = self._fine_sine_fixed[enemy_fine_angle]
        enemy_cosine_fixed = self._fine_sine_fixed[
            (enemy_fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ]
        enemy_radius_thrust_denominator = torch.round(
            self._enemy_mass[enemy_type][:, None, :] * (2 * _FIXED_UNIT)
        ).to(torch.int64)
        enemy_radius_thrust_x = torch.div(
            enemy_cosine_fixed * enemy_splash_points_fixed,
            enemy_radius_thrust_denominator,
            rounding_mode="trunc",
        )
        enemy_radius_thrust_y = torch.div(
            enemy_sine_fixed * enemy_splash_points_fixed,
            enemy_radius_thrust_denominator,
            rounding_mode="trunc",
        )
        enemy_center_delta_z_fixed = torch.round(
            (
                self.enemy_z[:, None, :]
                + self._enemy_height[enemy_type][:, None, :] * 0.5
                - current_z[:, :, None]
            )
            * _FIXED_UNIT
        ).to(torch.int64)
        enemy_radius_vertical_denominator = torch.round(
            self._enemy_mass[enemy_type][:, None, :] * (4 * _FIXED_UNIT)
        ).to(torch.int64)
        enemy_radius_thrust_z = torch.div(
            enemy_center_delta_z_fixed * enemy_splash_points_fixed,
            enemy_radius_vertical_denominator,
            rounding_mode="trunc",
        )
        enemy_thrust_x = torch.sum(
            direct_enemy_thrust_x
            + splash_enemy_thrust_x
            + enemy_radius_thrust_x,
            dim=1,
        )
        enemy_thrust_y = torch.sum(
            direct_enemy_thrust_y
            + splash_enemy_thrust_y
            + enemy_radius_thrust_y,
            dim=1,
        )
        self._enemy_velocity_z_fixed.add_(
            torch.sum(enemy_radius_thrust_z, dim=1)
        )

        player_splash_damage, player_splash_points_fixed = self._rocket_radius_damage(
            current_x,
            current_y,
            current_z,
            self.x[:, None],
            self.y[:, None],
            self.z[:, None],
            torch.full_like(current_x, _PLAYER_RADIUS),
            torch.full_like(current_x, _PLAYER_HEIGHT),
        )
        visible_to_player = ~self._rocket_splash_blocked(
            current_x,
            current_y,
            current_z,
            self.x[:, None],
            self.y[:, None],
            self.z[:, None],
            torch.full((self.num_envs, 1), _PLAYER_HEIGHT, device=self.device),
        )[:, :, 0]
        direct_doll_total = torch.sum(direct_doll_damage, dim=1)
        player_survives_direct = direct_doll_total < self.health
        player_splash = (
            rocket_impact
            & visible_to_player
            & player_survives_direct[:, None]
        )
        player_splash_damage *= player_splash.to(torch.float32)
        player_splash_points_fixed *= player_splash.to(torch.int64)
        if doll_count:
            doll_splash_damage, _ = self._rocket_radius_damage(
                current_x[:, :, None],
                current_y[:, :, None],
                current_z[:, :, None],
                doll_x[None, None, :],
                doll_y[None, None, :],
                doll_z[None, None, :],
                torch.full(
                    (1, 1, doll_count),
                    _PLAYER_RADIUS,
                    device=self.device,
                ),
                torch.full(
                    (1, 1, doll_count),
                    _PLAYER_HEIGHT,
                    device=self.device,
                ),
            )
            visible_to_doll = ~self._rocket_splash_blocked(
                current_x,
                current_y,
                current_z,
                doll_x[None, :].expand(self.num_envs, -1),
                doll_y[None, :].expand(self.num_envs, -1),
                doll_z[None, :].expand(self.num_envs, -1),
                torch.full(
                    (self.num_envs, doll_count),
                    _PLAYER_HEIGHT,
                    device=self.device,
                ),
            )
            doll_splash = (
                rocket_impact[:, :, None]
                & visible_to_doll
                & player_survives_direct[:, None, None]
            )
            doll_splash_damage *= doll_splash.to(torch.float32)
            total_doll_splash_damage = torch.sum(doll_splash_damage, dim=(1, 2))
            doll_armor_absorb_request = torch.sum(
                torch.floor(
                    doll_splash_damage * self.armor_save_fraction[:, None, None]
                ),
                dim=(1, 2),
            )
        else:
            total_doll_splash_damage = torch.zeros_like(self.health)
            doll_armor_absorb_request = torch.zeros_like(self.health)
        self_damage = (
            torch.sum(player_splash_damage, dim=1)
            + direct_doll_total
            + total_doll_splash_damage
        )

        player_fine_angle = self._doom_fine_angle(
            torch.round((self.x[:, None] - current_x) * _FIXED_UNIT).to(torch.int64),
            torch.round((self.y[:, None] - current_y) * _FIXED_UNIT).to(torch.int64),
        )
        player_sine_fixed = self._fine_sine_fixed[player_fine_angle]
        player_cosine_fixed = self._fine_sine_fixed[
            (player_fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ]
        direct_thrust_fixed = (
            player_splash_damage.to(torch.int64) * _PLAYER_DAMAGE_THRUST_PER_POINT_FIXED
        ).clamp(0, _PLAYER_MAX_DAMAGE_THRUST_FIXED)
        radius_thrust_x_fixed = torch.div(
            player_cosine_fixed * player_splash_points_fixed,
            _PLAYER_RADIUS_THRUST_DENOMINATOR_FIXED,
            rounding_mode="trunc",
        )
        radius_thrust_y_fixed = torch.div(
            player_sine_fixed * player_splash_points_fixed,
            _PLAYER_RADIUS_THRUST_DENOMINATOR_FIXED,
            rounding_mode="trunc",
        )
        player_center_delta_z_fixed = torch.round(
            (self.z[:, None] + _PLAYER_HEIGHT * 0.5 - current_z) * _FIXED_UNIT
        ).to(torch.int64)
        radius_thrust_z_fixed = torch.div(
            player_center_delta_z_fixed * player_splash_points_fixed * 4,
            _PLAYER_SELF_RADIUS_VERTICAL_THRUST_DENOMINATOR_FIXED,
            rounding_mode="trunc",
        )
        self._add_player_thrust_fixed(
            torch.sum(
                (direct_thrust_fixed * player_cosine_fixed >> 16)
                + radius_thrust_x_fixed,
                dim=1,
            ),
            torch.sum(
                (direct_thrust_fixed * player_sine_fixed >> 16)
                + radius_thrust_y_fixed,
                dim=1,
            ),
        )
        next_velocity_z = self.velocity_z + torch.sum(
            radius_thrust_z_fixed,
            dim=1,
        ).to(torch.float32) / _FIXED_UNIT
        self.velocity_z.copy_(
            torch.where(
                (self.z <= self.player_floor_z) & (next_velocity_z < 0),
                torch.zeros_like(next_velocity_z),
                next_velocity_z,
            )
        )
        self._apply_player_damage(
            self_damage,
            armor_absorb_request=(
                torch.sum(
                    torch.floor(
                        player_splash_damage * self.armor_save_fraction[:, None]
                    ),
                    dim=1,
                )
                + torch.sum(
                    torch.floor(
                        direct_doll_damage * self.armor_save_fraction[:, None]
                    ),
                    dim=1,
                )
                + doll_armor_absorb_request
            ),
        )
        reward = self._apply_enemy_damage(
            damage_by_enemy,
            thrust_x_fixed=enemy_thrust_x,
            thrust_y_fixed=enemy_thrust_y,
        )

        self.projectile_x.copy_(torch.where(alive, current_x, self.projectile_x))
        self.projectile_y.copy_(torch.where(alive, current_y, self.projectile_y))
        self.projectile_z.copy_(torch.where(alive, current_z, self.projectile_z))
        self.projectile_age.add_(alive.to(torch.int32))
        impact_type = self.projectile_type.clamp(0, 1)
        self.projectile_impact_type.copy_(
            torch.where(impact, impact_type, self.projectile_impact_type)
        )
        self.projectile_impact_tics.copy_(
            torch.where(
                impact,
                self.map.projectile_explosion_total_tics[impact_type].to(torch.int32),
                self.projectile_impact_tics,
            )
        )
        self.projectile_alive &= ~impact
        self.projectile_type.masked_fill_(impact, -1)
        return reward

    def _apply_player_melee(
        self,
        weapon: torch.Tensor,
        fires: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_radius: torch.Tensor,
        target_height: torch.Tensor,
        target_alive: torch.Tensor,
        solid_sight: torch.Tensor,
        opening_bottom: torch.Tensor,
        opening_top: torch.Tensor,
    ) -> torch.Tensor:
        """Trace and apply Doom's fist and chainsaw attacks."""
        melee_fires = fires & (weapon <= 1)
        damage, spread = self._melee_attack_rolls(weapon, melee_fires)
        attack_angle = self.angle + spread
        actor_distance = self._player_ray_actor_distance(
            attack_angle[:, None],
            target_x,
            target_y,
            target_radius,
        ).squeeze(1)
        wall_distance = self._player_ray_wall_distance(attack_angle[:, None]).squeeze(1)
        wall_sectors = self.map.portal_wall_sectors
        solid_wall = self.map.portal_wall_blocks_sight | torch.any(
            wall_sectors < 0,
            dim=1,
        )
        nearest_solid_wall = torch.amin(
            torch.where(
                solid_wall[None, :],
                wall_distance,
                torch.full_like(wall_distance, torch.inf),
            ),
            dim=1,
        )

        center_dx = target_x - self.x[:, None]
        center_dy = target_y - self.y[:, None]
        center_distance = torch.sqrt(
            center_dx * center_dx + center_dy * center_dy
        ).clamp_min_(1e-4)
        shoot_z = self.z[:, None] + 36.0
        target_bottom_delta = target_z - shoot_z
        target_top_delta = target_z + target_height - shoot_z
        bottom_slope = torch.maximum(
            target_bottom_delta / center_distance,
            opening_bottom / center_distance,
        )
        top_slope = torch.minimum(
            target_top_delta / center_distance,
            opening_top / center_distance,
        )
        aim_range = 35.0 * math.pi / 180.0
        maximum_pitch = self.pitch + aim_range
        minimum_aim_slope = torch.where(
            maximum_pitch >= math.pi / 2,
            torch.full_like(maximum_pitch, -torch.inf),
            -torch.tan(maximum_pitch),
        )
        minimum_pitch = self.pitch - aim_range
        maximum_aim_slope = torch.where(
            minimum_pitch <= -math.pi / 2,
            torch.full_like(minimum_pitch, torch.inf),
            -torch.tan(minimum_pitch),
        )
        bottom_slope = torch.maximum(bottom_slope, minimum_aim_slope[:, None])
        top_slope = torch.minimum(top_slope, maximum_aim_slope[:, None])
        target_visible = target_alive & ~solid_sight & (top_slope > bottom_slope)
        melee_range = torch.where(
            weapon == 1,
            torch.full_like(self.angle, _CHAINSAW_RANGE),
            torch.full_like(self.angle, _FIST_RANGE),
        )
        valid = (
            target_visible
            & (actor_distance <= melee_range[:, None])
            & (actor_distance < nearest_solid_wall[:, None])
        )
        target_distance = torch.where(
            valid,
            actor_distance,
            torch.full_like(actor_distance, torch.inf),
        )
        target = torch.argmin(target_distance, dim=1)
        row = torch.arange(self.num_envs, device=self.device)
        has_target = melee_fires & torch.isfinite(target_distance[row, target])
        enemy_target = target.clamp_max(self.enemy_slots - 1)
        hits_enemy = has_target & (target < self.enemy_slots)
        hits_doll = has_target & (target >= self.enemy_slots)
        damage_by_enemy = torch.zeros_like(self.enemy_health)
        damage_by_enemy.scatter_add_(
            1,
            enemy_target[:, None],
            torch.where(hits_enemy, damage, torch.zeros_like(damage))[:, None],
        )
        self._apply_player_damage(
            torch.where(hits_doll, damage, torch.zeros_like(damage))
        )
        kickback = torch.where(
            weapon == 1,
            torch.zeros_like(damage),
            torch.full_like(damage, 100.0),
        )
        reward = self._apply_enemy_damage(
            damage_by_enemy,
            self.x[:, None],
            self.y[:, None],
            kickback=kickback[:, None],
        )

        target_angle = torch.atan2(
            target_y[row, target] - self.y,
            target_x[row, target] - self.x,
        )
        relative_angle = self._wrap_angle(target_angle - self.angle)
        far_turn = relative_angle.abs() > _CHAINSAW_TURN_STEP
        left_turn = relative_angle < 0
        chainsaw_angle = torch.where(
            left_turn,
            torch.where(
                far_turn,
                target_angle + _CHAINSAW_TURN_OFFSET,
                self.angle - _CHAINSAW_TURN_STEP,
            ),
            torch.where(
                far_turn,
                target_angle - _CHAINSAW_TURN_OFFSET,
                self.angle + _CHAINSAW_TURN_STEP,
            ),
        )
        fist_hit = has_target & (weapon == 0)
        chainsaw_hit = has_target & (weapon == 1)
        next_angle = torch.where(
            fist_hit,
            target_angle,
            torch.where(chainsaw_hit, chainsaw_angle, self.angle),
        )
        self.angle.copy_(torch.remainder(next_angle, 2.0 * math.pi))
        self.chainsaw_pull |= chainsaw_hit
        return reward

    def _player_autoaim(
        self,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_radius: torch.Tensor,
        target_height: torch.Tensor,
        target_alive: torch.Tensor,
        solid_sight: torch.Tensor,
        opening_bottom: torch.Tensor,
        opening_top: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return P_AimLineAttack's selected probe angle, pitch, and target flag."""
        shoot_z = self.z[:, None] + 36.0
        center_dx = target_x - self.x[:, None]
        center_dy = target_y - self.y[:, None]
        center_distance = torch.sqrt(
            center_dx * center_dx + center_dy * center_dy
        ).clamp_min_(1e-4)
        target_bottom_delta = target_z - shoot_z
        target_top_delta = target_z + target_height - shoot_z
        portal_bottom_slope = torch.where(
            opening_bottom > target_bottom_delta,
            opening_bottom / center_distance,
            torch.full_like(center_distance, -torch.inf),
        )
        portal_top_slope = torch.where(
            opening_top < target_top_delta,
            opening_top / center_distance,
            torch.full_like(center_distance, torch.inf),
        )

        # P_BulletSlope and P_SpawnPlayerMissile both probe center,
        # +5.625 degrees, then -5.625 degrees, stopping at the first probe
        # that crosses a shootable actor within 16 map blocks.
        aim_angles = torch.stack(
            (
                self.angle,
                self.angle + _BULLET_AUTOAIM_OFFSET,
                self.angle - _BULLET_AUTOAIM_OFFSET,
            ),
            dim=1,
        )
        aim_distance = self._player_ray_actor_distance(
            aim_angles,
            target_x,
            target_y,
            target_radius,
        )
        safe_aim_distance = torch.where(
            torch.isfinite(aim_distance),
            aim_distance.clamp_min(1e-4),
            torch.ones_like(aim_distance),
        )
        bottom_slope = torch.maximum(
            target_bottom_delta[:, None, :] / safe_aim_distance,
            portal_bottom_slope[:, None, :],
        )
        top_slope = torch.minimum(
            target_top_delta[:, None, :] / safe_aim_distance,
            portal_top_slope[:, None, :],
        )
        aim_range = 35.0 * math.pi / 180.0
        minimum_pitch = self.pitch - aim_range
        maximum_pitch = self.pitch + aim_range
        minimum_aim_slope = torch.where(
            maximum_pitch >= math.pi / 2,
            torch.full_like(maximum_pitch, -torch.inf),
            -torch.tan(maximum_pitch),
        )
        maximum_aim_slope = torch.where(
            minimum_pitch <= -math.pi / 2,
            torch.full_like(minimum_pitch, torch.inf),
            -torch.tan(minimum_pitch),
        )
        bottom_slope = torch.maximum(bottom_slope, minimum_aim_slope[:, None, None])
        top_slope = torch.minimum(top_slope, maximum_aim_slope[:, None, None])
        aim_wall_distance = self._player_ray_wall_distance(aim_angles)
        wall_sectors = self.map.portal_wall_sectors
        solid_wall = self.map.portal_wall_blocks_sight | torch.any(
            wall_sectors < 0,
            dim=1,
        )
        nearest_solid_wall = torch.amin(
            torch.where(
                solid_wall[None, None, :],
                aim_wall_distance,
                torch.full_like(aim_wall_distance, torch.inf),
            ),
            dim=2,
        )
        aim_valid = (
            target_alive[:, None, :]
            & ~solid_sight[:, None, :]
            & (top_slope > bottom_slope)
            & (aim_distance <= _BULLET_AUTOAIM_RANGE)
            & (aim_distance < nearest_solid_wall[:, :, None])
        )
        aim_target_distance = torch.where(
            aim_valid,
            aim_distance,
            torch.full_like(aim_distance, torch.inf),
        )
        aim_target = torch.argmin(aim_target_distance, dim=2)
        aim_exists = torch.isfinite(
            aim_target_distance.gather(2, aim_target[:, :, None]).squeeze(2)
        )
        selected_top_slope = top_slope.gather(
            2,
            aim_target[:, :, None],
        ).squeeze(2)
        selected_bottom_slope = bottom_slope.gather(
            2,
            aim_target[:, :, None],
        ).squeeze(2)
        selected_top_pitch = torch.maximum(
            -torch.atan(selected_top_slope),
            minimum_pitch[:, None],
        )
        selected_bottom_pitch = torch.minimum(
            -torch.atan(selected_bottom_slope),
            maximum_pitch[:, None],
        )
        # PTR_AimTraverse averages the clipped top and bottom pitch angles,
        # not their slopes. The distinction is visible for close actors
        # straddling the player's 36-unit hitscan origin.
        pitch_by_aim = (selected_top_pitch + selected_bottom_pitch) * 0.5
        selected_probe = torch.argmax(aim_exists.to(torch.int32), dim=1)
        row = torch.arange(self.num_envs, device=self.device)
        has_autoaim = torch.any(aim_exists, dim=1)
        selected_angle = torch.where(
            has_autoaim,
            aim_angles[row, selected_probe],
            self.angle,
        )
        selected_pitch = torch.where(
            has_autoaim,
            pitch_by_aim[row, selected_probe],
            self.pitch,
        )
        return selected_angle, selected_pitch, has_autoaim

    def _apply_player_hitscan(
        self,
        weapon: torch.Tensor,
        fires: torch.Tensor,
        accurate: torch.Tensor,
        base_pitch: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_radius: torch.Tensor,
        target_height: torch.Tensor,
        target_alive: torch.Tensor,
    ) -> torch.Tensor:
        """Trace and apply every Doom bullet pellet independently."""
        hitscan_fires = fires & (weapon >= 2) & (weapon <= 5)
        shoot_z = self.z + 36.0

        pellet_damage, horizontal_spread, vertical_spread = self._hitscan_pellet_rolls(
            weapon, hitscan_fires, accurate
        )
        pellet_angle = self.angle[:, None] + horizontal_spread
        pellet_pitch = base_pitch[:, None] + vertical_spread
        actor_distance = self._player_ray_actor_distance(
            pellet_angle,
            target_x,
            target_y,
            target_radius,
        )
        actor_intercept = torch.isfinite(actor_distance)
        safe_actor_distance = torch.where(
            actor_intercept,
            actor_distance,
            torch.zeros_like(actor_distance),
        )
        cosine_pitch, sine_pitch = self._fine_direction(pellet_pitch)
        vertical_slope = -sine_pitch / cosine_pitch.clamp_min_(1.0 / _FIXED_UNIT)
        intercept_z = shoot_z[:, None, None] + vertical_slope[:, :, None] * safe_actor_distance
        target_bottom = target_z[:, None, :]
        target_top = target_bottom + target_height[:, None, :]
        enters_actor_side = (
            actor_intercept & (intercept_z >= target_bottom) & (intercept_z <= target_top)
        )

        safe_vertical_slope = torch.where(
            vertical_slope.abs() < 1e-6,
            torch.ones_like(vertical_slope),
            vertical_slope,
        )
        top_plane_distance = (target_top - shoot_z[:, None, None]) / safe_vertical_slope[:, :, None]
        bottom_plane_distance = (target_bottom - shoot_z[:, None, None]) / safe_vertical_slope[
            :, :, None
        ]
        ray_cosine, ray_sine = self._fine_direction(pellet_angle)
        ray_cosine = ray_cosine[:, :, None]
        ray_sine = ray_sine[:, :, None]

        def inside_target_box(distance: torch.Tensor) -> torch.Tensor:
            hit_x = self.x[:, None, None] + ray_cosine * distance
            hit_y = self.y[:, None, None] + ray_sine * distance
            return ((hit_x - target_x[:, None, :]).abs() <= target_radius[:, None, :]) & (
                (hit_y - target_y[:, None, :]).abs() <= target_radius[:, None, :]
            )

        enters_actor_top = (
            actor_intercept
            & (intercept_z > target_top)
            & (vertical_slope[:, :, None] < 0)
            & (top_plane_distance >= 0)
            & inside_target_box(top_plane_distance)
        )
        enters_actor_bottom = (
            actor_intercept
            & (intercept_z < target_bottom)
            & (vertical_slope[:, :, None] > 0)
            & (bottom_plane_distance >= 0)
            & inside_target_box(bottom_plane_distance)
        )
        hit_distance = torch.where(
            enters_actor_top,
            top_plane_distance,
            torch.where(
                enters_actor_bottom,
                bottom_plane_distance,
                safe_actor_distance,
            ),
        )
        actor_hit = enters_actor_side | enters_actor_top | enters_actor_bottom

        maximum_horizontal_distance = _PLAYER_HITSCAN_RANGE * cosine_pitch
        actor_hit &= hit_distance <= maximum_horizontal_distance[:, :, None]

        pellet_wall_distance = self._player_ray_wall_distance(pellet_angle)
        wall_intercept = torch.isfinite(pellet_wall_distance)
        safe_wall_distance = torch.where(
            wall_intercept,
            pellet_wall_distance,
            torch.zeros_like(pellet_wall_distance),
        )
        wall_hit_z = shoot_z[:, None, None] + vertical_slope[:, :, None] * safe_wall_distance
        wall_sectors = self.map.portal_wall_sectors
        valid_portal = torch.all(wall_sectors >= 0, dim=1)
        safe_sectors = wall_sectors.clamp_min(0)
        portal_bottom = torch.amax(
            self.map.sector_heights[safe_sectors, 0],
            dim=1,
        )
        portal_top = torch.amin(
            self.map.sector_heights[safe_sectors, 1],
            dim=1,
        )
        wall_blocks_pellet = wall_intercept & (
            self.map.portal_wall_blocks_sight[None, None, :]
            | ~valid_portal[None, None, :]
            | (wall_hit_z <= portal_bottom[None, None, :])
            | (wall_hit_z >= portal_top[None, None, :])
        )
        wall_blocks_pellet &= pellet_wall_distance < maximum_horizontal_distance[:, :, None]
        nearest_blocking_wall = torch.amin(
            torch.where(
                wall_blocks_pellet,
                pellet_wall_distance,
                torch.full_like(pellet_wall_distance, torch.inf),
            ),
            dim=2,
        )
        actor_hit &= hit_distance < nearest_blocking_wall[:, :, None]
        actor_hit &= target_alive[:, None, :]

        # Pellets are processed in reference order. Once a pellet kills a
        # monster it stops being shootable, allowing later pellets through to
        # an actor behind it during the same shotgun blast.
        shootable = target_alive.clone()
        remaining_enemy_health = self.enemy_health.clone()
        damage_by_enemy_pellet: list[torch.Tensor] = []
        damage_by_doll_pellet: list[torch.Tensor] = []
        hurt_by_enemy_pellet: list[torch.Tensor] = []
        target_count = target_x.shape[1]
        for pellet_index in range(_HITSCAN_MAX_PELLETS):
            candidate_distance = torch.where(
                actor_hit[:, pellet_index, :] & shootable,
                hit_distance[:, pellet_index, :],
                torch.full_like(hit_distance[:, pellet_index, :], torch.inf),
            )
            target = torch.argmin(candidate_distance, dim=1)
            has_target = (pellet_damage[:, pellet_index] > 0) & torch.isfinite(
                candidate_distance.gather(1, target[:, None]).squeeze(1)
            )
            damage_by_target = torch.zeros(
                (self.num_envs, target_count),
                device=self.device,
            )
            damage_by_target.scatter_add_(
                1,
                target[:, None],
                torch.where(
                    has_target,
                    pellet_damage[:, pellet_index],
                    torch.zeros_like(pellet_damage[:, pellet_index]),
                )[:, None],
            )
            enemy_damage = damage_by_target[:, : self.enemy_slots]
            damage_by_enemy_pellet.append(enemy_damage)
            damage_by_doll_pellet.append(torch.sum(damage_by_target[:, self.enemy_slots :], dim=1))
            next_enemy_health = remaining_enemy_health - enemy_damage
            hurt_by_enemy_pellet.append(
                (enemy_damage > 0) & (next_enemy_health > 0)
            )
            remaining_enemy_health.copy_(next_enemy_health)
            shootable[:, : self.enemy_slots] &= remaining_enemy_health > 0

        damage_by_enemy = torch.stack(damage_by_enemy_pellet, dim=1)
        damage_by_doll = torch.stack(damage_by_doll_pellet, dim=1)
        hurt_by_enemy = torch.stack(hurt_by_enemy_pellet, dim=1)
        self._apply_player_damage(
            torch.sum(damage_by_doll, dim=1),
            armor_absorb_request=torch.sum(
                torch.floor(damage_by_doll * self.armor_save_fraction[:, None]),
                dim=1,
            ),
        )
        thrust_x, thrust_y = self._enemy_damage_thrust_components(
            damage_by_enemy,
            self.x[:, None, None],
            self.y[:, None, None],
        )
        pain_random = self._random_u32(torch.any(hurt_by_enemy, dim=(1, 2)))[:, None, None]
        pellet = torch.arange(
            _HITSCAN_MAX_PELLETS,
            device=self.device,
            dtype=torch.int64,
        )[None, :, None]
        enemy_slot = torch.arange(
            self.enemy_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, None, :]
        pain_mixed = (
            pain_random
            ^ (pellet * _HASH_MURMUR_SIGNED)
            ^ (enemy_slot * _HASH_GOLDEN_RATIO_SIGNED)
        )
        pain_mixed ^= pain_mixed >> 16
        pain_roll = torch.remainder(pain_mixed, 256)
        pain_chance = self._enemy_pain_chance[self.enemy_type.clamp_min(0)][:, None, :]
        pain_override = torch.any(
            hurt_by_enemy & (pain_roll < pain_chance),
            dim=1,
        )
        return self._apply_enemy_damage(
            torch.sum(damage_by_enemy, dim=1),
            thrust_x_fixed=torch.sum(thrust_x, dim=1),
            thrust_y_fixed=torch.sum(thrust_y, dim=1),
            pain_override=pain_override,
        )

    def _player_attack(self, buttons: torch.Tensor) -> torch.Tensor:
        reward = torch.zeros(self.num_envs, device=self.device)
        pending = self.pending_attack_weapon >= 0
        next_delay = torch.clamp_min(self.pending_attack_delay - 1, 0)
        pending_valid = (
            pending
            & ~self.player_dead
            & (self.episode_time < self.episode_timeout)
        )
        execute_pending = pending_valid & (next_delay <= 0)
        pending_weapon_to_execute = self.pending_attack_weapon.clamp_min(0)
        pending_accuracy_to_execute = self.pending_attack_accurate.clone()
        keep_pending = pending_valid & ~execute_pending
        self.pending_attack_delay.copy_(
            torch.where(keep_pending, next_delay, torch.zeros_like(next_delay))
        )
        self.pending_attack_weapon.copy_(
            torch.where(
                keep_pending,
                self.pending_attack_weapon,
                torch.full_like(self.pending_attack_weapon, -1),
            )
        )
        self.pending_attack_accurate.copy_(
            torch.where(
                keep_pending,
                self.pending_attack_accurate,
                torch.zeros_like(self.pending_attack_accurate),
            )
        )

        weapon = self._active_weapon()
        ammo_slot = self._weapon_ammo_slot[weapon]
        safe_ammo_slot = ammo_slot.clamp_min(0)
        ammo = self.ammo.gather(1, safe_ammo_slot[:, None]).squeeze(1)
        cost = self._weapon_ammo_cost[weapon]
        refire_tail = torch.clamp_min(
            self._weapon_ready_duration[weapon] - self._weapon_cooldown[weapon],
            0,
        )
        weapon_action_ready = (self.weapon_state_cooldown <= 0) | (
            self.weapon_state_cooldown == refire_tail
        )
        attempted_empty_fire = (
            buttons[:, 0]
            & (self.attack_cooldown <= 0)
            & weapon_action_ready
            & (self.weapon_raise_cooldown <= 0)
            & ~self.player_dead
            & (self.episode_time < self.episode_timeout)
            & self._weapon_owned(weapon)
            & (ammo_slot >= 0)
            & (ammo < cost)
            & ~pending
        )
        replacement = self._best_ready_weapon()
        self._set_active_weapon(replacement, attempted_empty_fire)
        fires = (
            buttons[:, 0]
            & (self.attack_cooldown <= 0)
            & weapon_action_ready
            & (self.weapon_raise_cooldown <= 0)
            & (self.pending_weapon < 0)
            & ~self.player_dead
            & (self.episode_time < self.episode_timeout)
            & self._weapon_owned(weapon)
            & ((ammo_slot < 0) | (ammo >= cost))
            & ~pending
        )
        action_delay = self._weapon_action_delay[weapon]
        self.attack_cooldown.copy_(
            torch.where(fires, self._weapon_cooldown[weapon], self.attack_cooldown)
        )
        self.weapon_state_cooldown.copy_(
            torch.where(
                fires,
                self._weapon_ready_duration[weapon],
                self.weapon_state_cooldown,
            )
        )
        delayed = fires & (action_delay > 0)
        accurate = self.attack_held_tics <= 1
        self.pending_attack_weapon.copy_(
            torch.where(delayed, weapon, self.pending_attack_weapon)
        )
        self.pending_attack_delay.copy_(
            torch.where(delayed, action_delay, self.pending_attack_delay)
        )
        self.pending_attack_accurate.copy_(
            torch.where(delayed, accurate, self.pending_attack_accurate)
        )
        immediate = fires & ~delayed
        execute_weapon = torch.where(
            execute_pending,
            pending_weapon_to_execute,
            weapon,
        )
        execute_accurate = torch.where(
            execute_pending,
            pending_accuracy_to_execute,
            accurate,
        )
        reward.add_(
            self._execute_player_attack(
                execute_weapon,
                execute_pending | immediate,
                execute_accurate,
            )
        )
        forced_second_action = fires & ((weapon == 1) | (weapon == 5))
        self.pending_attack_weapon.copy_(
            torch.where(forced_second_action, weapon, self.pending_attack_weapon)
        )
        self.pending_attack_delay.copy_(
            torch.where(
                forced_second_action,
                torch.full_like(self.pending_attack_delay, 4),
                self.pending_attack_delay,
            )
        )
        self.pending_attack_accurate.copy_(
            torch.where(
                forced_second_action,
                accurate,
                self.pending_attack_accurate,
            )
        )
        return reward

    def _execute_player_attack(
        self,
        weapon: torch.Tensor,
        fires: torch.Tensor,
        accurate: torch.Tensor,
    ) -> torch.Tensor:
        ammo_slot = self._weapon_ammo_slot[weapon]
        safe_ammo_slot = ammo_slot.clamp_min(0)
        ammo = self.ammo.gather(1, safe_ammo_slot[:, None]).squeeze(1)
        cost = self._weapon_ammo_cost[weapon]
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
        self.weapon_fire_count.add_(fires.to(torch.int32))
        self.enemy_heard_player |= (
            fires[:, None]
            & self.enemy_alive
            & (self.enemy_target_slot < -1)
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
            target_radius = torch.cat(
                (
                    self._enemy_radius[self.enemy_type.clamp_min(0)],
                    torch.full_like(doll_x, _PLAYER_RADIUS),
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
            target_radius = self._enemy_radius[self.enemy_type.clamp_min(0)]
            target_alive = self.enemy_alive
        shoot_z = self.z[:, None] + 36.0
        solid_sight, opening_bottom, opening_top = self._sight_opening(
            self.x[:, None],
            self.y[:, None],
            shoot_z,
            target_x,
            target_y,
            target_z,
            target_height,
        )
        autoaim_angle, autoaim_pitch, _ = self._player_autoaim(
            target_x,
            target_y,
            target_z,
            target_radius,
            target_height,
            target_alive,
            solid_sight,
            opening_bottom,
            opening_top,
        )
        self._spawn_player_projectile(
            weapon,
            fires,
            autoaim_angle,
            autoaim_pitch,
        )
        melee_reward = self._apply_player_melee(
            weapon,
            fires,
            target_x,
            target_y,
            target_z,
            target_radius,
            target_height,
            target_alive,
            solid_sight,
            opening_bottom,
            opening_top,
        )
        hitscan_reward = self._apply_player_hitscan(
            weapon,
            fires,
            accurate,
            autoaim_pitch,
            target_x,
            target_y,
            target_z,
            target_radius,
            target_height,
            target_alive,
        )
        return hitscan_reward + melee_reward

    def _enemy_damage_roll(
        self,
        enemy_type: torch.Tensor,
        attacks: torch.Tensor,
        in_melee_range: torch.Tensor,
    ) -> torch.Tensor:
        random_bits = self._random_u32(torch.any(attacks, dim=1))[:, None]
        slot = torch.arange(self.enemy_slots, device=self.device, dtype=torch.int64)[None, :]

        def die(draw: int, sides: int) -> torch.Tensor:
            mixed = random_bits ^ (slot * _HASH_GOLDEN_RATIO_SIGNED) ^ (draw * _HASH_MURMUR_SIGNED)
            mixed ^= mixed >> 16
            return torch.remainder(mixed, sides).to(torch.float32) + 1

        damage = torch.zeros_like(in_melee_range, dtype=torch.float32)
        damage = torch.where(enemy_type == 0, die(0, 5) * 3, damage)
        shotgun = (die(0, 5) + die(1, 5) + die(2, 5)) * 3
        damage = torch.where(enemy_type == 1, shotgun, damage)
        damage = torch.where(enemy_type == 2, die(0, 10) * 2, damage)
        damage = torch.where(enemy_type == 3, die(0, 5) * 3, damage)
        damage = torch.where(enemy_type == 4, die(0, 10) * 4, damage)
        knight_multiplier = torch.where(in_melee_range, 10.0, 8.0)
        damage = torch.where(enemy_type == 5, die(0, 8) * knight_multiplier, damage)
        return torch.where(attacks, damage, torch.zeros_like(damage))

    def _enemy_hitscan_rolls(
        self,
        enemy_type: torch.Tensor,
        fires: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Roll Doom's per-pellet monster bullet spread and damage."""
        pellet_count = torch.where(
            enemy_type == 1,
            torch.full_like(enemy_type, 3),
            ((enemy_type == 0) | (enemy_type == 3)).to(enemy_type.dtype),
        )
        pellet = torch.arange(3, device=self.device, dtype=torch.int64)[None, None, :]
        active = fires[:, :, None] & (pellet < pellet_count[:, :, None])
        draw_mask = torch.any(active, dim=(1, 2))
        enemy_slot = torch.arange(
            self.enemy_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, :, None]

        def mixed_draw(draw: int) -> torch.Tensor:
            bits = self._random_u32(draw_mask)[:, None, None]
            mixed = (
                bits
                ^ (enemy_slot * _HASH_GOLDEN_RATIO_SIGNED)
                ^ (draw * 0x27D4EB2D)
            )
            mixed ^= mixed >> 16
            return torch.bitwise_right_shift(mixed, pellet * 8)

        first = torch.bitwise_and(mixed_draw(0), 255)
        second = torch.bitwise_and(mixed_draw(1), 255)
        damage_roll = torch.remainder(mixed_draw(2), 5).to(torch.float32) + 1.0
        spread_bam = (first - second) * (1 << 20)
        return (
            torch.where(active, damage_roll * 3.0, 0.0),
            torch.where(active, spread_bam, 0),
        )

    def _enemy_target_geometry(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Resolve each monster's target pointer to tensor geometry."""
        target_is_monster = self.enemy_target_slot >= 0
        target_is_player = self.enemy_target_slot == -1
        safe_target = self.enemy_target_slot.clamp(0, self.enemy_slots - 1)
        target_x = torch.where(
            target_is_monster,
            self.enemy_x.gather(1, safe_target),
            self.x[:, None],
        )
        target_y = torch.where(
            target_is_monster,
            self.enemy_y.gather(1, safe_target),
            self.y[:, None],
        )
        target_z = torch.where(
            target_is_monster,
            self.enemy_z.gather(1, safe_target),
            self.z[:, None],
        )
        target_type = self._effective_enemy_type().gather(1, safe_target)
        target_height = torch.where(
            target_is_monster,
            self._enemy_height[target_type],
            torch.full_like(target_x, _PLAYER_HEIGHT),
        )
        target_radius = torch.where(
            target_is_monster,
            self._enemy_radius[target_type],
            torch.full_like(target_x, _PLAYER_RADIUS),
        )
        target_alive = torch.where(
            target_is_monster,
            self.enemy_alive.gather(1, safe_target),
            target_is_player & ~self.player_dead[:, None],
        )
        return (
            target_x,
            target_y,
            target_z,
            target_height,
            target_radius,
            target_is_monster,
            target_alive,
        )

    def _enemy_ray_actor_distances(
        self,
        ray_angle: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_radius: torch.Tensor,
    ) -> torch.Tensor:
        """Return PT_COMPATIBLE intercepts for arbitrary shootable actors."""
        cosine, sine = (
            self._fine_direction(ray_angle)
            if ray_angle.dtype.is_floating_point
            else self._fine_direction_from_index(ray_angle)
        )
        cosine = cosine[..., None]
        sine = sine[..., None]
        same_sign = (cosine >= 0) == (sine >= 0)
        radius = target_radius[:, None, None, :]
        diagonal_x = target_x[:, None, None, :] - radius
        diagonal_y = target_y[:, None, None, :] + torch.where(
            same_sign,
            radius,
            -radius,
        )
        diagonal_dx = radius * 2.0
        diagonal_dy = torch.where(
            same_sign,
            -radius * 2.0,
            radius * 2.0,
        )
        offset_x = diagonal_x - self.enemy_x[:, :, None, None]
        offset_y = diagonal_y - self.enemy_y[:, :, None, None]
        denominator = cosine * diagonal_dy - sine * diagonal_dx
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        along_ray = (offset_x * diagonal_dy - offset_y * diagonal_dx) / safe
        along_diagonal = (offset_x * sine - offset_y * cosine) / safe
        intersects = (
            (denominator.abs() >= 1e-6)
            & (along_ray >= 0)
            & (along_diagonal >= 0)
            & (along_diagonal <= 1)
        )
        return torch.where(
            intersects,
            along_ray,
            torch.full_like(along_ray, torch.inf),
        )

    def _enemy_ray_player_actor_distances(
        self,
        ray_angle: torch.Tensor,
    ) -> torch.Tensor:
        """Return PT_COMPATIBLE intercepts for the player and voodoo dolls."""
        target_x = self.x[:, None]
        target_y = self.y[:, None]
        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            target_x = torch.cat(
                (
                    target_x,
                    self.map.player_starts[None, :-1, 0].expand(
                        self.num_envs,
                        -1,
                    ),
                ),
                dim=1,
            )
            target_y = torch.cat(
                (
                    target_y,
                    self.map.player_starts[None, :-1, 1].expand(
                        self.num_envs,
                        -1,
                    ),
                ),
                dim=1,
            )
        target_radius = torch.full_like(target_x, _PLAYER_RADIUS)
        return self._enemy_ray_actor_distances(
            ray_angle,
            target_x,
            target_y,
            target_radius,
        )

    def _enemy_ray_player_distance(self, ray_angle: torch.Tensor) -> torch.Tensor:
        """Return the controlled player's monster-bullet intercepts."""
        return self._enemy_ray_player_actor_distances(ray_angle)[..., 0]

    def _enemy_ray_wall_distance(self, ray_angle: torch.Tensor) -> torch.Tensor:
        """Return linedef intercepts for each monster bullet ray."""
        cosine, sine = (
            self._fine_direction(ray_angle)
            if ray_angle.dtype.is_floating_point
            else self._fine_direction_from_index(ray_angle)
        )
        cosine = cosine[..., None]
        sine = sine[..., None]
        walls = self.map.portal_walls
        start_x = walls[:, 0]
        start_y = walls[:, 1]
        segment_x = walls[:, 2] - start_x
        segment_y = walls[:, 3] - start_y
        offset_x = start_x - self.enemy_x[:, :, None, None]
        offset_y = start_y - self.enemy_y[:, :, None, None]
        denominator = cosine * segment_y - sine * segment_x
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        along_ray = (offset_x * segment_y - offset_y * segment_x) / safe
        along_wall = (offset_x * sine - offset_y * cosine) / safe
        intersects = (
            (denominator.abs() >= 1e-6)
            & (along_ray > 1e-4)
            & (along_wall >= 0)
            & (along_wall <= 1)
        )
        return torch.where(
            intersects,
            along_ray,
            torch.full_like(along_ray, torch.inf),
        )

    def _enemy_hitscan_damage(
        self,
        enemy_type: torch.Tensor,
        fires: torch.Tensor,
        distance: torch.Tensor,
        visible: torch.Tensor,
        *,
        base_bam: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trace monster bullets through walls and every shootable actor."""
        pellet_damage, spread_bam = self._enemy_hitscan_rolls(enemy_type, fires)
        (
            _intended_x,
            _intended_y,
            intended_z,
            intended_height,
            _intended_radius,
            intended_is_monster,
            _intended_alive,
        ) = self._enemy_target_geometry()
        if base_bam is None:
            safe_target = self.enemy_target_slot.clamp(0, self.enemy_slots - 1)
            enemy_x_fixed = self._public_or_retained_fixed(
                self.enemy_x,
                self._enemy_x_fixed,
            )
            enemy_y_fixed = self._public_or_retained_fixed(
                self.enemy_y,
                self._enemy_y_fixed,
            )
            player_x_fixed = self._public_or_retained_fixed(self.x, self._x_fixed)
            player_y_fixed = self._public_or_retained_fixed(self.y, self._y_fixed)
            intended_x_fixed = torch.where(
                intended_is_monster,
                enemy_x_fixed.gather(1, safe_target),
                player_x_fixed[:, None],
            )
            intended_y_fixed = torch.where(
                intended_is_monster,
                enemy_y_fixed.gather(1, safe_target),
                player_y_fixed[:, None],
            )
            base_bam = self._doom_bam_angle(
                intended_x_fixed - enemy_x_fixed,
                intended_y_fixed - enemy_y_fixed,
            )
        pellet_fine_angle = (
            (base_bam[:, :, None] + spread_bam) & _UINT32_MASK
        ) >> _ANGLE_TO_FINE_SHIFT

        doll_count = max(len(self.map.player_starts) - 1, 0)
        actor_x = torch.cat((self.enemy_x, self.x[:, None]), dim=1)
        actor_y = torch.cat((self.enemy_y, self.y[:, None]), dim=1)
        actor_z = torch.cat((self.enemy_z, self.z[:, None]), dim=1)
        actor_height = torch.cat(
            (
                self._enemy_height[self._effective_enemy_type()],
                torch.full_like(self.x[:, None], _PLAYER_HEIGHT),
            ),
            dim=1,
        )
        actor_radius = torch.cat(
            (
                self._enemy_radius[self._effective_enemy_type()],
                torch.full_like(self.x[:, None], _PLAYER_RADIUS),
            ),
            dim=1,
        )
        actor_alive = torch.cat(
            (self.enemy_alive, (~self.player_dead)[:, None]),
            dim=1,
        )
        if doll_count:
            doll_x = self.map.player_starts[None, :-1, 0].expand(
                self.num_envs,
                -1,
            )
            doll_y = self.map.player_starts[None, :-1, 1].expand(
                self.num_envs,
                -1,
            )
            actor_x = torch.cat((actor_x, doll_x), dim=1)
            actor_y = torch.cat((actor_y, doll_y), dim=1)
            actor_z = torch.cat(
                (
                    actor_z,
                    self._player_start_z[None, :-1].expand(
                        self.num_envs,
                        -1,
                    ),
                ),
                dim=1,
            )
            actor_height = torch.cat(
                (actor_height, torch.full_like(doll_x, _PLAYER_HEIGHT)),
                dim=1,
            )
            actor_radius = torch.cat(
                (actor_radius, torch.full_like(doll_x, _PLAYER_RADIUS)),
                dim=1,
            )
            actor_alive = torch.cat(
                (
                    actor_alive,
                    (~self.player_dead)[:, None].expand(-1, doll_count),
                ),
                dim=1,
            )
        actor_distance = self._enemy_ray_actor_distances(
            pellet_fine_angle,
            actor_x,
            actor_y,
            actor_radius,
        )

        shoot_z = self.enemy_z + 36.0
        bottom_slope = torch.maximum(
            (intended_z - shoot_z) / distance,
            torch.full_like(distance, -_BULLET_AUTOAIM_MAX_SLOPE),
        )
        top_slope = torch.minimum(
            (intended_z + intended_height - shoot_z) / distance,
            torch.full_like(distance, _BULLET_AUTOAIM_MAX_SLOPE),
        )
        pitch = (-torch.atan(top_slope) - torch.atan(bottom_slope)) * 0.5
        cosine_pitch, sine_pitch = self._fine_direction(pitch)
        vertical_slope = -sine_pitch / cosine_pitch.clamp_min_(1.0 / _FIXED_UNIT)
        intercept_z = (
            shoot_z[:, :, None, None]
            + vertical_slope[:, :, None, None]
            * torch.where(
                torch.isfinite(actor_distance),
                actor_distance,
                torch.zeros_like(actor_distance),
            )
        )
        maximum_horizontal_distance = 2048.0 * cosine_pitch
        pellet_wall_distance = self._enemy_ray_wall_distance(pellet_fine_angle)
        wall_intercept = torch.isfinite(pellet_wall_distance)
        safe_wall_distance = torch.where(
            wall_intercept,
            pellet_wall_distance,
            torch.zeros_like(pellet_wall_distance),
        )
        wall_hit_z = (
            shoot_z[:, :, None, None]
            + vertical_slope[:, :, None, None] * safe_wall_distance
        )
        wall_sectors = self.map.portal_wall_sectors
        valid_portal = torch.all(wall_sectors >= 0, dim=1)
        safe_sectors = wall_sectors.clamp_min(0)
        portal_bottom = torch.amax(
            self.map.sector_heights[safe_sectors, 0],
            dim=1,
        )
        portal_top = torch.amin(
            self.map.sector_heights[safe_sectors, 1],
            dim=1,
        )
        wall_blocks_pellet = wall_intercept & (
            self.map.portal_wall_blocks_sight[None, None, None, :]
            | ~valid_portal[None, None, None, :]
            | (wall_hit_z <= portal_bottom[None, None, None, :])
            | (wall_hit_z >= portal_top[None, None, None, :])
        )
        wall_blocks_pellet &= (
            pellet_wall_distance
            < maximum_horizontal_distance[:, :, None, None]
        )
        nearest_blocking_wall = torch.amin(
            torch.where(
                wall_blocks_pellet,
                pellet_wall_distance,
                torch.full_like(pellet_wall_distance, torch.inf),
            ),
            dim=3,
        )
        actor_count = actor_x.shape[1]
        attacker_slot = torch.arange(
            self.enemy_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, :, None, None]
        actor_slot = torch.arange(
            actor_count,
            device=self.device,
            dtype=torch.int64,
        )[None, None, None, :]
        not_self = (actor_slot >= self.enemy_slots) | (
            actor_slot != attacker_slot
        )
        actor_hit = (
            fires[:, :, None, None]
            & visible[:, :, None, None]
            & actor_alive[:, None, None, :]
            & not_self
            & torch.isfinite(actor_distance)
            & (actor_distance <= maximum_horizontal_distance[:, :, None, None])
            & (actor_distance < nearest_blocking_wall[:, :, :, None])
            & (intercept_z >= actor_z[:, None, None, :])
            & (
                intercept_z
                <= actor_z[:, None, None, :] + actor_height[:, None, None, :]
            )
        )
        candidate_distance = torch.where(
            actor_hit,
            actor_distance,
            torch.full_like(actor_distance, torch.inf),
        )
        target = torch.argmin(candidate_distance, dim=3)
        has_target = torch.isfinite(
            candidate_distance.gather(3, target[..., None]).squeeze(3)
        )
        damage = torch.where(
            has_target,
            pellet_damage,
            torch.zeros_like(pellet_damage),
        )
        hits_enemy = has_target & (target < self.enemy_slots)
        hits_player_actor = has_target & (target >= self.enemy_slots)
        player_damage = torch.where(
            hits_player_actor,
            damage,
            torch.zeros_like(damage),
        )
        actual_player_damage = torch.where(
            has_target & (target == self.enemy_slots),
            damage,
            torch.zeros_like(damage),
        )
        enemy_target = target.clamp_max(self.enemy_slots - 1)
        enemy_damage = torch.zeros(
            (
                self.num_envs,
                self.enemy_slots,
                pellet_damage.shape[2],
                self.enemy_slots,
            ),
            device=self.device,
        )
        enemy_damage.scatter_add_(
            3,
            enemy_target[..., None],
            torch.where(
                hits_enemy,
                damage,
                torch.zeros_like(damage),
            )[..., None],
        )
        return player_damage, actual_player_damage, enemy_damage

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

    def _enemy_chaingun_refire_decision(
        self,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 40/256 A_CPosRefire bypass decision per actor."""

        random_bits = self._random_u32(torch.any(candidates, dim=1))[:, None]
        slot = torch.arange(self.enemy_slots, device=self.device, dtype=torch.int64)[None, :]
        mixed = random_bits ^ (slot * _HASH_GOLDEN_RATIO_SIGNED)
        mixed ^= mixed >> 16
        return candidates & (torch.bitwise_and(mixed, 255) < 40)

    def _spawn_enemy_projectiles(
        self,
        requested: torch.Tensor,
        dx: torch.Tensor,
        dy: torch.Tensor,
    ) -> None:
        # P_SpawnMissile creates an independent actor on every call. Allocate
        # requested sources into the first free projectile slots instead of
        # coupling projectile index to monster index, which incorrectly
        # limited each Hell Knight to one in-flight BaronBall.
        free = ~self.enemy_projectile_alive & (
            self.enemy_projectile_impact_tics <= 0
        )
        requested_prefix = torch.cumsum(requested.to(torch.int64), dim=1)
        requested_count = requested_prefix[:, -1]
        free_rank = torch.cumsum(free.to(torch.int64), dim=1) - 1
        spawn = free & (free_rank < requested_count[:, None])
        source_slot = torch.searchsorted(
            requested_prefix.contiguous(),
            (free_rank + 1).clamp_min(1),
            right=False,
        ).clamp(0, self.enemy_slots - 1)
        source_x = self.enemy_x.gather(1, source_slot)
        source_y = self.enemy_y.gather(1, source_slot)
        source_z = self.enemy_z.gather(1, source_slot)
        source_dx = dx.gather(1, source_slot)
        source_dy = dy.gather(1, source_slot)
        target_z = self._enemy_target_geometry()[2].gather(1, source_slot)
        dz = target_z - source_z
        dx_fixed = torch.round(source_dx * _FIXED_UNIT).to(torch.int64)
        dy_fixed = torch.round(source_dy * _FIXED_UNIT).to(torch.int64)
        dz_fixed = torch.round(dz * _FIXED_UNIT).to(torch.int64)
        aim_norm = torch.sqrt(
            dx_fixed.to(torch.float32) * dx_fixed.to(torch.float32)
            + dy_fixed.to(torch.float32) * dy_fixed.to(torch.float32)
            + dz_fixed.to(torch.float32) * dz_fixed.to(torch.float32)
        ).clamp_min_(1.0)
        speed_fixed = _ENEMY_PROJECTILE_SPEED * _FIXED_UNIT
        velocity_x_fixed = torch.trunc(
            dx_fixed.to(torch.float32) / aim_norm * speed_fixed
        ).to(torch.int64)
        velocity_y_fixed = torch.trunc(
            dy_fixed.to(torch.float32) / aim_norm * speed_fixed
        ).to(torch.int64)
        velocity_z_fixed = torch.trunc(
            dz_fixed.to(torch.float32) / aim_norm * speed_fixed
        ).to(torch.int64)
        velocity_x = velocity_x_fixed.to(torch.float32) / _FIXED_UNIT
        velocity_y = velocity_y_fixed.to(torch.float32) / _FIXED_UNIT
        velocity_z = velocity_z_fixed.to(torch.float32) / _FIXED_UNIT
        self.enemy_projectile_x.copy_(
            torch.where(
                spawn,
                source_x
                + (velocity_x_fixed >> 1).to(torch.float32) / _FIXED_UNIT,
                self.enemy_projectile_x,
            )
        )
        self.enemy_projectile_y.copy_(
            torch.where(
                spawn,
                source_y
                + (velocity_y_fixed >> 1).to(torch.float32) / _FIXED_UNIT,
                self.enemy_projectile_y,
            )
        )
        self.enemy_projectile_z.copy_(
            torch.where(
                spawn,
                source_z
                + 32.0
                + (velocity_z_fixed >> 1).to(torch.float32) / _FIXED_UNIT,
                self.enemy_projectile_z,
            )
        )
        self.enemy_projectile_velocity_x.copy_(
            torch.where(
                spawn,
                velocity_x,
                self.enemy_projectile_velocity_x,
            )
        )
        self.enemy_projectile_velocity_y.copy_(
            torch.where(
                spawn,
                velocity_y,
                self.enemy_projectile_velocity_y,
            )
        )
        self.enemy_projectile_velocity_z.copy_(
            torch.where(
                spawn,
                velocity_z,
                self.enemy_projectile_velocity_z,
            )
        )
        self.enemy_projectile_age.copy_(
            torch.where(
                spawn, torch.zeros_like(self.enemy_projectile_age), self.enemy_projectile_age
            )
        )
        self.enemy_projectile_source_slot.copy_(
            torch.where(
                spawn,
                source_slot,
                self.enemy_projectile_source_slot,
            )
        )
        self.enemy_projectile_alive |= spawn

    def _enemy_projectile_tick(self, active: torch.Tensor) -> None:
        self.enemy_projectile_impact_tics.copy_(
            torch.where(
                active[:, None],
                torch.clamp_min(self.enemy_projectile_impact_tics - 1, 0),
                self.enemy_projectile_impact_tics,
            )
        )
        alive = self.enemy_projectile_alive & active[:, None]
        projectile_radius = torch.full_like(self.enemy_projectile_x, 6.0)
        dominant_speed = torch.maximum(
            self.enemy_projectile_velocity_x.abs(),
            self.enemy_projectile_velocity_y.abs(),
        )
        movement_steps = torch.where(
            dominant_speed > 5.0,
            1 + torch.floor(dominant_speed / 5.0).to(torch.int32),
            torch.ones_like(self.enemy_projectile_age),
        )
        start_x = self.enemy_projectile_x.clone()
        start_y = self.enemy_projectile_y.clone()
        current_x = start_x.clone()
        current_y = start_y.clone()
        current_z = self.enemy_projectile_z.clone()
        moving = alive.clone()
        impact = torch.zeros_like(alive)
        player_impact = torch.zeros_like(alive)
        doll_impact = torch.zeros_like(alive)
        enemy_impact = torch.zeros_like(alive)
        nearest_enemy = torch.zeros_like(
            self.enemy_projectile_age,
            dtype=torch.int64,
        )
        solid_enemy_type = self._effective_enemy_type()
        solid_enemy = self._enemy_solid_mask()
        projectile_slot = torch.arange(
            self.enemy_projectile_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, :]
        source_slot = torch.where(
            self.enemy_projectile_source_slot >= 0,
            self.enemy_projectile_source_slot,
            projectile_slot,
        ).clamp(0, self.enemy_slots - 1)
        enemy_slot = torch.arange(
            self.enemy_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, None, :]
        not_source = enemy_slot != source_slot[:, :, None]
        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            doll_x = self.map.player_starts[:-1, 0]
            doll_y = self.map.player_starts[:-1, 1]
            doll_z = self._player_start_z[:-1]
        # BaronBall has radius 6 and speed 15, so cardinal movement requires
        # four fixed-size subdivisions in the reference P_XYMovement path.
        for step in range(1, 5):
            enabled = moving & (movement_steps >= step)
            fraction = step / movement_steps.clamp_min(1).to(torch.float32)
            candidate_x = start_x + self.enemy_projectile_velocity_x * fraction
            candidate_y = start_y + self.enemy_projectile_velocity_y * fraction
            wall_impact = enabled & self._points_collide(
                candidate_x,
                candidate_y,
                projectile_radius,
            )
            sector = self._sector_at(candidate_x.reshape(-1), candidate_y.reshape(-1)).reshape_as(
                candidate_x
            )
            floor = self.map.sector_heights[sector, 0]
            ceiling = self.map.sector_heights[sector, 1]
            opening_impact = enabled & ((current_z < floor) | (current_z + 16.0 > ceiling))
            player_dx = candidate_x - self.x[:, None]
            player_dy = candidate_y - self.y[:, None]
            player_distance = torch.sqrt(
                player_dx * player_dx + player_dy * player_dy
            )
            player_vertical_overlap = self._vertical_overlap(
                current_z,
                16.0,
                self.z[:, None],
                _PLAYER_HEIGHT,
            )
            step_player_impact = (
                enabled
                & player_vertical_overlap
                & (player_dx.abs() < 22.0)
                & (player_dy.abs() < 22.0)
            )
            if doll_count:
                doll_dx = candidate_x[:, :, None] - doll_x[None, None, :]
                doll_dy = candidate_y[:, :, None] - doll_y[None, None, :]
                doll_distance = torch.sqrt(doll_dx * doll_dx + doll_dy * doll_dy)
                doll_overlap = self._vertical_overlap(
                    current_z[:, :, None],
                    16.0,
                    doll_z[None, None, :],
                    _PLAYER_HEIGHT,
                )
                doll_candidate = (
                    enabled[:, :, None]
                    & ~self.player_dead[:, None, None]
                    & doll_overlap
                    & (doll_dx.abs() < 22.0)
                    & (doll_dy.abs() < 22.0)
                )
                nearest_doll_distance = torch.amin(
                    torch.where(
                        doll_candidate,
                        doll_distance,
                        torch.full_like(doll_distance, torch.inf),
                    ),
                    dim=2,
                )
                step_doll_impact = torch.isfinite(nearest_doll_distance) & (
                    ~step_player_impact
                    | (nearest_doll_distance < player_distance)
                )
                step_player_impact &= ~step_doll_impact
            else:
                step_doll_impact = torch.zeros_like(step_player_impact)
                nearest_doll_distance = torch.full_like(player_distance, torch.inf)

            enemy_dx = candidate_x[:, :, None] - self.enemy_x[:, None, :]
            enemy_dy = candidate_y[:, :, None] - self.enemy_y[:, None, :]
            enemy_distance = torch.sqrt(enemy_dx * enemy_dx + enemy_dy * enemy_dy)
            enemy_overlap = self._vertical_overlap(
                current_z[:, :, None],
                16.0,
                self.enemy_z[:, None, :],
                self._effective_enemy_height()[:, None, :],
            )
            enemy_candidate = (
                enabled[:, :, None]
                & solid_enemy[:, None, :]
                & not_source
                & enemy_overlap
                & (
                    enemy_dx.abs()
                    < 6.0 + self._enemy_radius[solid_enemy_type][:, None, :]
                )
                & (
                    enemy_dy.abs()
                    < 6.0 + self._enemy_radius[solid_enemy_type][:, None, :]
                )
            )
            nearest_enemy_distance, step_nearest_enemy = torch.min(
                torch.where(
                    enemy_candidate,
                    enemy_distance,
                    torch.full_like(enemy_distance, torch.inf),
                ),
                dim=2,
            )
            nearest_player_actor_distance = torch.where(
                step_player_impact,
                player_distance,
                torch.where(
                    step_doll_impact,
                    nearest_doll_distance,
                    torch.full_like(player_distance, torch.inf),
                ),
            )
            step_enemy_impact = torch.isfinite(nearest_enemy_distance) & (
                nearest_enemy_distance < nearest_player_actor_distance
            )
            step_player_impact &= ~step_enemy_impact
            step_doll_impact &= ~step_enemy_impact
            step_actor_impact = (
                step_player_impact | step_doll_impact | step_enemy_impact
            )
            step_impact = enabled & (wall_impact | opening_impact | step_actor_impact)
            successful = enabled & ~step_impact
            current_x.copy_(torch.where(successful, candidate_x, current_x))
            current_y.copy_(torch.where(successful, candidate_y, current_y))
            player_impact |= step_impact & step_player_impact
            doll_impact |= step_impact & step_doll_impact
            nearest_enemy.copy_(
                torch.where(
                    step_impact & step_enemy_impact,
                    step_nearest_enemy,
                    nearest_enemy,
                )
            )
            enemy_impact |= step_impact & step_enemy_impact
            impact |= step_impact
            moving &= ~step_impact

        next_z = current_z + self.enemy_projectile_velocity_z
        sector = self._sector_at(current_x.reshape(-1), current_y.reshape(-1)).reshape_as(current_x)
        floor = self.map.sector_heights[sector, 0]
        ceiling = self.map.sector_heights[sector, 1]
        plane_impact = moving & ((next_z < floor) | (next_z + 16.0 > ceiling))
        clipped_next_z = torch.where(
            next_z < floor,
            floor,
            torch.where(next_z + 16.0 > ceiling, ceiling - 16.0, next_z),
        )
        current_z.copy_(torch.where(moving, clipped_next_z, current_z))
        impact |= plane_impact
        random_bits = self._random_u32(torch.any(impact, dim=1))[:, None]
        slot_bits = torch.arange(
            self.enemy_projectile_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, :]
        mixed = random_bits ^ (slot_bits * _HASH_GOLDEN_RATIO_SIGNED)
        mixed ^= mixed >> 16
        damage = (torch.remainder(mixed, 8).to(torch.float32) + 1) * 8.0
        damage_by_projectile = torch.where(
            player_impact | doll_impact,
            damage,
            torch.zeros_like(damage),
        )
        incoming = torch.sum(damage_by_projectile, dim=1)
        damaging_slot = torch.argmax(
            damage_by_projectile,
            dim=1,
        )
        thrust_x_by_projectile, thrust_y_by_projectile = (
            self._player_damage_thrust_components(
                torch.where(
                    player_impact,
                    damage_by_projectile,
                    torch.zeros_like(damage_by_projectile),
                ),
                current_x,
                current_y,
            )
        )
        armor_absorb_request = torch.sum(
            torch.floor(
                damage_by_projectile * self.armor_save_fraction[:, None]
            ),
            dim=1,
        )
        row = torch.arange(self.num_envs, device=self.device)
        self._apply_player_damage(
            incoming,
            current_x[row, damaging_slot],
            current_y[row, damaging_slot],
            thrust_x_fixed=torch.sum(thrust_x_by_projectile, dim=1),
            thrust_y_fixed=torch.sum(thrust_y_by_projectile, dim=1),
            armor_absorb_request=armor_absorb_request,
        )
        target_enemy_type = solid_enemy_type.gather(1, nearest_enemy)
        live_enemy_impact = enemy_impact & self.enemy_alive.gather(
            1,
            nearest_enemy,
        )
        enemy_damage_by_projectile = torch.where(
            live_enemy_impact & (target_enemy_type != 5),
            damage,
            torch.zeros_like(damage),
        )
        damage_by_projectile_enemy = torch.zeros(
            (
                self.num_envs,
                self.enemy_projectile_slots,
                self.enemy_slots,
            ),
            device=self.device,
        )
        damage_by_projectile_enemy.scatter_add_(
            2,
            nearest_enemy[:, :, None],
            enemy_damage_by_projectile[:, :, None],
        )
        enemy_thrust_x, enemy_thrust_y = self._enemy_damage_thrust_components(
            damage_by_projectile_enemy,
            current_x[:, :, None],
            current_y[:, :, None],
        )
        monster_damage_by_source = torch.zeros(
            (
                self.num_envs,
                self.enemy_slots,
                self.enemy_slots,
            ),
            device=self.device,
        )
        monster_damage_by_source.scatter_add_(
            1,
            source_slot[:, :, None].expand_as(damage_by_projectile_enemy),
            damage_by_projectile_enemy,
        )
        self._apply_enemy_damage(
            torch.sum(monster_damage_by_source, dim=1),
            thrust_x_fixed=torch.sum(enemy_thrust_x, dim=1),
            thrust_y_fixed=torch.sum(enemy_thrust_y, dim=1),
            credit_player=False,
            attacker_is_player=False,
            monster_damage_by_source=monster_damage_by_source,
        )
        self.enemy_projectile_x.copy_(
            torch.where(alive, current_x, self.enemy_projectile_x)
        )
        self.enemy_projectile_y.copy_(
            torch.where(alive, current_y, self.enemy_projectile_y)
        )
        self.enemy_projectile_z.copy_(
            torch.where(alive, current_z, self.enemy_projectile_z)
        )
        self.enemy_projectile_age.add_(alive.to(torch.int32))
        self.enemy_projectile_impact_tics.copy_(
            torch.where(
                impact,
                self.map.projectile_explosion_total_tics[2].to(torch.int32),
                self.enemy_projectile_impact_tics,
            )
        )
        self.enemy_projectile_alive &= ~impact

    def _move_enemy_thrust(self, active: torch.Tensor) -> None:
        visible_x = self._enemy_x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_y = self._enemy_y_fixed.to(torch.float32) / _FIXED_UNIT
        visible_z = self._enemy_z_fixed.to(torch.float32) / _FIXED_UNIT
        position_resynchronized = (self.enemy_x != visible_x) | (
            self.enemy_y != visible_y
        )
        self._enemy_x_fixed.copy_(
            torch.where(
                self.enemy_x != visible_x,
                torch.round(self.enemy_x * _FIXED_UNIT).to(torch.int64),
                self._enemy_x_fixed,
            )
        )
        self._enemy_y_fixed.copy_(
            torch.where(
                self.enemy_y != visible_y,
                torch.round(self.enemy_y * _FIXED_UNIT).to(torch.int64),
                self._enemy_y_fixed,
            )
        )
        self._enemy_z_fixed.copy_(
            torch.where(
                self.enemy_z != visible_z,
                torch.round(self.enemy_z * _FIXED_UNIT).to(torch.int64),
                self._enemy_z_fixed,
            )
        )
        actor_exists = active[:, None] & (
            self.enemy_alive
            | ((self.enemy_death_type >= 0) & (self.enemy_death_tics > 0))
        )
        actor_type = torch.where(
            self.enemy_type >= 0,
            self.enemy_type,
            self.enemy_death_type,
        ).clamp_min(0)
        actor_height = torch.where(
            self.enemy_alive,
            self._enemy_height[actor_type],
            self._enemy_height[actor_type] * 0.25,
        )
        if self.debug_checks:
            opening_resynchronized = actor_exists & (
                position_resynchronized | ~self._enemy_opening_initialized
            )
            current_floor, current_ceiling = self._actor_opening_at(
                self.enemy_x,
                self.enemy_y,
                self._enemy_radius[actor_type],
            )
            self._enemy_floor_z_fixed.copy_(
                torch.where(
                    opening_resynchronized,
                    torch.round(current_floor * _FIXED_UNIT).to(torch.int64),
                    self._enemy_floor_z_fixed,
                )
            )
            self._enemy_ceiling_z_fixed.copy_(
                torch.where(
                    opening_resynchronized,
                    torch.round(current_ceiling * _FIXED_UNIT).to(torch.int64),
                    self._enemy_ceiling_z_fixed,
                )
            )
            self._enemy_opening_initialized |= opening_resynchronized
        old_floor_z_fixed = self._enemy_floor_z_fixed.clone()
        proposed_x_fixed = self._enemy_x_fixed + torch.where(
            actor_exists,
            self._enemy_momentum_x_fixed,
            torch.zeros_like(self._enemy_momentum_x_fixed),
        )
        proposed_y_fixed = self._enemy_y_fixed + torch.where(
            actor_exists,
            self._enemy_momentum_y_fixed,
            torch.zeros_like(self._enemy_momentum_y_fixed),
        )
        proposed_x = proposed_x_fixed.to(torch.float32) / _FIXED_UNIT
        proposed_y = proposed_y_fixed.to(torch.float32) / _FIXED_UNIT
        horizontal_motion = actor_exists & (
            (self._enemy_momentum_x_fixed != 0)
            | (self._enemy_momentum_y_fixed != 0)
        )
        collision = horizontal_motion & self._enemy_collides(
            proposed_x,
            proposed_y,
            actor_type,
            allow_dropoff=True,
            actor_height=actor_height,
        )
        moved = horizontal_motion & ~collision
        self._enemy_x_fixed.copy_(
            torch.where(moved, proposed_x_fixed, self._enemy_x_fixed)
        )
        self._enemy_y_fixed.copy_(
            torch.where(moved, proposed_y_fixed, self._enemy_y_fixed)
        )
        self.enemy_x.copy_(self._enemy_x_fixed.to(torch.float32) / _FIXED_UNIT)
        self.enemy_y.copy_(self._enemy_y_fixed.to(torch.float32) / _FIXED_UNIT)
        actor_floor, actor_ceiling = self._actor_opening_at(
            self.enemy_x,
            self.enemy_y,
            self._enemy_radius[actor_type],
        )
        self._enemy_floor_z_fixed.copy_(
            torch.where(
                moved,
                torch.round(actor_floor * _FIXED_UNIT).to(torch.int64),
                self._enemy_floor_z_fixed,
            )
        )
        self._enemy_ceiling_z_fixed.copy_(
            torch.where(
                moved,
                torch.round(actor_ceiling * _FIXED_UNIT).to(torch.int64),
                self._enemy_ceiling_z_fixed,
            )
        )
        floor_z_fixed = self._enemy_floor_z_fixed
        ceiling_z_fixed = self._enemy_ceiling_z_fixed
        actor_height_fixed = torch.round(
            actor_height * _FIXED_UNIT
        ).to(torch.int64)
        proposed_z_fixed = self._enemy_z_fixed + torch.where(
            actor_exists,
            self._enemy_velocity_z_fixed,
            torch.zeros_like(self._enemy_velocity_z_fixed),
        )
        above_floor = proposed_z_fixed > floor_z_fixed
        walked_off_ledge = (
            (self._enemy_velocity_z_fixed == 0)
            & (old_floor_z_fixed > floor_z_fixed)
            & (proposed_z_fixed == old_floor_z_fixed)
        )
        gravity_fixed = torch.where(
            walked_off_ledge,
            torch.full_like(self._enemy_velocity_z_fixed, 2 * _FIXED_UNIT),
            torch.full_like(self._enemy_velocity_z_fixed, _FIXED_UNIT),
        )
        next_velocity_z = torch.where(
            above_floor,
            self._enemy_velocity_z_fixed - gravity_fixed,
            self._enemy_velocity_z_fixed,
        )
        hit_floor = proposed_z_fixed <= floor_z_fixed
        ceiling_limit_fixed = ceiling_z_fixed - actor_height_fixed
        hit_ceiling = proposed_z_fixed > ceiling_limit_fixed
        clipped_z_fixed = torch.minimum(
            torch.maximum(proposed_z_fixed, floor_z_fixed),
            ceiling_limit_fixed,
        )
        next_velocity_z = torch.where(
            hit_floor & (next_velocity_z < 0),
            torch.zeros_like(next_velocity_z),
            next_velocity_z,
        )
        next_velocity_z = torch.where(
            hit_ceiling & (next_velocity_z > 0),
            torch.zeros_like(next_velocity_z),
            next_velocity_z,
        )
        self._enemy_z_fixed.copy_(
            torch.where(actor_exists, clipped_z_fixed, self._enemy_z_fixed)
        )
        self._enemy_velocity_z_fixed.copy_(
            torch.where(
                actor_exists,
                next_velocity_z,
                torch.zeros_like(next_velocity_z),
            )
        )
        self.enemy_z.copy_(self._enemy_z_fixed.to(torch.float32) / _FIXED_UNIT)

        retained_x = torch.where(
            moved,
            self._enemy_momentum_x_fixed,
            torch.zeros_like(self._enemy_momentum_x_fixed),
        )
        retained_y = torch.where(
            moved,
            self._enemy_momentum_y_fixed,
            torch.zeros_like(self._enemy_momentum_y_fixed),
        )
        stopped = (
            (retained_x > -_ACTOR_STOP_SPEED_FIXED)
            & (retained_x < _ACTOR_STOP_SPEED_FIXED)
            & (retained_y > -_ACTOR_STOP_SPEED_FIXED)
            & (retained_y < _ACTOR_STOP_SPEED_FIXED)
        )
        next_x = torch.where(
            stopped,
            torch.zeros_like(retained_x),
            retained_x * _PLAYER_FRICTION_FIXED >> 16,
        )
        next_y = torch.where(
            stopped,
            torch.zeros_like(retained_y),
            retained_y * _PLAYER_FRICTION_FIXED >> 16,
        )
        self._enemy_momentum_x_fixed.copy_(
            torch.where(actor_exists, next_x, torch.zeros_like(next_x))
        )
        self._enemy_momentum_y_fixed.copy_(
            torch.where(actor_exists, next_y, torch.zeros_like(next_y))
        )

    def _try_enemy_chase_step(
        self,
        requested: torch.Tensor,
        direction: torch.Tensor,
        enemy_type: torch.Tensor,
    ) -> torch.Tensor:
        """Attempt one P_Move-style fixed-point step in a discrete direction."""

        valid_direction = direction < 8
        safe_direction = direction.clamp(0, 7)
        delta_x_fixed = self._enemy_chase_step_x_fixed[
            enemy_type,
            safe_direction,
        ]
        delta_y_fixed = self._enemy_chase_step_y_fixed[
            enemy_type,
            safe_direction,
        ]
        proposed_x_fixed = self._enemy_x_fixed + delta_x_fixed
        proposed_y_fixed = self._enemy_y_fixed + delta_y_fixed
        proposed_x = proposed_x_fixed.to(torch.float32) / _FIXED_UNIT
        proposed_y = proposed_y_fixed.to(torch.float32) / _FIXED_UNIT
        collision = self._enemy_collides(
            proposed_x,
            proposed_y,
            enemy_type,
        )
        moved = requested & valid_direction & ~collision
        self._enemy_x_fixed.copy_(
            torch.where(moved, proposed_x_fixed, self._enemy_x_fixed)
        )
        self._enemy_y_fixed.copy_(
            torch.where(moved, proposed_y_fixed, self._enemy_y_fixed)
        )
        self.enemy_x.copy_(self._enemy_x_fixed.to(torch.float32) / _FIXED_UNIT)
        self.enemy_y.copy_(self._enemy_y_fixed.to(torch.float32) / _FIXED_UNIT)
        self.enemy_move_direction.copy_(
            torch.where(requested, direction, self.enemy_move_direction)
        )
        return moved

    def _enemy_chase_move(
        self,
        requested: torch.Tensor,
        enemy_type: torch.Tensor,
        delta_x: torch.Tensor,
        delta_y: torch.Tensor,
    ) -> torch.Tensor:
        """Run P_Move/P_NewChaseDir for every due monster."""

        decremented_count = self.enemy_move_count - 1
        self.enemy_move_count.copy_(
            torch.where(requested, decremented_count, self.enemy_move_count)
        )
        current_direction = self.enemy_move_direction.clone()
        keep_direction = requested & (decremented_count >= 0)
        moved = self._try_enemy_chase_step(
            keep_direction,
            current_direction,
            enemy_type,
        )
        reroute = requested & ~moved
        old_direction = current_direction
        turnaround = self._enemy_opposite_direction[old_direction.clamp(0, 8)]

        lane_reroute = torch.any(reroute, dim=1)
        swap_roll = self._enemy_chase_random(lane_reroute)
        search_roll = self._enemy_chase_random(lane_reroute)
        walk_roll = self._enemy_chase_random(lane_reroute).to(torch.int32)

        east_west = torch.where(
            delta_x > 10.0,
            torch.zeros_like(old_direction),
            torch.where(
                delta_x < -10.0,
                torch.full_like(old_direction, 4),
                torch.full_like(old_direction, 8),
            ),
        )
        north_south = torch.where(
            delta_y < -10.0,
            torch.full_like(old_direction, 6),
            torch.where(
                delta_y > 10.0,
                torch.full_like(old_direction, 2),
                torch.full_like(old_direction, 8),
            ),
        )
        diagonal_index = (
            (delta_y < 0).to(torch.int64) * 2
            + (delta_x > 0).to(torch.int64)
        )
        diagonal = self._enemy_diagonal_direction[diagonal_index]

        remaining = reroute
        rerouted = torch.zeros_like(requested)

        def attempt(candidate: torch.Tensor, allowed: torch.Tensor) -> None:
            nonlocal remaining, rerouted
            request = remaining & allowed & (candidate < 8)
            success = self._try_enemy_chase_step(request, candidate, enemy_type)
            rerouted |= success
            remaining &= ~success

        attempt(
            diagonal,
            (east_west < 8) & (north_south < 8) & (diagonal != turnaround),
        )

        swap_axes = (swap_roll > 200) | (delta_y.abs() > delta_x.abs())
        first = torch.where(swap_axes, north_south, east_west)
        second = torch.where(swap_axes, east_west, north_south)
        first = torch.where(first == turnaround, torch.full_like(first, 8), first)
        second = torch.where(second == turnaround, torch.full_like(second, 8), second)
        attempt(first, torch.ones_like(requested))
        attempt(second, torch.ones_like(requested))
        attempt(
            old_direction,
            (old_direction < 8) & (old_direction != turnaround),
        )

        ascending = torch.bitwise_and(search_roll, 1).bool()
        for search_index in range(8):
            candidate = torch.where(
                ascending,
                torch.full_like(old_direction, search_index),
                torch.full_like(old_direction, 7 - search_index),
            )
            attempt(candidate, candidate != turnaround)
        attempt(turnaround, turnaround < 8)

        self.enemy_move_direction.masked_fill_(remaining, 8)
        self.enemy_move_count.copy_(
            torch.where(
                rerouted,
                torch.bitwise_and(walk_roll, 15),
                torch.where(
                    remaining,
                    torch.zeros_like(self.enemy_move_count),
                    self.enemy_move_count,
                ),
            )
        )
        return moved | rerouted

    def _enemy_tick(self, active: torch.Tensor | None = None) -> None:
        if active is None:
            active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._move_enemy_thrust(active)
        in_pain = self.enemy_alive & active[:, None] & (self.enemy_pain_tics > 0)
        self.enemy_pain_tics.copy_(
            torch.where(
                active[:, None],
                torch.clamp_min(self.enemy_pain_tics - 1, 0),
                self.enemy_pain_tics,
            )
        )
        available = self.enemy_alive & active[:, None] & ~in_pain
        enemy_type = self.enemy_type.clamp_min(0)
        target_is_monster = self.enemy_target_slot >= 0
        safe_target = self.enemy_target_slot.clamp(0, self.enemy_slots - 1)
        lost_target = target_is_monster & ~self.enemy_alive.gather(1, safe_target)
        self.enemy_target_slot.masked_fill_(lost_target, -1)
        self.enemy_target_threshold.masked_fill_(lost_target, 0)
        idle = available & (self.enemy_target_slot < -1)
        look_ready = idle & (self.enemy_move_cooldown <= 0)
        player_dx = self.x[:, None] - self.enemy_x
        player_dy = self.y[:, None] - self.enemy_y
        player_dx_fixed = (self._x_fixed[:, None] - self._enemy_x_fixed).abs()
        player_dy_fixed = (self._y_fixed[:, None] - self._enemy_y_fixed).abs()
        player_approximate_distance_fixed = (
            player_dx_fixed
            + player_dy_fixed
            - (torch.minimum(player_dx_fixed, player_dy_fixed) >> 1)
        )
        player_direction = torch.atan2(player_dy, player_dx)
        relative_player_direction = self._wrap_angle(
            player_direction - self.enemy_angle
        )
        in_front = (relative_player_direction.abs() <= math.pi / 2.0) | (
            player_approximate_distance_fixed <= 64 * _FIXED_UNIT
        )
        sees_player = (
            ~self.player_dead[:, None]
            & in_front
            & ~self._sight_blocked(
                self.enemy_x,
                self.enemy_y,
                self.enemy_z + self._enemy_height[enemy_type] * 0.75,
                self.x[:, None],
                self.y[:, None],
                self.z[:, None],
                torch.full_like(self.z[:, None], _PLAYER_HEIGHT),
            )
        )
        wakes = look_ready & (self.enemy_heard_player | sees_player)
        self.enemy_target_slot.masked_fill_(wakes, -1)
        self.enemy_target_threshold.masked_fill_(wakes, 0)
        self.enemy_heard_player.masked_fill_(wakes, False)
        self.enemy_move_cooldown.copy_(
            torch.where(
                wakes,
                torch.zeros_like(self.enemy_move_cooldown),
                torch.where(
                    look_ready,
                    self._enemy_look_interval[enemy_type],
                    self.enemy_move_cooldown,
                ),
            )
        )
        remaining_idle = idle & ~wakes
        self.enemy_animation_tics.copy_(
            torch.where(
                remaining_idle,
                self.enemy_animation_tics + 1,
                torch.where(
                    wakes,
                    torch.zeros_like(self.enemy_animation_tics),
                    self.enemy_animation_tics,
                ),
            )
        )
        alive = available & (self.enemy_target_slot >= -1)
        (
            target_x,
            target_y,
            target_z,
            target_height,
            target_radius,
            target_is_monster,
            target_alive,
        ) = self._enemy_target_geometry()
        dx = target_x - self.enemy_x
        dy = target_y - self.enemy_y
        distance = torch.sqrt(dx * dx + dy * dy).clamp_min_(1e-4)
        target_x_fixed = torch.where(
            target_is_monster,
            self._enemy_x_fixed.gather(1, safe_target),
            self._x_fixed[:, None],
        )
        target_y_fixed = torch.where(
            target_is_monster,
            self._enemy_y_fixed.gather(1, safe_target),
            self._y_fixed[:, None],
        )
        melee_dx_fixed = (target_x_fixed - self._enemy_x_fixed).abs()
        melee_dy_fixed = (target_y_fixed - self._enemy_y_fixed).abs()
        melee_minimum_fixed = torch.minimum(melee_dx_fixed, melee_dy_fixed)
        approximate_distance_fixed = (
            melee_dx_fixed + melee_dy_fixed - (melee_minimum_fixed >> 1)
        )
        melee_limit_fixed = torch.round(
            (44.0 + target_radius) * _FIXED_UNIT
        ).to(torch.int64)
        in_melee_range = approximate_distance_fixed < melee_limit_fixed
        line_of_sight = ~self._sight_blocked(
            self.enemy_x,
            self.enemy_y,
            self.enemy_z + self._enemy_height[enemy_type] * 0.75,
            target_x,
            target_y,
            target_z,
            target_height,
        )
        line_of_sight &= target_alive
        shoot_z = self.enemy_z + 36.0
        target_bottom_slope = (target_z - shoot_z) / distance
        target_top_slope = (target_z + target_height - shoot_z) / distance
        max_autoaim_slope = math.tan(35.0 * math.pi / 180.0)
        visible = line_of_sight & (target_top_slope >= -max_autoaim_slope) & (
            target_bottom_slope <= max_autoaim_slope
        )
        melee_vertical_overlap = self._vertical_overlap(
            self.enemy_z,
            self._enemy_height[enemy_type],
            target_z,
            target_height,
        )
        melee_target = line_of_sight & melee_vertical_overlap & in_melee_range
        target_bam_angle = self._doom_bam_angle(
            target_x_fixed - self._enemy_x_fixed,
            target_y_fixed - self._enemy_y_fixed,
        )

        attack_phase = self.enemy_attack_phase
        phase_due = alive & (attack_phase > 0) & (self.enemy_cooldown <= 1)
        phase_one_due = phase_due & (attack_phase == 1)
        phase_two_due = phase_due & (attack_phase == 2)
        phase_three_due = phase_due & (attack_phase == 3)
        phase_four_due = phase_due & (attack_phase == 4)
        finite_type = (enemy_type == 0) | (enemy_type == 1) | (enemy_type == 4) | (enemy_type == 5)
        chainsaw_target = melee_target
        chainsaw_action = phase_one_due & (enemy_type == 2)
        second_prefire_face = (
            alive
            & (attack_phase == 1)
            & ((enemy_type == 4) | (enemy_type == 5))
            & (self.enemy_cooldown == 9)
        )
        finite_fire = phase_one_due & finite_type
        chainsaw_fire = chainsaw_action & chainsaw_target
        chaingun_fire = (enemy_type == 3) & (
            phase_one_due | phase_two_due | phase_four_due
        )
        chaingun_refire = phase_three_due & (enemy_type == 3)
        random_refire = self._enemy_chaingun_refire_decision(chaingun_refire)
        chaingun_continue = chaingun_refire & (random_refire | line_of_sight)
        fire_event = finite_fire | chainsaw_fire | chaingun_fire
        knight_projectile = fire_event & (enemy_type == 5) & ~melee_target
        self._spawn_enemy_projectiles(knight_projectile, dx, dy)
        hitscan_type = (enemy_type == 0) | (enemy_type == 1) | (enemy_type == 3)
        direct_attack = fire_event & ~knight_projectile
        direct_attack &= torch.where(hitscan_type, visible, melee_target)
        direct_attack &= ~hitscan_type | (
            distance < self._enemy_attack_range[enemy_type]
        )
        (
            hitscan_player_damage,
            hitscan_actual_player_damage,
            hitscan_enemy_damage,
        ) = self._enemy_hitscan_damage(
            enemy_type,
            direct_attack & hitscan_type,
            distance,
            visible,
            base_bam=target_bam_angle,
        )
        direct_damage_by_attacker = self._enemy_damage_roll(
            enemy_type,
            direct_attack & ~hitscan_type,
            in_melee_range,
        )
        direct_player_damage = torch.where(
            target_is_monster,
            torch.zeros_like(direct_damage_by_attacker),
            direct_damage_by_attacker,
        )
        direct_enemy_damage = torch.zeros(
            (
                self.num_envs,
                self.enemy_slots,
                self.enemy_slots,
            ),
            device=self.device,
        )
        direct_enemy_damage.scatter_add_(
            2,
            self.enemy_target_slot.clamp(0, self.enemy_slots - 1)[..., None],
            torch.where(
                target_is_monster,
                direct_damage_by_attacker,
                torch.zeros_like(direct_damage_by_attacker),
            )[..., None],
        )
        monster_damage_by_source = direct_enemy_damage + torch.sum(
            hitscan_enemy_damage,
            dim=2,
        )
        incoming = torch.sum(direct_player_damage, dim=1) + torch.sum(
            hitscan_player_damage,
            dim=(1, 2),
        )
        total_player_damage_by_attacker = direct_player_damage + torch.sum(
            hitscan_player_damage,
            dim=2,
        )
        damaging_slot = torch.argmax(total_player_damage_by_attacker, dim=1)
        melee_thrust_x, melee_thrust_y = self._player_damage_thrust_components(
            direct_player_damage,
            self.enemy_x,
            self.enemy_y,
        )
        hitscan_thrust_x, hitscan_thrust_y = self._player_damage_thrust_components(
            hitscan_actual_player_damage,
            self.enemy_x[:, :, None],
            self.enemy_y[:, :, None],
        )
        armor_absorb_request = (
            torch.sum(
                torch.floor(
                    direct_player_damage * self.armor_save_fraction[:, None]
                ),
                dim=1,
            )
            + torch.sum(
                torch.floor(
                    hitscan_player_damage
                    * self.armor_save_fraction[:, None, None]
                ),
                dim=(1, 2),
            )
        )
        row = torch.arange(self.num_envs, device=self.device)
        self._apply_player_damage(
            incoming,
            self.enemy_x[row, damaging_slot],
            self.enemy_y[row, damaging_slot],
            thrust_x_fixed=torch.sum(melee_thrust_x, dim=1)
            + torch.sum(hitscan_thrust_x, dim=(1, 2)),
            thrust_y_fixed=torch.sum(melee_thrust_y, dim=1)
            + torch.sum(hitscan_thrust_y, dim=(1, 2)),
            armor_absorb_request=armor_absorb_request,
        )
        monster_thrust_x, monster_thrust_y = self._enemy_damage_thrust_components(
            monster_damage_by_source,
            self.enemy_x[:, :, None],
            self.enemy_y[:, :, None],
        )
        self._apply_enemy_damage(
            torch.sum(monster_damage_by_source, dim=1),
            thrust_x_fixed=torch.sum(monster_thrust_x, dim=1),
            thrust_y_fixed=torch.sum(monster_thrust_y, dim=1),
            credit_player=False,
            attacker_is_player=False,
            monster_damage_by_source=monster_damage_by_source,
        )
        alive &= self.enemy_alive & (self.enemy_pain_tics <= 0)
        phase_one_due &= alive
        phase_two_due &= alive
        phase_three_due &= alive

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

        # Phase 1 is the initial E prefire. Phase 4 is the one-tic, non-BRIGHT
        # F gap left by A_CPosRefire; both enter the first bright F shot.
        chaingun_first = (phase_one_due | phase_four_due) & (enemy_type == 3)
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
            self._enemy_attack_recovery[enemy_type],
            next_cooldown,
        )
        # Phase 3 retains the second bright E shot for four tics. Its due
        # action is A_CPosRefire, not another attack.
        next_phase = torch.where(
            chaingun_refire,
            torch.where(
                chaingun_continue,
                torch.full_like(next_phase, 4),
                torch.zeros_like(next_phase),
            ),
            next_phase,
        )
        next_cooldown = torch.where(
            chaingun_refire,
            torch.where(
                chaingun_continue,
                torch.ones_like(next_cooldown),
                torch.zeros_like(next_cooldown),
            ),
            next_cooldown,
        )

        move_ready = alive & (next_phase == 0) & (self.enemy_move_cooldown <= 0)
        # Doom sets MF_JUSTATTACKED when a missile state is selected. The
        # first A_Chase after that state completes must choose and attempt a
        # fresh chase direction instead of checking for another attack.
        post_attack_chase = move_ready & self.enemy_just_attacked
        next_reaction_time = torch.where(
            move_ready,
            torch.clamp_min(self.enemy_reaction_time - 1, 0),
            self.enemy_reaction_time,
        )
        self.enemy_reaction_time.copy_(next_reaction_time)
        attack_ready = move_ready & (next_cooldown <= 0) & ~self.enemy_just_attacked
        turning = move_ready & (self.enemy_move_direction < 8)
        quantized_angle = torch.floor(
            torch.remainder(self.enemy_angle, 2 * math.pi) / (math.pi / 4)
        ) * (math.pi / 4)
        movement_direction_angle = self.enemy_move_direction.to(torch.float32) * (
            math.pi / 4
        )
        turn_delta = self._wrap_angle(quantized_angle - movement_direction_angle)
        turned_angle = torch.where(
            turn_delta > 0,
            quantized_angle - math.pi / 4,
            torch.where(
                turn_delta < 0,
                quantized_angle + math.pi / 4,
                quantized_angle,
            ),
        )
        self.enemy_angle.copy_(
            torch.where(turning, self._wrap_angle(turned_angle), self.enemy_angle)
        )
        melee_type = (enemy_type == 2) | (enemy_type == 4) | (enemy_type == 5)
        melee_attack = attack_ready & melee_type & melee_target
        ranged_type = (enemy_type == 0) | (enemy_type == 1) | (enemy_type == 3) | (enemy_type == 5)
        ranged_check = (
            attack_ready
            & line_of_sight
            & ranged_type
            & (self.enemy_move_count <= 0)
            & (distance < self._enemy_attack_range[enemy_type])
            & ~((enemy_type == 5) & melee_vertical_overlap & in_melee_range)
        )
        forced_retaliation = ranged_check & self.enemy_just_hit
        ranged_candidate = (
            ranged_check & (next_reaction_time <= 0) & ~self.enemy_just_hit
        )
        ranged_attack = forced_retaliation | self._enemy_missile_decision(
            enemy_type,
            ranged_candidate,
            dx,
            dy,
        )
        self.enemy_just_hit.masked_fill_(forced_retaliation, False)
        can_attack = melee_attack | ranged_attack
        moving = move_ready & ~post_attack_chase & ~can_attack & target_alive
        forced_post_attack_move = post_attack_chase & target_alive
        self.enemy_move_count.masked_fill_(forced_post_attack_move, 0)
        self._enemy_chase_move(
            moving | forced_post_attack_move,
            enemy_type,
            dx,
            dy,
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
        decrement_target_threshold = move_ready & (self.enemy_target_threshold > 0)
        self.enemy_target_threshold.copy_(
            torch.where(
                decrement_target_threshold,
                torch.clamp_min(self.enemy_target_threshold - 1, 0),
                self.enemy_target_threshold,
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
        self.enemy_just_attacked.copy_(
            torch.where(
                ranged_attack,
                torch.ones_like(self.enemy_just_attacked),
                torch.where(
                    post_attack_chase,
                    torch.zeros_like(self.enemy_just_attacked),
                    self.enemy_just_attacked,
                ),
            )
        )
        action_faces_target = (fire_event & (enemy_type != 5)) | chainsaw_action
        face_target = (
            can_attack
            | action_faces_target
            | chaingun_refire
            | second_prefire_face
        )
        chainsaw_hit = direct_attack & (enemy_type == 2)
        facing_bam_angle = torch.where(
            chainsaw_hit,
            (target_bam_angle + (_ANGLE_90 // 20)) & _UINT32_MASK,
            target_bam_angle,
        )
        target_angle = facing_bam_angle.to(torch.float32) * _BAM_TO_RADIANS
        self.enemy_angle.copy_(
            torch.where(
                face_target,
                target_angle,
                self.enemy_angle,
            )
        )
        returning_to_walk = alive & (attack_phase > 0) & (next_phase == 0)
        walking = alive & (next_phase == 0)
        next_animation_tics = torch.where(
            walking,
            self.enemy_animation_tics + 1,
            self.enemy_animation_tics,
        )
        self.enemy_animation_tics.copy_(
            torch.where(returning_to_walk | can_attack, 0, next_animation_tics)
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
        vertical_reach = (item_z - self.z[:, None] <= _PLAYER_HEIGHT) & (
            item_z - self.z[:, None] >= -_PICKUP_REACH_BELOW
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
        self.mugshot_grin |= newly_owned
        self.mugshot_grin_tics.copy_(
            torch.where(
                newly_owned,
                torch.full_like(self.mugshot_grin_tics, _MUGSHOT_GRIN_TICS),
                self.mugshot_grin_tics,
            )
        )
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
        self.bonus_count.copy_(
            torch.where(
                torch.any(consumed, dim=1),
                torch.full_like(self.bonus_count, 6),
                self.bonus_count,
            )
        )

    def _collect_drops(self) -> None:
        self.teleport_fog_tics.sub_(1).clamp_min_(0)
        # Doom death states hold their final frame forever (duration -1).
        # A value of one therefore means a persistent corpse, not one tic left.
        self.enemy_death_tics.copy_(
            torch.where(
                self.enemy_death_tics > 1,
                self.enemy_death_tics - 1,
                self.enemy_death_tics,
            )
        )
        corpse = self.enemy_death_type >= 0
        self.enemy_death_elapsed.copy_(
            torch.where(
                corpse,
                self.enemy_death_elapsed + 1,
                self.enemy_death_elapsed,
            )
        )
        self.drop_delay.sub_(1).clamp_min_(0)

        # Public coordinates are mutable for tests and advanced callers. Keep
        # their fixed-point mirrors authoritative only while they still match.
        visible_enemy_x = self._enemy_x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_enemy_y = self._enemy_y_fixed.to(torch.float32) / _FIXED_UNIT
        visible_enemy_z = self._enemy_z_fixed.to(torch.float32) / _FIXED_UNIT
        self._enemy_x_fixed.copy_(
            torch.where(
                self.enemy_x != visible_enemy_x,
                torch.round(self.enemy_x * _FIXED_UNIT).to(torch.int64),
                self._enemy_x_fixed,
            )
        )
        self._enemy_y_fixed.copy_(
            torch.where(
                self.enemy_y != visible_enemy_y,
                torch.round(self.enemy_y * _FIXED_UNIT).to(torch.int64),
                self._enemy_y_fixed,
            )
        )
        self._enemy_z_fixed.copy_(
            torch.where(
                self.enemy_z != visible_enemy_z,
                torch.round(self.enemy_z * _FIXED_UNIT).to(torch.int64),
                self._enemy_z_fixed,
            )
        )
        visible_drop_x = self._drop_x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_drop_y = self._drop_y_fixed.to(torch.float32) / _FIXED_UNIT
        visible_drop_z = self._drop_z_fixed.to(torch.float32) / _FIXED_UNIT
        self._drop_x_fixed.copy_(
            torch.where(
                self.drop_spawned & (self.drop_x != visible_drop_x),
                torch.round(self.drop_x * _FIXED_UNIT).to(torch.int64),
                self._drop_x_fixed,
            )
        )
        self._drop_y_fixed.copy_(
            torch.where(
                self.drop_spawned & (self.drop_y != visible_drop_y),
                torch.round(self.drop_y * _FIXED_UNIT).to(torch.int64),
                self._drop_y_fixed,
            )
        )
        self._drop_z_fixed.copy_(
            torch.where(
                self.drop_spawned & (self.drop_z != visible_drop_z),
                torch.round(self.drop_z * _FIXED_UNIT).to(torch.int64),
                self._drop_z_fixed,
            )
        )

        has_drop = self.drop_type >= 0
        spawn_drop = has_drop & ~self.drop_spawned & (self.drop_delay <= 0)
        spawn_lanes = torch.any(spawn_drop, dim=1)
        random_draws = torch.stack(
            [self._random_u32(spawn_lanes).clone() for _ in range(5)],
            dim=1,
        )
        slot = torch.arange(
            self.enemy_slots,
            device=self.device,
            dtype=torch.int64,
        )[None, None, :]
        mixed = random_draws[:, :, None] ^ (slot * _HASH_GOLDEN_RATIO_SIGNED)
        mixed = torch.bitwise_xor(mixed, mixed >> 16)
        random_bytes = torch.bitwise_and(mixed, 255)
        toss_x_fixed = (random_bytes[:, 0] - random_bytes[:, 1]) << 8
        toss_y_fixed = (random_bytes[:, 2] - random_bytes[:, 3]) << 8
        toss_z_fixed = 5 * _FIXED_UNIT + (random_bytes[:, 4] << 10)
        corpse_type = self.enemy_death_type.clamp(0, len(_ENEMY_HEIGHT) - 1)
        drop_spawn_z_fixed = self._enemy_z_fixed + torch.round(
            self._enemy_height[corpse_type] * (0.125 * _FIXED_UNIT)
        ).to(torch.int64)
        self._drop_x_fixed.copy_(
            torch.where(spawn_drop, self._enemy_x_fixed, self._drop_x_fixed)
        )
        self._drop_y_fixed.copy_(
            torch.where(spawn_drop, self._enemy_y_fixed, self._drop_y_fixed)
        )
        self._drop_z_fixed.copy_(
            torch.where(spawn_drop, drop_spawn_z_fixed, self._drop_z_fixed)
        )
        self._drop_velocity_x_fixed.copy_(
            torch.where(spawn_drop, toss_x_fixed, self._drop_velocity_x_fixed)
        )
        self._drop_velocity_y_fixed.copy_(
            torch.where(spawn_drop, toss_y_fixed, self._drop_velocity_y_fixed)
        )
        self._drop_velocity_z_fixed.copy_(
            torch.where(spawn_drop, toss_z_fixed, self._drop_velocity_z_fixed)
        )
        self.drop_spawned |= spawn_drop
        active_drop = has_drop & self.drop_spawned

        current_x = self._drop_x_fixed.to(torch.float32) / _FIXED_UNIT
        current_y = self._drop_y_fixed.to(torch.float32) / _FIXED_UNIT
        current_z = self._drop_z_fixed.to(torch.float32) / _FIXED_UNIT
        old_floor, _old_ceiling = self._actor_opening_at(
            current_x,
            current_y,
            _PICKUP_RADIUS,
        )
        old_floor_fixed = torch.round(old_floor * _FIXED_UNIT).to(torch.int64)
        grounded_before_move = self._drop_z_fixed <= old_floor_fixed
        proposed_x_fixed = self._drop_x_fixed + torch.where(
            active_drop,
            self._drop_velocity_x_fixed,
            torch.zeros_like(self._drop_velocity_x_fixed),
        )
        proposed_y_fixed = self._drop_y_fixed + torch.where(
            active_drop,
            self._drop_velocity_y_fixed,
            torch.zeros_like(self._drop_velocity_y_fixed),
        )
        proposed_x = proposed_x_fixed.to(torch.float32) / _FIXED_UNIT
        proposed_y = proposed_y_fixed.to(torch.float32) / _FIXED_UNIT
        proposed_floor, proposed_ceiling = self._actor_opening_at(
            proposed_x,
            proposed_y,
            _PICKUP_RADIUS,
        )
        horizontal_collision = active_drop & (
            self._points_collide(proposed_x, proposed_y, _PICKUP_RADIUS)
            | (proposed_floor > current_z + 24.0)
            | (
                proposed_ceiling - torch.maximum(current_z, proposed_floor)
                < _DROP_HEIGHT
            )
        )
        moved = active_drop & ~horizontal_collision
        self._drop_x_fixed.copy_(
            torch.where(moved, proposed_x_fixed, self._drop_x_fixed)
        )
        self._drop_y_fixed.copy_(
            torch.where(moved, proposed_y_fixed, self._drop_y_fixed)
        )
        retained_x = torch.where(
            moved,
            self._drop_velocity_x_fixed,
            torch.zeros_like(self._drop_velocity_x_fixed),
        )
        retained_y = torch.where(
            moved,
            self._drop_velocity_y_fixed,
            torch.zeros_like(self._drop_velocity_y_fixed),
        )
        stopped = (
            (retained_x > -_ACTOR_STOP_SPEED_FIXED)
            & (retained_x < _ACTOR_STOP_SPEED_FIXED)
            & (retained_y > -_ACTOR_STOP_SPEED_FIXED)
            & (retained_y < _ACTOR_STOP_SPEED_FIXED)
        )
        friction_x = torch.where(
            stopped,
            torch.zeros_like(retained_x),
            retained_x * _PLAYER_FRICTION_FIXED >> 16,
        )
        friction_y = torch.where(
            stopped,
            torch.zeros_like(retained_y),
            retained_y * _PLAYER_FRICTION_FIXED >> 16,
        )
        self._drop_velocity_x_fixed.copy_(
            torch.where(
                active_drop & grounded_before_move,
                friction_x,
                retained_x,
            )
        )
        self._drop_velocity_y_fixed.copy_(
            torch.where(
                active_drop & grounded_before_move,
                friction_y,
                retained_y,
            )
        )

        moved_x = self._drop_x_fixed.to(torch.float32) / _FIXED_UNIT
        moved_y = self._drop_y_fixed.to(torch.float32) / _FIXED_UNIT
        floor, ceiling = self._actor_opening_at(
            moved_x,
            moved_y,
            _PICKUP_RADIUS,
        )
        floor_fixed = torch.round(floor * _FIXED_UNIT).to(torch.int64)
        ceiling_fixed = torch.round(ceiling * _FIXED_UNIT).to(torch.int64)
        proposed_z_fixed = self._drop_z_fixed + torch.where(
            active_drop,
            self._drop_velocity_z_fixed,
            torch.zeros_like(self._drop_velocity_z_fixed),
        )
        above_floor = proposed_z_fixed > floor_fixed
        next_velocity_z = torch.where(
            above_floor,
            self._drop_velocity_z_fixed - _DROP_GRAVITY_FIXED,
            self._drop_velocity_z_fixed,
        )
        hit_floor = proposed_z_fixed <= floor_fixed
        ceiling_limit_fixed = ceiling_fixed - int(_DROP_HEIGHT * _FIXED_UNIT)
        hit_ceiling = proposed_z_fixed > ceiling_limit_fixed
        clipped_z_fixed = torch.minimum(
            torch.maximum(proposed_z_fixed, floor_fixed),
            ceiling_limit_fixed,
        )
        next_velocity_z = torch.where(
            hit_floor & (next_velocity_z < 0),
            torch.zeros_like(next_velocity_z),
            next_velocity_z,
        )
        next_velocity_z = torch.where(
            hit_ceiling & (next_velocity_z > 0),
            torch.zeros_like(next_velocity_z),
            next_velocity_z,
        )
        self._drop_z_fixed.copy_(
            torch.where(active_drop, clipped_z_fixed, self._drop_z_fixed)
        )
        self._drop_velocity_z_fixed.copy_(
            torch.where(
                active_drop,
                next_velocity_z,
                torch.zeros_like(next_velocity_z),
            )
        )
        self.drop_x.copy_(self._drop_x_fixed.to(torch.float32) / _FIXED_UNIT)
        self.drop_y.copy_(self._drop_y_fixed.to(torch.float32) / _FIXED_UNIT)
        self.drop_z.copy_(self._drop_z_fixed.to(torch.float32) / _FIXED_UNIT)

        touched = active_drop & self._touching(self.drop_x, self.drop_y, self.drop_z)
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
        self.drop_delay.masked_fill_(consumed, 0)
        self.drop_spawned.masked_fill_(consumed, False)
        self._drop_velocity_x_fixed.masked_fill_(consumed, 0)
        self._drop_velocity_y_fixed.masked_fill_(consumed, 0)
        self._drop_velocity_z_fixed.masked_fill_(consumed, 0)
        self.bonus_count.copy_(
            torch.where(
                torch.any(consumed, dim=1),
                torch.full_like(self.bonus_count, 6),
                self.bonus_count,
            )
        )

    def _collect_items(self) -> None:
        self._collect_map_items()
        self._collect_drops()

    def step(
        self, buttons: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.debug_checks and torch.any(self.pending_reset):
            lanes = torch.nonzero(self.pending_reset).flatten().to("cpu").tolist()
            raise RuntimeError(f"terminal lanes must be reset before step: {lanes}")
        hud_weapon = self._active_weapon()
        hud_ammo_slot = self._weapon_ammo_slot[hud_weapon]
        hud_ammo = self.ammo.gather(1, hud_ammo_slot.clamp_min(0)[:, None]).squeeze(1)
        self.hud_ready_ammo.copy_(
            torch.where(hud_ammo_slot < 0, torch.zeros_like(hud_ammo), hud_ammo)
        )
        reward = torch.zeros(self.num_envs, device=self.device)
        for _ in range(self.frame_skip):
            self.player_dead |= self.health <= 0
            active = ~self.player_dead & (self.episode_time < self.episode_timeout)
            previous_mugshot_override = (self.mugshot_pain_tics > 0) | (
                self.mugshot_grin_tics > 0
            )
            self.damage_count.copy_(
                torch.where(active, torch.clamp_min(self.damage_count - 1, 0), self.damage_count)
            )
            self.bonus_count.copy_(
                torch.where(active, torch.clamp_min(self.bonus_count - 1, 0), self.bonus_count)
            )
            decayed_pain = torch.clamp_min(self.mugshot_pain_tics - 1, 0)
            self.mugshot_pain_tics.copy_(
                torch.where(
                    active & (self.damage_count > 0),
                    torch.full_like(self.mugshot_pain_tics, _MUGSHOT_STATE_TICS),
                    torch.where(active, decayed_pain, self.mugshot_pain_tics),
                )
            )
            self.mugshot_ouch &= self.mugshot_pain_tics > 0
            self.mugshot_grin_tics.copy_(
                torch.where(
                    active,
                    torch.clamp_min(self.mugshot_grin_tics - 1, 0),
                    self.mugshot_grin_tics,
                )
            )
            self.mugshot_grin.copy_(self.mugshot_grin_tics > 0)
            mugshot_override = (self.mugshot_pain_tics > 0) | (
                self.mugshot_grin_tics > 0
            )
            resumed_normal_face = active & previous_mugshot_override & ~mugshot_override
            next_face_tics = torch.clamp_min(self.mugshot_face_tics - 1, 0)
            neutral_face = active & ~mugshot_override
            change_face = neutral_face & (
                (next_face_tics <= 0) | resumed_normal_face
            )
            mugshot_random = self.mugshot_rng_state
            next_mugshot_random = torch.bitwise_xor(
                mugshot_random,
                torch.bitwise_and(mugshot_random << 13, _UINT32_MASK),
            )
            next_mugshot_random = torch.bitwise_xor(
                next_mugshot_random,
                next_mugshot_random >> 17,
            )
            next_mugshot_random = torch.bitwise_xor(
                next_mugshot_random,
                torch.bitwise_and(next_mugshot_random << 5, _UINT32_MASK),
            )
            next_mugshot_random = torch.bitwise_and(
                next_mugshot_random,
                _UINT32_MASK,
            )
            self.mugshot_rng_state.copy_(
                torch.where(change_face, next_mugshot_random, mugshot_random)
            )
            self.mugshot_face_index.copy_(
                torch.where(
                    change_face,
                    torch.remainder(next_mugshot_random, 3),
                    self.mugshot_face_index,
                )
            )
            self.mugshot_face_tics.copy_(
                torch.where(
                    change_face,
                    torch.full_like(next_face_tics, _MUGSHOT_NORMAL_FRAME_TICS),
                    torch.where(neutral_face, next_face_tics, self.mugshot_face_tics),
                )
            )
            active_buttons = buttons & active[:, None]
            self.attack_held_tics.copy_(
                torch.where(
                    active_buttons[:, 0],
                    torch.clamp_max(self.attack_held_tics + 1, _MUGSHOT_RAMPAGE_DELAY),
                    torch.zeros_like(self.attack_held_tics),
                )
            )
            decremented_attack = torch.clamp_min(self.attack_cooldown - 1, 0)
            self.attack_cooldown.copy_(
                torch.where(active, decremented_attack, self.attack_cooldown)
            )
            decremented_weapon_state = torch.clamp_min(
                self.weapon_state_cooldown - 1,
                0,
            )
            self.weapon_state_cooldown.copy_(
                torch.where(active, decremented_weapon_state, self.weapon_state_cooldown)
            )
            self._weapon_switch_tick(active)
            self._select_weapons(active_buttons)
            self._move_player(active_buttons)
            self._vertical_player_tick(active)
            reward.add_(self._player_attack(active_buttons))
            weapon_ready = (
                active
                & (self.weapon_state_cooldown <= 0)
                & (self.weapon_raise_cooldown <= 0)
                & (self.pending_weapon < 0)
            )
            self.weapon_ready_tics.copy_(
                torch.where(
                    weapon_ready,
                    self.weapon_ready_tics + 1,
                    torch.zeros_like(self.weapon_ready_tics),
                )
            )
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

    def _pitch_projection_offset(self, focal_length: float) -> torch.Tensor:
        """Return ZDoom's fixed-point vertical view-panning offset."""
        tangent_index = torch.bitwise_right_shift(
            _ANGLE_90 - self._pitch_bam,
            _ANGLE_TO_FINE_SHIFT,
        ).clamp(0, _FINE_ANGLES // 2 - 1)
        tangent = self._fine_tangent_fixed[tangent_index]
        focal_fixed = round(focal_length * _FIXED_UNIT)
        offset_fixed = tangent * focal_fixed >> 16
        return offset_fixed.to(torch.float32) / _FIXED_UNIT

    def _render_flats(
        self,
        sector: torch.Tensor,
        view_z: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        ray_angles = self.angle[:, None] + self._ray_offsets[None, :]
        cosine_correction = torch.cos(self._ray_offsets)[None, :]
        pixel_delta = self._pixel_y.to(torch.float32) - center[:, None, None]
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
        center: torch.Tensor,
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
                return center[:, None] - (
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
                (center[:, None, None] - pixel_y)
                * distance[:, None, :]
                / _PROJECTION_FOCAL_LENGTH
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
        center = self.observation_height * 0.48 + self._pitch_projection_offset(
            _PROJECTION_FOCAL_LENGTH
        )
        sector = self._current_sector()
        view_z = self.view_z
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
        drop_visible = (self.drop_type >= 0) & self.drop_spawned
        drop_visual_type = torch.full_like(self.drop_type, 18)
        drop_visual_type = torch.where(self.drop_type == 2007, 12, drop_visual_type)
        drop_visual_type = torch.where(self.drop_type == 2002, 20, drop_visual_type)
        actor_x = torch.cat((actor_x, map_item_x, self.drop_x), dim=1)
        actor_y = torch.cat((actor_y, map_item_y, self.drop_y), dim=1)
        actor_z = torch.cat((actor_z, map_item_z, self.drop_z), dim=1)
        actor_alive = torch.cat((actor_alive, self.item_available, drop_visible), dim=1)
        actor_type = torch.cat((actor_type, map_item_type, drop_visual_type), dim=1)
        dx = actor_x - self.x[:, None]
        dy = actor_y - self.y[:, None]
        actor_distance = torch.sqrt(dx * dx + dy * dy).clamp_min_(1)
        relative = self._wrap_angle(torch.atan2(dy, dx) - self.angle[:, None])
        screen_center = (0.5 - relative / (math.pi / 2)) * self.observation_width
        safe_actor_type = actor_type.clamp_min(0)
        projection_scale = (self.observation_width * 0.5) / actor_distance
        sprite_width = self.map.sprite_widths[safe_actor_type].to(torch.float32)
        sprite_height = self.map.sprite_heights[safe_actor_type].to(torch.float32)
        sprite_left = (
            screen_center - self.map.sprite_left_offsets[safe_actor_type] * projection_scale
        )
        sprite_top = (
            center[:, None]
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
        flash = self._damage_to_alpha[self.damage_count.clamp(0, 113).to(torch.int64)] / 255.0
        bonus = torch.minimum(
            self.bonus_count.to(torch.float32) * 8.0, torch.full_like(self.health, 128.0)
        )
        bonus = (bonus / 255.0)[:, None, None]
        frame = frame * (1 - bonus) + 184.89 * bonus
        flash = flash[:, None, None]
        frame = frame * (1 - flash) + 53.55 * flash
        if self.mask_hud:
            frame[:, -11:, :] = 0
        return frame.clamp(0, 255).to(torch.uint8)

    def _native_raycast(self) -> torch.Tensor:
        direction = self._native_wall_ray_directions()
        origin = torch.stack((self.x, self.y), dim=-1)[:, None, None, :]
        start = self.map.walls[None, None, :, :2]
        segment = self.map.walls[None, None, :, 2:] - start
        ray = direction[:, :, None, :]
        offset = start - origin
        denominator = ray[..., 0] * segment[..., 1] - ray[..., 1] * segment[..., 0]
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        distance = (offset[..., 0] * segment[..., 1] - offset[..., 1] * segment[..., 0]) / safe
        along = (offset[..., 0] * ray[..., 1] - offset[..., 1] * ray[..., 0]) / safe
        valid = (denominator.abs() >= 1e-6) & (distance > 0) & (along >= 0) & (along <= 1)
        distance = torch.where(valid, distance, torch.full_like(distance, torch.inf))
        nearest_distance = torch.min(distance, dim=2).values
        return nearest_distance.clamp(1, 4096)

    def _native_sector_grid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        sectors: list[torch.Tensor] = []
        for start in range(0, flat_x.numel(), 2048):
            sectors.append(
                self._sector_at(
                    flat_x[start : start + 2048],
                    flat_y[start : start + 2048],
                )
            )
        return torch.cat(sectors).reshape_as(x)

    def _native_apply_colormap(
        self,
        indices: torch.Tensor,
        light: torch.Tensor,
        distance: torch.Tensor,
    ) -> torch.Tensor:
        base_shade = 61.0 - light / 4.0
        visibility = (1280.0 / distance.clamp_min(1)).clamp_max(24.0)
        shade = torch.floor(base_shade - visibility).to(torch.int64)
        shade = shade.clamp(0, 31)
        return self.map.colormap[shade, indices.to(torch.int64)]

    def _native_animated_texture_ids(self, texture_ids: torch.Tensor) -> torch.Tensor:
        # ViZDoom's certified deathmatch runtime never advances texture
        # translations: BFALL1 remains BFALL1 across consecutive rendered
        # tics. Preserve that observable behavior in the raw-fidelity path.
        return texture_ids

    def _native_view_angle_bam(self) -> torch.Tensor:
        """Return the retained or externally overridden unsigned view BAM."""

        visible_angle = self._angle_bam.to(torch.float32) * _BAM_TO_RADIANS
        public_angle_bam = torch.bitwise_and(
            torch.round(torch.remainder(self.angle, 2.0 * math.pi) / _BAM_TO_RADIANS).to(
                torch.int64
            ),
            _UINT32_MASK,
        )
        return torch.where(self.angle != visible_angle, public_angle_bam, self._angle_bam)

    def _native_wall_ray_directions(self) -> torch.Tensor:
        """Build wall rays from the software renderer's fine-angle view basis."""

        fine_angle = self._native_view_angle_bam() >> _ANGLE_TO_FINE_SHIFT
        view_cosine, view_sine = self._fine_direction_from_index(fine_angle)
        columns = (
            self._native_pixel_x[0, 0].to(torch.float32) - self.native_screen_width / 2.0
        ) / (self.native_screen_width / 2.0)
        direction_x = view_cosine[:, None] + view_sine[:, None] * columns[None, :]
        direction_y = view_sine[:, None] - view_cosine[:, None] * columns[None, :]
        return torch.stack((direction_x, direction_y), dim=-1)

    def _native_wall_vertical_steps(self) -> torch.Tensor:
        """Build PrepWall's per-column fixed-point vertical texture steps."""

        walls_fixed = torch.round(self.map.portal_walls * _FIXED_UNIT).to(torch.int64)
        visible_x_fixed = self._x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_y_fixed = self._y_fixed.to(torch.float32) / _FIXED_UNIT
        view_x_fixed = torch.where(
            self.x != visible_x_fixed,
            torch.round(self.x * _FIXED_UNIT).to(torch.int64),
            self._x_fixed,
        )
        view_y_fixed = torch.where(
            self.y != visible_y_fixed,
            torch.round(self.y * _FIXED_UNIT).to(torch.int64),
            self._y_fixed,
        )
        relative_x = walls_fixed[None, :, 0::2] - view_x_fixed[:, None, None]
        relative_y = walls_fixed[None, :, 1::2] - view_y_fixed[:, None, None]

        fine_angle = self._native_view_angle_bam() >> _ANGLE_TO_FINE_SHIFT
        view_sine = self._fine_sine_fixed[fine_angle][:, None, None]
        view_cosine = self._fine_sine_fixed[
            (fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ][:, None, None]
        transformed_x = (
            relative_x * view_sine - relative_y * view_cosine
        ) >> 20
        transformed_y = (
            relative_x * view_cosine + relative_y * view_sine
        ) >> 20
        tx1, tx2 = transformed_x[..., 0], transformed_x[..., 1]
        ty1, ty2 = transformed_y[..., 0], transformed_y[..., 1]

        # FWallTmapVals first narrows these values to float, while PrepWall
        # promotes the stored values to double for its perspective division.
        u_over_z_origin = (tx1.to(torch.float32) * (self.native_screen_width // 2)).to(
            torch.float64
        )
        u_over_z_step = (-ty1.to(torch.float32)).to(torch.float64)
        inv_z_origin = (
            (tx1 - tx2).to(torch.float32) * (self.native_screen_width // 2)
        ).to(torch.float64)
        inv_z_step_float = (ty2 - ty1).to(torch.float32)
        inv_z_step = inv_z_step_float.to(torch.float64)
        columns = (
            self._native_pixel_x[0, 0].to(torch.float64)
            - self.native_screen_width // 2
        )[None, :, None]
        top = u_over_z_origin[:, None, :] + u_over_z_step[:, None, :] * columns
        bottom = inv_z_origin[:, None, :] + inv_z_step[:, None, :] * columns
        safe_bottom = torch.where(bottom == 0, torch.ones_like(bottom), bottom)
        fraction = top / safe_bottom

        inverse_vertical_aspect = torch.tensor(
            self.native_screen_width * 200.0,
            device=self.device,
            dtype=torch.float32,
        )
        inverse_vertical_aspect /= 320.0
        inverse_vertical_aspect /= float(self.native_screen_height)
        wall_map_scale = inverse_vertical_aspect * 64.0 / (
            self.native_screen_width / 2.0
        )
        depth_scale = (inv_z_step_float * wall_map_scale).to(torch.float64)
        depth_origin = (-u_over_z_step.to(torch.float32) * wall_map_scale).to(
            torch.float64
        )
        return torch.floor(
            fraction * depth_scale[:, None, :] + depth_origin[:, None, :] + 0.5
        ).to(torch.int64)

    def _native_flat_texture_coordinates(
        self,
        texture_ids: torch.Tensor,
        plane_height: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map flat texels with R_DrawNormalPlane's fixed-point span math."""

        texture_width = self.map.texture_widths[texture_ids]
        texture_height = self.map.texture_heights[texture_ids]
        width_bits = torch.floor(torch.log2(texture_width.to(torch.float32))).to(torch.int64)
        height_bits = torch.floor(torch.log2(texture_height.to(torch.float32))).to(torch.int64)
        xscale = torch.bitwise_left_shift(torch.ones_like(width_bits), 32 - width_bits)
        yscale = torch.bitwise_left_shift(torch.ones_like(height_bits), 32 - height_bits)

        angle_bam = self._native_view_angle_bam()
        fine_angle = torch.bitwise_right_shift(angle_bam, _ANGLE_TO_FINE_SHIFT) & (
            _FINE_ANGLES - 1
        )
        fine_sine = self._fine_sine_fixed[fine_angle][:, None, None]
        fine_cosine = self._fine_sine_fixed[
            (fine_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)
        ][:, None, None]

        xstep_scale = self._trunc_divide(
            xscale * fine_sine,
            torch.full_like(xscale, _NATIVE_FOCAL_X_FIXED),
        )
        ystep_scale = self._trunc_divide(
            yscale * fine_cosine,
            torch.full_like(yscale, _NATIVE_FOCAL_X_FIXED),
        )
        flat_columns = self._native_flat_columns[None, None, :]
        basexfrac = ((xscale * fine_cosine) >> 16) + flat_columns * xstep_scale
        baseyfrac = ((yscale * -fine_sine) >> 16) + flat_columns * ystep_scale

        visible_x_fixed = self._x_fixed.to(torch.float32) / _FIXED_UNIT
        visible_y_fixed = self._y_fixed.to(torch.float32) / _FIXED_UNIT
        view_x_fixed = torch.where(
            self.x != visible_x_fixed,
            torch.round(self.x * _FIXED_UNIT).to(torch.int64),
            self._x_fixed,
        )[:, None, None]
        view_y_fixed = torch.where(
            self.y != visible_y_fixed,
            torch.round(self.y * _FIXED_UNIT).to(torch.int64),
            self._y_fixed,
        )[:, None, None]
        pviewx = (xscale * view_x_fixed) >> 16
        pviewy = (yscale * -view_y_fixed) >> 16

        tangent_index = torch.bitwise_right_shift(
            _ANGLE_90 - self._pitch_bam,
            _ANGLE_TO_FINE_SHIFT,
        ).clamp(0, _FINE_ANGLES // 2 - 1)
        pitch_offset_fixed = (
            _NATIVE_FOCAL_Y_FIXED * self._fine_tangent_fixed[tangent_index]
        ) >> 16
        center_fixed = (self.native_view_height // 2) * _FIXED_UNIT + pitch_offset_fixed
        center_row = center_fixed >> 16
        pixel_y = self._native_pixel_y.to(torch.int64)
        pixel_y_fixed = pixel_y * _FIXED_UNIT
        denominator = torch.where(
            pixel_y < center_row[:, None, None],
            center_fixed[:, None, None] - pixel_y_fixed - _FIXED_UNIT // 2,
            pixel_y_fixed - center_fixed[:, None, None] + _FIXED_UNIT // 2,
        ).clamp_min(1)
        slope_overflow = denominator <= (_NATIVE_FOCAL_Y_FIXED >> 15)
        yslope = self._trunc_divide(
            torch.full_like(denominator, _NATIVE_FOCAL_Y_FIXED << 16),
            denominator,
        )
        yslope = torch.where(slope_overflow, torch.full_like(yslope, (1 << 31) - 1), yslope)
        plane_height_fixed = torch.round(plane_height * _FIXED_UNIT).to(torch.int64)
        distance = (plane_height_fixed * yslope) >> 16
        xfrac = (((distance * basexfrac) >> 16) + pviewx) & _UINT32_MASK
        yfrac = (((distance * baseyfrac) >> 16) + pviewy) & _UINT32_MASK

        texture_u = torch.bitwise_right_shift(xfrac, 32 - width_bits)
        texture_v = torch.bitwise_right_shift(yfrac, 32 - height_bits)
        return texture_u, texture_v

    def _native_render_flats(
        self,
        current_sector: torch.Tensor,
        view_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        focal_length = self.native_screen_width / 2.0 * self.native_vertical_aspect
        # R_SetupFreelook builds yslope through pixel centers, subtracting or
        # adding FRACUNIT/2 around centeryfrac. Walls and sprites retain their
        # integer-edge projection convention, so this half pixel is plane-only.
        center = self.native_view_height / 2.0 - 0.5 + self._pitch_projection_offset(
            focal_length
        )
        ray_angles = self.angle[:, None] + self._native_ray_offsets[None, :]
        cosine_correction = torch.cos(self._native_ray_offsets)[None, None, :]
        pixel_delta = self._native_pixel_y.to(torch.float32) - center[:, None, None]
        floor_pixels = pixel_delta > 0
        shape = (self.num_envs, self.native_view_height, self.native_screen_width)
        sectors = current_sector[:, None, None].expand(shape).clone()
        ray_distance = torch.full(shape, torch.inf, device=self.device)
        ray_cos = torch.cos(ray_angles)[:, None, :]
        ray_sin = torch.sin(ray_angles)[:, None, :]
        denominator = pixel_delta.abs().clamp_min(0.5)

        # A height transition cannot be resolved by repeatedly guessing a sector:
        # the projected point can alternate between the upper floor and a pit.
        # Intersect every sector plane, clip that point to the sector polygon, and
        # retain the nearest valid surface, as Doom's subsector traversal does.
        for sector_index in range(len(self.map.sector_heights)):
            floor_height = view_z - self.map.sector_heights[sector_index, 0]
            ceiling_height = self.map.sector_heights[sector_index, 1] - view_z
            plane_height = torch.where(
                floor_pixels,
                floor_height[:, None, None],
                ceiling_height[:, None, None],
            )
            perpendicular_depth = plane_height * focal_length / denominator
            candidate_distance = perpendicular_depth / cosine_correction
            candidate_x = self.x[:, None, None] + ray_cos * candidate_distance
            candidate_y = self.y[:, None, None] + ray_sin * candidate_distance

            edges = self.map.sector_edges[self.map.sector_edge_mask[sector_index]]
            edge_x1 = edges[:, 0]
            edge_y1 = edges[:, 1]
            edge_x2 = edges[:, 2]
            edge_y2 = edges[:, 3]
            edge_dy = edge_y2 - edge_y1
            safe_edge_dy = torch.where(
                edge_dy.abs() < 1e-6,
                torch.ones_like(edge_dy),
                edge_dy,
            )
            point_x = candidate_x[..., None]
            point_y = candidate_y[..., None]
            crosses_y = (edge_y1 > point_y) != (edge_y2 > point_y)
            crossing_x = edge_x1 + (point_y - edge_y1) * (edge_x2 - edge_x1) / safe_edge_dy
            inside = torch.remainder(
                torch.sum(
                    crosses_y & (point_x < crossing_x),
                    dim=3,
                ),
                2,
            ).bool()
            nearer = (
                inside
                & (plane_height > 0)
                & torch.isfinite(candidate_distance)
                & (candidate_distance > 0)
                & (candidate_distance < ray_distance)
            )
            sectors = torch.where(nearer, sector_index, sectors)
            ray_distance = torch.where(nearer, candidate_distance, ray_distance)

        unresolved = ~torch.isfinite(ray_distance)
        surface_depth = torch.where(
            unresolved,
            torch.full_like(ray_distance, torch.inf),
            ray_distance * cosine_correction,
        )

        # R_DrawNormalPlane anchors span coordinates at x - halfviewwidth,
        # where halfviewwidth is centerx - 1. Keep surface selection on the
        # wall projection rays, but sample flats with that plane-only origin.
        selected_floor_height = view_z[:, None, None] - self.map.sector_heights[sectors, 0]
        selected_ceiling_height = self.map.sector_heights[sectors, 1] - view_z[:, None, None]
        selected_plane_height = torch.where(
            floor_pixels,
            selected_floor_height,
            selected_ceiling_height,
        )
        floor_texture = self.map.sector_floor_texture_ids[sectors]
        ceiling_texture = self.map.sector_ceiling_texture_ids[sectors]
        texture_id = torch.where(floor_pixels, floor_texture, ceiling_texture)
        texture_id = self._native_animated_texture_ids(texture_id)
        texture_u, texture_v = self._native_flat_texture_coordinates(
            texture_id,
            selected_plane_height,
        )
        indices = self.map.texture_index_atlas[texture_id, texture_v, texture_u]
        _weapon_frame, _weapon_flash, flash_light = self._native_weapon_frame_selection()
        light = self.map.sector_lights[sectors] + flash_light[:, None, None] * 16
        frame = self._native_apply_colormap(indices, light, surface_depth.clamp(1, 4096))
        return frame, surface_depth

    def _native_portal_intersections(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction = self._native_wall_ray_directions()
        origin = torch.stack((self.x, self.y), dim=-1)[:, None, None, :]
        start = self.map.portal_walls[None, None, :, :2]
        segment = self.map.portal_walls[None, None, :, 2:] - start
        ray = direction[:, :, None, :]
        offset = start - origin
        denominator = ray[..., 0] * segment[..., 1] - ray[..., 1] * segment[..., 0]
        safe = torch.where(
            denominator.abs() < 1e-6,
            torch.ones_like(denominator),
            denominator,
        )
        distance = (offset[..., 0] * segment[..., 1] - offset[..., 1] * segment[..., 0]) / safe
        along = (offset[..., 0] * ray[..., 1] - offset[..., 1] * ray[..., 0]) / safe
        valid = (denominator.abs() >= 1e-6) & (distance > 0) & (along >= 0) & (along <= 1)
        distance = torch.where(valid, distance, torch.full_like(distance, torch.inf))
        return distance, along.clamp(0, 1)

    def _native_render_portal_walls(
        self,
        frame: torch.Tensor,
        view_z: torch.Tensor,
        surface_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        focal_length = self.native_screen_width / 2.0 * self.native_vertical_aspect
        center = self.native_view_height / 2.0 - 1.0 + self._pitch_projection_offset(
            focal_length
        )
        distances, wall_along = self._native_portal_intersections()
        wall_vertical_steps = self._native_wall_vertical_steps()
        filled = torch.zeros_like(frame, dtype=torch.bool)
        scene_depth = surface_depth.clone()
        pixel_y = self._native_pixel_y.to(torch.float32)
        current_sector = (
            self._current_sector()[:, None].expand(-1, self.native_screen_width).clone()
        )
        previous_distance = torch.zeros_like(current_sector, dtype=torch.float32)
        all_sectors = self.map.portal_wall_sectors
        for _ in range(32):
            current = current_sector.clamp_min(0)
            incident = (all_sectors[None, None, :, 0] == current[:, :, None]) | (
                all_sectors[None, None, :, 1] == current[:, :, None]
            )
            candidates = torch.where(
                incident
                & (current_sector[:, :, None] >= 0)
                & (distances > previous_distance[:, :, None] + 1e-3),
                distances,
                torch.full_like(distances, torch.inf),
            )
            distance, wall_index = torch.min(candidates, dim=2)
            valid = torch.isfinite(distance)
            along = wall_along.gather(2, wall_index[:, :, None]).squeeze(2)
            sectors = self.map.portal_wall_sectors[wall_index]
            front = sectors[..., 0]
            back = sectors[..., 1]
            from_front = current_sector == front
            side_index = (~from_front).to(torch.int64)
            other_sector = torch.where(from_front, back, front)
            safe_other = other_sector.clamp_min(0)
            view_floor = self.map.sector_heights[current, 0]
            view_ceiling = self.map.sector_heights[current, 1]
            other_floor = self.map.sector_heights[safe_other, 0]
            other_ceiling = self.map.sector_heights[safe_other, 1]
            one_sided = other_sector < 0

            def project(
                world_z: torch.Tensor,
                layer_distance: torch.Tensor = distance,
            ) -> torch.Tensor:
                return center[:, None] - (
                    (world_z - view_z[:, None]) * focal_length / layer_distance
                )

            one_top = project(view_ceiling)
            one_bottom = project(view_floor)
            lower_top = project(other_floor)
            lower_bottom = project(view_floor)
            upper_top = project(view_ceiling)
            upper_bottom = project(other_ceiling)
            one_span = (
                (one_sided & valid)[:, None, :]
                & (pixel_y >= one_top[:, None, :])
                & (pixel_y <= one_bottom[:, None, :])
            )
            lower_span = (
                (~one_sided & valid & (view_floor < other_floor))[:, None, :]
                & (pixel_y >= lower_top[:, None, :])
                & (pixel_y <= lower_bottom[:, None, :])
            )
            upper_span = (
                (~one_sided & valid & (view_ceiling > other_ceiling))[:, None, :]
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
            in_front_of_surface = distance[:, None, :] <= surface_depth + 1e-3
            span = (
                (one_span | lower_span | upper_span)
                & (texture_id >= 0)
                & in_front_of_surface
                & ~filled
            )
            safe_texture_id = self._native_animated_texture_ids(texture_id.clamp_min(0))
            texture_width = self.map.texture_widths[safe_texture_id]
            texture_height = self.map.texture_heights[safe_texture_id]
            texture_offset = self.map.portal_side_texture_offsets[wall_index, side_index]
            texture_along = torch.where(side_index == 0, along, 1.0 - along)
            texture_u = torch.remainder(
                torch.floor(
                    texture_along * self.map.portal_wall_lengths[wall_index]
                    + texture_offset[..., 0]
                ).to(torch.int64)[:, None, :],
                texture_width,
            ).expand(-1, self.native_view_height, -1)
            texture_origin_z = torch.where(
                one_span,
                view_ceiling[:, None, :],
                torch.where(
                    lower_span,
                    other_floor[:, None, :],
                    other_ceiling[:, None, :],
                ),
            )
            height_bits = torch.floor(torch.log2(texture_height.to(torch.float32))).to(
                torch.int64
            )
            vertical_step = wall_vertical_steps.gather(
                2,
                wall_index[:, :, None],
            ).squeeze(2)[:, None, :] * torch.bitwise_left_shift(
                torch.ones_like(height_bits),
                14 - height_bits,
            )
            tangent_index = torch.bitwise_right_shift(
                _ANGLE_90 - self._pitch_bam,
                _ANGLE_TO_FINE_SHIFT,
            ).clamp(0, _FINE_ANGLES // 2 - 1)
            pitch_offset_fixed = (
                _NATIVE_FOCAL_Y_FIXED * self._fine_tangent_fixed[tangent_index]
            ) >> 16
            center_fixed = (
                self.native_view_height // 2 * _FIXED_UNIT + pitch_offset_fixed
            )
            screen_delta_fixed = (
                self._native_pixel_y.to(torch.int64) * _FIXED_UNIT
                - center_fixed[:, None, None]
                + _FIXED_UNIT
            )
            texture_mid_fixed = torch.round(
                (texture_origin_z - view_z[:, None, None]) * _FIXED_UNIT
                + texture_offset[:, None, :, 1] * _FIXED_UNIT
            ).to(torch.int64)
            texture_fraction = torch.bitwise_left_shift(
                texture_mid_fixed,
                16 - height_bits,
            ) + torch.bitwise_right_shift(
                vertical_step * screen_delta_fixed,
                16,
            )
            texture_v = torch.bitwise_right_shift(
                torch.bitwise_and(texture_fraction, _UINT32_MASK),
                32 - height_bits,
            )
            texture = self.map.texture_index_atlas[
                safe_texture_id,
                texture_v,
                texture_u,
            ]
            _weapon_frame, _weapon_flash, flash_light = self._native_weapon_frame_selection()
            light = self.map.sector_lights[current][:, None, :] + flash_light[:, None, None] * 16
            wall = self.map.portal_walls[wall_index]
            horizontal = (wall[..., 3] - wall[..., 1]).abs() < 1e-6
            vertical = (wall[..., 2] - wall[..., 0]).abs() < 1e-6
            fake_contrast = torch.where(
                vertical,
                16.0,
                torch.where(horizontal, -16.0, 0.0),
            )
            light = light + fake_contrast[:, None, :]
            wall_value = self._native_apply_colormap(
                texture,
                light,
                distance[:, None, :],
            )
            frame = torch.where(span, wall_value, frame)
            scene_depth = torch.where(span, distance[:, None, :], scene_depth)
            filled |= span
            previous_distance = torch.where(valid, distance, previous_distance)
            current_sector = torch.where(
                valid & (other_sector >= 0),
                other_sector,
                torch.full_like(current_sector, -1),
            )
        return frame, scene_depth

    @staticmethod
    def _doom_sprite_rotation(
        viewer_angle: torch.Tensor,
        actor_angle: torch.Tensor,
    ) -> torch.Tensor:
        """Select Doom's clockwise 1..8 sprite rotation as a zero-based index."""
        relative = torch.remainder(
            viewer_angle - actor_angle + math.pi / 8,
            2 * math.pi,
        )
        return torch.floor(relative / (math.pi / 4)).to(torch.int64)

    def _native_enemy_sprite_ids(self) -> torch.Tensor:
        enemy_type = self.enemy_type.clamp(0, 5)
        viewer_angle = torch.atan2(
            self.y[:, None] - self.enemy_y,
            self.x[:, None] - self.enemy_x,
        )
        rotation = self._doom_sprite_rotation(viewer_angle, self.enemy_angle)
        walk_frame = torch.remainder(
            self.enemy_animation_tics // self._enemy_walk_frame_tics[enemy_type],
            4,
        ).to(torch.int64)
        idle_frame = torch.remainder(
            self.enemy_animation_tics // self._enemy_idle_frame_tics[enemy_type],
            2,
        ).to(torch.int64)
        # ScriptedMarine's spawn loop uses PLAY A exclusively; the other
        # certified actors alternate their A/B spawn frames every ten tics.
        idle_frame = torch.where(
            enemy_type == 2,
            torch.zeros_like(idle_frame),
            idle_frame,
        )
        walk_frame = torch.where(
            self.enemy_target_slot < -1,
            idle_frame,
            walk_frame,
        )
        phase = self.enemy_attack_phase
        cooldown = self.enemy_cooldown
        ranged_recovery_frame = (cooldown > (self._enemy_attack_recovery[enemy_type] // 2)).to(
            torch.int64
        )
        attack_frame = torch.where(
            (enemy_type == 0) | (enemy_type == 1),
            torch.where(phase == 2, ranged_recovery_frame, 0),
            torch.zeros_like(enemy_type),
        )
        attack_frame = torch.where(
            ((enemy_type == 4) | (enemy_type == 5)) & (phase == 1),
            (cooldown <= 8).to(torch.int64),
            attack_frame,
        )
        attack_frame = torch.where(
            ((enemy_type == 4) | (enemy_type == 5)) & (phase == 2),
            torch.full_like(attack_frame, 2),
            attack_frame,
        )
        attack_frame = torch.where(
            (enemy_type == 3) & (phase == 4),
            torch.ones_like(attack_frame),
            attack_frame,
        )
        attack_frame = torch.where(
            (enemy_type == 3) & (phase == 2),
            torch.ones_like(attack_frame),
            attack_frame,
        )
        attack_frame = torch.where(
            (enemy_type == 3) & (phase == 3),
            torch.full_like(attack_frame, 2),
            attack_frame,
        )
        walk = self.map.enemy_walk_sprite_ids[enemy_type, walk_frame, rotation]
        attack = self.map.enemy_attack_sprite_ids[enemy_type, attack_frame, rotation]
        pain = self.map.enemy_pain_sprite_ids[enemy_type, rotation]
        animated = torch.where(phase > 0, attack, walk)
        return torch.where(self.enemy_pain_tics > 0, pain, animated)

    def _native_enemy_death_sprite_ids(self) -> torch.Tensor:
        death_type = self.enemy_death_type.clamp(0, 5)
        death_elapsed = self.enemy_death_elapsed.to(torch.int64)
        death_count = self.map.enemy_death_frame_counts[death_type]
        death_durations = self.map.enemy_death_frame_durations[death_type]
        death_frame_ends = torch.cumsum(death_durations, dim=2)
        death_frame = torch.sum(
            death_elapsed[:, :, None] >= death_frame_ends,
            dim=2,
        )
        death_frame = torch.minimum(death_frame, death_count - 1)
        death_sprite = self.map.enemy_death_sprite_ids[death_type, death_frame]
        xdeath_count = self.map.enemy_xdeath_frame_counts[death_type]
        xdeath_durations = self.map.enemy_xdeath_frame_durations[death_type]
        xdeath_frame_ends = torch.cumsum(xdeath_durations, dim=2)
        xdeath_frame = torch.sum(
            death_elapsed[:, :, None] >= xdeath_frame_ends,
            dim=2,
        )
        xdeath_frame = torch.minimum(xdeath_frame, xdeath_count - 1)
        xdeath_sprite = self.map.enemy_xdeath_sprite_ids[death_type, xdeath_frame]
        return torch.where(
            self.enemy_death_extreme,
            xdeath_sprite,
            death_sprite,
        )

    def _native_projectile_explosion_sprite_ids(
        self,
        projectile_type: torch.Tensor,
        remaining_tics: torch.Tensor,
    ) -> torch.Tensor:
        safe_type = projectile_type.clamp(0, 2)
        elapsed = self.map.projectile_explosion_total_tics[safe_type] - remaining_tics.to(
            torch.int64
        )
        durations = self.map.projectile_explosion_frame_durations[safe_type]
        frame_ends = torch.cumsum(durations, dim=-1)
        frame = torch.sum(elapsed[..., None] >= frame_ends, dim=-1)
        frame = torch.minimum(
            frame,
            self.map.projectile_explosion_frame_counts[safe_type] - 1,
        )
        return self.map.raw_projectile_explosion_sprite_ids[safe_type, frame]

    @staticmethod
    def _native_enemy_fullbright(
        enemy_type: torch.Tensor,
        attack_phase: torch.Tensor,
        cooldown: torch.Tensor,
        attack_recovery: torch.Tensor,
    ) -> torch.Tensor:
        """Return the BRIGHT flag from the certified actors' current states."""
        shotgun_muzzle = (enemy_type == 1) & (attack_phase == 2) & (
            cooldown > (attack_recovery // 2)
        )
        chaingun_muzzle = (enemy_type == 3) & (
            (attack_phase == 2) | (attack_phase == 3)
        )
        return shotgun_muzzle | chaingun_muzzle

    def _native_render_sprites(
        self,
        frame: torch.Tensor,
        wall_distance: torch.Tensor,
        view_z: torch.Tensor,
        scene_depth: torch.Tensor,
    ) -> torch.Tensor:
        horizontal_focal_length = self.native_screen_width / 2.0
        vertical_focal_length = horizontal_focal_length * self.native_vertical_aspect
        center = self.native_view_height / 2.0 - 1.0 + self._pitch_projection_offset(
            vertical_focal_length
        )
        enemy_sprite = self._native_enemy_sprite_ids()
        static = self.map.raw_static_sprite_ids
        enemy_slots = torch.nonzero(torch.any(self.enemy_alive, dim=0)).flatten()
        actor_x = self.enemy_x[:, enemy_slots]
        actor_y = self.enemy_y[:, enemy_slots]
        actor_z = self.enemy_z[:, enemy_slots]
        actor_alive = self.enemy_alive[:, enemy_slots]
        actor_sprite = enemy_sprite[:, enemy_slots]
        enemy_type = self.enemy_type[:, enemy_slots].clamp(0, 5)
        actor_fullbright = self._native_enemy_fullbright(
            enemy_type,
            self.enemy_attack_phase[:, enemy_slots],
            self.enemy_cooldown[:, enemy_slots],
            self._enemy_attack_recovery[enemy_type],
        )
        actor_additive_style = torch.full_like(actor_sprite, -1, dtype=torch.int64)

        doll_count = max(len(self.map.player_starts) - 1, 0)
        if doll_count:
            dolls = self.map.player_starts[:-1]
            doll_x = dolls[None, :, 0].expand(self.num_envs, -1)
            doll_y = dolls[None, :, 1].expand(self.num_envs, -1)
            doll_angle = torch.deg2rad(dolls[None, :, 2]).expand(self.num_envs, -1)
            viewer_angle = torch.atan2(
                self.y[:, None] - doll_y,
                self.x[:, None] - doll_x,
            )
            doll_rotation = self._doom_sprite_rotation(viewer_angle, doll_angle)
            doll_sprite = self.map.enemy_walk_sprite_ids[2, 0, doll_rotation]
            actor_x = torch.cat((actor_x, doll_x), dim=1)
            actor_y = torch.cat((actor_y, doll_y), dim=1)
            actor_z = torch.cat(
                (
                    actor_z,
                    self._player_start_z[:-1][None, :].expand(self.num_envs, -1),
                ),
                dim=1,
            )
            actor_alive = torch.cat(
                (
                    actor_alive,
                    (~self.player_dead)[:, None].expand(-1, doll_count),
                ),
                dim=1,
            )
            actor_sprite = torch.cat((actor_sprite, doll_sprite), dim=1)
            actor_fullbright = torch.cat(
                (actor_fullbright, torch.zeros_like(doll_sprite, dtype=torch.bool)),
                dim=1,
            )
            actor_additive_style = torch.cat(
                (actor_additive_style, torch.full_like(doll_sprite, -1, dtype=torch.int64)),
                dim=1,
            )

        death_sprite = self._native_enemy_death_sprite_ids()
        death_slots = torch.nonzero(torch.any(self.enemy_death_tics > 0, dim=0)).flatten()
        if death_slots.numel():
            visible_death_sprite = death_sprite[:, death_slots]
            actor_x = torch.cat((actor_x, self.enemy_x[:, death_slots]), dim=1)
            actor_y = torch.cat((actor_y, self.enemy_y[:, death_slots]), dim=1)
            actor_z = torch.cat((actor_z, self.enemy_z[:, death_slots]), dim=1)
            actor_alive = torch.cat(
                (actor_alive, self.enemy_death_tics[:, death_slots] > 0),
                dim=1,
            )
            actor_sprite = torch.cat((actor_sprite, visible_death_sprite), dim=1)
            actor_fullbright = torch.cat(
                (
                    actor_fullbright,
                    torch.zeros_like(visible_death_sprite, dtype=torch.bool),
                ),
                dim=1,
            )
            actor_additive_style = torch.cat(
                (
                    actor_additive_style,
                    torch.full_like(visible_death_sprite, -1, dtype=torch.int64),
                ),
                dim=1,
            )

        player_projectile_type = self.projectile_type.clamp(0, 1)
        player_projectile_angle = torch.atan2(
            self.projectile_velocity_y,
            self.projectile_velocity_x,
        )
        player_projectile_viewer_angle = torch.atan2(
            self.y[:, None] - self.projectile_y,
            self.x[:, None] - self.projectile_x,
        )
        player_projectile_rotation = self._doom_sprite_rotation(
            player_projectile_viewer_angle,
            player_projectile_angle,
        )
        player_projectile_frame = torch.where(
            player_projectile_type == 1,
            torch.remainder(self.projectile_age // 6, 2).to(torch.int64),
            torch.zeros_like(player_projectile_type),
        )
        player_projectile_sprite = self.map.raw_projectile_flight_sprite_ids[
            player_projectile_type,
            player_projectile_frame,
            player_projectile_rotation,
        ]
        player_flight_slots = torch.nonzero(
            torch.any(self.projectile_alive, dim=0),
        ).flatten()
        if player_flight_slots.numel():
            player_flight_alive = self.projectile_alive[:, player_flight_slots]
            player_flight_type = player_projectile_type[:, player_flight_slots]
            actor_x = torch.cat((actor_x, self.projectile_x[:, player_flight_slots]), dim=1)
            actor_y = torch.cat((actor_y, self.projectile_y[:, player_flight_slots]), dim=1)
            actor_z = torch.cat((actor_z, self.projectile_z[:, player_flight_slots]), dim=1)
            actor_alive = torch.cat((actor_alive, player_flight_alive), dim=1)
            actor_sprite = torch.cat(
                (actor_sprite, player_projectile_sprite[:, player_flight_slots]),
                dim=1,
            )
            actor_fullbright = torch.cat((actor_fullbright, player_flight_alive), dim=1)
            actor_additive_style = torch.cat(
                (
                    actor_additive_style,
                    torch.where(
                        player_flight_type == 1,
                        torch.zeros_like(player_flight_type),
                        torch.full_like(player_flight_type, -1),
                    ),
                ),
                dim=1,
            )

        enemy_projectile_angle = torch.atan2(
            self.enemy_projectile_velocity_y,
            self.enemy_projectile_velocity_x,
        )
        enemy_projectile_viewer_angle = torch.atan2(
            self.y[:, None] - self.enemy_projectile_y,
            self.x[:, None] - self.enemy_projectile_x,
        )
        enemy_projectile_rotation = self._doom_sprite_rotation(
            enemy_projectile_viewer_angle,
            enemy_projectile_angle,
        )
        enemy_projectile_frame = torch.remainder(self.enemy_projectile_age // 4, 2).to(
            torch.int64
        )
        enemy_projectile_sprite = self.map.raw_projectile_flight_sprite_ids[
            2,
            enemy_projectile_frame,
            enemy_projectile_rotation,
        ]
        enemy_flight_slots = torch.nonzero(
            torch.any(self.enemy_projectile_alive, dim=0),
        ).flatten()
        if enemy_flight_slots.numel():
            enemy_flight_alive = self.enemy_projectile_alive[:, enemy_flight_slots]
            actor_x = torch.cat(
                (actor_x, self.enemy_projectile_x[:, enemy_flight_slots]),
                dim=1,
            )
            actor_y = torch.cat(
                (actor_y, self.enemy_projectile_y[:, enemy_flight_slots]),
                dim=1,
            )
            actor_z = torch.cat(
                (actor_z, self.enemy_projectile_z[:, enemy_flight_slots]),
                dim=1,
            )
            actor_alive = torch.cat((actor_alive, enemy_flight_alive), dim=1)
            actor_sprite = torch.cat(
                (actor_sprite, enemy_projectile_sprite[:, enemy_flight_slots]),
                dim=1,
            )
            actor_fullbright = torch.cat((actor_fullbright, enemy_flight_alive), dim=1)
            actor_additive_style = torch.cat(
                (
                    actor_additive_style,
                    torch.ones_like(enemy_projectile_frame[:, enemy_flight_slots]),
                ),
                dim=1,
            )

        player_impact_sprite = self._native_projectile_explosion_sprite_ids(
            self.projectile_impact_type,
            self.projectile_impact_tics,
        )
        player_impact_slots = torch.nonzero(
            torch.any(self.projectile_impact_tics > 0, dim=0),
        ).flatten()
        if player_impact_slots.numel():
            player_impact_alive = self.projectile_impact_tics[:, player_impact_slots] > 0
            player_impact_type = self.projectile_impact_type[:, player_impact_slots]
            actor_x = torch.cat((actor_x, self.projectile_x[:, player_impact_slots]), dim=1)
            actor_y = torch.cat((actor_y, self.projectile_y[:, player_impact_slots]), dim=1)
            actor_z = torch.cat((actor_z, self.projectile_z[:, player_impact_slots]), dim=1)
            actor_alive = torch.cat((actor_alive, player_impact_alive), dim=1)
            actor_sprite = torch.cat(
                (actor_sprite, player_impact_sprite[:, player_impact_slots]),
                dim=1,
            )
            actor_fullbright = torch.cat((actor_fullbright, player_impact_alive), dim=1)
            actor_additive_style = torch.cat(
                (
                    actor_additive_style,
                    torch.where(
                        player_impact_type == 1,
                        torch.zeros_like(player_impact_type),
                        torch.full_like(player_impact_type, -1),
                    ),
                ),
                dim=1,
            )

        enemy_impact_type = torch.full_like(self.enemy_projectile_age, 2, dtype=torch.int64)
        enemy_impact_sprite = self._native_projectile_explosion_sprite_ids(
            enemy_impact_type,
            self.enemy_projectile_impact_tics,
        )
        enemy_impact_slots = torch.nonzero(
            torch.any(self.enemy_projectile_impact_tics > 0, dim=0),
        ).flatten()
        if enemy_impact_slots.numel():
            enemy_impact_alive = self.enemy_projectile_impact_tics[:, enemy_impact_slots] > 0
            actor_x = torch.cat(
                (actor_x, self.enemy_projectile_x[:, enemy_impact_slots]),
                dim=1,
            )
            actor_y = torch.cat(
                (actor_y, self.enemy_projectile_y[:, enemy_impact_slots]),
                dim=1,
            )
            actor_z = torch.cat(
                (actor_z, self.enemy_projectile_z[:, enemy_impact_slots]),
                dim=1,
            )
            actor_alive = torch.cat((actor_alive, enemy_impact_alive), dim=1)
            actor_sprite = torch.cat(
                (actor_sprite, enemy_impact_sprite[:, enemy_impact_slots]),
                dim=1,
            )
            actor_fullbright = torch.cat((actor_fullbright, enemy_impact_alive), dim=1)
            actor_additive_style = torch.cat(
                (
                    actor_additive_style,
                    torch.ones_like(enemy_impact_type[:, enemy_impact_slots]),
                ),
                dim=1,
            )

        # Native rendering is intentionally diagnostic and dynamically trims
        # inactive effect slots to avoid doubling every sprite-layer tensor.
        fog_slots = torch.nonzero(
            torch.any(self.teleport_fog_tics > 0, dim=0),
        ).flatten()
        if fog_slots.numel():
            fog_tics = self.teleport_fog_tics[:, fog_slots]
            fog_elapsed = _TELEPORT_FOG_TOTAL_TICS - fog_tics.to(torch.int64)
            fog_frame = torch.clamp(fog_elapsed // 6, max=11)
            fog_sprite = self.map.raw_teleport_fog_sprite_ids[fog_frame]
            actor_x = torch.cat((actor_x, self.teleport_fog_x[:, fog_slots]), dim=1)
            actor_y = torch.cat((actor_y, self.teleport_fog_y[:, fog_slots]), dim=1)
            actor_z = torch.cat((actor_z, self.teleport_fog_z[:, fog_slots]), dim=1)
            actor_alive = torch.cat((actor_alive, fog_tics > 0), dim=1)
            actor_sprite = torch.cat((actor_sprite, fog_sprite), dim=1)
            actor_fullbright = torch.cat((actor_fullbright, fog_tics > 0), dim=1)
            actor_additive_style = torch.cat(
                (actor_additive_style, torch.ones_like(fog_sprite)),
                dim=1,
            )

        map_item_x = self.map.item_spawns[None, :, 0].expand(self.num_envs, -1)
        map_item_y = self.map.item_spawns[None, :, 1].expand(self.num_envs, -1)
        map_item_z = self._item_z[None, :].expand(self.num_envs, -1)
        map_item_sprite = self.map.item_raw_visual_types[None, :].expand(self.num_envs, -1)
        item_types = self.map.item_types[None, :]
        item_animation = self.map.raw_item_animation_sprite_ids
        bonus_phase = torch.remainder((self.episode_time - 1) // 6, 6)[:, None]
        health_bonus_frames = torch.stack(
            (
                map_item_sprite,
                item_animation[0].expand_as(map_item_sprite),
                item_animation[1].expand_as(map_item_sprite),
                item_animation[2].expand_as(map_item_sprite),
                item_animation[1].expand_as(map_item_sprite),
                item_animation[0].expand_as(map_item_sprite),
            ),
            dim=2,
        )
        armor_bonus_frames = torch.stack(
            (
                map_item_sprite,
                item_animation[3].expand_as(map_item_sprite),
                item_animation[4].expand_as(map_item_sprite),
                item_animation[5].expand_as(map_item_sprite),
                item_animation[4].expand_as(map_item_sprite),
                item_animation[3].expand_as(map_item_sprite),
            ),
            dim=2,
        )
        item_row = torch.arange(self.num_envs, device=self.device)[:, None]
        item_column = torch.arange(len(self.map.item_types), device=self.device)[None, :]
        map_item_sprite = torch.where(
            item_types == 2014,
            health_bonus_frames[item_row, item_column, bonus_phase],
            map_item_sprite,
        )
        map_item_sprite = torch.where(
            item_types == 2015,
            armor_bonus_frames[item_row, item_column, bonus_phase],
            map_item_sprite,
        )
        green_armor_phase = torch.remainder(self.episode_time - 1, 13)[:, None]
        blue_armor_phase = torch.remainder(self.episode_time - 1, 12)[:, None]
        green_armor_bright = (item_types == 2018) & (green_armor_phase >= 6)
        blue_armor_bright = (item_types == 2019) & (blue_armor_phase >= 6)
        map_item_sprite = torch.where(
            green_armor_bright,
            item_animation[6],
            map_item_sprite,
        )
        map_item_sprite = torch.where(
            blue_armor_bright,
            item_animation[7],
            map_item_sprite,
        )
        map_item_fullbright = green_armor_bright | blue_armor_bright
        item_slots = torch.nonzero(torch.any(self.item_available, dim=0)).flatten()
        if item_slots.numel():
            visible_item_sprite = map_item_sprite[:, item_slots]
            actor_x = torch.cat((actor_x, map_item_x[:, item_slots]), dim=1)
            actor_y = torch.cat((actor_y, map_item_y[:, item_slots]), dim=1)
            actor_z = torch.cat((actor_z, map_item_z[:, item_slots]), dim=1)
            actor_alive = torch.cat(
                (actor_alive, self.item_available[:, item_slots]),
                dim=1,
            )
            actor_sprite = torch.cat((actor_sprite, visible_item_sprite), dim=1)
            actor_fullbright = torch.cat(
                (actor_fullbright, map_item_fullbright[:, item_slots]),
                dim=1,
            )
            actor_additive_style = torch.cat(
                (
                    actor_additive_style,
                    torch.full_like(visible_item_sprite, -1, dtype=torch.int64),
                ),
                dim=1,
            )

        drop_visible = (self.drop_type >= 0) & self.drop_spawned
        drop_sprite = static[12].expand_as(self.drop_type)
        drop_sprite = torch.where(self.drop_type == 2007, static[6], drop_sprite)
        drop_sprite = torch.where(self.drop_type == 2002, static[14], drop_sprite)
        drop_slots = torch.nonzero(torch.any(drop_visible, dim=0)).flatten()
        if drop_slots.numel():
            visible_drop_sprite = drop_sprite[:, drop_slots]
            actor_x = torch.cat((actor_x, self.drop_x[:, drop_slots]), dim=1)
            actor_y = torch.cat((actor_y, self.drop_y[:, drop_slots]), dim=1)
            actor_z = torch.cat((actor_z, self.drop_z[:, drop_slots]), dim=1)
            actor_alive = torch.cat((actor_alive, drop_visible[:, drop_slots]), dim=1)
            actor_sprite = torch.cat((actor_sprite, visible_drop_sprite), dim=1)
            actor_fullbright = torch.cat(
                (
                    actor_fullbright,
                    torch.zeros_like(visible_drop_sprite, dtype=torch.bool),
                ),
                dim=1,
            )
            actor_additive_style = torch.cat(
                (
                    actor_additive_style,
                    torch.full_like(visible_drop_sprite, -1, dtype=torch.int64),
                ),
                dim=1,
            )

        dx = actor_x - self.x[:, None]
        dy = actor_y - self.y[:, None]
        actor_distance = torch.sqrt(dx * dx + dy * dy).clamp_min_(1)
        relative = self._wrap_angle(torch.atan2(dy, dx) - self.angle[:, None])
        actor_depth = actor_distance * torch.cos(relative)
        screen_center = (
            self.native_screen_width / 2.0
            - 1.0
            - torch.tan(relative) * horizontal_focal_length
        )
        horizontal_scale = horizontal_focal_length / actor_depth.clamp_min(1)
        vertical_scale = vertical_focal_length / actor_depth.clamp_min(1)
        sprite_width = self.map.raw_sprite_widths[actor_sprite].to(torch.float32)
        sprite_height = self.map.raw_sprite_heights[actor_sprite].to(torch.float32)
        sprite_left = (
            screen_center - self.map.raw_sprite_left_offsets[actor_sprite] * horizontal_scale
        )
        sprite_top = (
            center[:, None]
            + (view_z[:, None] - actor_z) * vertical_scale
            - self.map.raw_sprite_top_offsets[actor_sprite] * vertical_scale
        )
        sprite_right = sprite_left + sprite_width * horizontal_scale
        column_inside = (self._native_pixel_x >= sprite_left[:, :, None]) & (
            self._native_pixel_x < sprite_right[:, :, None]
        )
        candidate = (
            column_inside
            & actor_alive[:, :, None]
            & (relative[:, :, None].abs() < math.pi / 4)
            & (actor_depth[:, :, None] > 0)
            & (actor_depth[:, :, None] < wall_distance[:, None, :])
        )
        candidate_distance = torch.where(
            candidate,
            actor_depth[:, :, None],
            torch.full_like(actor_depth[:, :, None], torch.inf),
        )
        actor_sector = self._sector_at(actor_x.reshape(-1), actor_y.reshape(-1)).reshape_as(actor_x)
        actor_light = self.map.sector_lights[actor_sector]
        _weapon_frame, _weapon_flash, flash_light = self._native_weapon_frame_selection()
        # Native rendering is a diagnostic fidelity path, not the compiled
        # training observation path. Resolve every horizontally overlapping
        # sprite so transparent foreground texels can reveal arbitrarily deep
        # actors, while avoiding work for inactive/off-screen slots.
        layer_count = int(torch.amax(torch.sum(candidate, dim=1)).item())
        if layer_count == 0:
            return frame
        nearest_distances, nearest_actors = torch.topk(
            candidate_distance,
            k=layer_count,
            dim=1,
            largest=False,
            sorted=True,
        )
        composited = frame
        # Paint far-to-near so transparent texels reveal the next valid sprite
        # and additive missiles blend over any sprite behind them.
        for layer in range(layer_count - 1, -1, -1):
            selected_actor = nearest_actors[:, layer, :]
            selected_distance = nearest_distances[:, layer, :]
            selected_sprite = actor_sprite.gather(1, selected_actor)
            selected_horizontal_scale = horizontal_scale.gather(1, selected_actor)
            selected_vertical_scale = vertical_scale.gather(1, selected_actor)
            selected_left = sprite_left.gather(1, selected_actor)
            selected_top = sprite_top.gather(1, selected_actor)
            selected_width = sprite_width.gather(1, selected_actor).to(torch.int64)
            selected_height = sprite_height.gather(1, selected_actor).to(torch.int64)
            sprite_u = torch.floor(
                (self._native_pixel_x[:, 0, :] - selected_left) / selected_horizontal_scale
            ).to(torch.int64)
            sprite_v = torch.floor(
                (self._native_pixel_y - selected_top[:, None, :])
                / selected_vertical_scale[:, None, :]
            ).to(torch.int64)
            inside_sprite = (
                torch.isfinite(selected_distance)[:, None, :]
                & (selected_distance[:, None, :] < scene_depth)
                & (sprite_u[:, None, :] >= 0)
                & (sprite_u[:, None, :] < selected_width[:, None, :])
                & (sprite_v >= 0)
                & (sprite_v < selected_height[:, None, :])
            )
            sprite_u = sprite_u.clamp_min(0)[:, None, :].expand(
                -1, self.native_view_height, -1
            )
            sprite_v = sprite_v.clamp_min(0)
            sprite_u = torch.minimum(sprite_u, (selected_width - 1)[:, None, :])
            sprite_v = torch.minimum(sprite_v, (selected_height - 1)[:, None, :])
            sprite_type = selected_sprite[:, None, :].expand(
                -1, self.native_view_height, -1
            )
            sprite_opaque = self.map.raw_sprite_opaque[sprite_type, sprite_v, sprite_u]
            sprite_value = self.map.raw_sprite_atlas[sprite_type, sprite_v, sprite_u]
            selected_light = actor_light.gather(1, selected_actor)[:, None, :]
            selected_light = selected_light + flash_light[:, None, None] * 16
            selected_fullbright = actor_fullbright.gather(1, selected_actor)[:, None, :]
            selected_light = torch.where(
                selected_fullbright,
                torch.full_like(selected_light, 255),
                selected_light,
            )
            lit_sprite = self._native_apply_colormap(
                sprite_value,
                selected_light,
                selected_distance[:, None, :],
            )
            selected_additive_style = actor_additive_style.gather(
                1, selected_actor
            )[:, None, :]
            additive_style = selected_additive_style.clamp(0, 1)
            additive_sprite = self.map.projectile_additive_luts[
                additive_style,
                composited.to(torch.int64),
                lit_sprite.to(torch.int64),
            ]
            rendered_sprite = torch.where(
                selected_additive_style >= 0,
                additive_sprite,
                lit_sprite,
            )
            composited = torch.where(
                inside_sprite & sprite_opaque,
                rendered_sprite,
                composited,
            )
        return composited

    def _native_weapon_frame_selection(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weapon = self._active_weapon().clamp(0, 7)
        cooldown = self.weapon_state_cooldown.to(torch.int64).clamp(
            0,
            self.map.native_weapon_frame_ids.shape[2] - 1,
        )
        parity = torch.remainder(self.weapon_fire_count, 2).to(torch.int64)
        idle_chainsaw = (weapon == 1) & (cooldown == 0)
        idle_phase = torch.remainder(
            torch.clamp_min(self.weapon_ready_tics - 1, 0) // 4,
            2,
        ).to(torch.int64)
        parity = torch.where(idle_chainsaw, idle_phase, parity)
        frame_id = self.map.native_weapon_frame_ids[weapon, parity, cooldown]
        flash_id = self.map.native_weapon_flash_ids[weapon, parity, cooldown]
        flash_light = self.map.native_weapon_flash_lights[weapon, parity, cooldown]
        return frame_id, flash_id, flash_light

    def _native_shift_weapon_overlay(
        self,
        value: torch.Tensor,
        alpha: torch.Tensor,
        horizontal_pixels: torch.Tensor,
        vertical_pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_y = self._native_pixel_y.to(torch.int64) - vertical_pixels[:, None, None]
        source_x = self._native_pixel_x.to(torch.int64) - horizontal_pixels[:, None, None]
        valid = (
            (source_y >= 0)
            & (source_y < self.native_view_height)
            & (source_x >= 0)
            & (source_x < self.native_screen_width)
        )
        source_y = source_y.clamp(0, self.native_view_height - 1).expand(
            -1, -1, self.native_screen_width
        )
        source_x = source_x.clamp(0, self.native_screen_width - 1).expand(
            -1, self.native_view_height, -1
        )
        return value.gather(1, source_y).gather(2, source_x), (
            alpha.gather(1, source_y).gather(2, source_x) & valid
        )

    def _native_render_weapon(self, frame: torch.Tensor) -> torch.Tensor:
        frame_id, flash_id, _flash_light = self._native_weapon_frame_selection()
        value = self.map.native_weapon_frame_values[frame_id]
        alpha = self.map.native_weapon_frame_alpha[frame_id]
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
        spawn_raise_tics = torch.clamp(
            _WEAPON_SPAWN_RAISE_TICS - (self.episode_time - 1),
            0,
            _WEAPON_SPAWN_RAISE_TICS,
        )
        vertical_tics = torch.maximum(vertical_tics, spawn_raise_tics)
        raise_pixels = torch.floor(
            vertical_tics.to(torch.float32) * _WEAPON_VERTICAL_STEP_PIXELS
        ).to(torch.int64)
        ready = (
            (self.weapon_state_cooldown <= 0)
            & (self.weapon_raise_cooldown <= 0)
            & (self.pending_weapon < 0)
        )
        bob_angle = (self.episode_time.to(torch.int64) * 128) & (_FINE_ANGLES - 1)
        bob_x_fixed = (
            self._player_bob_fixed
            * self._fine_sine_fixed[(bob_angle + _FINE_ANGLES // 4) & (_FINE_ANGLES - 1)]
        ) >> 16
        bob_y_fixed = (
            self._player_bob_fixed
            * self._fine_sine_fixed[bob_angle & (_FINE_ANGLES // 2 - 1)]
        ) >> 16
        # The software renderer converts the fixed psprite origin to screen
        # coordinates by dropping fractional bits.  Flooring matters for both
        # the negative horizontal swing and the positive vertical swing.
        bob_x = torch.floor(bob_x_fixed.to(torch.float32) / _FIXED_UNIT).to(torch.int64)
        bob_y = torch.floor(
            bob_y_fixed.to(torch.float32) / _FIXED_UNIT * self.native_vertical_aspect
        ).to(torch.int64)
        bob_x = torch.where(ready, bob_x, torch.zeros_like(bob_x))
        bob_y = torch.where(ready, bob_y, torch.zeros_like(bob_y))
        value, alpha = self._native_shift_weapon_overlay(
            value,
            alpha,
            bob_x,
            raise_pixels + bob_y,
        )
        visible = ~self.player_dead[:, None, None]
        frame = torch.where(alpha & visible, value, frame)

        has_flash = flash_id >= 0
        safe_flash_id = flash_id.clamp_min(0)
        flash_value = self.map.native_weapon_frame_values[safe_flash_id]
        flash_alpha = self.map.native_weapon_frame_alpha[safe_flash_id]
        flash_value, flash_alpha = self._native_shift_weapon_overlay(
            flash_value,
            flash_alpha,
            bob_x,
            raise_pixels + bob_y,
        )
        return torch.where(
            flash_alpha & has_flash[:, None, None] & visible,
            flash_value,
            frame,
        )

    def _native_draw_hud_patch(
        self,
        canvas: torch.Tensor,
        patch_index: int,
        x: int,
        y: int,
    ) -> None:
        x -= int(self.map.hud_patch_left_offsets[patch_index].item())
        y -= int(self.map.hud_patch_top_offsets[patch_index].item())
        width = int(self.map.hud_patch_widths[patch_index].item())
        height = int(self.map.hud_patch_heights[patch_index].item())
        if width <= 0 or height <= 0:
            return
        source_x = max(-x, 0)
        source_y = max(-y, 0)
        target_x = max(x, 0)
        target_y = max(y, 0)
        copy_width = min(width - source_x, canvas.shape[1] - target_x)
        copy_height = min(height - source_y, canvas.shape[0] - target_y)
        if copy_width <= 0 or copy_height <= 0:
            return
        source = np.s_[
            source_y : source_y + copy_height,
            source_x : source_x + copy_width,
        ]
        target = np.s_[
            target_y : target_y + copy_height,
            target_x : target_x + copy_width,
        ]
        value = self.map.hud_patch_atlas[patch_index][source]
        opaque = self.map.hud_patch_opaque[patch_index][source]
        canvas[target].copy_(torch.where(opaque, value, canvas[target]))

    def _native_draw_hud_number(
        self,
        canvas: torch.Tensor,
        value: int,
        right: int,
        y: int,
        *,
        small: bool = False,
    ) -> None:
        text = str(max(-99, min(value, 999)))
        digit_width = 4 if small else 14
        base = 28 if small else 2
        x = right - len(text) * digit_width
        for character in text:
            if character == "-":
                x += digit_width
                continue
            patch = base + int(character)
            glyph_x = x - int(self.map.hud_patch_left_offsets[patch].item())
            glyph_y = y - int(self.map.hud_patch_top_offsets[patch].item())
            self._native_draw_hud_patch(canvas, patch, glyph_x, glyph_y)
            x += digit_width

    def _native_mugshot_patch_index(self, lane: int, health: int) -> int:
        pain = max(0, min((100 - health) // 20, 4))
        straight = int(self.mugshot_face_index[lane].item())
        face_patch = 13 + pain * 3 + straight
        if bool(self.mugshot_grin_tics[lane] > 0):
            face_patch = 64 + pain
        elif bool(self.mugshot_pain_tics[lane] > 0):
            if bool(self.mugshot_ouch[lane]):
                face_patch = 59 + pain
            else:
                direction = int(self.mugshot_pain_direction[lane].item())
                face_patch = (44, 49, 54)[direction] + pain
        elif bool(self.attack_held_tics[lane] >= _MUGSHOT_RAMPAGE_DELAY):
            face_patch = 49 + pain
        if health <= 0:
            face_patch = 69
        return face_patch

    def _native_render_hud(self) -> torch.Tensor:
        hud = torch.zeros(
            (self.num_envs, 32, self.native_screen_width),
            device=self.device,
            dtype=torch.uint8,
        )
        for lane in range(self.num_envs):
            canvas = hud[lane]
            self._native_draw_hud_patch(canvas, 0, 0, 0)
            self._native_draw_hud_patch(canvas, 1, 104, 0)
            active_weapon = int(self._active_weapon()[lane].item())
            ammo_slot = int(self._weapon_ammo_slot[active_weapon].item())
            ammo = 0 if ammo_slot < 0 else int(self.hud_ready_ammo[lane].item())
            health = int(self.health[lane].clamp(0, 999).item())
            armor = int(self.armor[lane].clamp(0, 999).item())
            self._native_draw_hud_number(canvas, ammo, 44, 3)
            self._native_draw_hud_number(canvas, health, 90, 3)
            self._native_draw_hud_patch(canvas, 12, 90, 3)
            self._native_draw_hud_number(canvas, armor, 221, 3)
            self._native_draw_hud_patch(canvas, 12, 221, 3)
            face_patch = self._native_mugshot_patch_index(lane, health)
            self._native_draw_hud_patch(canvas, face_patch, 143, 0)
            for weapon_index, (x, y) in enumerate(
                ((111, 4), (123, 4), (135, 4), (111, 14), (123, 14), (135, 14))
            ):
                owned = weapon_index < 5 and bool(self.weapons[lane, weapon_index + 1])
                patch = 28 + weapon_index + 2 if owned else 38 + weapon_index
                self._native_draw_hud_patch(canvas, patch, x, y)
            ammo_values = (
                int(self.ammo[lane, 1].item()),
                int(self.ammo[lane, 2].item()),
                int(self.ammo[lane, 4].item()),
                int(self.ammo[lane, 5].item()),
            )
            for row, (value, maximum) in enumerate(
                zip(ammo_values, (200, 50, 50, 300), strict=True)
            ):
                y = 5 + row * 6
                self._native_draw_hud_number(canvas, value, 288, y, small=True)
                self._native_draw_hud_number(canvas, maximum, 314, y, small=True)
        return hud

    def render_native_frame(self, *, include_hud: bool = True) -> torch.Tensor:
        """Render the unprocessed ViZDoom-compatible 320x240 RGB24 view."""

        wall_distance = self._native_raycast()
        sector = self._current_sector()
        view_z = self.view_z
        frame, surface_depth = self._native_render_flats(sector, view_z)
        frame, scene_depth = self._native_render_portal_walls(frame, view_z, surface_depth)
        frame = self._native_render_sprites(frame, wall_distance, view_z, scene_depth)
        frame = self._native_render_weapon(frame)
        if include_hud:
            frame = torch.cat((frame, self._native_render_hud()), dim=1)
        rgb = self.map.playpal[frame.to(torch.int64)]
        bonus = torch.minimum(
            self.bonus_count.to(torch.float32) * 8.0, torch.full_like(self.health, 128.0)
        )
        bonus = (bonus / 255.0)[:, None, None, None]
        gold = torch.tensor((215.0, 186.0, 69.0), device=self.device)
        rgb = rgb.to(torch.float32) * (1 - bonus) + gold * bonus
        flash = self._damage_to_alpha[self.damage_count.clamp(0, 113).to(torch.int64)] / 255.0
        flash = flash[:, None, None, None]
        red = torch.tensor((255.0, 0.0, 0.0), device=self.device)
        rgb = rgb * (1 - flash) + red * flash
        return rgb.clamp(0, 255).to(torch.uint8)

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
