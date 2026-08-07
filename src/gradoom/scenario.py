"""Ahead-of-time compilation of the pinned ViZDoom deathmatch map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .textures import (
    compile_grayscale_atlas,
    compile_indexed_atlas,
    compile_indexed_patch_atlas,
    compile_indexed_sprite_atlas,
    compile_indexed_weapon_overlays,
    compile_sprite_atlas,
    compile_weapon_overlays,
)
from .wad import UdmfDocument, WadArchive, parse_udmf

PINNED_DEATHMATCH_WAD_SHA256 = "1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d"
KNOWN_DOOM2_WAD_SHA256 = "10d67824b11025ddd9198e8cfc87ca335ee6e2d3e63af4180fa9b8a471893255"
DEATHMATCH_SPRITE_FRAMES = (
    "POSSA1",
    "SPOSA1",
    "PLAYA1",
    "CPOSA1",
    "SARGA1",
    "BOS2A1",
    "STIMA0",
    "MEDIA0",
    "BON1A0",
    "BON2A0",
    "ARM1A0",
    "ARM2A0",
    "CLIPA0",
    "AMMOA0",
    "SBOXA0",
    "BROKA0",
    "CELPA0",
    "CSAWA0",
    "SHOTA0",
    "SGN2A0",
    "MGUNA0",
    "LAUNA0",
    "PLASA0",
    "MISLA1",
    "PLSSA0",
    "BAL7A1",
)
DEATHMATCH_WEAPON_FRAMES = (
    "PUNGA0",
    "SAWGC0",
    "PISGA0",
    "SHTGA0",
    "SHT2A0",
    "CHGGA0",
    "MISGA0",
    "PLSGA0",
)
DEATHMATCH_NATIVE_WEAPON_FRAMES = (
    "PUNGA0",
    "PUNGB0",
    "PUNGC0",
    "PUNGD0",
    "SAWGA0",
    "SAWGB0",
    "SAWGC0",
    "SAWGD0",
    "PISGA0",
    "PISGB0",
    "PISGC0",
    "PISFA0",
    "SHTGA0",
    "SHTGB0",
    "SHTGC0",
    "SHTGD0",
    "SHTFA0",
    "SHTFB0",
    *(f"SHT2{frame}0" for frame in "ABCDEFGHIJ"),
    "CHGGA0",
    "CHGGB0",
    "CHGFA0",
    "CHGFB0",
    "MISGA0",
    "MISGB0",
    *(f"MISF{frame}0" for frame in "ABCD"),
    "PLSGA0",
    "PLSGB0",
    "PLSFA0",
    "PLSFB0",
)
DEATHMATCH_ENEMY_PREFIXES = ("POSS", "SPOS", "PLAY", "CPOS", "SARG", "BOS2")
DEATHMATCH_ENEMY_ATTACK_FRAMES = (
    ("E", "F", "E", "F"),
    ("E", "F", "E", "F"),
    ("E", "F", "E", "F"),
    ("E", "F", "E", "F"),
    ("E", "F", "G", "H"),
    ("E", "F", "G", "H"),
)
DEATHMATCH_ENEMY_DEATH_FRAMES = (
    tuple("HIJKL"),
    tuple("HIJKL"),
    tuple("HIJKLMN"),
    tuple("HIJKLMN"),
    tuple("IJKLMN"),
    tuple("IJKLMNO"),
)
DEATHMATCH_ENEMY_DEATH_DURATIONS = (
    (5, 5, 5, 5, 1),
    (5, 5, 5, 5, 1),
    (10, 10, 10, 10, 10, 10, 1),
    (5, 5, 5, 5, 5, 5, 1),
    (8, 8, 4, 4, 4, 1),
    (8, 8, 8, 8, 8, 8, 1),
)
DEATHMATCH_ENEMY_PAIN_FRAMES = ("G", "G", "G", "G", "H", "H")
DEATHMATCH_PROJECTILE_EXPLOSION_FRAMES = (
    ("MISLB0", "MISLC0", "MISLD0"),
    tuple(f"PLSE{frame}0" for frame in "ABCDE"),
    tuple(f"BAL7{frame}0" for frame in "CDE"),
)
DEATHMATCH_PROJECTILE_EXPLOSION_DURATIONS = (
    (8, 6, 4),
    (4, 4, 4, 4, 4),
    (6, 6, 6),
)
DEATHMATCH_TELEPORT_FOG_FRAMES = tuple(f"TFOG{frame}0" for frame in "ABABCDEFGHIJ")
DEATHMATCH_HUD_PATCHES = (
    "STBAR",
    "STARMS",
    *(f"STTNUM{digit}" for digit in range(10)),
    "STTPRCNT",
    *(f"STFST{pain}{straight}" for pain in range(5) for straight in range(3)),
    *(f"STYSNUM{digit}" for digit in range(10)),
    *(f"STGNUM{digit}" for digit in range(2, 8)),
    *(f"STFTR{pain}0" for pain in range(5)),
    *(f"STFKILL{pain}" for pain in range(5)),
    *(f"STFTL{pain}0" for pain in range(5)),
    *(f"STFOUCH{pain}" for pain in range(5)),
    *(f"STFEVL{pain}" for pain in range(5)),
    "STFDEAD0",
)
DEATHMATCH_TEXTURE_ANIMATIONS = (
    ("BFALL1", "BFALL2", "BFALL3", "BFALL4"),
    ("BLOOD1", "BLOOD2", "BLOOD3"),
)
DEATHMATCH_ITEM_ANIMATION_FRAMES = (
    "BON1B0",
    "BON1C0",
    "BON1D0",
    "BON2B0",
    "BON2C0",
    "BON2D0",
    "ARM1B0",
    "ARM2B0",
)


def _compile_projectile_additive_luts(playpal: np.ndarray) -> np.ndarray:
    """Build the indexed-color tables used by ZDoom-style additive missiles."""

    palette = playpal.astype(np.int64)
    palette_candidates = palette[1:255]
    rgb32k = np.empty(32 * 32 * 32, dtype=np.uint8)
    for red_5bit in range(32):
        green_5bit = np.repeat(np.arange(32, dtype=np.int64), 32)
        blue_5bit = np.tile(np.arange(32, dtype=np.int64), 32)
        targets = np.stack(
            (
                np.full(1024, (red_5bit << 3) | (red_5bit >> 2), dtype=np.int64),
                (green_5bit << 3) | (green_5bit >> 2),
                (blue_5bit << 3) | (blue_5bit >> 2),
            ),
            axis=1,
        )
        delta = targets[:, None, :] - palette_candidates[None, :, :]
        start = red_5bit * 1024
        rgb32k[start : start + 1024] = (
            np.argmin(np.sum(delta * delta, axis=2), axis=1) + 1
        ).astype(np.uint8)

    def col2rgb8(level: int) -> np.ndarray:
        swizzled = (
            (((palette[:, 0] * level) >> 4) << 20)
            | ((palette[:, 1] * level) >> 4)
            | (((palette[:, 2] * level) >> 4) << 10)
        )
        if level not in (0, 64):
            swizzled &= 0x3FEFFBFF
        return swizzled

    tables = np.empty((2, 256, 256), dtype=np.uint8)
    background = col2rgb8(64)[:, None]
    for style, foreground_level in enumerate((48, 64)):
        added = col2rgb8(foreground_level)[None, :] + background
        overflow = added.copy() & 0x40100400
        clamped = (added | 0x01F07C1F) & 0x3FFFFFFF
        clamped |= overflow - (overflow >> 5)
        rgb32k_index = (clamped & (clamped >> 15)).astype(np.int64)
        tables[style] = rgb32k[rgb32k_index]
    return tables


@dataclass(frozen=True)
class CompiledScenario:
    """Host representation copied once into immutable device tensors."""

    scenario_sha256: str
    iwad_sha256: str
    namespace: str
    vertices: np.ndarray
    wall_segments: np.ndarray
    blocking_segments: np.ndarray
    blocking_wall_indices: np.ndarray
    wall_texture_ids: np.ndarray
    wall_texture_offsets: np.ndarray
    wall_side_texture_ids: np.ndarray
    wall_side_texture_offsets: np.ndarray
    wall_sectors: np.ndarray
    sector_edge_mask: np.ndarray
    sector_heights: np.ndarray
    sector_lights: np.ndarray
    sector_floor_texture_ids: np.ndarray
    sector_ceiling_texture_ids: np.ndarray
    player_starts: np.ndarray
    item_spawns: np.ndarray
    item_types: np.ndarray
    playpal: np.ndarray
    texture_names: tuple[str, ...]
    texture_atlas: np.ndarray
    texture_widths: np.ndarray
    texture_heights: np.ndarray
    sprite_names: tuple[str, ...]
    sprite_atlas: np.ndarray
    sprite_opaque: np.ndarray
    sprite_widths: np.ndarray
    sprite_heights: np.ndarray
    sprite_left_offsets: np.ndarray
    sprite_top_offsets: np.ndarray
    weapon_sprite_names: tuple[str, ...]
    weapon_screen_values: np.ndarray
    weapon_screen_alpha: np.ndarray
    texture_index_atlas: np.ndarray | None = None
    colormap: np.ndarray | None = None
    raw_sprite_names: tuple[str, ...] = ()
    raw_sprite_atlas: np.ndarray | None = None
    raw_sprite_opaque: np.ndarray | None = None
    raw_sprite_widths: np.ndarray | None = None
    raw_sprite_heights: np.ndarray | None = None
    raw_sprite_left_offsets: np.ndarray | None = None
    raw_sprite_top_offsets: np.ndarray | None = None
    enemy_walk_sprite_ids: np.ndarray | None = None
    enemy_attack_sprite_ids: np.ndarray | None = None
    enemy_death_sprite_ids: np.ndarray | None = None
    enemy_death_frame_counts: np.ndarray | None = None
    enemy_death_frame_durations: np.ndarray | None = None
    enemy_death_total_tics: np.ndarray | None = None
    enemy_pain_sprite_ids: np.ndarray | None = None
    raw_projectile_flight_sprite_ids: np.ndarray | None = None
    raw_projectile_explosion_sprite_ids: np.ndarray | None = None
    raw_teleport_fog_sprite_ids: np.ndarray | None = None
    projectile_explosion_frame_counts: np.ndarray | None = None
    projectile_explosion_frame_durations: np.ndarray | None = None
    projectile_explosion_total_tics: np.ndarray | None = None
    projectile_additive_luts: np.ndarray | None = None
    raw_static_sprite_ids: np.ndarray | None = None
    raw_item_animation_sprite_ids: np.ndarray | None = None
    native_weapon_screen_values: np.ndarray | None = None
    native_weapon_screen_alpha: np.ndarray | None = None
    native_weapon_frame_values: np.ndarray | None = None
    native_weapon_frame_alpha: np.ndarray | None = None
    native_weapon_frame_ids: np.ndarray | None = None
    native_weapon_flash_ids: np.ndarray | None = None
    native_weapon_flash_lights: np.ndarray | None = None
    hud_patch_names: tuple[str, ...] = ()
    hud_patch_atlas: np.ndarray | None = None
    hud_patch_opaque: np.ndarray | None = None
    hud_patch_widths: np.ndarray | None = None
    hud_patch_heights: np.ndarray | None = None
    hud_patch_left_offsets: np.ndarray | None = None
    hud_patch_top_offsets: np.ndarray | None = None
    texture_animation_ids: np.ndarray | None = None
    texture_animation_counts: np.ndarray | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.vertices[:, 0].min()),
            float(self.vertices[:, 0].max()),
            float(self.vertices[:, 1].min()),
            float(self.vertices[:, 1].max()),
        )


def _required_blocks(document: UdmfDocument, name: str):
    try:
        return document.blocks[name]
    except KeyError as exc:
        raise ValueError(f"deathmatch TEXTMAP has no {name!r} blocks") from exc


def compile_deathmatch_scenario(
    scenario_wad: str | Path,
    iwad: str | Path,
    *,
    require_pinned_scenario: bool = True,
) -> CompiledScenario:
    """Compile UDMF geometry and actor placements without retaining WAD bytes."""

    scenario = WadArchive.from_path(scenario_wad)
    game = WadArchive.from_path(iwad)
    if require_pinned_scenario and scenario.sha256 != PINNED_DEATHMATCH_WAD_SHA256:
        raise ValueError(
            "scenario WAD does not match the certified ViZDoom deathmatch asset: "
            f"expected {PINNED_DEATHMATCH_WAD_SHA256}, got {scenario.sha256}"
        )
    if scenario.identity != "PWAD":
        raise ValueError("deathmatch scenario must be a PWAD")
    if game.identity != "IWAD":
        raise ValueError("base game data must be an IWAD")
    document = parse_udmf(scenario.read("TEXTMAP"))
    vertices_raw = _required_blocks(document, "vertex")
    linedefs = _required_blocks(document, "linedef")
    sidedefs = _required_blocks(document, "sidedef")
    sectors = _required_blocks(document, "sector")
    things = _required_blocks(document, "thing")

    vertices = np.asarray(
        [(float(vertex["x"]), float(vertex["y"])) for vertex in vertices_raw],
        dtype=np.float32,
    )
    sidedef_sectors = np.asarray([int(side.get("sector", -1)) for side in sidedefs], dtype=np.int32)
    wall_segments = np.empty((len(linedefs), 4), dtype=np.float32)
    blocking: list[np.ndarray] = []
    blocking_indices: list[int] = []
    wall_texture_names = [""] * len(linedefs)
    wall_texture_offsets = np.zeros((len(linedefs), 2), dtype=np.float32)
    wall_side_texture_names = [[["", "", ""], ["", "", ""]] for _ in range(len(linedefs))]
    wall_side_texture_offsets = np.zeros((len(linedefs), 2, 2), dtype=np.float32)
    wall_sectors = np.full((len(linedefs), 2), -1, dtype=np.int32)
    for index, line in enumerate(linedefs):
        v1 = int(line["v1"])
        v2 = int(line["v2"])
        if not 0 <= v1 < len(vertices) or not 0 <= v2 < len(vertices):
            raise ValueError(f"linedef {index} references an invalid vertex")
        wall_segments[index] = (*vertices[v1], *vertices[v2])
        front = int(line.get("sidefront", -1))
        back = int(line.get("sideback", -1))
        if front >= len(sidedefs) or back >= len(sidedefs):
            raise ValueError(f"linedef {index} references an invalid sidedef")
        wall_sectors[index, 0] = sidedef_sectors[front] if front >= 0 else -1
        wall_sectors[index, 1] = sidedef_sectors[back] if back >= 0 else -1
        for side_slot, side_index in enumerate((front, back)):
            if side_index < 0:
                continue
            side = sidedefs[side_index]
            wall_side_texture_offsets[index, side_slot] = (
                float(side.get("offsetx", 0)),
                float(side.get("offsety", 0)),
            )
            for texture_slot, key in enumerate(("texturemiddle", "texturebottom", "texturetop")):
                value = str(side.get(key, "")).upper()
                if value not in {"", "-"}:
                    wall_side_texture_names[index][side_slot][texture_slot] = value
        texture_name = None
        texture_side = None
        for side_index in (front, back):
            if side_index < 0:
                continue
            side = sidedefs[side_index]
            texture_name = next(
                (
                    str(side[key]).upper()
                    for key in ("texturemiddle", "texturebottom", "texturetop")
                    if key in side and str(side[key]) not in {"", "-"}
                ),
                None,
            )
            if texture_name is not None:
                texture_side = side
                break
        if texture_name is not None and texture_side is not None:
            wall_texture_names[index] = texture_name
            wall_texture_offsets[index] = (
                float(texture_side.get("offsetx", 0)),
                float(texture_side.get("offsety", 0)),
            )
        if back < 0 or bool(line.get("blocking", False)):
            if front < 0:
                raise ValueError(f"blocking linedef {index} has no front sidedef")
            if texture_name is None:
                raise ValueError(f"blocking linedef {index} has no visible wall texture")
            blocking.append(wall_segments[index])
            blocking_indices.append(index)

    sector_floor_texture_names = tuple(
        str(sector.get("texturefloor", "")).upper() for sector in sectors
    )
    sector_ceiling_texture_names = tuple(
        str(sector.get("textureceiling", "")).upper() for sector in sectors
    )
    if any(not name for name in (*sector_floor_texture_names, *sector_ceiling_texture_names)):
        raise ValueError("every deathmatch sector must declare floor and ceiling textures")
    referenced_texture_names = (
        {name for name in wall_texture_names if name}
        | {
            name
            for sides_for_line in wall_side_texture_names
            for textures_for_side in sides_for_line
            for name in textures_for_side
            if name
        }
        | set(sector_floor_texture_names)
        | set(sector_ceiling_texture_names)
    )
    for animation in DEATHMATCH_TEXTURE_ANIMATIONS:
        if referenced_texture_names.intersection(animation):
            referenced_texture_names.update(animation)
    texture_names = tuple(sorted(referenced_texture_names))
    texture_atlas, texture_widths, texture_heights = compile_grayscale_atlas(game, texture_names)
    texture_index_atlas, _, _ = compile_indexed_atlas(game, texture_names)
    (
        sprite_names,
        sprite_atlas,
        sprite_opaque,
        sprite_widths,
        sprite_heights,
        sprite_left_offsets,
        sprite_top_offsets,
    ) = compile_sprite_atlas(game, DEATHMATCH_SPRITE_FRAMES)
    weapon_sprite_names, weapon_screen_values, weapon_screen_alpha = compile_weapon_overlays(
        game,
        DEATHMATCH_WEAPON_FRAMES,
    )
    raw_sprite_requests: list[str] = []

    def request_raw_sprite(name: str) -> int:
        raw_sprite_requests.append(name)
        return len(raw_sprite_requests) - 1

    enemy_walk_sprite_ids = np.empty((6, 4, 8), dtype=np.int32)
    enemy_attack_sprite_ids = np.empty((6, 4, 8), dtype=np.int32)
    for enemy_type, prefix in enumerate(DEATHMATCH_ENEMY_PREFIXES):
        for frame_index, frame in enumerate(("A", "B", "C", "D")):
            for rotation in range(1, 9):
                enemy_walk_sprite_ids[enemy_type, frame_index, rotation - 1] = request_raw_sprite(
                    f"{prefix}{frame}{rotation}"
                )
        for frame_index, frame in enumerate(DEATHMATCH_ENEMY_ATTACK_FRAMES[enemy_type]):
            for rotation in range(1, 9):
                enemy_attack_sprite_ids[enemy_type, frame_index, rotation - 1] = request_raw_sprite(
                    f"{prefix}{frame}{rotation}"
                )
    max_death_frames = max(len(frames) for frames in DEATHMATCH_ENEMY_DEATH_FRAMES)
    enemy_death_sprite_ids = np.empty((6, max_death_frames), dtype=np.int32)
    enemy_death_frame_counts = np.asarray(
        [len(frames) for frames in DEATHMATCH_ENEMY_DEATH_FRAMES],
        dtype=np.int32,
    )
    enemy_death_frame_durations = np.zeros((6, max_death_frames), dtype=np.int32)
    for enemy_type, durations in enumerate(DEATHMATCH_ENEMY_DEATH_DURATIONS):
        enemy_death_frame_durations[enemy_type, : len(durations)] = durations
    enemy_death_total_tics = enemy_death_frame_durations.sum(axis=1, dtype=np.int32)
    for enemy_type, prefix in enumerate(DEATHMATCH_ENEMY_PREFIXES):
        frames = DEATHMATCH_ENEMY_DEATH_FRAMES[enemy_type]
        for frame_index in range(max_death_frames):
            frame = frames[min(frame_index, len(frames) - 1)]
            enemy_death_sprite_ids[enemy_type, frame_index] = request_raw_sprite(
                f"{prefix}{frame}0"
            )
    enemy_pain_sprite_ids = np.empty((6, 8), dtype=np.int32)
    for enemy_type, (prefix, frame) in enumerate(
        zip(DEATHMATCH_ENEMY_PREFIXES, DEATHMATCH_ENEMY_PAIN_FRAMES, strict=True)
    ):
        for rotation in range(1, 9):
            enemy_pain_sprite_ids[enemy_type, rotation - 1] = request_raw_sprite(
                f"{prefix}{frame}{rotation}"
            )

    raw_projectile_flight_sprite_ids = np.empty((3, 2, 8), dtype=np.int32)
    for rotation in range(1, 9):
        rocket = request_raw_sprite(f"MISLA{rotation}")
        raw_projectile_flight_sprite_ids[0, :, rotation - 1] = rocket
        raw_projectile_flight_sprite_ids[1, 0, rotation - 1] = request_raw_sprite("PLSSA0")
        raw_projectile_flight_sprite_ids[1, 1, rotation - 1] = request_raw_sprite("PLSSB0")
        raw_projectile_flight_sprite_ids[2, 0, rotation - 1] = request_raw_sprite(
            f"BAL7A{rotation}"
        )
        raw_projectile_flight_sprite_ids[2, 1, rotation - 1] = request_raw_sprite(
            f"BAL7B{rotation}"
        )
    max_explosion_frames = max(
        len(frames) for frames in DEATHMATCH_PROJECTILE_EXPLOSION_FRAMES
    )
    raw_projectile_explosion_sprite_ids = np.empty((3, max_explosion_frames), dtype=np.int32)
    projectile_explosion_frame_counts = np.asarray(
        [len(frames) for frames in DEATHMATCH_PROJECTILE_EXPLOSION_FRAMES],
        dtype=np.int32,
    )
    projectile_explosion_frame_durations = np.zeros(
        (3, max_explosion_frames), dtype=np.int32
    )
    for projectile_type, (frames, durations) in enumerate(
        zip(
            DEATHMATCH_PROJECTILE_EXPLOSION_FRAMES,
            DEATHMATCH_PROJECTILE_EXPLOSION_DURATIONS,
            strict=True,
        )
    ):
        projectile_explosion_frame_durations[projectile_type, : len(durations)] = durations
        for frame_index in range(max_explosion_frames):
            frame = frames[min(frame_index, len(frames) - 1)]
            raw_projectile_explosion_sprite_ids[projectile_type, frame_index] = (
                request_raw_sprite(frame)
            )
    projectile_explosion_total_tics = projectile_explosion_frame_durations.sum(
        axis=1, dtype=np.int32
    )
    raw_teleport_fog_sprite_ids = np.asarray(
        [request_raw_sprite(name) for name in DEATHMATCH_TELEPORT_FOG_FRAMES],
        dtype=np.int32,
    )
    raw_static_sprite_ids = np.asarray(
        [request_raw_sprite(name) for name in DEATHMATCH_SPRITE_FRAMES[6:]],
        dtype=np.int32,
    )
    raw_item_animation_sprite_ids = np.asarray(
        [request_raw_sprite(name) for name in DEATHMATCH_ITEM_ANIMATION_FRAMES],
        dtype=np.int32,
    )
    (
        raw_sprite_names,
        raw_sprite_atlas,
        raw_sprite_opaque,
        raw_sprite_widths,
        raw_sprite_heights,
        raw_sprite_left_offsets,
        raw_sprite_top_offsets,
    ) = compile_indexed_sprite_atlas(game, tuple(raw_sprite_requests))
    _, native_weapon_screen_values, native_weapon_screen_alpha = compile_indexed_weapon_overlays(
        game, DEATHMATCH_WEAPON_FRAMES
    )
    _, native_weapon_frame_values, native_weapon_frame_alpha = compile_indexed_weapon_overlays(
        game, DEATHMATCH_NATIVE_WEAPON_FRAMES
    )
    native_weapon_ids_by_name = {
        name: index for index, name in enumerate(DEATHMATCH_NATIVE_WEAPON_FRAMES)
    }
    max_weapon_cooldown = 61
    native_weapon_frame_ids = np.empty((8, 2, max_weapon_cooldown + 1), dtype=np.int32)
    native_weapon_flash_ids = np.full_like(native_weapon_frame_ids, -1)
    native_weapon_flash_lights = np.zeros_like(native_weapon_frame_ids)
    ready_names = ("PUNGA0", "SAWGC0", "PISGA0", "SHTGA0", "SHT2A0", "CHGGA0", "MISGA0", "PLSGA0")
    for weapon, ready_name in enumerate(ready_names):
        native_weapon_frame_ids[weapon] = native_weapon_ids_by_name[ready_name]

    def set_weapon_cycle(
        weapon: int,
        parity: int,
        frames: tuple[tuple[str, int], ...],
    ) -> None:
        cooldown = sum(duration for _name, duration in frames)
        for name, duration in frames:
            next_cooldown = cooldown - duration
            native_weapon_frame_ids[
                weapon,
                parity,
                next_cooldown + 1 : cooldown + 1,
            ] = native_weapon_ids_by_name[name]
            cooldown = next_cooldown

    def set_weapon_flash_cycle(
        weapon: int,
        parity: int,
        frames: tuple[tuple[str | None, int, int], ...],
    ) -> None:
        cooldown = sum(duration for _name, duration, _light in frames)
        for name, duration, light in frames:
            next_cooldown = cooldown - duration
            if name is not None:
                native_weapon_flash_ids[
                    weapon,
                    parity,
                    next_cooldown + 1 : cooldown + 1,
                ] = native_weapon_ids_by_name[name]
            native_weapon_flash_lights[
                weapon,
                parity,
                next_cooldown + 1 : cooldown + 1,
            ] = light
            cooldown = next_cooldown

    for parity in range(2):
        set_weapon_cycle(
            0,
            parity,
            (("PUNGB0", 4), ("PUNGC0", 4), ("PUNGD0", 5), ("PUNGC0", 4), ("PUNGB0", 4)),
        )
        set_weapon_cycle(
            1,
            parity,
            (("SAWGA0", 4), ("SAWGB0", 3)),
        )
        set_weapon_cycle(
            2,
            parity,
            (("PISGA0", 4), ("PISGB0", 6), ("PISGC0", 4), ("PISGB0", 4)),
        )
        set_weapon_cycle(
            3,
            parity,
            (
                ("SHTGA0", 3),
                ("SHTGA0", 7),
                ("SHTGB0", 5),
                ("SHTGC0", 5),
                ("SHTGD0", 4),
                ("SHTGC0", 5),
                ("SHTGB0", 5),
                ("SHTGA0", 3),
                ("SHTGA0", 6),
            ),
        )
        set_weapon_cycle(
            4,
            parity,
            (
                ("SHT2A0", 3),
                ("SHT2A0", 7),
                ("SHT2B0", 7),
                ("SHT2C0", 7),
                ("SHT2D0", 7),
                ("SHT2E0", 7),
                ("SHT2F0", 7),
                ("SHT2G0", 6),
                ("SHT2H0", 6),
                ("SHT2A0", 4),
            ),
        )
        set_weapon_cycle(
            5,
            parity,
            (("CHGGA0", 4), ("CHGGB0", 3)),
        )
        set_weapon_cycle(6, parity, (("MISGB0", 19),))
        set_weapon_cycle(7, parity, (("PLSGA0", 3), ("PLSGB0", 19)))
        set_weapon_flash_cycle(
            2,
            parity,
            ((None, 4, 0), ("PISFA0", 7, 1), (None, 7, 0)),
        )
        set_weapon_flash_cycle(
            3,
            parity,
            ((None, 3, 0), ("SHTFA0", 4, 1), ("SHTFB0", 3, 2), (None, 33, 0)),
        )
        set_weapon_flash_cycle(
            4,
            parity,
            ((None, 3, 0), ("SHT2I0", 4, 1), ("SHT2J0", 3, 2), (None, 51, 0)),
        )
        set_weapon_flash_cycle(
            6,
            parity,
            (
                ("MISFA0", 3, 1),
                ("MISFB0", 4, 1),
                ("MISFC0", 4, 2),
                ("MISFD0", 4, 2),
                (None, 4, 0),
            ),
        )
        set_weapon_flash_cycle(
            7,
            parity,
            (("PLSFA0" if parity else "PLSFB0", 4, 1), (None, 18, 0)),
        )

    set_weapon_flash_cycle(5, 1, (("CHGFA0", 7, 1),))
    set_weapon_flash_cycle(5, 0, (("CHGFB0", 7, 2),))
    native_weapon_frame_ids[1, 0, 0] = native_weapon_ids_by_name["SAWGC0"]
    native_weapon_frame_ids[1, 1, 0] = native_weapon_ids_by_name["SAWGD0"]
    (
        hud_patch_names,
        hud_patch_atlas,
        hud_patch_opaque,
        hud_patch_widths,
        hud_patch_heights,
        hud_patch_left_offsets,
        hud_patch_top_offsets,
    ) = compile_indexed_patch_atlas(game, DEATHMATCH_HUD_PATCHES)
    texture_ids_by_name = {name: index for index, name in enumerate(texture_names)}
    max_animation_frames = max(len(animation) for animation in DEATHMATCH_TEXTURE_ANIMATIONS)
    texture_animation_ids = np.repeat(
        np.arange(len(texture_names), dtype=np.int32)[:, None],
        max_animation_frames,
        axis=1,
    )
    texture_animation_counts = np.ones(len(texture_names), dtype=np.int32)
    for animation in DEATHMATCH_TEXTURE_ANIMATIONS:
        if not set(animation).issubset(texture_ids_by_name):
            continue
        frame_ids = [texture_ids_by_name[name] for name in animation]
        for texture_id in frame_ids:
            texture_animation_ids[texture_id, : len(frame_ids)] = frame_ids
            texture_animation_counts[texture_id] = len(frame_ids)
    wall_texture_ids = np.full(len(linedefs), -1, dtype=np.int32)
    for index, name in enumerate(wall_texture_names):
        if name:
            wall_texture_ids[index] = texture_ids_by_name[name]
    wall_side_texture_ids = np.full((len(linedefs), 2, 3), -1, dtype=np.int32)
    for line_index, sides_for_line in enumerate(wall_side_texture_names):
        for side_index, textures_for_side in enumerate(sides_for_line):
            for texture_index, name in enumerate(textures_for_side):
                if name:
                    wall_side_texture_ids[line_index, side_index, texture_index] = (
                        texture_ids_by_name[name]
                    )
    sector_floor_texture_ids = np.asarray(
        [texture_ids_by_name[name] for name in sector_floor_texture_names], dtype=np.int32
    )
    sector_ceiling_texture_ids = np.asarray(
        [texture_ids_by_name[name] for name in sector_ceiling_texture_names], dtype=np.int32
    )
    sector_edge_mask = np.zeros((len(sectors), len(linedefs)), dtype=np.bool_)
    for line_index, (front_sector, back_sector) in enumerate(wall_sectors):
        if front_sector >= 0:
            sector_edge_mask[front_sector, line_index] = True
        if back_sector >= 0:
            sector_edge_mask[back_sector, line_index] = True

    sector_heights = np.asarray(
        [
            (float(sector.get("heightfloor", 0)), float(sector.get("heightceiling", 128)))
            for sector in sectors
        ],
        dtype=np.float32,
    )
    sector_lights = np.asarray(
        [int(sector.get("lightlevel", 160)) for sector in sectors], dtype=np.int16
    )
    player_starts = np.asarray(
        [
            (float(thing["x"]), float(thing["y"]), float(thing.get("angle", 0)))
            for thing in things
            if int(thing.get("type", -1)) == 1
        ],
        dtype=np.float32,
    )
    if not len(player_starts):
        raise ValueError("deathmatch scenario contains no player starts")
    items = [thing for thing in things if int(thing.get("type", -1)) != 1]
    item_spawns = np.asarray(
        [(float(thing["x"]), float(thing["y"]), float(thing.get("height", 0))) for thing in items],
        dtype=np.float32,
    ).reshape(-1, 3)
    item_types = np.asarray([int(thing["type"]) for thing in items], dtype=np.int32)
    playpal_bytes = game.read("PLAYPAL")
    if len(playpal_bytes) < 256 * 3:
        raise ValueError("IWAD PLAYPAL lump is too small")
    playpal = np.frombuffer(playpal_bytes[: 256 * 3], dtype=np.uint8).reshape(256, 3).copy()
    projectile_additive_luts = _compile_projectile_additive_luts(playpal)
    colormap_bytes = game.read("COLORMAP")
    if len(colormap_bytes) < 34 * 256:
        raise ValueError("IWAD COLORMAP lump is too small")
    colormap = np.frombuffer(colormap_bytes[: 34 * 256], dtype=np.uint8).reshape(34, 256).copy()
    return CompiledScenario(
        scenario_sha256=scenario.sha256,
        iwad_sha256=game.sha256,
        namespace=document.namespace,
        vertices=vertices,
        wall_segments=wall_segments,
        blocking_segments=np.asarray(blocking, dtype=np.float32).reshape(-1, 4),
        blocking_wall_indices=np.asarray(blocking_indices, dtype=np.int32),
        wall_texture_ids=wall_texture_ids,
        wall_texture_offsets=wall_texture_offsets,
        wall_side_texture_ids=wall_side_texture_ids,
        wall_side_texture_offsets=wall_side_texture_offsets,
        wall_sectors=wall_sectors,
        sector_edge_mask=sector_edge_mask,
        sector_heights=sector_heights,
        sector_lights=sector_lights,
        sector_floor_texture_ids=sector_floor_texture_ids,
        sector_ceiling_texture_ids=sector_ceiling_texture_ids,
        player_starts=player_starts,
        item_spawns=item_spawns,
        item_types=item_types,
        playpal=playpal,
        texture_names=texture_names,
        texture_atlas=texture_atlas,
        texture_widths=texture_widths,
        texture_heights=texture_heights,
        sprite_names=sprite_names,
        sprite_atlas=sprite_atlas,
        sprite_opaque=sprite_opaque,
        sprite_widths=sprite_widths,
        sprite_heights=sprite_heights,
        sprite_left_offsets=sprite_left_offsets,
        sprite_top_offsets=sprite_top_offsets,
        weapon_sprite_names=weapon_sprite_names,
        weapon_screen_values=weapon_screen_values,
        weapon_screen_alpha=weapon_screen_alpha,
        texture_index_atlas=texture_index_atlas,
        colormap=colormap,
        raw_sprite_names=raw_sprite_names,
        raw_sprite_atlas=raw_sprite_atlas,
        raw_sprite_opaque=raw_sprite_opaque,
        raw_sprite_widths=raw_sprite_widths,
        raw_sprite_heights=raw_sprite_heights,
        raw_sprite_left_offsets=raw_sprite_left_offsets,
        raw_sprite_top_offsets=raw_sprite_top_offsets,
        enemy_walk_sprite_ids=enemy_walk_sprite_ids,
        enemy_attack_sprite_ids=enemy_attack_sprite_ids,
        enemy_death_sprite_ids=enemy_death_sprite_ids,
        enemy_death_frame_counts=enemy_death_frame_counts,
        enemy_death_frame_durations=enemy_death_frame_durations,
        enemy_death_total_tics=enemy_death_total_tics,
        enemy_pain_sprite_ids=enemy_pain_sprite_ids,
        raw_projectile_flight_sprite_ids=raw_projectile_flight_sprite_ids,
        raw_projectile_explosion_sprite_ids=raw_projectile_explosion_sprite_ids,
        raw_teleport_fog_sprite_ids=raw_teleport_fog_sprite_ids,
        projectile_explosion_frame_counts=projectile_explosion_frame_counts,
        projectile_explosion_frame_durations=projectile_explosion_frame_durations,
        projectile_explosion_total_tics=projectile_explosion_total_tics,
        projectile_additive_luts=projectile_additive_luts,
        raw_static_sprite_ids=raw_static_sprite_ids,
        raw_item_animation_sprite_ids=raw_item_animation_sprite_ids,
        native_weapon_screen_values=native_weapon_screen_values,
        native_weapon_screen_alpha=native_weapon_screen_alpha,
        native_weapon_frame_values=native_weapon_frame_values,
        native_weapon_frame_alpha=native_weapon_frame_alpha,
        native_weapon_frame_ids=native_weapon_frame_ids,
        native_weapon_flash_ids=native_weapon_flash_ids,
        native_weapon_flash_lights=native_weapon_flash_lights,
        hud_patch_names=hud_patch_names,
        hud_patch_atlas=hud_patch_atlas,
        hud_patch_opaque=hud_patch_opaque,
        hud_patch_widths=hud_patch_widths,
        hud_patch_heights=hud_patch_heights,
        hud_patch_left_offsets=hud_patch_left_offsets,
        hud_patch_top_offsets=hud_patch_top_offsets,
        texture_animation_ids=texture_animation_ids,
        texture_animation_counts=texture_animation_counts,
    )
