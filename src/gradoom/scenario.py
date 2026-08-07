"""Ahead-of-time compilation of the pinned ViZDoom deathmatch map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .textures import compile_grayscale_atlas, compile_sprite_atlas, compile_weapon_overlays
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
    texture_names = tuple(
        sorted(
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
    )
    texture_atlas, texture_widths, texture_heights = compile_grayscale_atlas(game, texture_names)
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
    texture_ids_by_name = {name: index for index, name in enumerate(texture_names)}
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
        [
            (float(thing["x"]), float(thing["y"]), float(thing.get("height", 0)))
            for thing in items
        ],
        dtype=np.float32,
    ).reshape(-1, 3)
    item_types = np.asarray([int(thing["type"]) for thing in items], dtype=np.int32)
    playpal_bytes = game.read("PLAYPAL")
    if len(playpal_bytes) < 256 * 3:
        raise ValueError("IWAD PLAYPAL lump is too small")
    playpal = np.frombuffer(playpal_bytes[: 256 * 3], dtype=np.uint8).reshape(256, 3).copy()
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
    )
