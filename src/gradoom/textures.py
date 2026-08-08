"""Strict Doom texture decoding for ahead-of-time GPU material compilation."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from .wad import WadArchive

_I16 = struct.Struct("<h")
_U16 = struct.Struct("<H")
_I32 = struct.Struct("<i")
_PATCH_HEADER = struct.Struct("<hhhh")
_MAP_TEXTURE_HEADER = struct.Struct("<8sIhhIh")
_MAP_PATCH = struct.Struct("<hhhhh")


@dataclass(frozen=True)
class IndexedTexture:
    name: str
    pixels: np.ndarray
    opaque: np.ndarray
    left_offset: int = 0
    top_offset: int = 0

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


@dataclass(frozen=True)
class TextureCatalog:
    patches: tuple[str, ...]
    textures: dict[str, tuple[int, int, tuple[tuple[int, int, int], ...]]]

    @classmethod
    def from_wad(cls, wad: WadArchive) -> TextureCatalog:
        pnames = wad.read("PNAMES")
        if len(pnames) < 4:
            raise ValueError("IWAD PNAMES lump is truncated")
        patch_count = _I32.unpack_from(pnames)[0]
        if patch_count < 0 or 4 + patch_count * 8 > len(pnames):
            raise ValueError("IWAD PNAMES lump has an invalid patch count")
        patches = tuple(
            pnames[4 + index * 8 : 12 + index * 8].rstrip(b"\0").decode("ascii").upper()
            for index in range(patch_count)
        )
        textures: dict[str, tuple[int, int, tuple[tuple[int, int, int], ...]]] = {}
        for lump_name in ("TEXTURE1", "TEXTURE2"):
            if lump_name not in wad.by_name:
                continue
            payload = wad.read(lump_name)
            if len(payload) < 4:
                raise ValueError(f"IWAD {lump_name} lump is truncated")
            count = _I32.unpack_from(payload)[0]
            if count < 0 or 4 + count * 4 > len(payload):
                raise ValueError(f"IWAD {lump_name} has an invalid texture count")
            for index in range(count):
                offset = _I32.unpack_from(payload, 4 + index * 4)[0]
                if offset < 0 or offset + _MAP_TEXTURE_HEADER.size > len(payload):
                    raise ValueError(f"IWAD {lump_name} texture {index} is out of bounds")
                raw_name, _masked, width, height, _column_directory, num_patches = (
                    _MAP_TEXTURE_HEADER.unpack_from(payload, offset)
                )
                if width <= 0 or height <= 0 or num_patches < 0:
                    raise ValueError(f"IWAD {lump_name} texture {index} has invalid dimensions")
                patch_offset = offset + _MAP_TEXTURE_HEADER.size
                patch_end = patch_offset + num_patches * _MAP_PATCH.size
                if patch_end > len(payload):
                    raise ValueError(f"IWAD {lump_name} texture {index} patches are truncated")
                placements: list[tuple[int, int, int]] = []
                for patch_index in range(num_patches):
                    origin_x, origin_y, pnames_index, _stepdir, _colormap = _MAP_PATCH.unpack_from(
                        payload, patch_offset + patch_index * _MAP_PATCH.size
                    )
                    if not 0 <= pnames_index < len(patches):
                        raise ValueError(
                            f"IWAD {lump_name} texture {index} references invalid PNAMES index"
                        )
                    placements.append((origin_x, origin_y, pnames_index))
                name = raw_name.rstrip(b"\0").decode("ascii").upper()
                textures[name] = (width, height, tuple(placements))
        return cls(patches=patches, textures=textures)

    def decode(self, wad: WadArchive, name: str) -> IndexedTexture:
        normalized = name.upper()
        try:
            width, height, placements = self.textures[normalized]
        except KeyError:
            payload = wad.read(normalized)
            if len(payload) != 64 * 64:
                raise KeyError(f"IWAD has no wall texture or 64x64 flat {normalized!r}") from None
            pixels = np.frombuffer(payload, dtype=np.uint8).reshape(64, 64).copy()
            return IndexedTexture(
                name=normalized,
                pixels=pixels,
                opaque=np.ones_like(pixels, dtype=np.bool_),
            )
        pixels = np.zeros((height, width), dtype=np.uint8)
        opaque = np.zeros((height, width), dtype=np.bool_)
        for origin_x, origin_y, patch_index in placements:
            patch = decode_patch(wad.read(self.patches[patch_index]), self.patches[patch_index])
            x0 = max(origin_x, 0)
            y0 = max(origin_y, 0)
            x1 = min(origin_x + patch.width, width)
            y1 = min(origin_y + patch.height, height)
            if x0 >= x1 or y0 >= y1:
                continue
            source_x = x0 - origin_x
            source_y = y0 - origin_y
            source = np.s_[source_y : source_y + (y1 - y0), source_x : source_x + (x1 - x0)]
            destination = np.s_[y0:y1, x0:x1]
            mask = patch.opaque[source]
            pixels[destination][mask] = patch.pixels[source][mask]
            opaque[destination][mask] = True
        return IndexedTexture(name=normalized, pixels=pixels, opaque=opaque)


def decode_patch(payload: bytes, name: str = "<patch>") -> IndexedTexture:
    if len(payload) < _PATCH_HEADER.size:
        raise ValueError(f"Doom patch {name!r} is truncated")
    width, height, left_offset, top_offset = _PATCH_HEADER.unpack_from(payload)
    if width <= 0 or height <= 0 or _PATCH_HEADER.size + width * 4 > len(payload):
        raise ValueError(f"Doom patch {name!r} has invalid dimensions")
    pixels = np.zeros((height, width), dtype=np.uint8)
    opaque = np.zeros((height, width), dtype=np.bool_)
    for column in range(width):
        cursor = _I32.unpack_from(payload, _PATCH_HEADER.size + column * 4)[0]
        if cursor < 0 or cursor >= len(payload):
            raise ValueError(f"Doom patch {name!r} column {column} is out of bounds")
        previous_top = -1
        while True:
            if cursor >= len(payload):
                raise ValueError(f"Doom patch {name!r} column {column} has no terminator")
            top = payload[cursor]
            cursor += 1
            if top == 0xFF:
                break
            if cursor + 2 > len(payload):
                raise ValueError(f"Doom patch {name!r} column {column} post is truncated")
            length = payload[cursor]
            cursor += 2
            if top <= previous_top:
                top += previous_top
            previous_top = top
            if cursor + length + 1 > len(payload):
                raise ValueError(f"Doom patch {name!r} column {column} pixels are truncated")
            destination_start = max(top, 0)
            destination_end = min(top + length, height)
            if destination_start < destination_end:
                source_start = cursor + destination_start - top
                source_end = source_start + destination_end - destination_start
                pixels[destination_start:destination_end, column] = np.frombuffer(
                    payload[source_start:source_end], dtype=np.uint8
                )
                opaque[destination_start:destination_end, column] = True
            cursor += length + 1
    return IndexedTexture(
        name=name.upper(),
        pixels=pixels,
        opaque=opaque,
        left_offset=left_offset,
        top_offset=top_offset,
    )


def grayscale_palette(playpal: np.ndarray) -> np.ndarray:
    if playpal.shape != (256, 3):
        raise ValueError("PLAYPAL must contain exactly 256 RGB colors")
    values = (
        playpal[:, 0].astype(np.float32) * 0.21
        + playpal[:, 1].astype(np.float32) * 0.72
        + playpal[:, 2].astype(np.float32) * 0.07
    )
    return values.astype(np.uint8)


def compile_grayscale_atlas(
    wad: WadArchive,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    catalog = TextureCatalog.from_wad(wad)
    palette_bytes = wad.read("PLAYPAL")
    if len(palette_bytes) < 256 * 3:
        raise ValueError("IWAD PLAYPAL lump is too small")
    playpal = np.frombuffer(palette_bytes[: 256 * 3], dtype=np.uint8).reshape(256, 3)
    grayscale = grayscale_palette(playpal)
    textures = tuple(catalog.decode(wad, name) for name in names)
    max_height = max((texture.height for texture in textures), default=1)
    max_width = max((texture.width for texture in textures), default=1)
    atlas = np.zeros((len(textures), max_height, max_width), dtype=np.uint8)
    widths = np.empty(len(textures), dtype=np.int32)
    heights = np.empty(len(textures), dtype=np.int32)
    for index, texture in enumerate(textures):
        atlas[index, : texture.height, : texture.width] = grayscale[texture.pixels]
        widths[index] = texture.width
        heights[index] = texture.height
    return atlas, widths, heights


def compile_indexed_atlas(
    wad: WadArchive,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compile textures without discarding Doom's PLAYPAL indices."""

    catalog = TextureCatalog.from_wad(wad)
    textures = tuple(catalog.decode(wad, name) for name in names)
    max_height = max((texture.height for texture in textures), default=1)
    max_width = max((texture.width for texture in textures), default=1)
    atlas = np.zeros((len(textures), max_height, max_width), dtype=np.uint8)
    widths = np.empty(len(textures), dtype=np.int32)
    heights = np.empty(len(textures), dtype=np.int32)
    for index, texture in enumerate(textures):
        atlas[index, : texture.height, : texture.width] = texture.pixels
        widths[index] = texture.width
        heights[index] = texture.height
    return atlas, widths, heights


def _resolve_sprite_frame(
    wad: WadArchive,
    requested_name: str,
) -> tuple[str, bool]:
    """Resolve a Doom sprite frame/rotation, including mirrored combined lumps."""

    normalized = requested_name.upper()
    if len(normalized) != 6 or not normalized[5].isdigit():
        raise ValueError(f"sprite request must be PREFIX+frame+rotation, got {requested_name!r}")
    prefix = normalized[:4]
    frame = normalized[4]
    rotation = normalized[5]
    rotation_zero: tuple[str, bool] | None = None
    for lump in wad.lumps:
        name = lump.name
        if len(name) < 6 or name[:4] != prefix:
            continue
        pairs = ((name[4], name[5], False),)
        if len(name) >= 8:
            pairs += ((name[6], name[7], True),)
        for candidate_frame, candidate_rotation, flipped in pairs:
            if candidate_frame != frame:
                continue
            if candidate_rotation == rotation:
                return name, flipped
            if candidate_rotation == "0":
                rotation_zero = (name, flipped)
    if rotation_zero is not None:
        return rotation_zero
    raise KeyError(f"IWAD has no sprite frame {normalized!r}")


def compile_indexed_sprite_atlas(
    wad: WadArchive,
    frame_names: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compile exact indexed sprite rotations in request order."""

    resolved_names: list[str] = []
    sprites: list[IndexedTexture] = []
    for requested_name in frame_names:
        resolved, flipped = _resolve_sprite_frame(wad, requested_name)
        sprite = decode_patch(wad.read(resolved), resolved)
        if flipped:
            sprite = IndexedTexture(
                name=f"{resolved}:FLIPPED",
                pixels=np.ascontiguousarray(np.fliplr(sprite.pixels)),
                opaque=np.ascontiguousarray(np.fliplr(sprite.opaque)),
                left_offset=sprite.width - sprite.left_offset - 1,
                top_offset=sprite.top_offset,
            )
        resolved_names.append(sprite.name)
        sprites.append(sprite)
    max_height = max((sprite.height for sprite in sprites), default=1)
    max_width = max((sprite.width for sprite in sprites), default=1)
    atlas = np.zeros((len(sprites), max_height, max_width), dtype=np.uint8)
    opaque = np.zeros_like(atlas, dtype=np.bool_)
    widths = np.empty(len(sprites), dtype=np.int32)
    heights = np.empty(len(sprites), dtype=np.int32)
    left_offsets = np.empty(len(sprites), dtype=np.int32)
    top_offsets = np.empty(len(sprites), dtype=np.int32)
    for index, sprite in enumerate(sprites):
        atlas[index, : sprite.height, : sprite.width] = sprite.pixels
        opaque[index, : sprite.height, : sprite.width] = sprite.opaque
        widths[index] = sprite.width
        heights[index] = sprite.height
        left_offsets[index] = sprite.left_offset
        top_offsets[index] = sprite.top_offset
    return (
        tuple(resolved_names),
        atlas,
        opaque,
        widths,
        heights,
        left_offsets,
        top_offsets,
    )


def compile_indexed_patch_atlas(
    wad: WadArchive,
    names: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compile named Doom patches as palette indices with transparency."""

    normalized_names = tuple(name.upper() for name in names)
    sprites = tuple(decode_patch(wad.read(name), name) for name in normalized_names)
    max_height = max((sprite.height for sprite in sprites), default=1)
    max_width = max((sprite.width for sprite in sprites), default=1)
    atlas = np.zeros((len(sprites), max_height, max_width), dtype=np.uint8)
    opaque = np.zeros_like(atlas, dtype=np.bool_)
    widths = np.empty(len(sprites), dtype=np.int32)
    heights = np.empty(len(sprites), dtype=np.int32)
    left_offsets = np.empty(len(sprites), dtype=np.int32)
    top_offsets = np.empty(len(sprites), dtype=np.int32)
    for index, sprite in enumerate(sprites):
        atlas[index, : sprite.height, : sprite.width] = sprite.pixels
        opaque[index, : sprite.height, : sprite.width] = sprite.opaque
        widths[index] = sprite.width
        heights[index] = sprite.height
        left_offsets[index] = sprite.left_offset
        top_offsets[index] = sprite.top_offset
    return (
        normalized_names,
        atlas,
        opaque,
        widths,
        heights,
        left_offsets,
        top_offsets,
    )


def compile_sprite_atlas(
    wad: WadArchive,
    frame_names: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Decode fixed sprite frames, accepting combined rotation names such as BOS2A1C1."""

    palette_bytes = wad.read("PLAYPAL")
    if len(palette_bytes) < 256 * 3:
        raise ValueError("IWAD PLAYPAL lump is too small")
    playpal = np.frombuffer(palette_bytes[: 256 * 3], dtype=np.uint8).reshape(256, 3)
    grayscale = grayscale_palette(playpal)
    resolved_names: list[str] = []
    sprites: list[IndexedTexture] = []
    for requested_name in frame_names:
        normalized = requested_name.upper()
        if normalized in wad.by_name:
            resolved = normalized
        else:
            resolved = next(
                (lump.name for lump in wad.lumps if lump.name.startswith(normalized)),
                "",
            )
            if not resolved:
                raise KeyError(f"IWAD has no sprite frame beginning with {normalized!r}")
        resolved_names.append(resolved)
        sprites.append(decode_patch(wad.read(resolved), resolved))
    max_height = max((sprite.height for sprite in sprites), default=1)
    max_width = max((sprite.width for sprite in sprites), default=1)
    atlas = np.zeros((len(sprites), max_height, max_width), dtype=np.uint8)
    opaque = np.zeros_like(atlas, dtype=np.bool_)
    widths = np.empty(len(sprites), dtype=np.int32)
    heights = np.empty(len(sprites), dtype=np.int32)
    left_offsets = np.empty(len(sprites), dtype=np.int32)
    top_offsets = np.empty(len(sprites), dtype=np.int32)
    for index, sprite in enumerate(sprites):
        atlas[index, : sprite.height, : sprite.width] = grayscale[sprite.pixels]
        opaque[index, : sprite.height, : sprite.width] = sprite.opaque
        widths[index] = sprite.width
        heights[index] = sprite.height
        left_offsets[index] = sprite.left_offset
        top_offsets[index] = sprite.top_offset
    return (
        tuple(resolved_names),
        atlas,
        opaque,
        widths,
        heights,
        left_offsets,
        top_offsets,
    )


def compile_weapon_overlays(
    wad: WadArchive,
    frame_names: tuple[str, ...],
    *,
    output_size: tuple[int, int] = (84, 84),
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Rasterize ready-state psprites through Doom's logical 320x200 view."""
    palette_bytes = wad.read("PLAYPAL")
    if len(palette_bytes) < 256 * 3:
        raise ValueError("IWAD PLAYPAL lump is too small")
    playpal = np.frombuffer(palette_bytes[: 256 * 3], dtype=np.uint8).reshape(256, 3)
    grayscale = grayscale_palette(playpal)
    resolved_names: list[str] = []
    values: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for requested_name in frame_names:
        normalized = requested_name.upper()
        resolved = normalized if normalized in wad.by_name else ""
        if not resolved:
            raise KeyError(f"IWAD has no weapon frame {normalized!r}")
        patch = decode_patch(wad.read(resolved), resolved)
        left = -patch.left_offset
        top = -patch.top_offset + 33
        x0 = max(left, 0)
        y0 = max(top, 0)
        x1 = min(left + patch.width, 320)
        y1 = min(top + patch.height, 200)
        if x0 >= x1 or y0 >= y1:
            raise ValueError(f"weapon frame {resolved!r} lies outside the logical view")
        alpha_canvas = torch.zeros((1, 1, 200, 320), dtype=torch.float32)
        value_canvas = torch.zeros_like(alpha_canvas)
        patch_x0 = x0 - left
        patch_y0 = y0 - top
        patch_x1 = patch_x0 + x1 - x0
        patch_y1 = patch_y0 + y1 - y0
        alpha = torch.from_numpy(
            patch.opaque[patch_y0:patch_y1, patch_x0:patch_x1].astype(np.float32)
        )
        value = (
            torch.from_numpy(
                grayscale[patch.pixels[patch_y0:patch_y1, patch_x0:patch_x1]].astype(np.float32)
            )
            * alpha
        )
        alpha_canvas[0, 0, y0:y1, x0:x1] = alpha
        value_canvas[0, 0, y0:y1, x0:x1] = value
        values.append(
            functional.interpolate(value_canvas, size=output_size, mode="area")[0, 0].numpy()
        )
        alphas.append(
            functional.interpolate(alpha_canvas, size=output_size, mode="area")[0, 0].numpy()
        )
        resolved_names.append(resolved)
    return (
        tuple(resolved_names),
        np.stack(values).astype(np.float32),
        np.stack(alphas).astype(np.float32),
    )


def compile_indexed_weapon_overlays(
    wad: WadArchive,
    frame_names: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Rasterize aspect-corrected ready-state psprites in ViZDoom's play view."""

    resolved_names: list[str] = []
    values: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    for requested_name in frame_names:
        normalized = requested_name.upper()
        if normalized not in wad.by_name:
            raise KeyError(f"IWAD has no weapon frame {normalized!r}")
        patch = decode_patch(wad.read(normalized), normalized)
        left = -patch.left_offset
        vertical_scale = 1.2
        ready_y = 32.0
        top = 104.0 - (100.0 - ready_y + patch.top_offset) * vertical_scale
        x0 = max(left, 0)
        x1 = min(left + patch.width, 320)
        source_rows = np.floor((np.arange(208, dtype=np.float32) - top) / vertical_scale).astype(
            np.int32
        )
        visible_rows = (source_rows >= 0) & (source_rows < patch.height)
        if x0 >= x1 or not visible_rows.any():
            raise ValueError(f"weapon frame {normalized!r} lies outside the logical view")
        value_canvas = np.zeros((208, 320), dtype=np.uint8)
        alpha_canvas = np.zeros((208, 320), dtype=np.bool_)
        patch_x0 = x0 - left
        patch_x1 = patch_x0 + x1 - x0
        destination_rows = np.flatnonzero(visible_rows)
        selected_rows = source_rows[visible_rows]
        value_canvas[destination_rows, x0:x1] = patch.pixels[
            selected_rows[:, None], np.arange(patch_x0, patch_x1)[None, :]
        ]
        alpha_canvas[destination_rows, x0:x1] = patch.opaque[
            selected_rows[:, None], np.arange(patch_x0, patch_x1)[None, :]
        ]
        values.append(value_canvas)
        alphas.append(alpha_canvas)
        resolved_names.append(normalized)
    return tuple(resolved_names), np.stack(values), np.stack(alphas)


__all__ = [
    "IndexedTexture",
    "TextureCatalog",
    "compile_grayscale_atlas",
    "compile_indexed_atlas",
    "compile_indexed_patch_atlas",
    "compile_indexed_sprite_atlas",
    "compile_indexed_weapon_overlays",
    "compile_sprite_atlas",
    "compile_weapon_overlays",
    "decode_patch",
    "grayscale_palette",
]
