from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np
import pytest

from env_doom_turbo_torch.textures import (
    TextureCatalog,
    compile_grayscale_atlas,
    compile_indexed_sprite_atlas,
    compile_sprite_atlas,
    decode_patch,
    grayscale_palette,
    policy_grayscale_palette,
)
from env_doom_turbo_torch.wad import WadArchive

DOOM2 = Path(os.environ.get("ENV_DOOM_TURBO_TORCH_IWAD", "/Users/tsilva/roms/vizdoom/doom2.wad"))
FREEDOOM2 = Path(
    os.environ.get(
        "ENV_DOOM_TURBO_TORCH_FREEDOOM_IWAD",
        Path(__file__).resolve().parents[2]
        / "env-ViZDoom-turbo/bin/python3.14/vizdoom/freedoom2.wad",
    )
)
DEATHMATCH_TEXTURES = ("BIGBRIK1", "BRICK12", "BFALL1", "FLAT5_3", "COMPBLUE")


def test_decode_patch_preserves_transparent_posts() -> None:
    first_column = bytes((0, 3, 0, 1, 2, 3, 0, 0xFF))
    second_column = bytes((1, 1, 0, 9, 0, 0xFF))
    payload = (
        struct.pack("<hhhh", 2, 3, 0, 0)
        + struct.pack("<ii", 16, 16 + len(first_column))
        + first_column
        + second_column
    )

    patch = decode_patch(payload, "fixture")

    np.testing.assert_array_equal(patch.pixels, np.asarray(((1, 0), (2, 9), (3, 0))))
    np.testing.assert_array_equal(
        patch.opaque,
        np.asarray(((True, False), (True, True), (True, False))),
    )
    assert patch.left_offset == 0
    assert patch.top_offset == 0


def test_grayscale_palette_matches_vizdoom_gray8_coefficients() -> None:
    playpal = np.zeros((256, 3), dtype=np.uint8)
    playpal[0] = (100, 200, 50)

    grayscale = grayscale_palette(playpal)

    assert grayscale.dtype == np.uint8
    assert int(grayscale[0]) == 168


def test_policy_grayscale_palette_matches_reference_rgb_pipeline() -> None:
    playpal = np.zeros((256, 3), dtype=np.uint8)
    playpal[0] = (100, 200, 50)

    grayscale = policy_grayscale_palette(playpal)

    assert grayscale.dtype == np.uint8
    assert int(grayscale[0]) == 153


@pytest.mark.skipif(not DOOM2.is_file(), reason="operator Doom2 IWAD absent")
def test_combined_sprite_mirror_uses_doom_left_offset_origin() -> None:
    wad = WadArchive.from_path(DOOM2)

    names, sprites, opaque, widths, _heights, left_offsets, _top_offsets = (
        compile_indexed_sprite_atlas(wad, ("POSSA3", "POSSA7"))
    )

    assert names == ("POSSA3A7", "POSSA3A7:FLIPPED")
    assert widths.tolist() == [43, 43]
    assert left_offsets.tolist() == [21, 21]
    np.testing.assert_array_equal(sprites[1], np.fliplr(sprites[0]))
    np.testing.assert_array_equal(opaque[1], np.fliplr(opaque[0]))


@pytest.mark.parametrize("iwad", (DOOM2, FREEDOOM2))
@pytest.mark.skipif(
    not DOOM2.is_file() or not FREEDOOM2.is_file(),
    reason="operator Doom2/Freedoom IWADs absent",
)
def test_deathmatch_textures_decode_from_supported_iwads(iwad: Path) -> None:
    wad = WadArchive.from_path(iwad)
    catalog = TextureCatalog.from_wad(wad)

    decoded = tuple(catalog.decode(wad, name) for name in DEATHMATCH_TEXTURES)
    atlas, widths, heights = compile_grayscale_atlas(wad, DEATHMATCH_TEXTURES)

    assert all(texture.opaque.all() for texture in decoded)
    assert all(np.unique(texture.pixels).size > 1 for texture in decoded)
    assert atlas.shape == (5, 128, 64)
    assert widths.tolist() == [64] * 5
    assert heights.tolist() == [128, 128, 128, 64, 128]
    assert np.unique(atlas[0]).size > 1

    names, sprites, opaque, sprite_widths, sprite_heights, left_offsets, top_offsets = (
        compile_sprite_atlas(wad, ("POSSA1", "BOS2A1", "STIMA0"))
    )
    assert names[0] == "POSSA1"
    assert names[1].startswith("BOS2A1")
    assert names[2] == "STIMA0"
    assert sprites.shape[0] == 3
    assert sprites.shape == opaque.shape
    assert opaque.any(axis=(1, 2)).all()
    assert (sprite_widths > 0).all()
    assert (sprite_heights > 0).all()
    assert (left_offsets >= 0).all()
    assert (top_offsets > 0).all()
