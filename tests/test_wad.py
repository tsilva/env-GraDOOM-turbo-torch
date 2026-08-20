from __future__ import annotations

import struct

import pytest

from env_doom_turbo_torch.wad import WadArchive, parse_udmf


def _wad(*lumps: tuple[str, bytes], identity: bytes = b"PWAD") -> bytes:
    payload = bytearray(b"\0" * 12)
    directory: list[tuple[int, int, bytes]] = []
    for name, data in lumps:
        offset = len(payload)
        payload.extend(data)
        directory.append((offset, len(data), name.encode("ascii").ljust(8, b"\0")))
    directory_offset = len(payload)
    for entry in directory:
        payload.extend(struct.pack("<II8s", *entry))
    struct.pack_into("<4sII", payload, 0, identity, len(directory), directory_offset)
    return bytes(payload)


def test_wad_archive_validates_and_reads_duplicate_lumps() -> None:
    archive = WadArchive(_wad(("MAP01", b""), ("DATA", b"first"), ("DATA", b"last")))
    assert archive.identity == "PWAD"
    assert archive.read("DATA") == b"last"
    assert archive.read("DATA", occurrence=0) == b"first"
    assert archive.find("MAP01").size == 0


def test_wad_archive_rejects_invalid_offsets() -> None:
    payload = bytearray(_wad(("DATA", b"ok")))
    _, _, directory_offset = struct.unpack_from("<4sII", payload)
    struct.pack_into("<I", payload, directory_offset, len(payload) + 1)
    with pytest.raises(ValueError, match="past end"):
        WadArchive(bytes(payload))


def test_udmf_parser_handles_comments_types_and_blocks() -> None:
    document = parse_udmf(
        b"""
        namespace = "zdoom";
        // line comment
        vertex { x = -1.5; y = 2; active = true; }
        /* block comment */
        vertex { x = 4; y = 8.25; label = "arena"; }
        """
    )
    assert document.namespace == "zdoom"
    assert document.blocks["vertex"][0] == {"x": -1.5, "y": 2, "active": True}
    assert document.blocks["vertex"][1]["label"] == "arena"
