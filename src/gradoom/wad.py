"""Small, strict WAD and UDMF readers for ahead-of-time scenario compilation."""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_HEADER = struct.Struct("<4sII")
_DIRECTORY_ENTRY = struct.Struct("<II8s")
_TOKEN = re.compile(
    r"""\s*(?:(?P<string>\"(?:\\.|[^\"\\])*\")|(?P<number>[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)|(?P<identifier>[A-Za-z_][A-Za-z0-9_]*)|(?P<punct>[{}=;]))"""
)
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)


@dataclass(frozen=True)
class Lump:
    name: str
    offset: int
    size: int
    ordinal: int


class WadArchive:
    """Immutable view over a validated Doom WAD archive."""

    def __init__(self, payload: bytes, *, source: str = "<memory>") -> None:
        if len(payload) < _HEADER.size:
            raise ValueError(f"{source} is too small to be a WAD")
        identity, count, directory_offset = _HEADER.unpack_from(payload)
        if identity not in {b"IWAD", b"PWAD"}:
            raise ValueError(f"{source} has invalid WAD identity {identity!r}")
        directory_end = directory_offset + count * _DIRECTORY_ENTRY.size
        if directory_offset < _HEADER.size or directory_end > len(payload):
            raise ValueError(f"{source} has an out-of-bounds WAD directory")
        lumps: list[Lump] = []
        by_name: dict[str, list[Lump]] = {}
        for ordinal in range(count):
            offset, size, raw_name = _DIRECTORY_ENTRY.unpack_from(
                payload, directory_offset + ordinal * _DIRECTORY_ENTRY.size
            )
            if offset + size > len(payload):
                raise ValueError(f"{source} lump {ordinal} extends past end of file")
            try:
                name = raw_name.rstrip(b"\0").decode("ascii").upper()
            except UnicodeDecodeError as exc:
                raise ValueError(f"{source} lump {ordinal} has a non-ASCII name") from exc
            lump = Lump(name=name, offset=offset, size=size, ordinal=ordinal)
            lumps.append(lump)
            by_name.setdefault(name, []).append(lump)
        self._payload = payload
        self.source = source
        self.identity = identity.decode("ascii")
        self.sha256 = hashlib.sha256(payload).hexdigest()
        self.lumps = tuple(lumps)
        self.by_name = MappingProxyType({name: tuple(values) for name, values in by_name.items()})

    @classmethod
    def from_path(cls, path: str | Path) -> WadArchive:
        resolved = Path(path).expanduser().resolve()
        return cls(resolved.read_bytes(), source=str(resolved))

    def find(self, name: str, *, occurrence: int = -1) -> Lump:
        normalized = name.upper()
        try:
            return self.by_name[normalized][occurrence]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"{self.source} does not contain lump {normalized!r}") from exc

    def read(self, lump: Lump | str, *, occurrence: int = -1) -> bytes:
        resolved = self.find(lump, occurrence=occurrence) if isinstance(lump, str) else lump
        return self._payload[resolved.offset : resolved.offset + resolved.size]


@dataclass(frozen=True)
class UdmfDocument:
    namespace: str
    assignments: MappingProxyType[str, Any]
    blocks: MappingProxyType[str, tuple[MappingProxyType[str, Any], ...]]


def _tokens(text: str) -> list[str]:
    stripped = _COMMENTS.sub("", text)
    result: list[str] = []
    position = 0
    while position < len(stripped):
        match = _TOKEN.match(stripped, position)
        if match is None:
            if stripped[position:].strip():
                excerpt = stripped[position : position + 48].replace("\n", "\\n")
                raise ValueError(f"unsupported UDMF syntax near {excerpt!r}")
            break
        result.append(match.group(match.lastgroup))
        position = match.end()
    return result


def _value(token: str) -> Any:
    if token.startswith('"'):
        return bytes(token[1:-1], "utf-8").decode("unicode_escape")
    normalized = token.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if any(character in token for character in ".eE"):
        return float(token)
    try:
        return int(token)
    except ValueError:
        return token


def parse_udmf(payload: bytes | str) -> UdmfDocument:
    """Parse the data subset used by Doom-family UDMF TEXTMAP lumps."""

    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    tokens = _tokens(text)
    cursor = 0
    assignments: dict[str, Any] = {}
    blocks: dict[str, list[MappingProxyType[str, Any]]] = {}

    def take() -> str:
        nonlocal cursor
        if cursor >= len(tokens):
            raise ValueError("unexpected end of UDMF document")
        token = tokens[cursor]
        cursor += 1
        return token

    while cursor < len(tokens):
        name = take()
        marker = take()
        if marker == "=":
            assignments[name] = _value(take())
            if take() != ";":
                raise ValueError(f"UDMF assignment {name!r} is missing a semicolon")
            continue
        if marker != "{":
            raise ValueError(f"expected '=' or '{{' after UDMF identifier {name!r}")
        properties: dict[str, Any] = {}
        while True:
            key = take()
            if key == "}":
                break
            if take() != "=":
                raise ValueError(f"expected '=' after UDMF property {key!r}")
            properties[key] = _value(take())
            if take() != ";":
                raise ValueError(f"UDMF property {key!r} is missing a semicolon")
        blocks.setdefault(name, []).append(MappingProxyType(properties))
    namespace = str(assignments.get("namespace", ""))
    if not namespace:
        raise ValueError("UDMF document does not declare a namespace")
    return UdmfDocument(
        namespace=namespace,
        assignments=MappingProxyType(assignments),
        blocks=MappingProxyType({name: tuple(values) for name, values in blocks.items()}),
    )
