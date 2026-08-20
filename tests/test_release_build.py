from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_BUILD = (
    REPO_ROOT / ".codex" / "skills" / "build-release" / "scripts" / "release_build.py"
)


@pytest.fixture(scope="module")
def release_build() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gradoom_release_build", RELEASE_BUILD)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ("0.1.0a0", "0.1.0a1"),
        ("0.1.0b4", "0.1.0b5"),
        ("0.1.0rc2", "0.1.0rc3"),
        ("0.1.0.dev7", "0.1.0.dev8"),
        ("1.2.3", "1.2.4"),
        ("1.2.3.post1", "1.2.4"),
    ],
)
def test_next_version_preserves_the_current_release_channel(
    release_build: ModuleType,
    current: str,
    expected: str,
) -> None:
    assert release_build.next_version(current) == expected


def test_select_release_version_keeps_an_unused_pending_version(
    release_build: ModuleType,
) -> None:
    assert release_build.select_release_version("0.1.0a0", {}, set()) == "0.1.0a0"


def test_select_release_version_skips_published_and_tagged_versions(
    release_build: ModuleType,
) -> None:
    releases = {
        "0.1.0a0": [{"filename": "env_doom_turbo_torch-0.1.0a0.tar.gz"}],
        "0.1.0a1": [{"filename": "env_doom_turbo_torch-0.1.0a1.tar.gz"}],
    }
    tags = {"0.1.0a2"}
    assert (
        release_build.select_release_version("0.1.0a0", releases, tags)
        == "0.1.0a3"
    )


def test_write_version_updates_all_release_metadata(
    release_build: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "gradoom"
    package.mkdir(parents=True)
    pyproject = tmp_path / "pyproject.toml"
    init = package / "__init__.py"
    lock = tmp_path / "uv.lock"
    pyproject.write_text(
        '[project]\nname = "env-doom-turbo-torch"\nversion = "0.1.0a0"\n',
        encoding="utf-8",
    )
    init.write_text('__version__ = "0.1.0a0"\n', encoding="utf-8")
    lock.write_text(
        'version = 1\n\n[[package]]\nname = "env-doom-turbo-torch"\nversion = "0.1.0a0"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(release_build, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release_build, "VERSION_FILES", (pyproject, init, lock))

    release_build.write_version("0.1.0a1")

    assert 'version = "0.1.0a1"' in pyproject.read_text(encoding="utf-8")
    assert init.read_text(encoding="utf-8") == '__version__ = "0.1.0a1"\n'
    assert 'version = "0.1.0a1"' in lock.read_text(encoding="utf-8")
    assert release_build.project_metadata() == ("env-doom-turbo-torch", "0.1.0a1")
    assert release_build.init_version() == "0.1.0a1"
    assert release_build.lock_version() == "0.1.0a1"
