from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gradoom.diagnostics import ActorAttributionStage, ActorSnapshot
from gradoom.evidence import reference_provider


class _Distribution:
    version = "1"

    def __init__(self, direct_url: dict[str, object] | None) -> None:
        self._direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self._direct_url is None else json.dumps(self._direct_url)


class _CurrentEnv:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def _pinned_direct_url(*, commit_id: str = reference_provider.REFERENCE_REVISION):
    return {
        "url": "https://github.com/tsilva/env-ViZDoom-turbo.git",
        "subdirectory": "turbo",
        "vcs_info": {
            "vcs": "git",
            "requested_revision": reference_provider.REFERENCE_REVISION,
            "commit_id": commit_id,
        },
    }


def _install_fake_provider(monkeypatch: pytest.MonkeyPatch, direct_url=None) -> None:
    module = ModuleType("env_vizdoom_turbo")
    module.EnvViZDoomTurboVecEnv = _CurrentEnv
    native = ModuleType("env_vizdoom_turbo._env_vizdoom_turbo")
    native.preprocess_into = lambda *args: args
    monkeypatch.setattr(
        reference_provider.importlib_metadata,
        "distribution",
        lambda name: _Distribution(direct_url or _pinned_direct_url()),
    )
    monkeypatch.setattr(
        reference_provider.importlib,
        "import_module",
        lambda name: {
            "env_vizdoom_turbo": module,
            "env_vizdoom_turbo._env_vizdoom_turbo": native,
        }[name],
    )


def test_pinned_provider_loads_the_current_public_export(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_provider(monkeypatch)

    provider = reference_provider.load_reference_provider()

    assert provider.revision == "5b74973e4fbb1a96550a1884805b51fd6dcfe90f"
    assert provider.env_class is _CurrentEnv
    assert provider.preprocess_into("frame") == ("frame",)


def test_provider_revision_mismatch_fails_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reference_provider.importlib_metadata,
        "distribution",
        lambda name: _Distribution(_pinned_direct_url(commit_id="0" * 40)),
    )

    with pytest.raises(
        reference_provider.ReferenceProviderError,
        match=r"revision mismatch.*5b74973e",
    ):
        reference_provider.load_reference_provider()


def test_unverifiable_provider_install_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reference_provider.importlib_metadata,
        "distribution",
        lambda name: _Distribution(None),
    )

    with pytest.raises(
        reference_provider.ReferenceProviderError,
        match=r"cannot verify.*direct_url.json",
    ):
        reference_provider.load_reference_provider()


def test_missing_player_killcount_never_falls_back_to_killcount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    provider = reference_provider.load_reference_provider()

    with pytest.raises(
        reference_provider.ReferenceProviderError,
        match="missing required player_killcount",
    ):
        provider.episode_kill_signals({"killcount": np.asarray([17.0])}, lane=0)


def test_episode_kills_keep_quality_and_compatibility_signals_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    provider = reference_provider.load_reference_provider()

    signals = provider.episode_kill_signals(
        {
            "player_killcount": np.asarray([4.0, 7.0]),
            "killcount": np.asarray([9.0, 12.0]),
        },
        lane=1,
    )

    assert signals == {
        "player_killcount": 7.0,
        "compatibility_killcount": 12.0,
    }


def test_reference_constructor_requires_both_explicit_kill_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    provider = reference_provider.load_reference_provider()

    with pytest.raises(
        reference_provider.ReferenceProviderError,
        match="must explicitly request player_killcount",
    ):
        provider.make_env("deathmatch.cfg", game_variables=("KILLCOUNT",))


def test_reference_constructor_forwards_the_shared_contract_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_provider(monkeypatch)
    provider = reference_provider.load_reference_provider()
    game_variables = ("KILLCOUNT", "PLAYER_KILLCOUNT")

    env = provider.make_env(
        "deathmatch.cfg",
        game_variables=game_variables,
        obs_crop=(0, 32, 0, 0),
        obs_resize=(84, 84),
    )

    assert type(env) is _CurrentEnv
    assert env.args == ("deathmatch.cfg",)
    assert env.kwargs == {
        "game_variables": game_variables,
        "obs_crop": (0, 32, 0, 0),
        "obs_resize": (84, 84),
    }


class _DiagnosticBase:
    pass


class _ObjectGame:
    def __init__(self, objects: list[SimpleNamespace]) -> None:
        self.objects = objects

    def get_state(self) -> SimpleNamespace:
        return SimpleNamespace(objects=self.objects)


def _actor(actor_id: int, category: str) -> SimpleNamespace:
    return SimpleNamespace(id=actor_id, category=category)


def test_reference_diagnostic_derives_third_party_identity_from_native_object_ids() -> None:
    cls = reference_provider._reference_attribution_env_class(_DiagnosticBase)
    env = object.__new__(cls)
    env._games = [_ObjectGame([_actor(0, "Self"), _actor(83, "Monster")])]
    env._diagnostic_attribution_stages = (
        ActorAttributionStage(
            token="native-stage",
            actors=(
                ActorSnapshot(0, "player", True),
                ActorSnapshot(71, "enemy", True),
                ActorSnapshot(83, "enemy", True),
            ),
        ),
    )

    events = env.diagnostic_kill_events(0)

    assert len(events) == 1
    assert events[0].stage_token == "native-stage"
    assert events[0].attacker_kind == "enemy"
    assert events[0].attacker_id == 83
    assert events[0].target_kind == "enemy"
    assert events[0].target_id == 71


def test_reference_diagnostic_derives_player_identity_from_native_object_ids() -> None:
    cls = reference_provider._reference_attribution_env_class(_DiagnosticBase)
    env = object.__new__(cls)
    env._games = [_ObjectGame([_actor(0, "Self")])]
    env._diagnostic_attribution_stages = (
        ActorAttributionStage(
            token="native-stage",
            actors=(
                ActorSnapshot(0, "player", True),
                ActorSnapshot(71, "enemy", True),
            ),
        ),
    )

    events = env.diagnostic_kill_events(0)

    assert events == (
        reference_provider.ActorKillEvent(
            stage_token="native-stage",
            attacker_id=0,
            attacker_kind="player",
            target_id=71,
            target_kind="enemy",
        ),
    )


def test_reference_diagnostic_surfaces_unexpected_surviving_actors() -> None:
    cls = reference_provider._reference_attribution_env_class(_DiagnosticBase)
    env = object.__new__(cls)
    env._games = [
        _ObjectGame(
            [
                _actor(0, "Self"),
                _actor(83, "Monster"),
                _actor(99, "Monster"),
            ]
        )
    ]
    env._diagnostic_attribution_stages = (
        ActorAttributionStage(
            token="native-stage",
            actors=(
                ActorSnapshot(0, "player", True),
                ActorSnapshot(71, "enemy", True),
                ActorSnapshot(83, "enemy", True),
            ),
        ),
    )

    events = env.diagnostic_kill_events(0)

    assert events[0].attacker_id == 83
    assert env.diagnostic_actor_snapshot(0) == (
        ActorSnapshot(0, "player", True),
        ActorSnapshot(83, "enemy", True),
        ActorSnapshot(99, "enemy", True),
    )


def test_reference_diagnostic_reset_invalidates_staged_actor_identities() -> None:
    class ResettableDiagnosticBase:
        def reset(self, *args: object, **kwargs: object) -> tuple[str, str]:
            del args, kwargs
            return "observation", "infos"

    cls = reference_provider._reference_attribution_env_class(ResettableDiagnosticBase)
    env = object.__new__(cls)
    env._games = [_ObjectGame([_actor(0, "Self")])]
    env._diagnostic_attribution_stages = (
        ActorAttributionStage(
            token="stale-stage",
            actors=(
                ActorSnapshot(0, "player", True),
                ActorSnapshot(71, "enemy", True),
            ),
        ),
    )

    assert env.reset() == ("observation", "infos")
    with pytest.raises(RuntimeError, match="has not been staged"):
        env.diagnostic_actor_snapshot(0)


def test_reference_distribution_identity_detects_modified_bytes_with_same_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "env_vizdoom_turbo"
    package.mkdir()
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    metadata = tmp_path / "env_vizdoom_turbo-1.dist-info"
    metadata.mkdir()
    direct_url = metadata / "direct_url.json"
    direct_url.write_text(json.dumps(_pinned_direct_url()), encoding="utf-8")

    class BoundDistribution(_Distribution):
        files = (
            Path("env_vizdoom_turbo/__init__.py"),
            Path("env_vizdoom_turbo-1.dist-info/direct_url.json"),
        )

        def locate_file(self, relative: object) -> Path:
            return tmp_path / Path(str(relative))

    distribution = BoundDistribution(_pinned_direct_url())
    monkeypatch.setattr(
        reference_provider.importlib_metadata,
        "distribution",
        lambda _name: distribution,
    )
    before = reference_provider.reference_distribution_identity()
    module.write_text("VALUE = 2\n", encoding="utf-8")
    after = reference_provider.reference_distribution_identity()

    assert before["revision"] == after["revision"] == reference_provider.REFERENCE_REVISION
    assert before["content_sha256"] != after["content_sha256"]
