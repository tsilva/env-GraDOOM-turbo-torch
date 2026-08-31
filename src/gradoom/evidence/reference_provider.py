from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gradoom.diagnostics import (
    ActorAttributionDiagnostics,
    ActorAttributionStage,
    ActorKillEvent,
    ActorSnapshot,
)

REFERENCE_DISTRIBUTION = "env-vizdoom-turbo"
REFERENCE_MODULE = "env_vizdoom_turbo"
REFERENCE_REVISION = "5b74973e4fbb1a96550a1884805b51fd6dcfe90f"


class ReferenceProviderError(RuntimeError):
    """The installed reference provider cannot satisfy the pinned evaluation contract."""


def _installed_revision() -> str:
    try:
        distribution = importlib_metadata.distribution(REFERENCE_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as error:
        raise ReferenceProviderError(
            f"{REFERENCE_DISTRIBUTION} is not installed; install revision {REFERENCE_REVISION}"
        ) from error
    direct_url_payload = distribution.read_text("direct_url.json")
    if direct_url_payload is None:
        raise ReferenceProviderError(
            f"cannot verify {REFERENCE_DISTRIBUTION} revision: direct_url.json is missing; "
            f"install immutable revision {REFERENCE_REVISION} from Git"
        )
    try:
        direct_url = json.loads(direct_url_payload)
        revision = direct_url["vcs_info"]["commit_id"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ReferenceProviderError(
            f"cannot verify {REFERENCE_DISTRIBUTION} revision from direct_url.json"
        ) from error
    if revision != REFERENCE_REVISION:
        raise ReferenceProviderError(
            f"{REFERENCE_DISTRIBUTION} revision mismatch: expected {REFERENCE_REVISION}, "
            f"found {revision!r}"
        )
    return revision


@dataclass(frozen=True)
class ReferenceProvider:
    revision: str
    env_class: type[Any]
    preprocess_into: Callable[..., Any]

    def make_env(
        self,
        *args: Any,
        game_variables: Sequence[str],
        actor_diagnostics: bool = False,
        **kwargs: Any,
    ) -> Any:
        requested = {str(name).casefold() for name in game_variables}
        if "player_killcount" not in requested:
            raise ReferenceProviderError(
                "reference evaluation must explicitly request player_killcount"
            )
        if "killcount" not in requested:
            raise ReferenceProviderError(
                "reference evaluation must explicitly request compatibility killcount"
            )
        env_class = (
            _reference_attribution_env_class(self.env_class)
            if actor_diagnostics
            else self.env_class
        )
        return env_class(*args, game_variables=game_variables, **kwargs)

    def episode_kill_signals(
        self,
        infos: Mapping[str, Any],
        *,
        lane: int,
    ) -> dict[str, float]:
        if "player_killcount" not in infos:
            raise ReferenceProviderError(
                "reference provider result is missing required player_killcount"
            )
        if "killcount" not in infos:
            raise ReferenceProviderError(
                "reference provider result is missing required compatibility killcount"
            )
        return {
            "player_killcount": float(infos["player_killcount"][lane]),
            "compatibility_killcount": float(infos["killcount"][lane]),
        }


def _reference_attribution_env_class(base: type[Any]) -> type[Any]:
    class ReferenceAttributionEnv(ActorAttributionDiagnostics, base):  # type: ignore[misc,valid-type]
        """Pinned Turbo provider with diagnostic-only native actor inspection."""

        _gradoom_reference_attribution_adapter = True

        def _new_game(self) -> Any:
            game = super()._new_game()
            game.set_objects_info_enabled(True)
            return game

        def _configure_action_space(self, value: Any, template: Any) -> None:
            super()._configure_action_space(value, template)
            # The native batch stepper intentionally transports only frames,
            # rewards, terminals, and game variables. Actor objects therefore
            # require the provider's equivalent ordinary step path. This does
            # not change the configured action or observation semantics.
            self._use_indexed_native = False

        def reset(self, *args: Any, **kwargs: Any) -> Any:
            # A public reset changes the native world and therefore invalidates
            # every actor identity captured for the previous staged event.
            if hasattr(self, "_diagnostic_attribution_stages"):
                del self._diagnostic_attribution_stages
            return super().reset(*args, **kwargs)

        def diagnostic_asset_sha256(self) -> dict[str, str]:
            iwad = Path(str(self._rom_path)).expanduser().resolve()
            pwad = self._scenario.config_path.with_name("deathmatch.wad").resolve()
            return {
                "iwad": hashlib.sha256(iwad.read_bytes()).hexdigest(),
                "pwad": hashlib.sha256(pwad.read_bytes()).hexdigest(),
            }

        @staticmethod
        def _actor_snapshot(game: Any) -> tuple[ActorSnapshot, ...]:
            state = game.get_state()
            if state is None or state.objects is None:
                raise RuntimeError("reference provider did not expose native actor objects")
            actors: list[ActorSnapshot] = []
            for actor in state.objects:
                category = str(actor.category).casefold()
                if category == "self":
                    actors.append(ActorSnapshot(int(actor.id), "player", True))
                elif category == "monster":
                    actors.append(ActorSnapshot(int(actor.id), "enemy", True))
            return tuple(sorted(actors, key=lambda actor: actor.actor_id))

        @staticmethod
        def _stage_game(game: Any, behavior: str) -> None:
            buttons = tuple(str(button.name) for button in game.get_available_buttons())
            noop = [0.0] * len(buttons)
            game.make_action(noop, 10)
            angle = float(game.get_game_variable(_vizdoom().GameVariable.ANGLE))
            angle_units = round(angle / 360.0 * 65536.0)
            game.send_game_command(
                f"turnspeeds {angle_units} {angle_units} {angle_units} {angle_units}"
            )
            game.make_action(noop, 1)
            turn = noop.copy()
            turn[buttons.index("TURN_RIGHT")] = 1.0
            game.make_action(turn, 1)
            game.send_game_command("turnspeeds 32768 32768 32768 32768")
            game.make_action(noop, 1)
            game.make_action(turn, 1)
            if behavior == "player_killcount":
                # The player faces west. Summon places the actor 56 units in
                # front of the temporary position, yielding exact x=412.
                game.send_game_command("warp 468 512")
                game.send_game_command("summon Zombieman")
            elif behavior == "player_killcount.enemy_on_enemy_exclusion":
                for name, x in (("Zombieman", 668), ("ShotgunGuy", 768)):
                    game.send_game_command(f"warp {x} 512")
                    game.send_game_command(f"summon {name}")
            else:
                raise ValueError(f"unsupported actor attribution behavior {behavior!r}")
            game.send_game_command("warp 512 512")
            wake = noop
            if behavior == "player_killcount.enemy_on_enemy_exclusion":
                # Fire west, away from both east-side actors. The sound wakes
                # them without assigning player damage to either actor.
                wake = noop.copy()
                wake[buttons.index("ATTACK")] = 1.0
            game.make_action(wake, 2)

        def diagnostic_stage_actor_attribution(
            self,
            behavior: str,
        ) -> tuple[ActorAttributionStage, ...]:
            stages: list[ActorAttributionStage] = []
            for game in self._games:
                before = self._actor_snapshot(game)
                if sum(actor.kind == "enemy" for actor in before):
                    raise RuntimeError("reference attribution stage began with unexpected enemies")
                self._stage_game(game, behavior)
                actors = self._actor_snapshot(game)
                expected_enemies = 1 if behavior == "player_killcount" else 2
                if (
                    sum(actor.kind == "player" for actor in actors) != 1
                    or sum(actor.kind == "enemy" for actor in actors) != expected_enemies
                ):
                    raise RuntimeError("reference attribution actors did not materialize exactly")
                stages.append(ActorAttributionStage(secrets.token_hex(16), actors))
            self._diagnostic_attribution_stages = tuple(stages)
            return self._diagnostic_attribution_stages

        def diagnostic_actor_snapshot(self, lane: int) -> tuple[ActorSnapshot, ...]:
            if not hasattr(self, "_diagnostic_attribution_stages"):
                raise RuntimeError("actor attribution has not been staged")
            return self._actor_snapshot(self._games[lane])

        def diagnostic_kill_events(self, lane: int) -> tuple[ActorKillEvent, ...]:
            try:
                stage = self._diagnostic_attribution_stages[lane]
            except (AttributeError, IndexError) as error:
                raise RuntimeError("actor attribution has not been staged") from error
            current_ids = {
                actor.actor_id for actor in self.diagnostic_actor_snapshot(lane) if actor.alive
            }
            dead = [
                actor
                for actor in stage.actors
                if actor.kind == "enemy" and actor.actor_id not in current_ids
            ]
            if not dead:
                return ()
            live = [
                actor for actor in stage.actors if actor.actor_id in current_ids and actor.alive
            ]
            player = next((actor for actor in live if actor.kind == "player"), None)
            enemies = [actor for actor in live if actor.kind == "enemy"]
            events: list[ActorKillEvent] = []
            for target in dead:
                source = (
                    player if len(stage.actors) == 2 else enemies[0] if len(enemies) == 1 else None
                )
                events.append(
                    ActorKillEvent(
                        stage_token=stage.token,
                        attacker_id=-1 if source is None else source.actor_id,
                        attacker_kind="enemy" if source is None else source.kind,
                        target_id=target.actor_id,
                        target_kind="enemy",
                    )
                )
            return tuple(events)

    ReferenceAttributionEnv.__name__ = f"{base.__name__}AttributionDiagnostics"
    ReferenceAttributionEnv.__qualname__ = ReferenceAttributionEnv.__name__
    return ReferenceAttributionEnv


def _vizdoom() -> Any:
    return importlib.import_module("vizdoom")


def load_reference_provider() -> ReferenceProvider:
    revision = _installed_revision()
    try:
        module = importlib.import_module(REFERENCE_MODULE)
        native_module = importlib.import_module(f"{REFERENCE_MODULE}._env_vizdoom_turbo")
        env_class = module.EnvViZDoomTurboVecEnv
        preprocess_into = native_module.preprocess_into
    except (AttributeError, ImportError) as error:
        raise ReferenceProviderError(
            "pinned env-vizdoom-turbo does not expose EnvViZDoomTurboVecEnv and preprocess_into"
        ) from error
    return ReferenceProvider(
        revision=revision,
        env_class=env_class,
        preprocess_into=preprocess_into,
    )


__all__ = [
    "REFERENCE_REVISION",
    "ReferenceProvider",
    "ReferenceProviderError",
    "load_reference_provider",
]
