from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

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
        return self.env_class(*args, game_variables=game_variables, **kwargs)

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
