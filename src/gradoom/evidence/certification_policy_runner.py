"""Repository-owned real-provider policy runner for parity certification."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from gradoom import GraDoomVecEnv
from gradoom.evidence.checkpoint_policy import load_policy_checkpoint
from gradoom.evidence.policy_execution import (
    canonical_policy_execution_identity,
    validate_loaded_policy_contract,
)
from gradoom.evidence.reference_provider import (
    installed_distribution_identity,
    load_reference_provider,
    reference_distribution_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_train() -> tuple[ModuleType, Path]:
    repository = Path(__import__("gradoom").__file__).resolve().parents[2]
    path = repository / "train.py"
    spec = importlib.util.spec_from_file_location("gradoom_certification_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the repository-owned policy implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, repository


def _source_identity(repository: Path, expected: dict[str, Any]) -> dict[str, Any]:
    entries = expected.get("files")
    if not isinstance(entries, list):
        raise ValueError("GraDOOM source binding is incomplete")
    actual = [
        {"path": entry["path"], "sha256": _sha256(repository / entry["path"])} for entry in entries
    ]
    encoded = json.dumps(actual, separators=(",", ":"), sort_keys=True).encode()
    return {
        "repository": expected.get("repository"),
        "revision": expected.get("revision"),
        "source_count": len(actual),
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "repository_root": str(repository),
        "files": actual,
    }


def _validate_runtime_distributions(value: object) -> None:
    runtime_distributions = value
    runtime_names = (
        {identity.get("name") for identity in runtime_distributions if isinstance(identity, dict)}
        if isinstance(runtime_distributions, list)
        else set()
    )
    if (
        not isinstance(runtime_distributions, list)
        or not {"torch", "numpy", "gymnasium"}.issubset(runtime_names)
        or len(runtime_names) != len(runtime_distributions)
    ):
        raise ValueError("authenticated runner dependency bindings are incomplete")
    for expected_distribution in runtime_distributions:
        if (
            not isinstance(expected_distribution, dict)
            or installed_distribution_identity(str(expected_distribution.get("name")))
            != expected_distribution
        ):
            raise ValueError("authenticated runner dependency bytes do not match their binding")


def _validated_context(request: dict[str, Any], repository: Path) -> dict[str, Any]:
    binding = request.get("execution_binding")
    if not isinstance(binding, dict) or binding.get("fixture") is not False:
        raise ValueError("authenticated runner accepts only non-fixture execution")
    provider = binding.get("provider")
    policy_identity = binding.get("policy_execution_identity")
    context = binding.get("runner_context")
    if (
        not isinstance(provider, dict)
        or provider.get("id") != request.get("provider_id")
        or provider.get("revision") != request.get("provider_revision")
        or policy_identity != request.get("policy", {}).get("execution_identity")
        or not isinstance(context, dict)
    ):
        raise ValueError("authenticated runner request binding is inconsistent")
    raw_providers = context.get("providers")
    if not isinstance(raw_providers, dict) or set(raw_providers) != {
        "gradoom",
        "env-vizdoom-turbo",
    }:
        raise ValueError("authenticated runner WAD bindings are incomplete")
    for provider_id, provider_binding in raw_providers.items():
        if not isinstance(provider_binding, dict):
            raise ValueError(f"{provider_id} WAD binding is invalid")
        for asset in ("iwad", "pwad"):
            asset_binding = provider_binding.get(asset)
            if not isinstance(asset_binding, dict):
                raise ValueError(f"{provider_id} {asset.upper()} binding is invalid")
            path = Path(str(asset_binding.get("path", "")))
            if not path.is_file() or _sha256(path) != asset_binding.get("sha256"):
                raise ValueError(f"{provider_id} {asset.upper()} binding changed")
    if any(
        raw_providers["gradoom"][asset]["sha256"]
        != raw_providers["env-vizdoom-turbo"][asset]["sha256"]
        for asset in ("iwad", "pwad")
    ):
        raise ValueError("providers are not bound to byte-identical WADs")
    if raw_providers["gradoom"].get("configuration") != raw_providers["env-vizdoom-turbo"].get(
        "configuration"
    ):
        raise ValueError("providers are not bound to one configuration")
    expected_sources = context.get("gradoom_sources")
    if request["provider_id"] == "gradoom" and (
        not isinstance(expected_sources, dict)
        or expected_sources.get("revision") != request["provider_revision"]
    ):
        raise ValueError("executed GraDOOM revision does not match the request")
    if (
        not isinstance(expected_sources, dict)
        or _source_identity(repository, expected_sources) != expected_sources
    ):
        raise ValueError("executed GraDOOM source bytes do not match their binding")
    if reference_distribution_identity() != context.get("reference_distribution"):
        raise ValueError("installed reference-provider bytes do not match their binding")
    _validate_runtime_distributions(context.get("runtime_distributions"))
    python = context.get("python")
    executable = Path(sys.executable)
    if (
        not isinstance(python, dict)
        or executable != Path(str(python.get("executable", "")))
        or _sha256(executable) != python.get("sha256")
    ):
        raise ValueError("executed Python interpreter does not match its binding")
    python_home = python.get("home")
    if not isinstance(python_home, dict) or Path(sys.prefix) != Path(
        str(python_home.get("root", ""))
    ):
        raise ValueError("executed Python standard library does not match its binding")
    actual_python_files = [
        {
            "path": entry["path"],
            "sha256": _sha256(Path(sys.prefix) / entry["path"]),
            "executable": entry.get("executable") is True,
        }
        for entry in python_home.get("files", [])
    ]
    if len(actual_python_files) != python_home.get("file_count") or hashlib.sha256(
        json.dumps(actual_python_files, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest() != python_home.get("content_sha256"):
        raise ValueError("executed Python standard-library bytes do not match their binding")
    scenario_config = Path(str(context.get("reference_scenario_config_path", "")))
    if not scenario_config.is_file() or _sha256(scenario_config) != context.get(
        "reference_scenario_config_sha256"
    ):
        raise ValueError("reference scenario configuration changed")
    return context


def _make_env(
    provider_id: str,
    *,
    context: dict[str, Any],
    train: ModuleType,
    device: torch.device,
) -> tuple[Any, Any | None]:
    binding = context["providers"][provider_id]
    configuration = binding["configuration"]
    observation = configuration["observation"]
    crop = observation["crop_or_mask"]
    resize = observation["resize"]
    common = {
        "use_restricted_actions": train.RESTRICTED_ACTIONS,
        "num_envs": 1,
        "obs_resize": tuple(resize["shape"]),
        "obs_crop": tuple(crop["edges"]),
        "obs_crop_mode": crop["kind"],
        "obs_crop_fill": crop["fill"],
        "obs_grayscale": observation["grayscale"]["enabled"],
        "obs_layout": observation["layout"],
        "obs_resize_algorithm": resize["algorithm"],
        "frame_skip": configuration["frame_skip"],
        "frame_stack": observation["frame_stack"],
        "maxpool_last_two": False,
        "noop_reset_max": 0,
        "sticky_action_prob": 0.0,
        "reward_clip": False,
        "info": "data",
        "info_filter": {"mode": "all", "keys": list(train.INFO_SIGNALS)},
        "doom_skill": configuration["skill"],
        "game_variables": train.GAME_VARIABLES,
        "treat_episode_timeout_as_truncation": configuration["scenario"][
            "episode_timeout_as_truncation"
        ],
    }
    if provider_id == "gradoom":
        env = GraDoomVecEnv(
            configuration["scenario"]["game"],
            scenario=binding["pwad"]["path"],
            rom_path=binding["iwad"]["path"],
            device=device,
            transport="torch",
            info_frame_stack_keys=train.MODEL_HISTORY_SIGNALS,
            wall_contact_damage_scale=1.0,
            observation_renderer="native-fused",
            require_pinned_scenario=True,
            vizdoom_config={
                "episode_timeout": configuration["episode_horizon_tics"],
                "render_screen_flashes": configuration["scenario"]["render_screen_flashes"],
            },
            **common,
        )
        return env, None
    reference = load_reference_provider()
    if reference.revision != context["providers"][provider_id].get("revision", reference.revision):
        raise ValueError("reference provider revision does not match its binding")
    env = reference.make_env(
        context["reference_scenario_config_path"],
        rom_path=binding["iwad"]["path"],
        num_threads=1,
        doom_map=configuration["map"],
        vizdoom_config={
            "episode_timeout": configuration["episode_horizon_tics"],
            "render_hud": configuration["scenario"]["render_hud"],
            "render_screen_flashes": configuration["scenario"]["render_screen_flashes"],
        },
        **common,
    )
    return env, reference


def _history(infos: dict[str, Any], train: ModuleType, device: torch.device) -> torch.Tensor:
    values = torch.stack(
        [torch.as_tensor(infos[name], device=device) for name in train.MODEL_HISTORY_SIGNALS],
        dim=1,
    )
    return values[..., None].expand(-1, -1, train.FRAME_STACK).clone()


def _validate_loaded_checkpoint(request: dict[str, Any], loaded: Any) -> None:
    validate_loaded_policy_contract(
        request["policy"]["model_runtime_contract"],
        loaded.contract.as_dict(),
    )
    actual_identity = canonical_policy_execution_identity(
        artifact_sha256=loaded.artifact_sha256,
        artifact_contract=request["policy"]["model_runtime_contract"],
        stochastic_actions=True,
    )
    if actual_identity != request["policy"]["execution_identity"]:
        raise ValueError("loaded policy execution identity does not match the request")


def _dispatch_loaded_checkpoint(
    request: dict[str, Any], loaded: Any, evaluator: Callable[[], Any]
) -> Any:
    """Expose the real post-validation dispatch boundary as a CPU-testable seam."""

    _validate_loaded_checkpoint(request, loaded)
    return evaluator()


def _evaluate(request: dict[str, Any]) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for authenticated policy evaluation")
    train, repository = _load_train()
    context = _validated_context(request, repository)
    device = torch.device(context["device"])
    if device.type != "cuda":
        raise ValueError("authenticated policy evaluation requires a CUDA device")
    artifact = Path(request["policy"]["resolved_artifact_path"])
    loaded = load_policy_checkpoint(artifact, map_location=device)
    _dispatch_loaded_checkpoint(request, loaded, lambda: None)
    _policy, calls, precision = train._bind_checkpoint_policy(loaded, device)
    context_encoder = train.CombatContextEncoder(train.MODEL_HISTORY_SIGNALS, device)
    env, reference = _make_env(request["provider_id"], context=context, train=train, device=device)
    if reference is not None and reference.revision != request["provider_revision"]:
        env.close()
        raise ValueError("executed reference-provider revision does not match the request")
    outcomes: list[dict[str, Any]] = []
    maximum_decisions = (
        math.ceil(
            context["providers"][request["provider_id"]]["configuration"]["episode_horizon_tics"]
            / context["providers"][request["provider_id"]]["configuration"]["frame_skip"]
        )
        + 1
    )
    try:
        for seed in request["seeds"]:
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            reset_seed: Any = (
                torch.tensor([seed], dtype=torch.int64, device=device)
                if request["provider_id"] == "gradoom"
                else [seed]
            )
            observations, infos = env.reset(seed=reset_seed)
            histories = _history(infos, train, device)
            for episode_length in range(1, maximum_decisions + 1):
                observation_device = torch.as_tensor(observations, device=device)
                with torch.no_grad(), precision.autocast():
                    actions, _values, _log_probs = calls.act(
                        observation_device, context_encoder.encode(histories)
                    )
                provider_actions: Any = (
                    actions if request["provider_id"] == "gradoom" else actions.cpu().numpy()
                )
                observations, _rewards, terminated, truncated, infos = env.step(provider_actions)
                done = bool(torch.as_tensor(terminated).item()) or bool(
                    torch.as_tensor(truncated).item()
                )
                if done:
                    kills = (
                        float(torch.as_tensor(infos["player_killcount"]).item())
                        if reference is None
                        else reference.episode_kill_signals(infos, lane=0)["player_killcount"]
                    )
                    outcomes.append(
                        {
                            "seed": seed,
                            "player_killcount": kills,
                            "termination_state": (
                                "terminated"
                                if bool(torch.as_tensor(terminated).item())
                                else "truncated"
                            ),
                            "episode_length": episode_length,
                            "execution_failure": None,
                        }
                    )
                    break
                histories = torch.roll(histories, shifts=-1, dims=2)
                histories[:, :, -1].copy_(
                    torch.stack(
                        [
                            torch.as_tensor(infos[name], device=device)
                            for name in train.MODEL_HISTORY_SIGNALS
                        ],
                        dim=1,
                    )
                )
            else:
                raise RuntimeError(f"episode watchdog expired for seed {seed}")
    finally:
        env.close()
    return outcomes


def main() -> int:
    request = json.load(sys.stdin)
    try:
        outcomes = _evaluate(request)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"[:4096]
        outcomes = [
            {
                "seed": seed,
                "player_killcount": None,
                "termination_state": None,
                "episode_length": None,
                "execution_failure": {
                    "code": "authenticated_provider_failure",
                    "message": message,
                },
            }
            for seed in request.get("seeds", [])
        ]
    json.dump(
        {
            "protocol_version": 2,
            "execution_binding": request.get("execution_binding"),
            "outcomes": outcomes,
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
