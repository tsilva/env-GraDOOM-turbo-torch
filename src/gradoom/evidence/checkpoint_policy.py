from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT = "standalone-gradoom-ppo-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {field} must be a mapping")
    return value


def _configured(
    name: str,
    *sections: Mapping[str, Any],
    default: object,
) -> object:
    for section in sections:
        if name in section:
            return section[name]
    return default


def _choice(value: object, *, field: str, choices: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"checkpoint {field} must be one of {choices}, got {value!r}")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"checkpoint {field} must be a boolean")
    return value


@dataclass(frozen=True)
class CheckpointPolicyContract:
    architecture: str
    memory_format: str
    observation_blur_kernel: int
    frozen_encoder_custom_conv: bool
    precision: str
    compile_policy: bool
    float32_matmul_precision: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CheckpointPolicyContract:
        effective_recipe = _mapping(
            config.get("effective_recipe", {}),
            field="config.effective_recipe",
        )
        policy_model = _mapping(
            config.get("policy_model", {}),
            field="config.policy_model",
        )
        observation_invariance = _mapping(
            config.get("observation_invariance", {}),
            field="config.observation_invariance",
        )
        architecture = _configured(
            "architecture",
            policy_model,
            default=effective_recipe.get("policy_architecture", "nature"),
        )
        memory_format = _configured(
            "memory_format",
            policy_model,
            default=effective_recipe.get("policy_memory_format", "contiguous"),
        )
        blur_kernel = _configured(
            "observation_blur_kernel",
            policy_model,
            effective_recipe,
            observation_invariance,
            default=1,
        )
        if not isinstance(blur_kernel, int) or isinstance(blur_kernel, bool):
            raise ValueError("checkpoint observation_blur_kernel must be an integer")
        if blur_kernel <= 0 or blur_kernel % 2 == 0:
            raise ValueError("checkpoint observation_blur_kernel must be a positive odd integer")
        frozen_custom_conv = _configured(
            "frozen_encoder_custom_conv",
            policy_model,
            default=(
                effective_recipe.get("freeze_observation_encoder", False)
                and effective_recipe.get("frozen_encoder_custom_conv", False)
            ),
        )
        return cls(
            architecture=str(architecture),
            memory_format=_choice(
                memory_format,
                field="policy memory_format",
                choices=("contiguous", "channels-last"),
            ),
            observation_blur_kernel=blur_kernel,
            frozen_encoder_custom_conv=_boolean(
                frozen_custom_conv,
                field="frozen_encoder_custom_conv",
            ),
            precision=_choice(
                effective_recipe.get("precision", "fp32"),
                field="precision",
                choices=("fp32", "amp-fp16", "amp-bf16"),
            ),
            compile_policy=_boolean(
                effective_recipe.get("compile_policy", True),
                field="compile_policy",
            ),
            float32_matmul_precision=_choice(
                effective_recipe.get("float32_matmul_precision", "high"),
                field="float32_matmul_precision",
                choices=("highest", "high", "medium"),
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "memory_format": self.memory_format,
            "observation_blur_kernel": self.observation_blur_kernel,
            "frozen_encoder_custom_conv": self.frozen_encoder_custom_conv,
            "precision": self.precision,
            "compile_policy": self.compile_policy,
            "float32_matmul_precision": self.float32_matmul_precision,
        }


@dataclass(frozen=True)
class LoadedPolicyCheckpoint:
    payload: Mapping[str, Any]
    contract: CheckpointPolicyContract
    artifact_sha256: str


def load_policy_checkpoint(
    path: Path,
    *,
    map_location: torch.device,
) -> LoadedPolicyCheckpoint:
    path = path.expanduser().resolve()
    digest_before = _file_sha256(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    digest_after = _file_sha256(path)
    if digest_after != digest_before:
        raise RuntimeError(f"checkpoint changed while it was being loaded: {path}")
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported standalone checkpoint: {path}")
    _mapping(payload.get("policy_state_dict"), field="policy_state_dict")
    config = _mapping(payload.get("config", {}), field="config")
    return LoadedPolicyCheckpoint(
        payload=payload,
        contract=CheckpointPolicyContract.from_config(config),
        artifact_sha256=digest_before,
    )


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointPolicyContract",
    "LoadedPolicyCheckpoint",
    "load_policy_checkpoint",
]
