"""Convert the published GradLab Deathmatch policy to the standalone format.

The converter reads the Torch policy member directly from the SB3-compatible
archive.  It does not import GradLab or Stable-Baselines3.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

PUBLISHED_CHECKPOINT_SHA256 = "e5596d939cef0b6d34e75874b4d917660e7b6efab672ad8d7bea85445a7bb100"
PUBLISHED_CHECKPOINT_STEP = 463_970_304


def _load_standalone_train() -> ModuleType:
    path = Path(__file__).parents[1] / "train.py"
    spec = importlib.util.spec_from_file_location("env_doom_turbo_torch_standalone_train", path)
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"cannot load standalone trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_policy(path: Path) -> Mapping[str, torch.Tensor]:
    with zipfile.ZipFile(path) as archive:
        try:
            payload = archive.read("policy.pth")
        except KeyError as exc:
            raise ValueError("GradLab checkpoint has no policy.pth member") from exc
    loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError("GradLab policy.pth is not a state dictionary")
    valid_items = all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in loaded.items()
    )
    if not valid_items:
        raise TypeError("GradLab policy.pth contains non-tensor state")
    return loaded


def _converted_state_dict(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    direct = {
        "observation_encoder.0.weight": "features_extractor.observation_encoder.cnn.0.weight",
        "observation_encoder.0.bias": "features_extractor.observation_encoder.cnn.0.bias",
        "observation_encoder.2.weight": "features_extractor.observation_encoder.cnn.2.weight",
        "observation_encoder.2.bias": "features_extractor.observation_encoder.cnn.2.bias",
        "observation_encoder.4.weight": "features_extractor.observation_encoder.cnn.4.weight",
        "observation_encoder.4.bias": "features_extractor.observation_encoder.cnn.4.bias",
        "observation_encoder.7.weight": "features_extractor.observation_encoder.linear.0.weight",
        "observation_encoder.7.bias": "features_extractor.observation_encoder.linear.0.bias",
        "fusion.0.bias": "features_extractor.fusion.0.bias",
        "action_head.weight": "action_net.weight",
        "action_head.bias": "action_net.bias",
        "value_head.weight": "value_net.weight",
        "value_head.bias": "value_net.bias",
    }
    converted: dict[str, torch.Tensor] = {}
    for target_name, source_name in direct.items():
        value = source[source_name]
        if value.shape != target[target_name].shape:
            raise ValueError(
                f"shape mismatch for {target_name}: {tuple(value.shape)} != "
                f"{tuple(target[target_name].shape)}"
            )
        converted[target_name] = value.detach().clone()

    source_fusion = source["features_extractor.fusion.0.weight"]
    target_fusion = target["fusion.0.weight"]
    image_features = 512
    per_frame_context = 21
    if source_fusion.shape != (256, image_features + per_frame_context):
        raise ValueError(f"unexpected source fusion shape: {tuple(source_fusion.shape)}")
    expected_target_shape = (256, image_features + 4 * per_frame_context)
    if target_fusion.shape != expected_target_shape:
        raise ValueError(f"unexpected target fusion shape: {tuple(target_fusion.shape)}")
    converted_fusion = torch.zeros_like(target_fusion)
    converted_fusion[:, :image_features].copy_(source_fusion[:, :image_features])
    converted_fusion[:, -per_frame_context:].copy_(source_fusion[:, image_features:])
    converted["fusion.0.weight"] = converted_fusion

    missing = set(target) - set(converted)
    if missing:
        raise ValueError(f"unmapped standalone policy tensors: {sorted(missing)}")
    return converted


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an immutable GradLab Deathmatch policy without runtime dependencies.",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256",
        default=PUBLISHED_CHECKPOINT_SHA256,
        help="Refuse any source archive whose SHA-256 differs from this value.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_path = args.source.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source_path}")
    actual_sha256 = _sha256(source_path)
    if actual_sha256 != str(args.expected_sha256):
        raise ValueError(
            f"source SHA-256 mismatch: expected {args.expected_sha256}, got {actual_sha256}"
        )

    train = _load_standalone_train()
    policy = train.NatureActorCritic()
    policy.load_state_dict(_converted_state_dict(_source_policy(source_path), policy.state_dict()))
    optimizer = train._make_optimizer(
        policy,
        learning_rate=train.REFERENCE_RECIPE.learning_rate,
        fused=False,
    )
    audit_args = train._parser().parse_args(("--config-only",))
    audit: dict[str, Any] = train._audit_config(audit_args)
    audit["operation"] = "convert_gradlab_checkpoint"
    audit["initialization"] = {
        "source": "huggingface:tsilva/VizdoomDeathmatch-v1_gradlab-ppo_b0330247",
        "source_checkpoint_sha256": actual_sha256,
        "source_checkpoint_step": PUBLISHED_CHECKPOINT_STEP,
        "context_adaptation": "current-21-to-newest-of-history-84-exact",
    }
    destination = train._save_checkpoint(
        args.output,
        policy=policy,
        optimizer=optimizer,
        step=PUBLISHED_CHECKPOINT_STEP,
        audit=audit,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
