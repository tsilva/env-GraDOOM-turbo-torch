from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_TOOL_PATH = Path(__file__).parents[1] / "tools" / "convert_gradlab_checkpoint.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("convert_gradlab_checkpoint", _TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = tool
_TOOL_SPEC.loader.exec_module(tool)
train = tool._load_standalone_train()


def test_context_conversion_is_exact_for_current_context() -> None:
    target = train.NatureActorCritic().state_dict()
    source = {}
    reverse = {
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
    for target_name, source_name in reverse.items():
        source[source_name] = torch.zeros_like(target[target_name])
    source_context = torch.arange(21, dtype=torch.float32).expand(256, -1).clone()
    source["features_extractor.fusion.0.weight"] = torch.cat(
        (torch.zeros((256, 512)), source_context),
        dim=1,
    )

    converted = tool._converted_state_dict(source, target)
    context_weights = converted["fusion.0.weight"][:, 512:]

    assert torch.count_nonzero(context_weights[:, :-21]) == 0
    assert torch.equal(context_weights[:, -21:], source_context)
