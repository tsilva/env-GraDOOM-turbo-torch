from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_TOOL_PATH = Path(__file__).parents[1] / "tools" / "compare_renderer.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("compare_renderer", _TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = tool
_TOOL_SPEC.loader.exec_module(tool)


def test_reference_policy_frame_uses_pinned_provider_preprocessing(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    def preprocess_into(*args) -> None:
        calls.append(args)
        args[1].fill(23)

    monkeypatch.setattr(
        tool,
        "load_reference_provider",
        lambda: SimpleNamespace(preprocess_into=preprocess_into),
    )

    result = tool._reference_policy_frame(torch.zeros((240, 320, 3), dtype=torch.uint8))

    assert np.asarray(calls[0][0]).shape == (1, 240, 320, 3)
    assert calls[0][2:] == ([0, 32, 0, 0], True, 0, "area")
    assert result.shape == (84, 84)
    assert torch.all(result == 23)
