from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

_TOOL_PATH = Path(__file__).parents[1] / "tools" / "compare_policy_observation_domains.py"
_TOOL_SPEC = importlib.util.spec_from_file_location(
    "compare_policy_observation_domains",
    _TOOL_PATH,
)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
comparison = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = comparison
_TOOL_SPEC.loader.exec_module(comparison)


def test_decal_visibility_uses_negative_serial_as_inactive() -> None:
    serial = torch.tensor([[-1, 0, 7], [-1, -1, -1]])

    active = comparison._visibility_state_active("hitscan_decal_serial", serial)

    assert active.tolist() == [[False, True, True], [False, False, False]]


def test_decal_ablation_hides_and_restores_serials() -> None:
    original = torch.tensor([[-1, 4, 9]], dtype=torch.int32)
    engine = SimpleNamespace(hitscan_decal_serial=original.clone())
    engine.render_reference_frame = lambda: engine.hitscan_decal_serial.clone()
    engine.render_fast_native_policy_frame = (
        lambda *, exact_weapon: engine.hitscan_decal_serial.clone() + int(exact_weapon)
    )
    env = SimpleNamespace(_engine=engine)

    exact, fast = comparison._current_ablation_frames(
        env,
        "native-fused",
        ("hitscan_decal_serial",),
    )

    assert exact.tolist() == [[-1, -1, -1]]
    assert fast.tolist() == [[-1, -1, -1]]
    assert torch.equal(engine.hitscan_decal_serial, original)
