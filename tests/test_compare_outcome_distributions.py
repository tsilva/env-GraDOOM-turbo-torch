from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_TOOL_PATH = Path(__file__).parents[1] / "tools" / "compare_outcome_distributions.py"
_TOOL_SPEC = importlib.util.spec_from_file_location(
    "compare_outcome_distributions",
    _TOOL_PATH,
)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = tool
_TOOL_SPEC.loader.exec_module(tool)


def _record(value: float, *, length: int = 100) -> dict[str, float | int | bool]:
    return {
        "kills": value,
        "length": length,
        "return": value,
        "terminated": bool(value),
        "hitcount": value,
        "damagecount": value,
        "hits_taken": value,
        "damage_taken": value,
        "health_gain": value,
        "health_loss": value,
        "armor_gain": value,
        "armor_loss": value,
    }


def test_normal_summary_reports_sample_uncertainty() -> None:
    summary = tool._normal_summary([1.0, 3.0, 5.0, 7.0])

    assert summary["count"] == 4
    assert summary["mean"] == 4.0
    assert math.isclose(summary["sample_stddev"], math.sqrt(20 / 3))
    assert math.isclose(summary["standard_error"], math.sqrt(5 / 3))
    assert summary["normal_95_ci"] == [
        4.0 - 1.96 * math.sqrt(5 / 3),
        4.0 + 1.96 * math.sqrt(5 / 3),
    ]


def test_distribution_comparison_uses_independent_provider_uncertainty() -> None:
    reference = [_record(1.0), _record(3.0)]
    gradoom = [_record(3.0), _record(5.0)]

    comparison = tool._distribution_comparison(reference, gradoom)
    damage = comparison["damage_taken"]
    assert damage["gradoom_minus_vizdoom"] == 2.0
    assert damage["gradoom_minus_vizdoom_standard_error"] == math.sqrt(2.0)
    rate = comparison["damage_taken_per_1000_decisions"]
    assert rate["gradoom_minus_vizdoom"] == 20.0
    assert rate["vizdoom"]["count"] == 2
    assert rate["gradoom"]["count"] == 2
