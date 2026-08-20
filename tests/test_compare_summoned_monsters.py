from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_TOOL_PATH = Path(__file__).parents[1] / "tools" / "compare_summoned_monsters.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("compare_summoned_monsters", _TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = tool
_TOOL_SPEC.loader.exec_module(tool)


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
    def record(damage: float, observed: bool) -> dict[str, float | int | None]:
        return {
            "damage_taken": damage,
            "final_displacement": damage,
            "first_damage_decision": 1 if observed else None,
            "first_motion_decision": 2 if observed else None,
            "hits_taken": damage,
            "minimum_monster_player_distance": damage,
            "player_displacement": damage,
        }

    reference = [record(1.0, False), record(3.0, True)]
    env_doom_turbo_torch = [record(3.0, True), record(5.0, True)]

    comparison = tool._distribution_comparison(reference, env_doom_turbo_torch)
    damage = comparison["damage_taken"]
    assert damage["env_doom_turbo_torch_minus_reference"] == 2.0
    assert damage["env_doom_turbo_torch_minus_reference_standard_error"] == math.sqrt(2.0)
    observed = comparison["first_damage_observed"]
    assert observed["env_doom_turbo_torch_minus_reference"] == 0.5
    assert observed["reference"]["count"] == 2
    assert observed["env_doom_turbo_torch"]["count"] == 2
