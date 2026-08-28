from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOL_PATH = Path(__file__).parents[1] / "tools" / "replay_vizdoom_action_traces.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("replay_vizdoom_action_traces", _TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = tool
_TOOL_SPEC.loader.exec_module(tool)


def test_summary_keeps_player_quality_separate_from_compatibility_killcount() -> None:
    summary = tool._summary(
        [
            {
                "player_killcount": 3.0,
                "compatibility_killcount": 103.0,
                "length": 100,
                "return": 1.0,
                "terminated": True,
                "hits_taken": 2.0,
                "damage_taken": 4.0,
            }
        ]
    )

    assert "kills_mean" not in summary
    assert summary["player_killcount_mean"] == 3.0
    assert summary["compatibility_killcount_mean"] == 103.0
