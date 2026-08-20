from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TOOL_PATH = Path(__file__).parents[1] / "tools" / "evaluate_vizdoom_checkpoint.py"
_TOOL_SPEC = importlib.util.spec_from_file_location("vizdoom_checkpoint_eval", _TOOL_PATH)
assert _TOOL_SPEC is not None and _TOOL_SPEC.loader is not None
tool = importlib.util.module_from_spec(_TOOL_SPEC)
sys.modules[_TOOL_SPEC.name] = tool
_TOOL_SPEC.loader.exec_module(tool)


def test_episode_quotas_are_balanced_and_exact() -> None:
    quotas = tool._load_standalone_train()._episode_quotas(100, 16)

    assert quotas == (7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)
    assert sum(quotas) == 100


def test_provider_and_game_seed_protocol_matches_known_first_lane() -> None:
    assert tool._provider_seed(123, 0, 0) == 123
    assert tool._provider_seed(123, 7, 0) == 130
    assert tool._provider_seed(123, 0, 1) == 1917560605
    assert tool._game_seed(123) == 66316749


def test_zero_shot_summary_uses_reference_kill_target() -> None:
    result = tool._summary(
        (
            {"kills": 31.0, "length": 100},
            {"kills": 33.0, "length": 200},
        )
    )

    assert result["evaluation/kills/mean"] == 32.0
    assert result["evaluation/episode/count"] == 2
    assert result["evaluation/target/kills/mean"] == 31.78
    assert result["evaluation/target/passed"] is True


def test_zero_shot_contract_keeps_the_training_hud_enabled() -> None:
    assert tool.REFERENCE_RENDER_HUD is True


def test_reference_contract_excludes_env_doom_turbo_torch_only_player_kill_signal() -> None:
    train = tool._load_standalone_train()

    game_variables = tool._reference_signal_names(train.GAME_VARIABLES)
    info_signals = tool._reference_signal_names(train.INFO_SIGNALS)

    assert "killcount" in game_variables
    assert "player_killcount" not in game_variables
    assert "player_killcount" not in info_signals
