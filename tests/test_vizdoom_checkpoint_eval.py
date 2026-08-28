from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

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
            {
                "player_killcount": 31.0,
                "compatibility_killcount": 91.0,
                "length": 100,
            },
            {
                "player_killcount": 33.0,
                "compatibility_killcount": 99.0,
                "length": 200,
            },
        )
    )

    assert result["evaluation/kills/mean"] == 32.0
    assert result["evaluation/kills/signal"] == "player_killcount"
    assert result["evaluation/compatibility_killcount/mean"] == 95.0
    assert result["evaluation/episode/count"] == 2
    assert result["evaluation/target/kills/mean"] == 31.78
    assert result["evaluation/target/kills/signal"] == "player_killcount"
    assert result["evaluation/target/passed"] is True


def test_reference_contract_uses_the_certified_profile_hud_setting() -> None:
    assert tool.REFERENCE_RENDER_HUD is False


def test_reference_contract_explicitly_requests_both_kill_signals() -> None:
    train = tool._load_standalone_train()

    game_variables = tool._reference_signal_names(train.GAME_VARIABLES)
    info_signals = tool._reference_signal_names(train.INFO_SIGNALS)

    assert "killcount" in game_variables
    assert "player_killcount" in game_variables
    assert "player_killcount" in info_signals


def test_checkpoint_execution_uses_frozen_non_default_model_and_runtime_contract(
    tmp_path: Path,
) -> None:
    train = tool._load_standalone_train()
    source = train.NatureActorCritic("nature-quarter", "channels-last", 3)
    checkpoint = tmp_path / "quarter-policy.pt"
    torch.save(
        {
            "format": "standalone-gradoom-ppo-v1",
            "policy_state_dict": source.state_dict(),
            "config": {
                "effective_recipe": {
                    "compile_policy": False,
                    "float32_matmul_precision": "highest",
                    "precision": "fp32",
                },
                "policy_model": {
                    "architecture": "nature-quarter",
                    "memory_format": "channels-last",
                    "observation_blur_kernel": 3,
                    "frozen_encoder_custom_conv": False,
                },
            },
        },
        checkpoint,
    )

    loaded, policy, calls, precision = tool._load_checkpoint_execution(
        checkpoint,
        train,
        torch.device("cpu"),
    )

    assert loaded.contract.as_dict() == {
        "architecture": "nature-quarter",
        "memory_format": "channels-last",
        "observation_blur_kernel": 3,
        "frozen_encoder_custom_conv": False,
        "precision": "fp32",
        "compile_policy": False,
        "float32_matmul_precision": "highest",
    }
    assert (
        policy.observation_feature_count
        == train.POLICY_ARCHITECTURES["nature-quarter"].observation_features
    )
    assert policy.channels_last is True
    assert policy.observation_blur_kernel == 3
    assert calls.act.__self__ is policy
    assert precision.name == "fp32"
