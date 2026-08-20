from __future__ import annotations

import pytest

from env_doom_turbo_torch.actions import (
    DEATHMATCH_ACTION_MEANINGS,
    DEATHMATCH_ACTION_TABLE_SHA256,
    DEATHMATCH_ACTIONS,
    normalize_action_table,
)


def test_pinned_action_table_matches_vizdoom_turbo() -> None:
    assert len(DEATHMATCH_ACTIONS) == 17
    assert DEATHMATCH_ACTION_MEANINGS[0] == "noop"
    assert DEATHMATCH_ACTION_MEANINGS[13] == "attack_turn_left"
    assert (
        DEATHMATCH_ACTION_TABLE_SHA256
        == "0bd9dd28d67a88ef6bc54734f53d55bc4af597e672665a7f20d4b204098036af"
    )


def test_action_table_rejects_unknown_and_duplicate_actions() -> None:
    with pytest.raises(ValueError, match="unknown button"):
        normalize_action_table(((), ("RIP_AND_TEAR",)))
    with pytest.raises(ValueError, match="duplicates"):
        normalize_action_table(((), ()))
    with pytest.raises(ValueError, match="labels must be strings"):
        normalize_action_table(((), (1,)))
    with pytest.raises(ValueError, match="non-empty sequence"):
        normalize_action_table("minimal")
