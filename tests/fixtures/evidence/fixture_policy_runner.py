from __future__ import annotations

import json
import sys

request = json.load(sys.stdin)
outcomes = []
for seed in request["seeds"]:
    failure = seed == request.get("fixture_failure_seed")
    outcomes.append(
        {
            "seed": seed,
            "player_killcount": None
            if failure
            else (seed % 7) + (request["provider_id"] == "gradoom"),
            "termination_state": None if failure else "terminated",
            "episode_length": None if failure else 100 + seed % 11,
            "execution_failure": (
                {"code": "fixture_failure", "message": "declared fixture execution failure"}
                if failure
                else None
            ),
        }
    )
json.dump({"protocol_version": 2, "outcomes": outcomes}, sys.stdout)
