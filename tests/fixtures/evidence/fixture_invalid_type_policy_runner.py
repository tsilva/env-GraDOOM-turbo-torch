from __future__ import annotations

import json
import sys

request = json.load(sys.stdin)
json.dump(
    {
        "protocol_version": 2,
        "outcomes": [
            {
                "seed": seed,
                "player_killcount": 1,
                "termination_state": [],
                "episode_length": 100,
                "execution_failure": None,
            }
            for seed in request["seeds"]
        ],
    },
    sys.stdout,
)
