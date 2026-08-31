from __future__ import annotations

import json
import sys

request = json.load(sys.stdin)
json.dump(
    {
        "protocol_version": 1,
        "outcomes": [
            {
                "seed": seed,
                "player_killcount": seed % 3,
                "termination_state": "terminated",
                "episode_length": 90,
                "execution_failure": None,
            }
            for seed in request["seeds"][:-1]
        ],
    },
    sys.stdout,
)
