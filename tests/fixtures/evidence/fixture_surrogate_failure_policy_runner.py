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
                "player_killcount": None,
                "termination_state": None,
                "episode_length": None,
                "execution_failure": {
                    "code": "\ud800",
                    "message": "malformed failure code",
                },
            }
            for seed in request["seeds"]
        ],
    },
    sys.stdout,
)
