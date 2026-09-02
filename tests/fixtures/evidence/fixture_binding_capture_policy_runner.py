from __future__ import annotations

import json
import os
import sys
from pathlib import Path

request = json.load(sys.stdin)
Path(os.environ["GRADOOM_BINDING_CAPTURE"]).write_text(
    json.dumps(request, sort_keys=True), encoding="utf-8"
)
json.dump(
    {
        "protocol_version": 2,
        "execution_binding": request["execution_binding"],
        "outcomes": [
            {
                "seed": seed,
                "player_killcount": seed % 5,
                "termination_state": "terminated",
                "episode_length": 80,
                "execution_failure": None,
            }
            for seed in request["seeds"]
        ],
    },
    sys.stdout,
)
