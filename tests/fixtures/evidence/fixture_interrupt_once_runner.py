from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

request = json.load(sys.stdin)
marker = Path(os.environ["GRADOOM_INTERRUPT_MARKER"])
log = Path(os.environ["GRADOOM_INVOCATION_LOG"])
unit = f"{request['provider_id']}:{request['policy']['id']}"
with log.open("a", encoding="utf-8") as stream:
    stream.write(unit + "\n")
    stream.flush()
    os.fsync(stream.fileno())
if unit == "gradoom:reference-policy" and not marker.exists():
    marker.write_text("interrupt once\n", encoding="utf-8")
    os.kill(os.getppid(), signal.SIGKILL)

outcomes = []
for seed in request["seeds"]:
    if seed == 17:
        outcomes.append(
            {
                "seed": seed,
                "player_killcount": None,
                "termination_state": None,
                "episode_length": None,
                "execution_failure": {
                    "code": "fixture_failure",
                    "message": "deterministic fixture failure",
                },
            }
        )
    else:
        outcomes.append(
            {
                "seed": seed,
                "player_killcount": seed % 4,
                "termination_state": "terminated",
                "episode_length": 88,
                "execution_failure": None,
            }
        )

json.dump(
    {
        "protocol_version": 2,
        "execution_binding": request["execution_binding"],
        "outcomes": outcomes,
    },
    sys.stdout,
)
