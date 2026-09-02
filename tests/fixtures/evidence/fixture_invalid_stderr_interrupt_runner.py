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

if unit == "gradoom:gradoom-policy":
    sys.stderr.buffer.write(b"bad\x00stderr")
    sys.stderr.buffer.flush()
    raise SystemExit(9)

if unit == "gradoom:reference-policy" and not marker.exists():
    marker.write_text("interrupt once\n", encoding="utf-8")
    os.kill(os.getppid(), signal.SIGKILL)

json.dump(
    {
        "protocol_version": 2,
        "outcomes": [
            {
                "seed": seed,
                "player_killcount": seed % 4,
                "termination_state": "terminated",
                "episode_length": 88,
                "execution_failure": None,
            }
            for seed in request["seeds"]
        ],
    },
    sys.stdout,
)
