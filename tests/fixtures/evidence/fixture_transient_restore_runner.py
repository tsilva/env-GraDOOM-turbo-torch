from __future__ import annotations

import json
import os
import sys
from pathlib import Path

request = json.load(sys.stdin)
artifact = Path(request["policy"]["resolved_artifact_path"])
try:
    artifact.write_bytes(b"replacement policy\n")
except OSError:
    pass
else:
    raise SystemExit("execution artifact was writable")
try:
    Path(__file__).write_bytes(b"replacement runner\n")
except OSError:
    pass
else:
    raise SystemExit("execution runner was writable")

for raw_path in json.loads(os.environ["GRADOOM_TRANSIENT_MUTATION_TARGETS"]):
    path = Path(raw_path)
    original = path.read_bytes()
    path.write_bytes(b"transient replacement\n")
    path.write_bytes(original)

if not artifact.read_bytes().startswith(b"frozen "):
    raise SystemExit("runner did not receive sealed policy bytes")
json.dump(
    {
        "protocol_version": 2,
        "outcomes": [
            {
                "seed": seed,
                "player_killcount": seed % 5,
                "termination_state": "terminated",
                "episode_length": 77,
                "execution_failure": None,
            }
            for seed in request["seeds"]
        ],
    },
    sys.stdout,
)
