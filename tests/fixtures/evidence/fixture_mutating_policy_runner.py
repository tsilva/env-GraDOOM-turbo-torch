from __future__ import annotations

import json
import sys
from pathlib import Path

request = json.load(sys.stdin)
Path(request["policy"]["resolved_artifact_path"]).write_bytes(b"post-seal mutation\n")
json.dump({"protocol_version": 1, "outcomes": []}, sys.stdout)
