from __future__ import annotations

import json
import sys

json.load(sys.stdin)
unexpected_field = "x" * (8 * 1024 * 1024 - 128)
json.dump(
    {"protocol_version": 2, "outcomes": [], unexpected_field: None},
    sys.stdout,
    separators=(",", ":"),
)
