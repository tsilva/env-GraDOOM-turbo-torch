from __future__ import annotations

import json
import sys

json.load(sys.stdin)
json.dump({"protocol_version": 2.0, "outcomes": []}, sys.stdout)
