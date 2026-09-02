"""Repository-owned entry point for real parity policy execution.

The evidence coordinator accepts this exact tracked script for non-fixture runs.  Provider
execution is intentionally fail-closed until the real evaluator installs its audited backend;
an unavailable backend is retained as ordinary failed evidence and can never issue a certificate.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.load(sys.stdin)
    outcomes = [
        {
            "seed": seed,
            "player_killcount": None,
            "termination_state": None,
            "episode_length": None,
            "execution_failure": {
                "code": "authenticated_provider_backend_unavailable",
                "message": "The audited real-provider policy backend is not installed.",
            },
        }
        for seed in request["seeds"]
    ]
    json.dump(
        {
            "protocol_version": 2,
            "execution_binding": request["execution_binding"],
            "outcomes": outcomes,
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
