"""Print deterministic scenario-compiler metadata without starting an engine."""

from __future__ import annotations

import argparse
import json

from .scenario import compile_deathmatch_scenario


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--iwad", required=True)
    parser.add_argument("--allow-unpinned-scenario", action="store_true")
    args = parser.parse_args()
    compiled = compile_deathmatch_scenario(
        args.scenario,
        args.iwad,
        require_pinned_scenario=not args.allow_unpinned_scenario,
    )
    print(
        json.dumps(
            {
                "bounds": compiled.bounds,
                "iwad_sha256": compiled.iwad_sha256,
                "items": len(compiled.item_types),
                "namespace": compiled.namespace,
                "player_starts": len(compiled.player_starts),
                "scenario_sha256": compiled.scenario_sha256,
                "sectors": len(compiled.sector_heights),
                "vertices": len(compiled.vertices),
                "walls": len(compiled.wall_segments),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
