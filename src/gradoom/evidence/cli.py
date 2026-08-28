from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from .report import EvidenceError, build_readiness_report, validate_merge_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradoom-evidence",
        description="Produce versioned GraDOOM evidence reports.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--merge",
        type=Path,
        help="validate and continue an existing report with the same run identity",
    )
    return parser


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_readiness_report(args.manifest)
        if args.merge is not None:
            validate_merge_report(args.merge, report["run_identity"])
        _write_report(args.output, report)
    except (EvidenceError, OSError) as error:
        print(f"gradoom-evidence: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
