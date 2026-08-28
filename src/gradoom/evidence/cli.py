from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .report import (
    EvidenceError,
    _paths_alias,
    _resolve_evidence_path,
    build_readiness_report,
    validate_merge_report,
)


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
    try:
        payload = json.dumps(
            report,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    except (RecursionError, ValueError) as error:
        raise EvidenceError("report cannot be encoded as standard JSON") from error
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


def _validate_output_path(
    output_path: Path,
    *,
    manifest_path: Path,
    report: dict[str, Any],
    merge_path: Path | None,
) -> None:
    working_directory = Path.cwd()
    resolved_output = _resolve_evidence_path(
        output_path,
        base_directory=working_directory,
    )
    resolved_manifest = _resolve_evidence_path(
        manifest_path,
        base_directory=working_directory,
    )
    if _paths_alias(resolved_output, resolved_manifest):
        raise EvidenceError("output path aliases the manifest")

    declared_inputs = report["declared_inputs"]
    assert isinstance(declared_inputs, list)
    for declared_input in declared_inputs:
        assert isinstance(declared_input, dict)
        resolved_input = _resolve_evidence_path(
            Path(declared_input["path"]),
            base_directory=resolved_manifest.parent,
        )
        if _paths_alias(resolved_output, resolved_input):
            raise EvidenceError(
                f"output path aliases declared input {declared_input['name']!r}"
            )

    if merge_path is not None:
        resolved_merge = _resolve_evidence_path(
            merge_path,
            base_directory=working_directory,
        )
        if _paths_alias(resolved_output, resolved_merge):
            raise EvidenceError("output path aliases the merge report")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_readiness_report(args.manifest)
        _validate_output_path(
            args.output,
            manifest_path=args.manifest,
            report=report,
            merge_path=args.merge,
        )
        if args.merge is not None:
            evidence_index = report["evidence_index"]
            assert isinstance(evidence_index, dict)
            entries = evidence_index["entries"]
            assert isinstance(entries, list)
            manifest_entry = next(
                entry
                for entry in entries
                if isinstance(entry, dict) and entry.get("name") == "manifest"
            )
            validate_merge_report(
                args.merge,
                report["run_identity"],
                expected_manifest_sha256=manifest_entry["sha256"],
                manifest_directory=args.manifest.parent,
            )
        _write_report(args.output, report)
    except (EvidenceError, OSError) as error:
        print(f"gradoom-evidence: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
