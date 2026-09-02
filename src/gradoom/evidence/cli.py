from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from .benchmark import build_development_benchmark_report
from .diagnostic import build_fixed_time_diagnostic_report
from .policy_corpus import build_policy_evaluation_report
from .report import (
    EvidenceError,
    _load_manifest,
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


def _fsync_directory(path: Path) -> None:
    directory_descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _write_report(
    path: Path,
    report: dict[str, object],
    *,
    revalidate: Any | None = None,
    rollback_on_failure: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            json.dumps(
                report,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (RecursionError, ValueError) as error:
        raise EvidenceError("report cannot be encoded as standard JSON") from error
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    backup_path: Path | None = None
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if revalidate is not None:
            revalidate()
        if rollback_on_failure and path.exists():
            backup_descriptor, backup_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".rollback",
            )
            os.close(backup_descriptor)
            backup_path = Path(backup_name)
            backup_path.unlink()
            os.replace(path, backup_path)
        os.replace(temporary_path, path)
        replaced = True
        if revalidate is not None:
            revalidate()
        _fsync_directory(path.parent)
        if backup_path is not None:
            backup_path.unlink()
            backup_path = None
            # The claim-bearing replacement was already validated and made durable.
            # Failure to durably remove the private rollback copy is non-fatal.
            with contextlib.suppress(OSError):
                _fsync_directory(path.parent)
    except BaseException:
        if rollback_on_failure and replaced:
            path.unlink(missing_ok=True)
        if rollback_on_failure and backup_path is not None:
            os.replace(backup_path, path)
            backup_path = None
        if rollback_on_failure:
            with contextlib.suppress(OSError):
                _fsync_directory(path.parent)
        raise
    finally:
        temporary_path.unlink(missing_ok=True)
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)


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
    manifest_directory = _resolve_evidence_path(
        manifest_path.parent,
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
            base_directory=manifest_directory,
        )
        if _paths_alias(resolved_output, resolved_input):
            raise EvidenceError(f"output path aliases declared input {declared_input['name']!r}")

    wad_profile = report.get("wad_profile")
    if isinstance(wad_profile, dict):
        providers = wad_profile.get("providers")
        assert isinstance(providers, list)
        for provider in providers:
            assert isinstance(provider, dict)
            for asset_name in ("iwad", "pwad"):
                asset = provider[asset_name]
                assert isinstance(asset, dict)
                raw_path = asset.get("path")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue
                resolved_asset = _resolve_evidence_path(
                    Path(raw_path),
                    base_directory=manifest_directory,
                )
                if _paths_alias(resolved_output, resolved_asset):
                    asset_id = f"{provider['id']}.{asset_name}"
                    raise EvidenceError(f"output path aliases WAD profile asset {asset_id!r}")

    generated_artifacts = report.get("generated_artifacts", [])
    assert isinstance(generated_artifacts, list)
    for artifact in generated_artifacts:
        assert isinstance(artifact, dict)
        resolved_artifact = _resolve_evidence_path(
            Path(artifact["path"]),
            base_directory=manifest_directory,
        )
        if _paths_alias(resolved_output, resolved_artifact):
            raise EvidenceError(
                f"output path aliases generated benchmark artifact {artifact['name']!r}"
            )

    policy_evaluation = report.get("policy_evaluation")
    if isinstance(policy_evaluation, dict):
        corpus = policy_evaluation.get("corpus")
        if isinstance(corpus, dict):
            policies = corpus.get("policies")
            if isinstance(policies, list):
                for policy in policies:
                    if not isinstance(policy, dict):
                        continue
                    raw_path = policy.get("resolved_artifact_path")
                    if not isinstance(raw_path, str):
                        continue
                    artifact_path = _resolve_evidence_path(
                        Path(raw_path), base_directory=manifest_directory
                    )
                    if _paths_alias(resolved_output, artifact_path):
                        raise EvidenceError(
                            f"output path aliases policy artifact {policy.get('id')!r}"
                        )

    if merge_path is not None:
        resolved_merge = _resolve_evidence_path(
            merge_path,
            base_directory=working_directory,
        )
        if _paths_alias(resolved_output, resolved_merge):
            raise EvidenceError("output path aliases the merge report")


def _validate_document_paths(
    output_path: Path,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    merge_path: Path | None,
) -> None:
    """Reject document/asset aliases before a workflow can perform expensive work."""
    working_directory = Path.cwd()
    resolved_output = _resolve_evidence_path(output_path, base_directory=working_directory)
    resolved_manifest = _resolve_evidence_path(manifest_path, base_directory=working_directory)
    if _paths_alias(resolved_output, resolved_manifest):
        raise EvidenceError("output path aliases the manifest")
    if merge_path is not None:
        resolved_merge = _resolve_evidence_path(merge_path, base_directory=working_directory)
        if _paths_alias(resolved_output, resolved_merge):
            raise EvidenceError("output path aliases the merge report")
    manifest_directory = _resolve_evidence_path(
        manifest_path.parent,
        base_directory=working_directory,
    )
    declared_inputs = manifest.get("declared_inputs")
    if isinstance(declared_inputs, list):
        for declared_input in declared_inputs:
            if not isinstance(declared_input, dict):
                continue
            raw_path = declared_input.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            resolved_input = _resolve_evidence_path(
                Path(raw_path),
                base_directory=manifest_directory,
            )
            if _paths_alias(resolved_output, resolved_input):
                raise EvidenceError(
                    f"output path aliases declared input {declared_input.get('name')!r}"
                )
    wad_profile = manifest.get("wad_profile")
    if isinstance(wad_profile, dict):
        providers = wad_profile.get("providers")
        if isinstance(providers, list):
            for provider in providers:
                if not isinstance(provider, dict):
                    continue
                for asset_name in ("iwad", "pwad"):
                    raw_path = provider.get(f"{asset_name}_path")
                    if not isinstance(raw_path, str) or not raw_path.strip():
                        continue
                    resolved_asset = _resolve_evidence_path(
                        Path(raw_path),
                        base_directory=manifest_directory,
                    )
                    if _paths_alias(resolved_output, resolved_asset):
                        asset_id = f"{provider.get('id')}.{asset_name}"
                        raise EvidenceError(f"output path aliases WAD profile asset {asset_id!r}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest, _payload = _load_manifest(args.manifest)
        _validate_document_paths(
            args.output,
            manifest_path=args.manifest,
            manifest=manifest,
            merge_path=args.merge,
        )
        workflow = manifest.get("workflow")
        final_report_published = False
        if workflow == "parity_readiness":
            report = build_readiness_report(args.manifest)
        elif workflow == "development_training_benchmark":
            if args.merge is not None:
                raise EvidenceError(
                    "development training benchmark continuation is not supported yet"
                )
            report = build_development_benchmark_report(args.manifest)
        elif workflow == "fixed_time_training_diagnostic":
            if args.merge is not None:
                raise EvidenceError("fixed-time diagnostic continuation is not supported yet")
            report = build_fixed_time_diagnostic_report(args.manifest)
        elif workflow == "parity_certification":

            def publish_policy_report(final_report: dict[str, Any], revalidate: Any) -> None:
                _validate_output_path(
                    args.output,
                    manifest_path=args.manifest,
                    report=final_report,
                    merge_path=args.merge,
                )
                _write_report(
                    args.output,
                    final_report,
                    revalidate=revalidate,
                    rollback_on_failure=True,
                )

            report = build_policy_evaluation_report(
                args.manifest,
                merge_path=args.merge,
                output_path=args.output,
                progress_callback=lambda progress: _write_report(args.output, progress),
                final_callback=publish_policy_report,
            )
            final_report_published = True
        else:
            raise EvidenceError(f"unsupported manifest workflow {workflow!r}")
        if not final_report_published:
            _validate_output_path(
                args.output,
                manifest_path=args.manifest,
                report=report,
                merge_path=args.merge,
            )
        if args.merge is not None and workflow == "parity_readiness":
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
                expected_wad_profile=report.get("wad_profile"),
                expected_invariant_suite=report.get("invariant_suite"),
                expected_evidence_entries=[
                    entry
                    for entry in entries
                    if isinstance(entry, dict)
                    and entry.get("name")
                    not in {
                        "manifest",
                        *(
                            item["name"]
                            for item in report["declared_inputs"]
                            if isinstance(item, dict)
                        ),
                    }
                ],
            )
        if not final_report_published:
            _write_report(args.output, report)
    except (EvidenceError, OSError) as error:
        print(f"gradoom-evidence: error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
