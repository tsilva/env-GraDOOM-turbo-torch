from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gradoom.evidence import policy_corpus

RUNNER = Path(__file__).parent / "fixtures" / "evidence" / "fixture_policy_runner.py"
INCOMPLETE_RUNNER = RUNNER.with_name("fixture_incomplete_policy_runner.py")
INVALID_TYPE_RUNNER = RUNNER.with_name("fixture_invalid_type_policy_runner.py")
FLOAT_PROTOCOL_RUNNER = RUNNER.with_name("fixture_float_protocol_policy_runner.py")
NEAR_LIMIT_MALFORMED_RUNNER = RUNNER.with_name("fixture_near_limit_malformed_policy_runner.py")
SURROGATE_FAILURE_RUNNER = RUNNER.with_name("fixture_surrogate_failure_policy_runner.py")
MUTATING_RUNNER = RUNNER.with_name("fixture_mutating_policy_runner.py")
TRANSIENT_RUNNER = RUNNER.with_name("fixture_transient_restore_runner.py")
INTERRUPT_RUNNER = RUNNER.with_name("fixture_interrupt_once_runner.py")
NOISY_RUNNER = RUNNER.with_name("fixture_noisy_policy_runner.py")
OVERSIZED_NUMERIC_RUNNER = RUNNER.with_name("fixture_oversized_numeric_policy_runner.py")
LIMITED_FILE_SIZE_HARNESS = """
import json
import resource
import sys
from pathlib import Path

from gradoom.evidence.policy_corpus import build_policy_evaluation_report

limit = int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
report = build_policy_evaluation_report(Path(sys.argv[1]))
outcomes = report["policy_evaluation"]["outcomes"]
print(json.dumps({
    "status": report["status"],
    "outcome_count": len(outcomes),
    "failure_count": report["policy_evaluation"]["failure_count"],
    "failure_codes": sorted({item["execution_failure"]["code"] for item in outcomes}),
}))
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _documents(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    artifacts = []
    policies = []
    for policy_id, provider in (
        ("gradoom-policy", "gradoom"),
        ("reference-policy", "env-vizdoom-turbo"),
    ):
        artifact = tmp_path / f"{policy_id}.bin"
        artifact.write_bytes(f"frozen {policy_id}\n".encode())
        artifacts.append(artifact)
        policies.append(
            {
                "id": policy_id,
                "training_provider": provider,
                "artifact_path": artifact.name,
                "artifact_sha256": _sha256(artifact),
                "model_runtime_contract": {
                    "contract_version": 1,
                    "artifact_format": "standalone-gradoom-ppo-v1",
                    "runtime": "torch",
                    "architecture": "nature",
                },
                "stochastic_actions": True,
                "adapted": False,
                "provider_specific_modifications": [],
            }
        )
    corpus_path = tmp_path / "corpus.json"
    _write_json(
        corpus_path,
        {
            "schema_version": 1,
            "corpus_version": "fixture-corpus-v1",
            "sealed": True,
            "shared_preprocessing_identity": (
                "6ff033ce02585302f78e84c16f6a86da99690e0d861092cab41e55b4257e08d0"
            ),
            "policies": policies,
        },
    )
    seeds_path = tmp_path / "seeds.json"
    _write_json(
        seeds_path,
        {"schema_version": 1, "seed_set_id": "fixture-256-v1", "seeds": list(range(256))},
    )
    manifest = {
        "schema_version": 1,
        "workflow": "parity_certification",
        "evidence_level": "formal",
        "fixture": True,
        "code_provenance": {
            "repository": "tsilva/env-GraDOOM-turbo-torch",
            "revision": "fixture-revision",
            "dirty": False,
        },
        "declared_inputs": [
            {"name": "policy_corpus", "path": corpus_path.name, "sha256": _sha256(corpus_path)},
            {"name": "episode_seeds", "path": seeds_path.name, "sha256": _sha256(seeds_path)},
            {"name": "policy_runner", "path": str(RUNNER), "sha256": _sha256(RUNNER)},
        ],
        "policy_evaluation": {
            "protocol_version": 2,
            "corpus_input": "policy_corpus",
            "seed_manifest_input": "episode_seeds",
            "runner_input": "policy_runner",
            "providers": [
                {"id": "gradoom", "revision": "fixture-gradoom"},
                {"id": "env-vizdoom-turbo", "revision": "fixture-reference"},
            ],
        },
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, corpus_path, manifest


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None
    return subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )


def _replace_runner(manifest_path: Path, manifest: dict[str, object], runner: Path) -> None:
    runner_input = manifest["declared_inputs"][2]  # type: ignore[index]
    runner_input["path"] = str(runner)
    runner_input["sha256"] = _sha256(runner)
    _write_json(manifest_path, manifest)


def _rehash_policy_report(report: dict[str, object]) -> None:
    evaluation = report["policy_evaluation"]
    evidence_index = report["evidence_index"]
    entry = next(item for item in evidence_index["entries"] if item["name"] == "policy_evaluation")
    entry["sha256"] = _canonical_sha256(evaluation)
    evidence_index["sha256"] = _canonical_sha256(evidence_index["entries"])


def test_public_command_executes_complete_sealed_fixture_corpus(tmp_path: Path) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    output = tmp_path / "report.json"

    result = _run("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["workflow"] == "parity_certification"
    assert report["status"] == "evaluation_complete"
    assert report["claim_eligible"] is False
    assert {reason["code"] for reason in report["claim_reasons"]} == {
        "fixture_evidence",
        "parity_verdict_pending",
    }
    evaluation = report["policy_evaluation"]
    assert evaluation["corpus"]["corpus_version"] == "fixture-corpus-v1"
    assert evaluation["corpus"]["sealed"] is True
    assert len(evaluation["outcomes"]) == 2 * 2 * 256
    expected_grid = list(range(256))
    for provider in ("gradoom", "env-vizdoom-turbo"):
        for policy in ("gradoom-policy", "reference-policy"):
            outcomes = [
                item
                for item in evaluation["outcomes"]
                if item["provider_id"] == provider and item["policy_id"] == policy
            ]
            assert [item["seed"] for item in outcomes] == expected_grid
            assert [item["seed_index"] for item in outcomes] == expected_grid
            assert all(item["execution_failure"] is None for item in outcomes)
            assert all(item["termination_state"] == "terminated" for item in outcomes)
            assert all(item["episode_length"] >= 100 for item in outcomes)
            assert all(item["unit_identity"] for item in outcomes)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda corpus: corpus["policies"].append(dict(corpus["policies"][0])), "duplicated"),
        (lambda corpus: corpus["policies"].pop(), "each training provider"),
        (lambda corpus: corpus["policies"][0].update(stochastic_actions=False), "stochastic"),
        (lambda corpus: corpus["policies"][0].update(adapted=True), "adapted"),
        (
            lambda corpus: corpus["policies"][0]["model_runtime_contract"].update(
                contract_version=2
            ),
            "unsupported",
        ),
        (
            lambda corpus: corpus["policies"][0]["model_runtime_contract"].update(
                architecture="definitely-unsupported"
            ),
            "unsupported",
        ),
        (
            lambda corpus: corpus["policies"][0]["model_runtime_contract"].update(
                provider_overrides={"gradoom": "compensated"}
            ),
            "invalid fields",
        ),
        (
            lambda corpus: corpus["policies"][0].update(
                provider_arguments={"gradoom": ["--compensate"]}
            ),
            "invalid fields",
        ),
        (lambda corpus: corpus.update(undeclared_corpus_field=True), "invalid fields"),
        (lambda corpus: corpus.update(sealed=False), "sealed"),
    ],
)
def test_invalid_corpus_is_rejected_before_execution(
    tmp_path: Path, mutate: object, message: str
) -> None:
    manifest_path, corpus_path, manifest = _documents(tmp_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    mutate(corpus)  # type: ignore[operator]
    _write_json(corpus_path, corpus)
    manifest["declared_inputs"][0]["sha256"] = _sha256(corpus_path)  # type: ignore[index]
    _write_json(manifest_path, manifest)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert message in result.stderr


def test_byte_identical_policy_artifacts_are_rejected_before_execution(
    tmp_path: Path,
) -> None:
    manifest_path, corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, INTERRUPT_RUNNER)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    first_artifact = tmp_path / corpus["policies"][0]["artifact_path"]
    second_artifact = tmp_path / corpus["policies"][1]["artifact_path"]
    second_artifact.write_bytes(first_artifact.read_bytes())
    corpus["policies"][1]["artifact_sha256"] = _sha256(second_artifact)
    _write_json(corpus_path, corpus)
    manifest["declared_inputs"][0]["sha256"] = _sha256(corpus_path)  # type: ignore[index]
    _write_json(manifest_path, manifest)
    output = tmp_path / "report.json"
    invocation_log = tmp_path / "invocations.log"

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
        env={
            "GRADOOM_INTERRUPT_MARKER": str(tmp_path / "interrupt.marker"),
            "GRADOOM_INVOCATION_LOG": str(invocation_log),
        },
    )

    assert result.returncode == 2
    assert "artifact_sha256 is duplicated" in result.stderr
    assert not output.exists()
    assert not invocation_log.exists()


@pytest.mark.parametrize("contract_version", [True, 1.0])
def test_model_contract_version_requires_an_exact_json_integer(
    tmp_path: Path, contract_version: object
) -> None:
    manifest_path, corpus_path, manifest = _documents(tmp_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["policies"][0]["model_runtime_contract"]["contract_version"] = contract_version
    _write_json(corpus_path, corpus)
    manifest["declared_inputs"][0]["sha256"] = _sha256(corpus_path)  # type: ignore[index]
    _write_json(manifest_path, manifest)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "unsupported model/runtime contract" in result.stderr


@pytest.mark.parametrize("architecture", [[], {}])
def test_model_contract_architecture_requires_a_json_string_without_a_traceback(
    tmp_path: Path, architecture: object
) -> None:
    manifest_path, corpus_path, manifest = _documents(tmp_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["policies"][0]["model_runtime_contract"]["architecture"] = architecture
    _write_json(corpus_path, corpus)
    manifest["declared_inputs"][0]["sha256"] = _sha256(corpus_path)  # type: ignore[index]
    _write_json(manifest_path, manifest)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "unsupported model/runtime contract" in result.stderr
    assert "Traceback" not in result.stderr


def test_policy_evaluation_protocol_requires_an_exact_json_integer(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    manifest["policy_evaluation"]["protocol_version"] = 2.0  # type: ignore[index]
    _write_json(manifest_path, manifest)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "unsupported protocol_version" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.update(undeclared_manifest_field=True),
        lambda manifest: manifest["code_provenance"].update(build_id="undeclared"),
        lambda manifest: manifest["declared_inputs"][0].update(role="undeclared"),
        lambda manifest: manifest["policy_evaluation"].update(
            provider_overrides={"gradoom": "compensated"}
        ),
        lambda manifest: manifest["policy_evaluation"]["providers"][0].update(
            arguments=["--compensate"]
        ),
    ],
)
def test_undeclared_manifest_fields_are_rejected(tmp_path: Path, mutation: object) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    mutation(manifest)  # type: ignore[operator]
    _write_json(manifest_path, manifest)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "invalid fields" in result.stderr


def test_undeclared_seed_manifest_fields_are_rejected(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    seeds_path = tmp_path / "seeds.json"
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    seeds["provider_seed_overrides"] = {"gradoom": [1]}
    _write_json(seeds_path, seeds)
    manifest["declared_inputs"][1]["sha256"] = _sha256(seeds_path)  # type: ignore[index]
    _write_json(manifest_path, manifest)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "invalid fields" in result.stderr


def test_changed_or_missing_artifact_is_rejected(tmp_path: Path) -> None:
    manifest_path, corpus_path, _manifest = _documents(tmp_path)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    (tmp_path / corpus["policies"][0]["artifact_path"]).write_bytes(b"mutated")

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))
    assert result.returncode == 2
    assert "artifact SHA-256 mismatch" in result.stderr

    (tmp_path / corpus["policies"][0]["artifact_path"]).unlink()
    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))
    assert result.returncode == 2
    assert "artifact is unavailable" in result.stderr


def test_execution_failure_is_retained_and_cannot_be_replaced_on_merge(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    manifest["policy_evaluation"]["fixture_failure_seed"] = 17  # type: ignore[index]
    _write_json(manifest_path, manifest)
    initial = tmp_path / "initial.json"
    result = _run("--manifest", str(manifest_path), "--output", str(initial))
    assert result.returncode == 0, result.stderr
    report = json.loads(initial.read_text(encoding="utf-8"))
    failed = [item for item in report["policy_evaluation"]["outcomes"] if item["seed"] == 17]
    assert len(failed) == 4
    assert all(item["execution_failure"]["code"] == "fixture_failure" for item in failed)

    resumed = tmp_path / "resumed.json"
    result = _run(
        "--manifest", str(manifest_path), "--output", str(resumed), "--merge", str(initial)
    )
    assert result.returncode == 0, result.stderr
    assert (
        json.loads(resumed.read_text(encoding="utf-8"))["policy_evaluation"]
        == report["policy_evaluation"]
    )


def test_omitted_runner_outcomes_are_retained_as_failures(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, INCOMPLETE_RUNNER)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    outcomes = report["policy_evaluation"]["outcomes"]
    assert len(outcomes) == 2 * 2 * 256
    omitted = [item for item in outcomes if item["seed_index"] == 255]
    assert len(omitted) == 4
    assert all(item["execution_failure"]["code"] == "missing_runner_outcome" for item in omitted)


def test_runner_output_is_bounded_and_retained_as_failures(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, NOISY_RUNNER)
    output = tmp_path / "report.json"

    result = _run("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy_evaluation"]["failure_count"] == 2 * 2 * 256
    assert {
        item["execution_failure"]["code"] for item in report["policy_evaluation"]["outcomes"]
    } == {"runner_output_limit"}
    assert output.stat().st_size < 2 * 1024 * 1024


def test_oversized_runner_numbers_are_retained_as_failures(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, OVERSIZED_NUMERIC_RUNNER)
    output = tmp_path / "report.json"

    result = _run("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy_evaluation"]["failure_count"] == 2 * 2 * 256
    assert {
        item["execution_failure"]["code"] for item in report["policy_evaluation"]["outcomes"]
    } == {"invalid_runner_response"}


def test_unhashable_runner_types_are_retained_as_failures(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, INVALID_TYPE_RUNNER)
    output = tmp_path / "report.json"

    result = _run("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy_evaluation"]["failure_count"] == 2 * 2 * 256
    assert {
        item["execution_failure"]["code"] for item in report["policy_evaluation"]["outcomes"]
    } == {"invalid_runner_response"}


def test_runner_protocol_requires_an_exact_json_integer(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, FLOAT_PROTOCOL_RUNNER)
    output = tmp_path / "report.json"

    result = _run("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy_evaluation"]["failure_count"] == 2 * 2 * 256
    assert {
        item["execution_failure"]["code"] for item in report["policy_evaluation"]["outcomes"]
    } == {"invalid_runner_response"}


def test_near_capture_limit_malformed_response_has_bounded_failure_evidence(
    tmp_path: Path,
) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, NEAR_LIMIT_MALFORMED_RUNNER)
    output = tmp_path / "report.json"

    result = _run("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    failures = [item["execution_failure"] for item in report["policy_evaluation"]["outcomes"]]
    assert len(failures) == 2 * 2 * 256
    assert {failure["code"] for failure in failures} == {"invalid_runner_response"}
    assert max(len(failure["message"].encode()) for failure in failures) <= 4096
    assert output.stat().st_size < 2 * 1024 * 1024


def test_lone_surrogate_runner_failure_is_retained_as_bounded_failures(
    tmp_path: Path,
) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, SURROGATE_FAILURE_RUNNER)
    output = tmp_path / "report.json"

    result = _run("--manifest", str(manifest_path), "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    failures = [item["execution_failure"] for item in report["policy_evaluation"]["outcomes"]]
    assert len(failures) == 2 * 2 * 256
    assert {failure["code"] for failure in failures} == {"invalid_runner_response"}
    assert max(len(failure["message"].encode()) for failure in failures) <= 4096
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("timeout_seconds", [10**400, 2**63 - 1])
def test_oversized_manifest_timeout_fails_without_a_traceback(
    tmp_path: Path, timeout_seconds: int
) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    manifest["policy_evaluation"]["timeout_seconds"] = timeout_seconds  # type: ignore[index]
    _write_json(manifest_path, manifest)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "supported range" in result.stderr or "positive finite number" in result.stderr
    assert "Traceback" not in result.stderr


def test_runner_respects_a_stricter_inherited_file_size_hard_limit(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, FLOAT_PROTOCOL_RUNNER)

    result = subprocess.run(
        [sys.executable, "-c", LIMITED_FILE_SIZE_HARNESS, str(manifest_path), "1024"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "status": "evaluation_complete",
        "outcome_count": 2 * 2 * 256,
        "failure_count": 2 * 2 * 256,
        "failure_codes": ["invalid_runner_response"],
    }
    assert "Traceback" not in result.stderr


def test_runner_setup_failure_is_retained_for_every_required_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)

    def fail_runner_setup() -> None:
        raise RuntimeError("forced runner setup failure")

    monkeypatch.setattr(policy_corpus, "_limit_runner_output_files", fail_runner_setup)

    report = policy_corpus.build_policy_evaluation_report(manifest_path)

    outcomes = report["policy_evaluation"]["outcomes"]
    assert report["status"] == "evaluation_complete"
    assert len(outcomes) == 2 * 2 * 256
    assert report["policy_evaluation"]["failure_count"] == 2 * 2 * 256
    assert {item["execution_failure"]["code"] for item in outcomes} == {"runner_failure"}


def test_execution_copy_rejects_policy_mutation(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, MUTATING_RUNNER)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["policy_evaluation"]["failure_count"] == 2 * 2 * 256
    assert {
        item["execution_failure"]["code"] for item in report["policy_evaluation"]["outcomes"]
    } == {"runner_process_failure"}


def test_transient_changed_and_restored_paths_cannot_change_execution_bytes(
    tmp_path: Path,
) -> None:
    manifest_path, corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, TRANSIENT_RUNNER)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    targets = [
        str(corpus_path),
        str(tmp_path / "seeds.json"),
        str(TRANSIENT_RUNNER),
        *(str(tmp_path / policy["artifact_path"]) for policy in corpus["policies"]),
    ]
    before = {path: Path(path).read_bytes() for path in targets}

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
        env={"GRADOOM_TRANSIENT_MUTATION_TARGETS": json.dumps(targets)},
    )

    assert result.returncode == 0, result.stderr
    assert {path: Path(path).read_bytes() for path in targets} == before
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["policy_evaluation"]["failure_count"] == 0
    assert {item["player_killcount"] for item in report["policy_evaluation"]["outcomes"]} == {
        0,
        1,
        2,
        3,
        4,
    }


def test_merge_rejects_mismatched_corpus_identity(tmp_path: Path) -> None:
    manifest_path, corpus_path, manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus["corpus_version"] = "replacement-v2"
    _write_json(corpus_path, corpus)
    manifest["declared_inputs"][0]["sha256"] = _sha256(corpus_path)  # type: ignore[index]
    _write_json(manifest_path, manifest)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert "cannot merge unlike policy evaluation identities" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(fixture=False),
        lambda report: report.update(evidence_level="development"),
        lambda report: report["code_provenance"].update(revision="tampered"),
        lambda report: report.update(declared_inputs=[]),
        lambda report: report.update(claim_eligible=True),
        lambda report: report.update(undeclared_top_level_field=True),
    ],
)
def test_merge_rejects_tampered_top_level_report(tmp_path: Path, mutation: object) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    mutation(report)  # type: ignore[operator]
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert "merge report" in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.update(schema_version=True),
        lambda report: report.update(fixture=1),
        lambda report: report.update(claim_eligible=0),
        lambda report: report["code_provenance"].update(dirty=0),
        lambda report: report["policy_evaluation"].update(protocol_version=2.0),
        lambda report: report["policy_evaluation"].update(expected_outcome_count=1024.0),
        lambda report: report["policy_evaluation"].update(failure_count=0.0),
        lambda report: report["policy_evaluation"]["corpus"].update(schema_version=True),
        lambda report: report["policy_evaluation"]["corpus"]["policies"][0][
            "model_runtime_contract"
        ].update(contract_version=True),
        lambda report: report["policy_evaluation"]["seed_manifest"].update(schema_version=True),
    ],
)
def test_merge_rejects_self_consistently_rehashed_json_type_substitutions(
    tmp_path: Path, mutation: object
) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    mutation(report)  # type: ignore[operator]
    _rehash_policy_report(report)
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert "merge report" in result.stderr


@pytest.mark.parametrize("status", [[], {}])
def test_merge_status_requires_a_json_string_without_a_traceback(
    tmp_path: Path, status: object
) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    report["status"] = status
    _rehash_policy_report(report)
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert "invalid policy evaluation status" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("message", "rehash", "diagnostic"),
    [
        ("tampered\0failure", True, "contains U+0000"),
        ("tampered\ud800failure", False, "contains invalid Unicode"),
    ],
)
def test_merge_rejects_failure_messages_with_disallowed_string_content(
    tmp_path: Path, message: str, rehash: bool, diagnostic: str
) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    manifest["policy_evaluation"]["fixture_failure_seed"] = 17  # type: ignore[index]
    _write_json(manifest_path, manifest)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    failed = next(
        item
        for item in report["policy_evaluation"]["outcomes"]
        if item["execution_failure"] is not None
    )
    failed["execution_failure"]["message"] = message
    if rehash:
        _rehash_policy_report(report)
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert "Traceback" not in result.stderr


def test_merge_rejects_tampered_policy_evidence(tmp_path: Path) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    report["policy_evaluation"]["outcomes"][0]["player_killcount"] = 999
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert "policy_evaluation SHA-256 mismatch" in result.stderr


def test_merge_rejects_unhashable_unit_identity_without_a_traceback(tmp_path: Path) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    report["policy_evaluation"]["outcomes"][0]["unit_identity"] = []
    _rehash_policy_report(report)
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert "unit_identity must be a lowercase SHA-256 digest" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        lambda outcome: outcome.update(player_killcount=-1),
        lambda outcome: outcome.update(player_killcount=float("inf")),
        lambda outcome: outcome.update(player_killcount=1e308),
        lambda outcome: outcome.update(player_killcount=10**400),
        lambda outcome: outcome.update(episode_length=-1),
        lambda outcome: outcome.update(termination_state=""),
        lambda outcome: outcome.update(termination_state=[]),
        lambda outcome: outcome.update(termination_state="finished"),
        lambda outcome: outcome.pop("termination_state"),
        lambda outcome: outcome.update(extra_evidence="undeclared"),
        lambda outcome: outcome.update(
            player_killcount=None,
            termination_state=None,
            episode_length=None,
            execution_failure={"code": "missing-message"},
        ),
        lambda outcome: outcome.update(
            player_killcount=None,
            termination_state=None,
            episode_length=None,
            execution_failure={"code": "", "message": "bad"},
        ),
    ],
)
def test_merge_reapplies_canonical_outcome_validation(tmp_path: Path, mutation: object) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    mutation(report["policy_evaluation"]["outcomes"][0])  # type: ignore[operator]
    _rehash_policy_report(report)
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert (
        "merge report contains an invalid policy outcome" in result.stderr
        or "merge report is not valid JSON" in result.stderr
    )


def test_interruption_durably_retains_and_resumes_completed_prefix(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, INTERRUPT_RUNNER)
    partial = tmp_path / "partial.json"
    marker = tmp_path / "interrupt.marker"
    invocation_log = tmp_path / "invocations.log"
    env = {
        "GRADOOM_INTERRUPT_MARKER": str(marker),
        "GRADOOM_INVOCATION_LOG": str(invocation_log),
    }

    interrupted = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(partial),
        env=env,
    )

    assert interrupted.returncode < 0
    progress = json.loads(partial.read_text(encoding="utf-8"))
    assert progress["status"] == "evaluation_in_progress"
    assert len(progress["policy_evaluation"]["outcomes"]) == 256
    assert progress["policy_evaluation"]["expected_outcome_count"] == 2 * 2 * 256
    assert progress["policy_evaluation"]["failure_count"] == 1
    retained_failure = next(
        outcome
        for outcome in progress["policy_evaluation"]["outcomes"]
        if outcome["execution_failure"] is not None
    )
    assert not list(tmp_path.glob(f".{partial.name}.*.tmp"))

    completed = tmp_path / "completed.json"
    resumed = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(completed),
        "--merge",
        str(partial),
        env=env,
    )

    assert resumed.returncode == 0, resumed.stderr
    report = json.loads(completed.read_text(encoding="utf-8"))
    assert report["status"] == "evaluation_complete"
    assert len(report["policy_evaluation"]["outcomes"]) == 2 * 2 * 256
    assert report["policy_evaluation"]["failure_count"] == 4
    assert retained_failure in report["policy_evaluation"]["outcomes"]
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert invocations.count("gradoom:gradoom-policy") == 1
    assert invocations.count("gradoom:reference-policy") == 2
    assert len(invocations) == 5


def test_merge_rejects_self_consistent_nonprefix_progress(tmp_path: Path) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    initial = tmp_path / "initial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(initial)).returncode == 0
    report = json.loads(initial.read_text(encoding="utf-8"))
    report["status"] = "evaluation_in_progress"
    del report["policy_evaluation"]["outcomes"][0]
    report["policy_evaluation"]["failure_count"] = 0
    _rehash_policy_report(report)
    _write_json(initial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(initial),
    )

    assert result.returncode == 2
    assert "completed policy outcomes must be an exact leading prefix" in result.stderr


def test_merge_rejects_progress_inside_a_provider_policy_batch(tmp_path: Path) -> None:
    manifest_path, _corpus_path, _manifest = _documents(tmp_path)
    partial = tmp_path / "partial.json"
    assert _run("--manifest", str(manifest_path), "--output", str(partial)).returncode == 0
    report = json.loads(partial.read_text(encoding="utf-8"))
    report["status"] = "evaluation_in_progress"
    report["policy_evaluation"]["outcomes"] = report["policy_evaluation"]["outcomes"][:1]
    report["policy_evaluation"]["failure_count"] = 0
    _rehash_policy_report(report)
    _write_json(partial, report)

    result = _run(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "resumed.json"),
        "--merge",
        str(partial),
    )

    assert result.returncode == 2
    assert "complete provider-policy batches" in result.stderr
