from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent / "fixtures" / "evidence" / "fixture_policy_runner.py"
INCOMPLETE_RUNNER = RUNNER.with_name("fixture_incomplete_policy_runner.py")
MUTATING_RUNNER = RUNNER.with_name("fixture_mutating_policy_runner.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            "protocol_version": 1,
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


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None
    return subprocess.run([command, *args], check=False, capture_output=True, text=True)


def _replace_runner(manifest_path: Path, manifest: dict[str, object], runner: Path) -> None:
    runner_input = manifest["declared_inputs"][2]  # type: ignore[index]
    runner_input["path"] = str(runner)
    runner_input["sha256"] = _sha256(runner)
    _write_json(manifest_path, manifest)


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


def test_post_seal_artifact_mutation_is_rejected(tmp_path: Path) -> None:
    manifest_path, _corpus_path, manifest = _documents(tmp_path)
    _replace_runner(manifest_path, manifest, MUTATING_RUNNER)

    result = _run("--manifest", str(manifest_path), "--output", str(tmp_path / "report.json"))

    assert result.returncode == 2
    assert "artifact changed after the corpus was sealed" in result.stderr
    assert not (tmp_path / "report.json").exists()


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
