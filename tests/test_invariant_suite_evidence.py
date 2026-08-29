from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from gradoom.evidence import invariant_runner, invariant_suite

FIXTURES = Path(__file__).parent / "fixtures" / "evidence"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "gradoom" / "evidence" / "invariant_runner.py"
PROFILE = json.loads(
    (ROOT / "src" / "gradoom" / "evidence" / "profiles" / "freedoom2-deathmatch-v1.json").read_text(
        encoding="utf-8"
    )
)
REFERENCE_CONFIG = """\
doom_scenario_path = deathmatch.wad
screen_resolution = RES_320X240
episode_start_time = 1
mode = PLAYER
"""


def semantic_probes() -> dict[str, object]:
    return {
        behavior: {"seeds": [101, 102], "actions": [[0, 0]], "max_steps": 32}
        for behavior in (
            "termination",
            "truncation",
            "player_killcount",
            "player_killcount.enemy_on_enemy_exclusion",
        )
    }


def run_evidence(*args: str) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None
    return subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def write_manifest(tmp_path: Path, *, mismatch: str | None = None) -> Path:
    source = FIXTURES / "invariant-readiness-manifest.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    manifest["declared_inputs"][0]["path"] = str(RUNNER)
    manifest["declared_inputs"][0]["sha256"] = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    if mismatch is not None:
        manifest["invariant_suite"]["fixture_case"] = (
            "reward_mismatch" if mismatch == "rewards" else mismatch
        )
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    return target


def test_public_readiness_runs_the_versioned_invariant_suite(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path)),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    suite = report["invariant_suite"]
    assert suite["version"] == "1.0.0"
    assert suite["status"] == "passed"
    assert suite["failures"] == []
    assert {check["behavior"] for check in suite["checks"]} == {
        "constructor",
        "action_meanings",
        "observation_shapes",
        "signal_shapes",
        "rewards",
        "reset",
        "step",
        "masked_reset",
        "termination",
        "truncation",
        "episode",
        "player_killcount",
        "player_killcount.enemy_on_enemy_exclusion",
        "gradoom.tensor_inputs",
        "gradoom.tensor_outputs",
        "gradoom.device",
    }
    assert all(check["status"] == "passed" for check in suite["checks"])
    assert suite["diagnostics"]["affects_verdict"] is False
    assert {item["id"] for item in suite["diagnostics"]["tools"]} == {
        "mechanics",
        "trace",
        "distribution",
        "observation",
        "rendering",
    }
    assert report["status"] == "unavailable"
    assert report["claim_eligible"] is False
    assert any(
        reason.get("prerequisite") == "real_pretrained_policy_corpus"
        for reason in report["claim_reasons"]
    )


def test_failed_public_behavior_blocks_readiness_and_is_named(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path, mismatch="rewards")),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["claim_eligible"] is False
    assert report["invariant_suite"]["status"] == "failed"
    assert report["invariant_suite"]["failures"] == [
        {
            "behavior": "rewards",
            "message": ("env-vizdoom-turbo public behavior does not match gradoom for rewards."),
            "provider": "env-vizdoom-turbo",
        }
    ]
    assert any(
        reason.get("code") == "invariant_failure" and reason.get("behavior") == "rewards"
        for reason in report["claim_reasons"]
    )


@pytest.mark.parametrize("fixture_case", ["ignored_masked_reset", "leaky_masked_reset"])
def test_invalid_masked_reset_behavior_is_named_through_the_public_command(
    tmp_path: Path,
    fixture_case: str,
) -> None:
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path, mismatch=fixture_case)),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["invariant_suite"]["failures"] == [
        {
            "behavior": "masked_reset",
            "message": "env-vizdoom-turbo public behavior is invalid for masked_reset.",
            "provider": "env-vizdoom-turbo",
        }
    ]


def test_missing_public_signal_is_a_named_failed_invariant(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path, mismatch="missing_player_killcount")),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    suite = report["invariant_suite"]
    assert suite["status"] == "failed"
    assert suite["unavailable_reasons"] == []
    assert {failure["behavior"] for failure in suite["failures"]} >= {
        "signal_shapes",
        "player_killcount",
        "player_killcount.enemy_on_enemy_exclusion",
    }
    assert all(failure["message"] for failure in suite["failures"])


def test_unobserved_lifecycle_event_is_a_named_failed_invariant(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path, mismatch="missing_termination")),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    suite = report["invariant_suite"]
    assert suite["status"] == "failed"
    assert suite["unavailable_reasons"] == []
    assert any(
        failure["behavior"] == "termination" and "event was not observed" in failure["message"]
        for failure in suite["failures"]
    )


def test_real_runner_handles_an_absent_optional_runtime_honestly(tmp_path: Path) -> None:
    iwad = tmp_path / "freedoom2.wad"
    pwad = tmp_path / "deathmatch.wad"
    config = tmp_path / "deathmatch.cfg"
    iwad.write_bytes(b"iwad")
    pwad.write_bytes(b"pwad")
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    providers = {
        provider: {
            "iwad_path": str(iwad),
            "iwad_sha256": hashlib.sha256(iwad.read_bytes()).hexdigest(),
            "pwad_path": str(pwad),
            "pwad_sha256": hashlib.sha256(pwad.read_bytes()).hexdigest(),
            "configuration": PROFILE["configuration"],
        }
        for provider in ("gradoom", "env-vizdoom-turbo")
    }

    response = invariant_suite._execute_runner(
        runner_sha256=hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        mode="real",
        fixture_case="pass",
        gradoom_revision="revision",
        real_configuration={
            "device": "cpu",
            "reference_scenario_config_path": str(config),
            "semantic_probes": semantic_probes(),
            "wad_binding_sha256": "b" * 64,
            "providers": providers,
        },
    )

    assert response["status"] in {"unavailable", "complete"}
    if response["status"] == "unavailable":
        assert response["contracts"] == []
        assert response["unavailable_reasons"]
    else:
        assert len(response["contracts"]) == 2
        assert response["unavailable_reasons"] == []


def test_real_invariants_require_the_validated_wad_profile_binding(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture"] = False
    manifest["invariant_suite"] = {
        "version": "1.0.0",
        "mode": "real",
        "runner_input": "invariant_runner",
        "real_configuration": {
            "device": "cpu",
            "reference_scenario_config_input": "reference_scenario_config",
            "semantic_probes": semantic_probes(),
        },
    }
    config = tmp_path / "deathmatch.cfg"
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    manifest["declared_inputs"].append(
        {
            "name": "reference_scenario_config",
            "path": str(config),
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "real invariant execution requires a matched wad_profile" in result.stderr


def test_real_runner_receives_only_the_validated_wad_profile_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    iwad = tmp_path / "freedoom2.wad"
    pwad = tmp_path / "deathmatch.wad"
    config = tmp_path / "deathmatch.cfg"
    iwad.write_bytes(b"iwad")
    pwad.write_bytes(b"pwad")
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    iwad_sha256 = hashlib.sha256(iwad.read_bytes()).hexdigest()
    pwad_sha256 = hashlib.sha256(pwad.read_bytes()).hexdigest()
    providers = [
        {
            "id": provider,
            "iwad": {"path": str(iwad), "sha256": iwad_sha256},
            "pwad": {"path": str(pwad), "sha256": pwad_sha256},
            "configuration": PROFILE["configuration"],
        }
        for provider in ("gradoom", "env-vizdoom-turbo")
    ]
    wad_profile = {
        "status": "matched",
        "profile": PROFILE,
        "providers": providers,
        "binding_sha256": "b" * 64,
    }
    captured: dict[str, object] = {}

    def capture_request(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "status": "unavailable",
            "contracts": [],
            "unavailable_reasons": [{"code": "provider_unavailable", "message": "test boundary"}],
        }

    monkeypatch.setattr(invariant_suite, "_execute_runner", capture_request)
    declaration = {
        "version": "1.0.0",
        "mode": "real",
        "runner_input": "invariant_runner",
        "real_configuration": {
            "device": "cpu",
            "reference_scenario_config_input": "reference_scenario_config",
            "semantic_probes": semantic_probes(),
        },
    }
    declared_inputs = [
        {
            "name": "invariant_runner",
            "path": str(RUNNER),
            "sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        },
        {
            "name": "reference_scenario_config",
            "path": str(config),
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        },
    ]

    report = invariant_suite.run_invariant_suite(
        declaration,
        base_directory=tmp_path,
        declared_inputs=declared_inputs,
        fixture=False,
        gradoom_revision="revision",
        wad_profile=wad_profile,
    )

    assert report["status"] == "unavailable"
    assert captured["real_configuration"] == {
        "device": "cpu",
        "reference_scenario_config_path": str(config),
        "semantic_probes": semantic_probes(),
        "wad_binding_sha256": "b" * 64,
        "providers": {
            provider["id"]: {
                "iwad_path": str(iwad),
                "iwad_sha256": iwad_sha256,
                "pwad_path": str(pwad),
                "pwad_sha256": pwad_sha256,
                "configuration": PROFILE["configuration"],
            }
            for provider in providers
        },
    }


def test_gradoom_probe_inputs_are_allocated_on_the_requested_device() -> None:
    requested = torch.device("meta")

    actions = invariant_runner._actions("gradoom", [0, 1], requested)
    mask = invariant_runner._mask("gradoom", [True, False], requested)

    assert actions.device == requested
    assert actions.dtype == torch.int64
    assert mask.device == requested
    assert mask.dtype == torch.bool


def test_semantic_probe_forwards_published_action_indices_without_reinterpretation() -> None:
    class ProgrammedPublicEnv:
        def __init__(self) -> None:
            self.actions: list[list[int]] = []

        def reset(
            self,
            *,
            seed: object = None,
            options: Mapping[str, object] | None = None,
        ) -> tuple[object, dict[str, object]]:
            del seed, options
            signals = {
                "player_killcount": [0.0, 0.0],
                "killcount": [0.0, 0.0],
            }
            return object(), signals

        def step(self, actions: object) -> tuple[object, object, object, object, object]:
            row = list(actions)
            self.actions.append(row)
            terminal = len(self.actions) == 2
            return (
                object(),
                [0.0, 0.0],
                [terminal, False],
                [False, False],
                {"player_killcount": [0.0, 0.0], "killcount": [0.0, 0.0]},
            )

        def close(self) -> None:
            pass

    env = ProgrammedPublicEnv()

    result = invariant_runner._semantic_probe(
        behavior="termination",
        provider="env-vizdoom-turbo",
        factory=lambda: env,
        probe={"seeds": [1, 2], "actions": [[7, 8]], "max_steps": 2},
        requested_device=None,
        kill_signal_reader=None,
    )

    assert result == {"reported_separately": True, "requires_reset": False}
    assert env.actions == [[7, 8], [7, 8], [7, 8]]


def test_reference_config_cannot_substitute_an_unbound_pwad(tmp_path: Path) -> None:
    iwad = tmp_path / "freedoom2.wad"
    bound_pwad = tmp_path / "deathmatch.wad"
    config_directory = tmp_path / "reference"
    config_directory.mkdir()
    config = config_directory / "deathmatch.cfg"
    substituted_pwad = config_directory / "deathmatch.wad"
    iwad.write_bytes(b"iwad")
    bound_pwad.write_bytes(b"bound pwad")
    substituted_pwad.write_bytes(b"different pwad")
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    providers = [
        {
            "id": provider,
            "iwad": {
                "path": str(iwad),
                "sha256": hashlib.sha256(iwad.read_bytes()).hexdigest(),
            },
            "pwad": {
                "path": str(bound_pwad),
                "sha256": hashlib.sha256(bound_pwad.read_bytes()).hexdigest(),
            },
            "configuration": PROFILE["configuration"],
        }
        for provider in ("gradoom", "env-vizdoom-turbo")
    ]

    with pytest.raises(
        invariant_suite.InvariantSuiteError,
        match="reference scenario configuration must consume the validated reference PWAD",
    ):
        invariant_suite.run_invariant_suite(
            {
                "version": "1.0.0",
                "mode": "real",
                "runner_input": "invariant_runner",
                "real_configuration": {
                    "device": "cpu",
                    "reference_scenario_config_input": "reference_scenario_config",
                    "semantic_probes": semantic_probes(),
                },
            },
            base_directory=tmp_path,
            declared_inputs=[
                {
                    "name": "invariant_runner",
                    "path": str(RUNNER),
                    "sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                },
                {
                    "name": "reference_scenario_config",
                    "path": str(config),
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                },
            ],
            fixture=False,
            gradoom_revision="revision",
            wad_profile={
                "status": "matched",
                "providers": providers,
                "binding_sha256": "b" * 64,
            },
        )


def test_merge_rejects_a_tampered_invariant_verdict(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    report_path = tmp_path / "report.json"
    initial = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(report_path),
    )
    assert initial.returncode == 0, initial.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["invariant_suite"]["status"] = "failed"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "merged.json"),
        "--merge",
        str(report_path),
    )

    assert result.returncode == 2
    assert "invariant_suite does not match the current invariant execution" in result.stderr


def test_fixture_runner_cannot_support_a_non_fixture_claim(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fixture"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert "invariant_suite.mode must match the manifest fixture state" in result.stderr
    assert not output.exists()


def test_provider_runner_hash_is_verified_before_execution(tmp_path: Path) -> None:
    sentinel = tmp_path / "executed"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"][0]["path"] = str(runner)
    manifest["declared_inputs"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "declared input 'invariant_runner' SHA-256 mismatch" in result.stderr
    assert not sentinel.exists()


def test_arbitrary_provider_commands_are_rejected_even_when_their_file_is_hashed(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["invariant_suite"]["providers"] = [
        {
            "id": "gradoom",
            "available": True,
            "runner_input": "invariant_runner",
            "command": ["python", str(RUNNER)],
        }
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 2
    assert "invariant_suite.providers is not supported" in result.stderr
    assert not output.exists()


def test_unrelated_hashed_executable_cannot_replace_the_trusted_runner(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("raise SystemExit(0)\n", encoding="utf-8")
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"].append(
        {
            "name": "unrelated_runner",
            "path": str(unrelated),
            "sha256": hashlib.sha256(unrelated.read_bytes()).hexdigest(),
        }
    )
    manifest["invariant_suite"]["runner_input"] = "unrelated_runner"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "must bind the repository-owned invariant runner" in result.stderr


def test_static_contract_emitter_cannot_impersonate_the_trusted_runner(tmp_path: Path) -> None:
    emitter = tmp_path / "static-emitter.py"
    emitter.write_text(
        "import json\n"
        "print(json.dumps({'protocol_version': 1, 'status': 'complete', "
        "'contracts': [], 'unavailable_reasons': []}))\n",
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"][0]["path"] = str(emitter)
    manifest["declared_inputs"][0]["sha256"] = hashlib.sha256(emitter.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "must bind the repository-owned invariant runner" in result.stderr


def test_minimal_bogus_contract_cannot_impersonate_complete_evidence(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus-contract.py"
    bogus.write_text(
        'print(\'{"schema_version":1,"provider":"gradoom","behaviors":{}}\')\n',
        encoding="utf-8",
    )
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["declared_inputs"][0].update(
        {
            "path": str(bogus),
            "sha256": hashlib.sha256(bogus.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(tmp_path / "report.json"),
    )

    assert result.returncode == 2
    assert "must bind the repository-owned invariant runner" in result.stderr


def test_authenticated_runner_contract_still_rejects_a_minimal_bogus_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    minimal_contracts = [
        {
            "schema_version": 1,
            "provider": provider,
            "revision": "bogus",
            "behaviors": {},
            **({"tensor_device": {}} if provider == "gradoom" else {}),
        }
        for provider in ("gradoom", "env-vizdoom-turbo")
    ]
    monkeypatch.setattr(
        invariant_suite,
        "_execute_runner",
        lambda **_kwargs: {
            "protocol_version": 1,
            "challenge": "ignored-after-authentication",
            "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            "status": "complete",
            "contracts": minimal_contracts,
            "unavailable_reasons": [],
        },
    )

    with pytest.raises(
        invariant_suite.InvariantSuiteError,
        match="gradoom invariant contract behavior set is incomplete",
    ):
        invariant_suite.run_invariant_suite(
            {
                "version": "1.0.0",
                "mode": "fixture",
                "runner_input": "invariant_runner",
            },
            base_directory=ROOT,
            declared_inputs=[
                {
                    "name": "invariant_runner",
                    "path": str(RUNNER),
                    "sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
                }
            ],
            fixture=True,
            gradoom_revision="fixture-revision",
        )


def test_provider_runtime_contract_failure_is_a_named_failed_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        invariant_suite,
        "_execute_runner",
        lambda **_kwargs: {
            "status": "unavailable",
            "contracts": [],
            "unavailable_reasons": [
                {
                    "code": "provider_contract_failure",
                    "provider": "env-vizdoom-turbo",
                    "behavior": "env-vizdoom-turbo.revision",
                    "message": "installed provider revision does not match the immutable pin",
                }
            ],
        },
    )

    report = invariant_suite.run_invariant_suite(
        {
            "version": "1.0.0",
            "mode": "fixture",
            "runner_input": "invariant_runner",
        },
        base_directory=ROOT,
        declared_inputs=[
            {
                "name": "invariant_runner",
                "path": str(RUNNER),
                "sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
            }
        ],
        fixture=True,
        gradoom_revision="fixture-revision",
    )

    assert report["status"] == "failed"
    assert report["unavailable_reasons"] == []
    assert report["failures"] == [
        {
            "behavior": "env-vizdoom-turbo.revision",
            "provider": "env-vizdoom-turbo",
            "message": "installed provider revision does not match the immutable pin",
        }
    ]


def test_omitted_invariant_suite_blocks_readiness(tmp_path: Path) -> None:
    manifest = json.loads((FIXTURES / "readiness-manifest.json").read_text(encoding="utf-8"))
    manifest["declared_inputs"][0]["path"] = str(FIXTURES / "provider-contract.json")
    manifest["prerequisites"] = [
        {"id": "pinned_reference_provider", "available": True},
        {"id": "real_pretrained_policy_corpus", "available": True},
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "unavailable"
    assert report["invariant_suite"]["status"] == "unavailable"
    assert any(
        reason["code"] == "invariant_suite_unavailable" for reason in report["claim_reasons"]
    )


def test_invariant_pass_without_a_declared_real_corpus_is_unavailable(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prerequisites"] = [
        {"id": "pinned_reference_provider", "available": True},
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(manifest_path),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["invariant_suite"]["status"] == "passed"
    assert report["status"] == "unavailable"
    assert report["claim_eligible"] is False
    assert any(
        reason.get("code") == "missing_required_prerequisite"
        and reason.get("prerequisite") == "real_pretrained_policy_corpus"
        for reason in report["claim_reasons"]
    )
