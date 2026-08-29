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
doom_skill = 1
screen_resolution = RES_320X240
render_hud = false
render_screen_flashes = false
episode_start_time = 1
episode_timeout = 4200
available_buttons =
    {
        ATTACK SPEED STRAFE MOVE_RIGHT MOVE_LEFT MOVE_BACKWARD MOVE_FORWARD
        TURN_RIGHT TURN_LEFT SELECT_WEAPON1 SELECT_WEAPON2 SELECT_WEAPON3
        SELECT_WEAPON4 SELECT_WEAPON5 SELECT_WEAPON6 SELECT_NEXT_WEAPON
        SELECT_PREV_WEAPON LOOK_UP_DOWN_DELTA TURN_LEFT_RIGHT_DELTA
        MOVE_LEFT_RIGHT_DELTA
    }
available_game_variables = { HEALTH KILLCOUNT PLAYER_KILLCOUNT }
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


@pytest.mark.parametrize(
    "fixture_case",
    ["ignored_masked_reset", "leaky_masked_reset", "forged_masked_reset"],
)
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


def test_broken_terminal_reset_is_named_through_the_public_command(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path, mismatch="broken_terminal_reset")),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    failures = report["invariant_suite"]["failures"]
    assert {failure["behavior"] for failure in failures} == {"termination", "truncation"}
    assert all("did not resume stepping" in failure["message"] for failure in failures)


def test_counter_only_kills_are_named_through_the_public_command(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    result = run_evidence(
        "--manifest",
        str(write_manifest(tmp_path, mismatch="counter_only_kills")),
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    failures = report["invariant_suite"]["failures"]
    assert {failure["behavior"] for failure in failures} == {
        "player_killcount",
        "player_killcount.enemy_on_enemy_exclusion",
    }
    assert all("actor/target attribution oracle" in failure["message"] for failure in failures)


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
            "timeout_seconds": 120,
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


def test_real_runner_separates_native_variables_and_observes_checkout_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gradoom
    from gradoom.evidence import reference_provider

    iwad = tmp_path / "freedoom2.wad"
    pwad = tmp_path / "deathmatch.wad"
    config = tmp_path / "deathmatch.cfg"
    iwad.write_bytes(b"iwad")
    pwad.write_bytes(b"pwad")
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    observed: dict[str, object] = {}

    class FakeGraDoom:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            observed["gradoom_game_variables"] = kwargs["game_variables"]
            observed["gradoom_info_filter"] = kwargs["info_filter"]

        def close(self) -> None:
            pass

    class FakeReference:
        revision = reference_provider.REFERENCE_REVISION
        env_class = invariant_runner._FixtureTurboEnv

        @staticmethod
        def make_env(*args: object, **kwargs: object) -> object:
            del args
            observed["reference_game_variables"] = kwargs["game_variables"]
            observed["reference_info_filter"] = kwargs["info_filter"]
            return type("Closable", (), {"close": lambda self: None})()

        @staticmethod
        def episode_kill_signals(*args: object, **kwargs: object) -> dict[str, float]:
            del args, kwargs
            return {"player_killcount": 0.0, "compatibility_killcount": 0.0}

    def capture_contract(**kwargs: object) -> dict[str, object]:
        env = kwargs["factory"]()  # type: ignore[operator]
        env.close()
        return {
            "provider": kwargs["provider"],
            "revision": kwargs["revision"],
        }

    monkeypatch.setattr(gradoom, "GraDoomVecEnv", FakeGraDoom)
    monkeypatch.setattr(reference_provider, "load_reference_provider", FakeReference)
    monkeypatch.setattr(invariant_runner, "_capture_contract", capture_contract)
    monkeypatch.setattr(invariant_runner, "_installed_gradoom_revision", lambda: "a" * 40)
    binding = {
        "iwad_path": str(iwad),
        "iwad_sha256": hashlib.sha256(iwad.read_bytes()).hexdigest(),
        "pwad_path": str(pwad),
        "pwad_sha256": hashlib.sha256(pwad.read_bytes()).hexdigest(),
        "configuration": PROFILE["configuration"],
    }

    contracts, unavailable = invariant_runner._real_contracts(
        {
            "gradoom_revision": "self-attested-manifest-value",
            "real_configuration": {
                "device": "cpu",
                "timeout_seconds": 120,
                "reference_scenario_config_path": str(config),
                "semantic_probes": semantic_probes(),
                "providers": {
                    "gradoom": dict(binding),
                    "env-vizdoom-turbo": dict(binding),
                },
            },
        }
    )

    assert unavailable == []
    assert contracts[0]["revision"] == "a" * 40
    assert contracts[0]["revision"] != "self-attested-manifest-value"
    expected_native = ("HEALTH", "KILLCOUNT", "PLAYER_KILLCOUNT")
    assert observed["gradoom_game_variables"] == expected_native
    assert observed["reference_game_variables"] == expected_native
    for provider in ("gradoom", "reference"):
        assert observed[f"{provider}_info_filter"] == {
            "mode": "all",
            "keys": ["health", "killcount", "player_killcount", "episode_return"],
        }


def test_real_runner_fails_closed_when_gradoom_revision_cannot_be_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gradoom.evidence import reference_provider

    iwad = tmp_path / "freedoom2.wad"
    pwad = tmp_path / "deathmatch.wad"
    config = tmp_path / "deathmatch.cfg"
    iwad.write_bytes(b"iwad")
    pwad.write_bytes(b"pwad")
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    monkeypatch.setattr(
        reference_provider,
        "load_reference_provider",
        lambda: type(
            "Provider",
            (),
            {"revision": reference_provider.REFERENCE_REVISION},
        )(),
    )
    monkeypatch.setattr(
        invariant_runner,
        "_installed_gradoom_revision",
        lambda: (_ for _ in ()).throw(RuntimeError("cannot prove checkout")),
    )
    binding = {
        "iwad_path": str(iwad),
        "iwad_sha256": hashlib.sha256(iwad.read_bytes()).hexdigest(),
        "pwad_path": str(pwad),
        "pwad_sha256": hashlib.sha256(pwad.read_bytes()).hexdigest(),
        "configuration": PROFILE["configuration"],
    }

    contracts, unavailable = invariant_runner._real_contracts(
        {
            "gradoom_revision": "declared",
            "real_configuration": {
                "device": "cpu",
                "timeout_seconds": 120,
                "reference_scenario_config_path": str(config),
                "semantic_probes": semantic_probes(),
                "providers": {
                    "gradoom": dict(binding),
                    "env-vizdoom-turbo": dict(binding),
                },
            },
        }
    )

    assert contracts == []
    assert unavailable == [
        {
            "code": "provider_contract_failure",
            "provider": "gradoom",
            "behavior": "gradoom.revision",
            "message": "cannot prove checkout",
        }
    ]


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
            "timeout_seconds": 120,
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
            "timeout_seconds": 120,
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
        "timeout_seconds": 120,
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


def synthetic_tensor_device_contract(
    declared_device: str,
    *,
    tensor_device: str,
    reward_device: str | None = None,
) -> dict[str, object]:
    def descriptor(
        shape: list[int], dtype: str, *, device: str = tensor_device
    ) -> dict[str, object]:
        return {
            "transport": "torch",
            "shape": shape,
            "dtype": dtype,
            "device": device,
        }

    signals = {
        name: descriptor([2], "float64")
        for name in ("health", "killcount", "player_killcount", "episode_return")
    }
    return {
        "tensor_device": {
            "declared_device": declared_device,
            "reset_mask_input": descriptor([2], "bool"),
            "step_action_input": descriptor([2], "int64"),
            "reset_outputs": {
                "observation": descriptor([2, 4, 84, 84], "uint8"),
                "signals": signals,
            },
            "step_outputs": {
                "observation": descriptor([2, 4, 84, 84], "uint8"),
                "reward": descriptor(
                    [2],
                    "float32",
                    device=reward_device or tensor_device,
                ),
                "terminated": descriptor([2], "bool"),
                "truncated": descriptor([2], "bool"),
                "signals": signals,
            },
        }
    }


@pytest.mark.parametrize(
    ("declared_device", "tensor_device"),
    [("cuda", "cuda:0"), ("cuda", "cuda:3"), ("cuda:0", "cuda:0")],
)
def test_requested_cuda_device_accepts_one_consistent_concrete_index(
    declared_device: str,
    tensor_device: str,
) -> None:
    checks = invariant_suite._gradoom_device_checks(
        synthetic_tensor_device_contract(declared_device, tensor_device=tensor_device)
    )

    assert all(check["status"] == "passed" for check in checks)


def test_requested_cuda_device_rejects_an_explicit_index_mismatch() -> None:
    checks = invariant_suite._gradoom_device_checks(
        synthetic_tensor_device_contract("cuda:1", tensor_device="cuda:0")
    )

    assert all(check["status"] == "failed" for check in checks)


def test_requested_cuda_shorthand_rejects_mixed_concrete_indices() -> None:
    checks = invariant_suite._gradoom_device_checks(
        synthetic_tensor_device_contract(
            "cuda",
            tensor_device="cuda:0",
            reward_device="cuda:1",
        )
    )

    assert {check["behavior"]: check["status"] for check in checks} == {
        "gradoom.tensor_inputs": "passed",
        "gradoom.tensor_outputs": "failed",
        "gradoom.device": "failed",
    }


@pytest.mark.parametrize("device", ["cuda:00", "cuda:01", "cuda:\u0661", "cuda:\uff11"])
def test_real_configuration_rejects_noncanonical_cuda_indices(
    tmp_path: Path,
    device: str,
) -> None:
    config = tmp_path / "deathmatch.cfg"
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    with pytest.raises(
        invariant_suite.InvariantSuiteError,
        match="device must be 'cpu', 'cuda', or 'cuda:N'",
    ):
        invariant_suite._prepare_real_configuration(
            {
                "device": device,
                "timeout_seconds": 120,
                "reference_scenario_config_input": "reference_scenario_config",
                "semantic_probes": semantic_probes(),
            },
            declared_inputs=[
                {
                    "name": "reference_scenario_config",
                    "path": str(config),
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                }
            ],
            base_directory=tmp_path,
            wad_profile={"providers": []},
        )


def test_reference_config_rejects_extra_provider_only_behavior(tmp_path: Path) -> None:
    config = tmp_path / "deathmatch.cfg"
    config.write_text(REFERENCE_CONFIG + "window_visible = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact validated scenario configuration"):
        invariant_runner._validate_reference_scenario_config(
            config,
            PROFILE["configuration"],
        )


def test_semantic_kill_probe_rejects_counter_only_provider() -> None:
    env = invariant_runner._FixtureTurboEnv(
        "VizdoomDeathmatch-v1",
        num_envs=2,
        fixture_transport="numpy",
    )

    with pytest.raises(RuntimeError, match="actor/target attribution oracle"):
        invariant_runner._semantic_probe(
            behavior="player_killcount",
            provider="env-vizdoom-turbo",
            factory=lambda: env,
            probe={"seeds": [1, 2], "actions": [[3, 0]], "max_steps": 1},
            requested_device=None,
            kill_signal_reader=None,
            attribution_oracle=None,
        )


def test_lifecycle_probe_rejects_a_broken_terminal_reset() -> None:
    env = invariant_runner._FixtureTurboEnv(
        "VizdoomDeathmatch-v1",
        num_envs=2,
        fixture_transport="numpy",
        fixture_terminal_reset="broken",
    )

    with pytest.raises(RuntimeError, match="resume stepping"):
        invariant_runner._semantic_probe(
            behavior="termination",
            provider="env-vizdoom-turbo",
            factory=lambda: env,
            probe={"seeds": [1, 2], "actions": [[1, 0]], "max_steps": 1},
            requested_device=None,
            kill_signal_reader=None,
            attribution_oracle=None,
        )


def test_masked_reset_rejects_forged_snapshots_without_internal_reset() -> None:
    contract = invariant_runner._capture_contract(
        provider="env-vizdoom-turbo",
        revision="fixture",
        env_class=invariant_runner._FixtureTurboEnv,
        factory=lambda: invariant_runner._FixtureTurboEnv(
            "VizdoomDeathmatch-v1",
            num_envs=2,
            fixture_transport="numpy",
            fixture_masked_reset="forge",
        ),
        fixture_case="pass",
        semantic_probes={
            "termination": {"seeds": [1, 2], "actions": [[1, 0]], "max_steps": 1},
            "truncation": {"seeds": [1, 2], "actions": [[0, 2]], "max_steps": 1},
            "player_killcount": {"seeds": [1, 2], "actions": [[3, 0]], "max_steps": 1},
            "player_killcount.enemy_on_enemy_exclusion": {
                "seeds": [1, 2],
                "actions": [[4, 0]],
                "max_steps": 1,
            },
        },
        attribution_oracle=invariant_runner._fixture_attribution_oracle,
    )

    assert contract["behaviors"]["masked_reset"] == {
        "supported": True,
        "selected_lane_state_and_signals_reset": True,
        "unselected_lane_state_and_signals_unchanged": True,
        "selected_lane_continues_from_reset_state": False,
        "unselected_lane_continues_without_reset": True,
    }


def test_accepted_probe_budget_controls_runner_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        del args
        captured.update(kwargs)
        request = json.loads(str(kwargs["input"]))
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "protocol_version": invariant_runner.RUNNER_PROTOCOL_VERSION,
                    "challenge": request["challenge"],
                    "runner_sha256": "a" * 64,
                    "status": "unavailable",
                    "contracts": [],
                    "unavailable_reasons": [{"code": "test", "message": "test"}],
                }
            ),
            "",
        )

    monkeypatch.setattr(invariant_suite.subprocess, "run", fake_run)
    invariant_suite._execute_runner(
        runner_sha256="a" * 64,
        mode="real",
        fixture_case="pass",
        gradoom_revision="revision",
        real_configuration={
            "timeout_seconds": 321,
            "semantic_probes": semantic_probes(),
        },
    )

    assert captured["timeout"] == 321


def test_probe_budget_rejects_an_incoherent_runner_timeout(tmp_path: Path) -> None:
    config = tmp_path / "deathmatch.cfg"
    config.write_text(REFERENCE_CONFIG, encoding="utf-8")
    probes = semantic_probes()
    for probe in probes.values():  # type: ignore[union-attr]
        probe["max_steps"] = 100_000  # type: ignore[index]

    with pytest.raises(
        invariant_suite.InvariantSuiteError,
        match=r"timeout_seconds must be an integer in \[20000, 86400\]",
    ):
        invariant_suite._prepare_real_configuration(
            {
                "device": "cpu",
                "timeout_seconds": 19_999,
                "reference_scenario_config_input": "reference_scenario_config",
                "semantic_probes": probes,
            },
            declared_inputs=[
                {
                    "name": "reference_scenario_config",
                    "path": str(config),
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                }
            ],
            base_directory=tmp_path,
            wad_profile={"providers": []},
        )


def test_runner_timeout_becomes_a_named_fail_closed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def time_out(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise subprocess.TimeoutExpired(["runner"], 120)

    monkeypatch.setattr(invariant_suite.subprocess, "run", time_out)

    result = invariant_suite._execute_runner(
        runner_sha256="a" * 64,
        mode="real",
        fixture_case="pass",
        gradoom_revision="revision",
        real_configuration={
            "timeout_seconds": 120,
            "semantic_probes": semantic_probes(),
        },
    )

    assert result["status"] == "unavailable"
    assert result["contracts"] == []
    assert result["unavailable_reasons"] == [
        {
            "code": "invariant_runner_timeout",
            "message": (
                "Authenticated invariant execution exhausted its predeclared 120-second timeout."
            ),
        }
    ]


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

    assert result == {
        "reported_separately": True,
        "requires_reset": False,
        "terminal_lane_reset": True,
        "stepping_resumed": True,
    }
    assert env.actions == [[7, 8], [7, 8], [7, 8], [7, 8]]


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
                    "timeout_seconds": 120,
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
