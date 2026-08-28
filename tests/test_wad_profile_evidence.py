from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

APPROVED_IWAD_SHA256 = "a8772e088847032510d97ba2312406a6998f21cbab44d4ff10696faa9c0ecd4b"
APPROVED_PWAD_SHA256 = "1d06c2113f2c1546062635ad599f49cd852287a08b7b07b26d30b8f4c362a42d"
APPROVED_CONFIGURATION = {
    "map": "MAP01",
    "skill": 1,
    "scenario": {
        "game": "VizdoomDeathmatch-v1",
        "mode": "PLAYER",
        "screen_resolution": [320, 240],
        "episode_start_time": 1,
        "player_death_termination": True,
        "episode_timeout_as_truncation": True,
        "render_hud": False,
        "render_screen_flashes": False,
    },
    "action_mode": {
        "kind": "custom_discrete",
        "count": 17,
        "table_sha256": "0bd9dd28d67a88ef6bc54734f53d55bc4af597e672665a7f20d4b204098036af",
    },
    "frame_skip": 2,
    "episode_horizon_tics": 4200,
    "observation": {
        "crop_or_mask": {"kind": "mask", "edges": [0, 32, 0, 0], "fill": 0},
        "resize": {"shape": [84, 84], "algorithm": "area"},
        "grayscale": {
            "enabled": True,
            "conversion": "env-vizdoom-turbo-rgb-area-gray8-v1",
        },
        "layout": "chw",
        "frame_stack": 4,
    },
}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def run_evidence(*args: str) -> subprocess.CompletedProcess[str]:
    command = shutil.which("gradoom-evidence")
    assert command is not None
    return subprocess.run(
        [command, *args],
        check=False,
        capture_output=True,
        text=True,
    )


def write_manifest(
    directory: Path,
    *,
    gradoom_iwad: Path,
    gradoom_pwad: Path,
    reference_iwad: Path,
    reference_pwad: Path,
    reference_configuration: dict[str, object] | None = None,
) -> Path:
    manifest = {
        "schema_version": 1,
        "workflow": "parity_readiness",
        "evidence_level": "development",
        "fixture": True,
        "code_provenance": {
            "repository": "tsilva/env-GraDOOM-turbo-torch",
            "revision": "fixture-revision",
            "dirty": False,
        },
        "declared_inputs": [],
        "prerequisites": [
            {"id": "certified_freedoom2_wad_profile", "available": True},
            {
                "id": "pinned_reference_provider",
                "available": False,
                "reason": "The provider adapter is implemented by issue #14.",
            },
            {
                "id": "real_pretrained_policy_corpus",
                "available": False,
                "reason": "The real corpus is not available.",
            },
        ],
        "wad_profile": {
            "profile_id": "freedoom2-deathmatch-v1",
            "providers": [
                {
                    "id": "gradoom",
                    "iwad_path": os.fspath(gradoom_iwad),
                    "pwad_path": os.fspath(gradoom_pwad),
                    "configuration": copy.deepcopy(APPROVED_CONFIGURATION),
                },
                {
                    "id": "env-vizdoom-turbo",
                    "iwad_path": os.fspath(reference_iwad),
                    "pwad_path": os.fspath(reference_pwad),
                    "configuration": copy.deepcopy(
                        reference_configuration or APPROVED_CONFIGURATION
                    ),
                },
            ],
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_asset_and_policy_mismatches_fail_readiness_in_the_report(tmp_path: Path) -> None:
    gradoom_iwad = tmp_path / "gradoom-iwad.wad"
    gradoom_pwad = tmp_path / "gradoom-pwad.wad"
    reference_iwad = tmp_path / "reference-iwad.wad"
    reference_pwad = tmp_path / "reference-pwad.wad"
    gradoom_iwad.write_bytes(b"fixture gradoom iwad")
    gradoom_pwad.write_bytes(b"fixture gradoom pwad")
    reference_iwad.write_bytes(b"different fixture reference iwad")
    reference_pwad.write_bytes(b"fixture gradoom pwad")
    mismatched_configuration = copy.deepcopy(APPROVED_CONFIGURATION)
    mismatched_configuration["skill"] = 3
    manifest = write_manifest(
        tmp_path,
        gradoom_iwad=gradoom_iwad,
        gradoom_pwad=gradoom_pwad,
        reference_iwad=reference_iwad,
        reference_pwad=reference_pwad,
        reference_configuration=mismatched_configuration,
    )
    report_path = tmp_path / "report.json"

    result = run_evidence("--manifest", str(manifest), "--output", str(report_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["wad_profile"]["status"] == "failed"
    failures = report["wad_profile"]["failures"]
    assert {failure["code"] for failure in failures} >= {
        "asset_hash_mismatch",
        "provider_asset_mismatch",
        "configuration_mismatch",
    }
    assert any(
        failure.get("provider") == "env-vizdoom-turbo"
        and failure.get("field") == "configuration.skill"
        and failure.get("expected") == 1
        and failure.get("actual") == 3
        for failure in failures
    )
    assert any(reason["code"] == "wad_profile_mismatch" for reason in report["claim_reasons"])
    assert report["prerequisites"][0] == {
        "id": "certified_freedoom2_wad_profile",
        "available": False,
        "reason": "The certified Freedoom2 WAD profile did not match.",
    }


def test_report_output_cannot_overwrite_a_profile_asset(tmp_path: Path) -> None:
    gradoom_iwad = tmp_path / "gradoom-iwad.wad"
    gradoom_pwad = tmp_path / "gradoom-pwad.wad"
    reference_iwad = tmp_path / "reference-iwad.wad"
    reference_pwad = tmp_path / "reference-pwad.wad"
    original = b"fixture gradoom iwad"
    gradoom_iwad.write_bytes(original)
    gradoom_pwad.write_bytes(b"fixture gradoom pwad")
    reference_iwad.write_bytes(b"fixture reference iwad")
    reference_pwad.write_bytes(b"fixture reference pwad")
    manifest = write_manifest(
        tmp_path,
        gradoom_iwad=gradoom_iwad,
        gradoom_pwad=gradoom_pwad,
        reference_iwad=reference_iwad,
        reference_pwad=reference_pwad,
    )

    result = run_evidence(
        "--manifest",
        str(manifest),
        "--output",
        str(gradoom_iwad),
    )

    assert result.returncode == 2
    assert "output path aliases WAD profile asset 'gradoom.iwad'" in result.stderr
    assert gradoom_iwad.read_bytes() == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("map", "MAP02"),
        ("skill", 3),
        ("scenario.mode", "SPECTATOR"),
        ("scenario.screen_resolution", [640, 480]),
        ("action_mode.kind", "full"),
        ("action_mode.table_sha256", "0" * 64),
        ("frame_skip", 4),
        ("episode_horizon_tics", 4_201),
        ("observation.crop_or_mask.kind", "remove"),
        ("observation.crop_or_mask.edges", [0, 0, 0, 0]),
        ("observation.resize.shape", [96, 96]),
        ("observation.resize.algorithm", "nearest"),
        ("observation.grayscale.enabled", False),
        ("observation.grayscale.conversion", "unbound-conversion"),
        ("observation.layout", "hwc"),
        ("observation.frame_stack", 1),
    ],
)
def test_every_policy_facing_setting_is_an_early_readiness_gate(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    asset = tmp_path / "same-provider-asset.wad"
    asset.write_bytes(b"small fixture asset")
    configuration = copy.deepcopy(APPROVED_CONFIGURATION)
    target = configuration
    parts = field.split(".")
    for part in parts[:-1]:
        child = target[part]
        assert isinstance(child, dict)
        target = child
    target[parts[-1]] = value
    manifest = write_manifest(
        tmp_path,
        gradoom_iwad=asset,
        gradoom_pwad=asset,
        reference_iwad=asset,
        reference_pwad=asset,
        reference_configuration=configuration,
    )
    report_path = tmp_path / "report.json"

    result = run_evidence("--manifest", str(manifest), "--output", str(report_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert any(
        failure["code"] == "configuration_mismatch"
        and failure["provider"] == "env-vizdoom-turbo"
        and failure["field"] == f"configuration.{field}"
        for failure in report["wad_profile"]["failures"]
    )


def test_a_distinct_wad_profile_cannot_inherit_the_first_profiles_identity(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "same-provider-asset.wad"
    asset.write_bytes(b"small fixture asset")
    manifest = write_manifest(
        tmp_path,
        gradoom_iwad=asset,
        gradoom_pwad=asset,
        reference_iwad=asset,
        reference_pwad=asset,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["wad_profile"]["profile_id"] = "ordinary-doom2-local-profile"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_evidence("--manifest", str(manifest), "--output", str(report_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert any(
        failure["code"] == "profile_id_mismatch"
        and failure["actual"] == "ordinary-doom2-local-profile"
        for failure in report["wad_profile"]["failures"]
    )


def _operator_assets() -> tuple[Path, Path] | None:
    configured_iwad = os.environ.get("GRADOOM_FREEDOOM_IWAD")
    configured_pwad = os.environ.get("GRADOOM_DEATHMATCH_WAD")
    candidates = []
    if configured_iwad and configured_pwad:
        candidates.append((Path(configured_iwad), Path(configured_pwad)))
    candidates.append(
        (
            Path(
                "/home/tsilva/repos/tsilva/gradlab/.venv/lib/python3.14/"
                "site-packages/vizdoom/freedoom2.wad"
            ),
            Path(
                "/home/tsilva/repos/tsilva/gradlab/.venv/lib/python3.14/"
                "site-packages/vizdoom/scenarios/deathmatch.wad"
            ),
        )
    )
    for iwad, pwad in candidates:
        if iwad.is_file() and pwad.is_file():
            return iwad, pwad
    return None


@pytest.mark.skipif(_operator_assets() is None, reason="approved operator WADs are absent")
def test_exact_profile_match_binds_complete_profile_and_provider_assets_to_run_identity(
    tmp_path: Path,
) -> None:
    assets = _operator_assets()
    assert assets is not None
    iwad, pwad = assets
    manifest = write_manifest(
        tmp_path,
        gradoom_iwad=iwad,
        gradoom_pwad=pwad,
        reference_iwad=iwad,
        reference_pwad=pwad,
    )
    report_path = tmp_path / "report.json"

    result = run_evidence("--manifest", str(manifest), "--output", str(report_path))

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    profile = report["wad_profile"]
    assert profile["status"] == "matched"
    assert profile["failures"] == []
    assert profile["profile"] == {
        "schema_version": 1,
        "profile_id": "freedoom2-deathmatch-v1",
        "assets": {
            "iwad": {"name": "Freedoom2", "sha256": APPROVED_IWAD_SHA256},
            "pwad": {"name": "ViZDoom deathmatch", "sha256": APPROVED_PWAD_SHA256},
        },
        "configuration": APPROVED_CONFIGURATION,
    }
    assert profile["profile_identity"] == canonical_sha256(profile["profile"])
    assert {provider["iwad"]["sha256"] for provider in profile["providers"]} == {
        APPROVED_IWAD_SHA256
    }
    assert {provider["pwad"]["sha256"] for provider in profile["providers"]} == {
        APPROVED_PWAD_SHA256
    }
    expected_binding = {
        "profile": profile["profile"],
        "providers": [
            {
                "id": provider["id"],
                "iwad_sha256": provider["iwad"]["sha256"],
                "pwad_sha256": provider["pwad"]["sha256"],
                "configuration": provider["configuration"],
            }
            for provider in profile["providers"]
        ],
    }
    assert profile["binding_identity"] == expected_binding
    assert profile["binding_sha256"] == canonical_sha256(expected_binding)
    assert report["run_identity"] == canonical_sha256(
        {
            "schema_version": 1,
            "workflow": "parity_readiness",
            "evidence_level": "development",
            "fixture": True,
            "code_provenance": {
                "repository": "tsilva/env-GraDOOM-turbo-torch",
                "revision": "fixture-revision",
                "dirty": False,
            },
            "declared_inputs": [],
            "prerequisites": [
                "certified_freedoom2_wad_profile",
                "pinned_reference_provider",
                "real_pretrained_policy_corpus",
            ],
            "wad_profile": expected_binding,
        }
    )
    assert report["prerequisites"][0] == {
        "id": "certified_freedoom2_wad_profile",
        "available": True,
    }
    assert report["status"] == "unavailable"


def test_uncertified_environment_instances_do_not_inherit_profile_evidence(
    square_scenario,
) -> None:
    from gradoom import GraDoomVecEnv

    env = GraDoomVecEnv(
        "VizdoomDeathmatch-v1",
        compiled_scenario=square_scenario,
        device="cpu",
        frame_skip=2,
        obs_crop=(0, 32, 0, 0),
        obs_crop_mode="mask",
    )
    try:
        assert env.iwad_sha256 == "1" * 64
        assert env.parity_certified is False
    finally:
        env.close()
