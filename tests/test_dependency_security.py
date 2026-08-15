from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_lstm_cell_rejects_invalid_state_shape_and_keeps_valid_inputs() -> None:
    cell = torch.nn.LSTMCell(3, 4)
    hidden, cell_state = cell(
        torch.zeros(2, 3),
        (torch.zeros(2, 4), torch.zeros(2, 4)),
    )
    assert hidden.shape == (2, 4)
    assert cell_state.shape == (2, 4)

    with pytest.raises(RuntimeError):
        cell(torch.zeros(2, 3), (torch.zeros(2, 5), torch.zeros(2, 5)))


def test_jit_script_handles_a_legitimate_local_module() -> None:
    scripted = torch.jit.script(torch.nn.ReLU())
    assert torch.equal(scripted(torch.tensor([-1.0, 2.0])), torch.tensor([0.0, 2.0]))


def test_lock_uses_patched_floors_and_registry_only_sources() -> None:
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text("utf-8"))
    versions = {package["name"]: package["version"] for package in lock["package"]}
    assert tuple(int(part) for part in versions["torch"].split(".")) >= (2, 13, 0)
    assert tuple(int(part) for part in versions["setuptools"].split(".")) >= (
        83,
        0,
        0,
    )

    for package in lock["package"]:
        source = package.get("source", {})
        assert "git" not in source, package["name"]
        assert "url" not in source, package["name"]
        assert "path" not in source, package["name"]
        if registry := source.get("registry"):
            assert registry == "https://pypi.org/simple"
