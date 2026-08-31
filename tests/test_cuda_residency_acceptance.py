from __future__ import annotations

import os

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from gradoom.evidence.cuda_residency import (
    CudaHostTransferGuard,
    CudaResidencyAcceptance,
)

EXPECTED_CATEGORIES = {
    "observations",
    "actions",
    "rewards",
    "resets",
    "rollout_state",
    "inference",
    "losses",
    "optimizer_state",
    "parameters",
    "updates",
}


def test_cuda_residency_contract_records_every_required_training_device() -> None:
    acceptance = CudaResidencyAcceptance(torch.device("cuda:0"))

    with FakeTensorMode():
        tensor = torch.empty(2, device="cuda:0")
        for category in EXPECTED_CATEGORIES:
            acceptance.observe(category, tensor)

    report = acceptance.report(
        checked_rollouts=1,
        checked_steps=4,
        workload={"num_envs": 2, "n_steps": 2, "global_step_start": 0, "global_step_end": 4},
        hardware={
            "gpu_model": "fixture GPU",
            "device": "cuda:0",
            "compute_capability": "9.9",
            "total_memory_bytes": 1,
        },
        software={
            "python": "fixture",
            "gradoom": "fixture",
            "torch": "fixture",
            "cuda": "fixture",
            "cudnn": "fixture",
            "numpy": "fixture",
        },
    )

    assert report["type"] == "cuda_residency_acceptance"
    assert report["contract"] == "gradoom-cuda-residency-v1"
    assert report["status"] == "passed"
    assert set(report["devices"]) == EXPECTED_CATEGORIES
    assert set(report["checked_categories"]) == EXPECTED_CATEGORIES
    assert all(devices == ["cuda:0"] for devices in report["devices"].values())
    assert report["host_transition_guard"] == {
        "status": "passed",
        "guarded_scopes": 0,
        "detected_transfers": 0,
    }


def test_cuda_residency_contract_fails_closed_on_cpu_tensor_or_missing_category() -> None:
    acceptance = CudaResidencyAcceptance(torch.device("cuda:0"))

    with pytest.raises(RuntimeError, match=r"observations.*expected cuda:0.*got cpu"):
        acceptance.observe("observations", torch.empty(1))

    with FakeTensorMode():
        acceptance.observe("observations", torch.empty(1, device="cuda:0"))
    with pytest.raises(RuntimeError, match="missing required device evidence"):
        acceptance.report(
            checked_rollouts=1,
            checked_steps=1,
            workload={},
            hardware={},
            software={},
        )


def test_host_transition_guard_rejects_an_accelerator_to_cpu_copy_before_execution() -> None:
    source = torch.empty(1, device="meta")

    with pytest.raises(
        RuntimeError, match=r"steady_state_step.*accelerator-to-host"
    ), CudaHostTransferGuard("steady_state_step"):
        source.cpu()


@pytest.mark.skipif(
    os.environ.get("GRADOOM_RUN_CUDA_ACCEPTANCE") != "1" or not torch.cuda.is_available(),
    reason=(
        "CUDA residency hardware execution requires an explicitly allocated GPU and "
        "GRADOOM_RUN_CUDA_ACCEPTANCE=1"
    ),
)
def test_cuda_host_transition_guard_rejects_cpu_and_numpy_round_trips() -> None:
    transition = torch.ones(1, device="cuda")

    with pytest.raises(
        RuntimeError, match="accelerator-to-host"
    ), CudaHostTransferGuard("steady_state_step"):
        transition.cpu()
    with pytest.raises(
        RuntimeError, match="accelerator-to-host"
    ), CudaHostTransferGuard("steady_state_step"):
        transition.cpu().numpy()
