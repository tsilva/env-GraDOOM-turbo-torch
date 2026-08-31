"""Opt-in CUDA residency evidence for the standalone training data plane."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

CUDA_RESIDENCY_CONTRACT = "gradoom-cuda-residency-v1"
CUDA_RESIDENCY_CATEGORIES = (
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
)


def _tensors(values: object) -> list[torch.Tensor]:
    if isinstance(values, torch.Tensor):
        return [values]
    if isinstance(values, Mapping):
        return [tensor for value in values.values() for tensor in _tensors(value)]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return [tensor for value in values for tensor in _tensors(value)]
    return []


class CudaHostTransferGuard(TorchDispatchMode):
    """Reject accelerator-to-host tensor copies inside one guarded data-plane scope."""

    def __init__(
        self,
        phase: str,
        *,
        on_detect: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.phase = phase
        self._on_detect = on_detect

    def _reject(self, operation: object) -> None:
        if self._on_detect is not None:
            self._on_detect()
        raise RuntimeError(
            f"{self.phase}: accelerator-to-host transition transport is forbidden "
            f"during CUDA residency acceptance ({operation})"
        )

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> Any:
        call_kwargs = {} if kwargs is None else kwargs
        operation = str(func)
        source_tensors = _tensors(args)
        if operation.startswith("aten._to_copy"):
            destination = call_kwargs.get("device")
            if (
                isinstance(destination, torch.device)
                and destination.type == "cpu"
                and any(tensor.device.type != "cpu" for tensor in source_tensors)
            ):
                self._reject(operation)
        if operation.startswith("aten.copy_") and len(args) >= 2:
            destination, source = args[:2]
            if (
                isinstance(destination, torch.Tensor)
                and destination.device.type == "cpu"
                and isinstance(source, torch.Tensor)
                and source.device.type != "cpu"
            ):
                self._reject(operation)
        return func(*args, **call_kwargs)  # type: ignore[operator]


class CudaResidencyAcceptance:
    """Collect fail-closed named device evidence for one checked training workload."""

    def __init__(self, expected_device: torch.device) -> None:
        if expected_device.type != "cuda":
            raise ValueError("CUDA residency acceptance requires a concrete CUDA device")
        if expected_device.index is None:
            raise ValueError("CUDA residency acceptance requires a concrete CUDA device index")
        self.expected_device = expected_device
        self._devices: dict[str, set[str]] = {}
        self._guarded_scopes = 0
        self._detected_transfers = 0

    def observe(self, category: str, *values: object) -> None:
        if category not in CUDA_RESIDENCY_CATEGORIES:
            raise ValueError(f"unknown CUDA residency category: {category}")
        observed = [tensor for value in values for tensor in _tensors(value)]
        if not observed:
            raise RuntimeError(f"{category}: no tensor device evidence was supplied")
        for tensor in observed:
            if tensor.device != self.expected_device:
                raise RuntimeError(
                    f"{category}: expected {self.expected_device}, got {tensor.device}"
                )
        self._devices.setdefault(category, set()).update(str(tensor.device) for tensor in observed)

    def guard(self, phase: str) -> CudaHostTransferGuard:
        self._guarded_scopes += 1

        def detected() -> None:
            self._detected_transfers += 1

        return CudaHostTransferGuard(phase, on_detect=detected)

    def report(
        self,
        *,
        checked_rollouts: int,
        checked_steps: int,
        workload: Mapping[str, object],
        hardware: Mapping[str, object],
        software: Mapping[str, object],
    ) -> dict[str, object]:
        missing = [
            category for category in CUDA_RESIDENCY_CATEGORIES if category not in self._devices
        ]
        if missing:
            raise RuntimeError(
                "CUDA residency acceptance missing required device evidence: " + ", ".join(missing)
            )
        if checked_rollouts <= 0 or checked_steps <= 0:
            raise RuntimeError("CUDA residency acceptance checked no steady-state training work")
        if self._detected_transfers:
            raise RuntimeError("CUDA residency acceptance detected forbidden host transport")
        return {
            "type": "cuda_residency_acceptance",
            "contract": CUDA_RESIDENCY_CONTRACT,
            "status": "passed",
            "checked_categories": list(CUDA_RESIDENCY_CATEGORIES),
            "devices": {
                category: sorted(self._devices[category])
                for category in CUDA_RESIDENCY_CATEGORIES
            },
            "host_transition_guard": {
                "status": "passed",
                "guarded_scopes": self._guarded_scopes,
                "detected_transfers": self._detected_transfers,
            },
            "checked_rollouts": checked_rollouts,
            "checked_steps": checked_steps,
            "workload": dict(workload),
            "hardware": dict(hardware),
            "software": dict(software),
        }
