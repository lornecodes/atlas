"""Hardware inventory and resource allocation for the execution pool.

Tracks pool capacity (GPUs, memory, CPU cores, devices) and manages
per-slot resource reservations. Used by SlotManager to gate slot
acquisition on hardware availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas.contract.types import HardwareSpec


class ResourceUnavailable(Exception):
    """Raised when requested hardware resources aren't available."""


@dataclass
class ResourceAllocation:
    """A resource reservation for a slot."""

    slot_id: str
    gpu_devices: list[int] = field(default_factory=list)
    memory_gb: int = 0
    cpu_cores: int = 0


@dataclass
class HardwareInventory:
    """Pool hardware capacity and allocation tracking.

    Configured at pool startup with total available resources.
    Allocations are tracked per-slot and released when slots are
    returned or destroyed.
    """

    # Configured capacity
    total_gpus: int = 0
    gpu_vram_gb: list[int] = field(default_factory=list)
    total_memory_gb: int = 0
    total_cpu_cores: int = 0
    architecture: str = "any"
    available_devices: list[str] = field(default_factory=list)

    # Active allocations (slot_id -> allocation)
    _allocations: dict[str, ResourceAllocation] = field(
        default_factory=dict, repr=False
    )

    @property
    def free_gpus(self) -> int:
        used = sum(len(a.gpu_devices) for a in self._allocations.values())
        return max(0, self.total_gpus - used)

    @property
    def free_memory_gb(self) -> int:
        used = sum(a.memory_gb for a in self._allocations.values())
        return max(0, self.total_memory_gb - used)

    @property
    def free_cpu_cores(self) -> int:
        used = sum(a.cpu_cores for a in self._allocations.values())
        return max(0, self.total_cpu_cores - used)

    @property
    def allocated_gpu_indices(self) -> set[int]:
        """Return the set of GPU device indices currently allocated."""
        indices: set[int] = set()
        for a in self._allocations.values():
            indices.update(a.gpu_devices)
        return indices

    def can_satisfy(self, spec: HardwareSpec) -> bool:
        """Check if spec can be satisfied with current free resources."""
        # Architecture check (static)
        if spec.architecture != "any" and self.architecture != "any":
            if spec.architecture != self.architecture:
                return False

        # GPU check
        if spec.gpu:
            if self.free_gpus < 1:
                return False
            # VRAM check — find a free GPU with enough VRAM
            if spec.gpu_vram_gb > 0 and self.gpu_vram_gb:
                allocated = self.allocated_gpu_indices
                has_suitable = any(
                    vram >= spec.gpu_vram_gb
                    for i, vram in enumerate(self.gpu_vram_gb)
                    if i not in allocated
                )
                if not has_suitable:
                    return False

        # Memory check
        if self.total_memory_gb > 0 and spec.min_memory_gb > self.free_memory_gb:
            return False

        # CPU check
        if self.total_cpu_cores > 0 and spec.min_cpu_cores > self.free_cpu_cores:
            return False

        # Device access check
        if spec.device_access:
            for device in spec.device_access:
                if device not in self.available_devices:
                    return False

        return True

    def allocate(
        self, slot_id: str, spec: HardwareSpec
    ) -> ResourceAllocation:
        """Reserve resources for a slot.

        Raises ResourceUnavailable if the spec can't be satisfied.
        """
        if not self.can_satisfy(spec):
            raise ResourceUnavailable(
                f"Cannot satisfy hardware requirements: "
                f"{describe_requirement(spec)}"
            )

        gpu_devices: list[int] = []
        if spec.gpu:
            # Assign lowest available GPU
            allocated = self.allocated_gpu_indices
            if spec.gpu_vram_gb > 0 and self.gpu_vram_gb:
                # Pick first free GPU with enough VRAM
                for i, vram in enumerate(self.gpu_vram_gb):
                    if i not in allocated and vram >= spec.gpu_vram_gb:
                        gpu_devices = [i]
                        break
            else:
                # Pick first free GPU
                for i in range(self.total_gpus):
                    if i not in allocated:
                        gpu_devices = [i]
                        break

        allocation = ResourceAllocation(
            slot_id=slot_id,
            gpu_devices=gpu_devices,
            memory_gb=spec.min_memory_gb,
            cpu_cores=spec.min_cpu_cores,
        )
        self._allocations[slot_id] = allocation
        return allocation

    def release(self, slot_id: str) -> None:
        """Free resources when a slot is released or destroyed."""
        self._allocations.pop(slot_id, None)

    def status(self) -> dict[str, Any]:
        """Return current hardware status for the health endpoint."""
        return {
            "total_gpus": self.total_gpus,
            "free_gpus": self.free_gpus,
            "gpu_vram_gb": list(self.gpu_vram_gb),
            "total_memory_gb": self.total_memory_gb,
            "free_memory_gb": self.free_memory_gb,
            "total_cpu_cores": self.total_cpu_cores,
            "free_cpu_cores": self.free_cpu_cores,
            "architecture": self.architecture,
            "allocations": [
                {
                    "slot_id": a.slot_id,
                    "gpu_devices": a.gpu_devices,
                    "memory_gb": a.memory_gb,
                    "cpu_cores": a.cpu_cores,
                }
                for a in self._allocations.values()
            ],
        }


def describe_requirement(spec: HardwareSpec) -> str:
    """Human-readable description of hardware requirements."""
    parts = []
    if spec.gpu:
        gpu_desc = "GPU"
        if spec.gpu_vram_gb:
            gpu_desc += f" ({spec.gpu_vram_gb}GB VRAM)"
        parts.append(gpu_desc)
    if spec.min_memory_gb > 1:
        parts.append(f"{spec.min_memory_gb}GB RAM")
    if spec.min_cpu_cores > 1:
        parts.append(f"{spec.min_cpu_cores} CPU cores")
    if spec.architecture != "any":
        parts.append(f"arch={spec.architecture}")
    if spec.device_access:
        parts.append(f"devices={spec.device_access}")
    return ", ".join(parts) if parts else "default"
