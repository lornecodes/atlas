"""PermissionsSpec — declarative security permissions for agent contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_ISOLATION_MODES = {"process", "container"}


@dataclass(frozen=True)
class PermissionsSpec:
    """What an agent is allowed to do, declared in agent.yaml.

    Defaults are conservative: read-only filesystem, outbound network only,
    no spawning, in-process isolation.
    """

    filesystem: list[str] = field(default_factory=lambda: ["read"])
    network: list[str] = field(default_factory=lambda: ["outbound"])
    spawn: bool = False
    max_memory_mb: int = 512
    max_cpu_seconds: int = 60
    secrets: list[str] = field(default_factory=list)
    isolation: str = "process"  # process | container
    container_image: str = ""   # override for container isolation

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> PermissionsSpec:
        """Parse a permissions block from agent.yaml."""
        if not d:
            return PermissionsSpec()

        isolation = d.get("isolation", "process")
        if isolation not in VALID_ISOLATION_MODES:
            raise ValueError(
                f"Invalid isolation mode '{isolation}', "
                f"must be one of: {sorted(VALID_ISOLATION_MODES)}"
            )

        return PermissionsSpec(
            filesystem=list(d.get("filesystem", ["read"])),
            network=list(d.get("network", ["outbound"])),
            spawn=d.get("spawn", False),
            max_memory_mb=int(d.get("max_memory_mb", 512)),
            max_cpu_seconds=int(d.get("max_cpu_seconds", 60)),
            secrets=list(d.get("secrets", [])),
            isolation=isolation,
            container_image=d.get("container_image", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for YAML/JSON."""
        return {
            "filesystem": list(self.filesystem),
            "network": list(self.network),
            "spawn": self.spawn,
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "secrets": list(self.secrets),
            "isolation": self.isolation,
            "container_image": self.container_image,
        }
