"""Core types for the Atlas agent contract."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any

from atlas.contract.permissions import PermissionsSpec


class SchemaSpec:
    """JSON Schema describing agent I/O.

    Single source of truth: stores the raw JSON Schema dict.
    Convenience properties compute from `raw` on access.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        object.__setattr__(self, "_raw", raw or {})

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("SchemaSpec is immutable")

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SchemaSpec):
            return self._raw == other._raw
        return NotImplemented

    def __hash__(self) -> int:
        return hash(_json.dumps(self._raw, sort_keys=True))

    def __repr__(self) -> str:
        return f"SchemaSpec({self._raw!r})"

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw

    @property
    def type(self) -> str:
        return self._raw.get("type", "object")

    @property
    def properties(self) -> dict[str, Any]:
        return self._raw.get("properties", {})

    @property
    def required(self) -> list[str]:
        return self._raw.get("required", [])

    def to_json_schema(self) -> dict[str, Any]:
        """Return the full JSON Schema dict for validation."""
        if self._raw:
            return dict(self._raw)
        return {"type": "object"}

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> SchemaSpec:
        if not d:
            return SchemaSpec()
        return SchemaSpec(raw=d)


@dataclass(frozen=True)
class HardwareSpec:
    """Hardware requirements for agent scheduling."""

    gpu: bool = False
    gpu_vram_gb: int = 0
    min_memory_gb: int = 1
    min_cpu_cores: int = 1
    architecture: str = "any"
    node_affinity: str = ""
    device_access: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> HardwareSpec:
        if not d:
            return HardwareSpec()
        return HardwareSpec(
            gpu=d.get("gpu", False),
            gpu_vram_gb=d.get("gpu_vram_gb", 0),
            min_memory_gb=d.get("min_memory_gb", 1),
            min_cpu_cores=d.get("min_cpu_cores", 1),
            architecture=d.get("architecture", "any"),
            node_affinity=d.get("node_affinity", ""),
            device_access=d.get("device_access", []),
        )


@dataclass(frozen=True)
class ModelSpec:
    """Model preferences for the agent."""

    preference: str = "balanced"  # fast | balanced | powerful
    override_allowed: bool = True

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> ModelSpec:
        if not d:
            return ModelSpec()
        return ModelSpec(
            preference=d.get("preference", "balanced"),
            override_allowed=d.get("override_allowed", True),
        )


@dataclass(frozen=True)
class RequiresSpec:
    """What the agent needs from the platform."""

    platform_tools: bool = False
    spawn_agents: bool = False
    skills: list[str] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> RequiresSpec:
        if not d:
            return RequiresSpec()
        return RequiresSpec(
            platform_tools=d.get("platform_tools", False),
            spawn_agents=d.get("spawn_agents", False),
            skills=d.get("skills", []),
        )


@dataclass(frozen=True)
class RetrySpec:
    """Retry policy for failed jobs."""

    max_retries: int = 0  # 0 = no retry (default)
    backoff_base: float = 2.0  # seconds, doubles each attempt

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> RetrySpec:
        if not d:
            return RetrySpec()
        return RetrySpec(
            max_retries=int(d.get("max_retries", 0)),
            backoff_base=float(d.get("backoff_base", 2.0)),
        )


@dataclass(frozen=True)
class AgentContract:
    """The parsed, validated agent.yaml — the foundational primitive.

    Immutable once loaded. Represents what an agent IS, not how it's
    configured at runtime. The runtime may augment this with context,
    but never mutates it.
    """

    name: str
    version: str
    type: str = "agent"  # agent | orchestrator
    description: str = ""
    input_schema: SchemaSpec = field(default_factory=SchemaSpec)
    output_schema: SchemaSpec = field(default_factory=SchemaSpec)
    capabilities: list[str] = field(default_factory=list)
    model: ModelSpec = field(default_factory=ModelSpec)
    requires: RequiresSpec = field(default_factory=RequiresSpec)
    hardware: HardwareSpec = field(default_factory=HardwareSpec)
    retry: RetrySpec = field(default_factory=RetrySpec)
    permissions: PermissionsSpec = field(default_factory=PermissionsSpec)
    execution_timeout: float = 60.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> AgentContract:
        """Parse a raw agent.yaml dict into an AgentContract."""
        agent = d.get("agent", d)  # Support both top-level and nested under 'agent' key
        return AgentContract(
            name=agent["name"],
            version=agent["version"],
            type=agent.get("type", "agent"),
            description=agent.get("description", ""),
            input_schema=SchemaSpec.from_dict(agent.get("input", {}).get("schema")),
            output_schema=SchemaSpec.from_dict(agent.get("output", {}).get("schema")),
            capabilities=agent.get("capabilities", []),
            model=ModelSpec.from_dict(agent.get("model")),
            requires=RequiresSpec.from_dict(agent.get("requires")),
            hardware=HardwareSpec.from_dict(agent.get("hardware")),
            retry=RetrySpec.from_dict(agent.get("retry")),
            permissions=PermissionsSpec.from_dict(agent.get("permissions")),
            execution_timeout=float(agent.get("execution_timeout", 60.0)),
        )
