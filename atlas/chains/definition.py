"""Chain definitions — declarative pipelines of agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ChainStep:
    """A single step in a chain."""

    agent_name: str
    input_map: dict[str, str] | None = None
    name: str = ""  # Optional name, defaults to step_{index} at runtime

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ChainStep:
        return ChainStep(
            agent_name=d["agent"],
            input_map=d.get("input_map"),
            name=d.get("name", ""),
        )


@dataclass
class ChainDefinition:
    """A declarative pipeline of agents."""

    name: str
    description: str = ""
    steps: list[ChainStep] = field(default_factory=list)
    orchestrator: str = ""  # Optional orchestrator name for per-chain routing

    def step_name(self, index: int) -> str:
        """Get the effective name for a step (explicit or default)."""
        if index < len(self.steps) and self.steps[index].name:
            return self.steps[index].name
        return f"step_{index}"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ChainDefinition:
        chain = d.get("chain", d)
        return ChainDefinition(
            name=chain["name"],
            description=chain.get("description", ""),
            steps=[ChainStep.from_dict(s) for s in chain.get("steps", [])],
            orchestrator=chain.get("orchestrator", ""),
        )

    @staticmethod
    def from_yaml(path: Path | str) -> ChainDefinition:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ChainDefinition.from_dict(raw)
