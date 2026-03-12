"""Core types for the Atlas skills system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from atlas.contract.types import SchemaSpec

# The skill function type — async callable with dict in/out
SkillCallable = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class SkillError(Exception):
    """Raised when a skill cannot be resolved or loaded."""


@dataclass(frozen=True)
class SkillSpec:
    """A skill definition loaded from skill.yaml.

    Skills are simpler than agents — no lifecycle, no warm pool.
    A skill is a named async callable with input/output schemas.
    """

    name: str
    version: str
    description: str = ""
    input_schema: SchemaSpec = field(default_factory=SchemaSpec)
    output_schema: SchemaSpec = field(default_factory=SchemaSpec)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> SkillSpec:
        """Parse a raw skill.yaml dict into a SkillSpec."""
        skill = d.get("skill", d)
        return SkillSpec(
            name=skill["name"],
            version=skill["version"],
            description=skill.get("description", ""),
            input_schema=SchemaSpec.from_dict(skill.get("input", {}).get("schema")),
            output_schema=SchemaSpec.from_dict(skill.get("output", {}).get("schema")),
        )
