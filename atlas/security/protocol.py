"""Container stdin/stdout JSON protocol types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContainerMessage:
    """Message sent to the container via stdin."""

    input: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"input": self.input, "context": self.context})


@dataclass
class ContainerResponse:
    """Response read from the container's stdout."""

    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    @staticmethod
    def from_json(raw: str) -> ContainerResponse:
        """Parse a JSON line from container stdout."""
        data = json.loads(raw)
        return ContainerResponse(
            output=data.get("output", {}),
            error=data.get("error", ""),
        )
