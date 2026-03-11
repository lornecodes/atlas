"""Shared fixtures for Atlas tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from atlas.contract.registry import AgentRegistry
from atlas.contract.types import SchemaSpec

AGENTS_DIR = Path(__file__).parent.parent / "agents"


@pytest.fixture
def agents_dir() -> Path:
    return AGENTS_DIR


@pytest.fixture
def registry(agents_dir: Path) -> AgentRegistry:
    """Registry pre-loaded with all test agents."""
    reg = AgentRegistry(search_paths=[agents_dir])
    reg.discover()
    return reg


def make_schema(
    props: dict[str, Any], required: list[str] | None = None
) -> SchemaSpec:
    """Create a SchemaSpec from properties dict — shared test helper.

    If required is None, defaults to all property keys (all fields required).
    Pass an empty list for no required fields.
    """
    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props.keys()),
    }
    return SchemaSpec(raw=schema)
