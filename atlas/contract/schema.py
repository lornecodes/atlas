"""YAML loading and JSON Schema validation for agent contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, ValidationError

from atlas.contract.types import AgentContract, SchemaSpec


# Meta-schema: validates the structure of agent.yaml itself
_CONTRACT_META_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "agent": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                "type": {"type": "string", "enum": ["agent", "orchestrator"]},
                "description": {"type": "string"},
                "input": {
                    "type": "object",
                    "properties": {"schema": {"type": "object"}},
                },
                "output": {
                    "type": "object",
                    "properties": {"schema": {"type": "object"}},
                },
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "model": {
                    "type": "object",
                    "properties": {
                        "preference": {
                            "type": "string",
                            "enum": ["fast", "balanced", "powerful"],
                        },
                        "override_allowed": {"type": "boolean"},
                    },
                },
                "requires": {
                    "type": "object",
                    "properties": {
                        "platform_tools": {"type": "boolean"},
                        "spawn_agents": {"type": "boolean"},
                        "skills": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "hardware": {"type": "object"},
            },
        },
    },
    "required": ["agent"],
}

_meta_validator = Draft202012Validator(_CONTRACT_META_SCHEMA)


class ContractError(Exception):
    """Raised when an agent contract is invalid."""


def load_contract(path: Path | str) -> AgentContract:
    """Load and validate an agent.yaml file.

    Raises ContractError if the file is missing, unparseable, or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise ContractError(f"Contract file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ContractError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ContractError(f"Contract must be a YAML mapping, got {type(raw).__name__}")

    # If top-level keys don't include 'agent', wrap it
    if "agent" not in raw and "name" in raw:
        raw = {"agent": raw}

    # Validate against meta-schema
    errors = list(_meta_validator.iter_errors(raw))
    if errors:
        msgs = [e.message for e in errors]
        raise ContractError(f"Contract validation failed: {'; '.join(msgs)}")

    return AgentContract.from_dict(raw)


def validate_input(contract: AgentContract, data: dict) -> list[str]:
    """Validate input data against the contract's input schema.

    Returns a list of error messages (empty if valid).
    """
    return _validate_against_schema(contract.input_schema, data)


def validate_output(contract: AgentContract, data: dict) -> list[str]:
    """Validate output data against the contract's output schema.

    Returns a list of error messages (empty if valid).
    """
    return _validate_against_schema(contract.output_schema, data)


def _validate_against_schema(schema: SchemaSpec, data: Any) -> list[str]:
    """Validate data against a SchemaSpec. Returns error messages."""
    json_schema = schema.to_json_schema()
    # Empty/trivial schemas accept anything
    if json_schema == {"type": "object"}:
        return []

    validator = Draft202012Validator(json_schema)
    return [err.message for err in validator.iter_errors(data)]
