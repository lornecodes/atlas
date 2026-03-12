"""YAML loading and validation for skill definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from atlas.skills.types import SkillError, SkillSpec

# Meta-schema: validates the structure of skill.yaml itself
_SKILL_META_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
                "description": {"type": "string"},
                "input": {
                    "type": "object",
                    "properties": {"schema": {"type": "object"}},
                },
                "output": {
                    "type": "object",
                    "properties": {"schema": {"type": "object"}},
                },
            },
        },
    },
    "required": ["skill"],
}

_meta_validator = Draft202012Validator(_SKILL_META_SCHEMA)


def load_skill(path: Path | str) -> SkillSpec:
    """Load and validate a skill.yaml file.

    Raises SkillError if the file is missing, unparseable, or invalid.
    """
    path = Path(path)
    if not path.exists():
        raise SkillError(f"Skill file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SkillError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise SkillError(f"Skill must be a YAML mapping, got {type(raw).__name__}")

    # If top-level keys don't include 'skill', wrap it
    if "skill" not in raw and "name" in raw:
        raw = {"skill": raw}

    # Validate against meta-schema
    errors = list(_meta_validator.iter_errors(raw))
    if errors:
        msgs = [e.message for e in errors]
        raise SkillError(f"Skill validation failed: {'; '.join(msgs)}")

    return SkillSpec.from_dict(raw)
