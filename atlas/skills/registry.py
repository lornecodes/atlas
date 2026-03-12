"""Skill registry — discover, register, and query skills."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.logging import get_logger
from atlas.skills.schema import SkillError, load_skill
from atlas.skills.types import SkillCallable, SkillSpec

logger = get_logger(__name__)


@dataclass
class RegisteredSkill:
    """A skill known to the registry."""

    spec: SkillSpec
    source_path: Path | None = None
    module_path: Path | None = None
    _callable: SkillCallable | None = field(default=None, repr=False)

    @property
    def callable(self) -> SkillCallable | None:
        """Lazy-load the skill's execute function from skill.py."""
        if self._callable is not None:
            return self._callable
        if self.module_path and self.module_path.exists():
            self._callable = _load_skill_callable(self.module_path)
        return self._callable


class SkillRegistry:
    """Discover, register, and query skills.

    Skills are discovered by scanning directories for skill.yaml files.
    Each skill.yaml sits next to a skill.py that exports an async execute function.
    """

    def __init__(self, search_paths: list[Path | str] | None = None) -> None:
        self._skills: dict[str, RegisteredSkill] = {}
        self._search_paths: list[Path] = [Path(p) for p in (search_paths or [])]

    def register(self, path: Path | str) -> SkillSpec:
        """Register a single skill from its skill.yaml path."""
        path = Path(path)
        spec = load_skill(path)
        skill_dir = path.parent
        module_path = skill_dir / "skill.py"

        entry = RegisteredSkill(
            spec=spec,
            source_path=path,
            module_path=module_path if module_path.exists() else None,
        )
        self._skills[spec.name] = entry
        return spec

    def register_callable(self, spec: SkillSpec, fn: SkillCallable) -> None:
        """Register a skill programmatically with a callable.

        Used by platform tools, remote federation, and tests.
        """
        self._skills[spec.name] = RegisteredSkill(
            spec=spec,
            _callable=fn,
        )

    def discover(self) -> int:
        """Scan search_paths for skill.yaml files, register all found.

        Returns the number of skills discovered.
        """
        count = 0
        for search_path in self._search_paths:
            search_path = Path(search_path)
            if not search_path.is_dir():
                logger.debug("Search path does not exist: %s", search_path)
                continue
            for yaml_path in search_path.rglob("skill.yaml"):
                try:
                    self.register(yaml_path)
                    count += 1
                except SkillError as e:
                    logger.warning("Skipping invalid skill %s: %s", yaml_path, e)
        logger.info("Discovered %d skills from %d search paths", count, len(self._search_paths))
        return count

    def get(self, name: str) -> RegisteredSkill | None:
        """Get a registered skill by name."""
        return self._skills.get(name)

    def list_all(self) -> list[RegisteredSkill]:
        """List all registered skills."""
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


def _load_skill_callable(module_path: Path) -> SkillCallable | None:
    """Dynamically load a skill's execute function from a Python module.

    Looks for a function with __skill__ = True, or falls back to 'execute'.
    """
    spec = importlib.util.spec_from_file_location(
        f"atlas.skills.{module_path.parent.name}",
        module_path,
    )
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None

    # Prefer function with __skill__ = True
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if callable(attr) and getattr(attr, "__skill__", False):
            return attr

    # Fall back to 'execute'
    execute = getattr(module, "execute", None)
    if callable(execute):
        return execute

    return None
