"""Skill resolver — resolve skill names to callables for injection."""

from __future__ import annotations

from atlas.skills.registry import SkillRegistry
from atlas.skills.types import SkillCallable, SkillError


class SkillResolver:
    """Resolves a list of skill names to callable wrappers.

    Given an agent's requires.skills list, looks up each skill in the
    registry and returns a dict of name → callable. Raises SkillError
    if any requested skill is not found or has no implementation.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> SkillRegistry:
        """The underlying skill registry."""
        return self._registry

    async def resolve(self, skill_names: list[str]) -> dict[str, SkillCallable]:
        """Resolve skill names to callables.

        Returns a dict of {name: callable} for all resolved skills.
        Raises SkillError if a skill is not found or has no callable.
        """
        result: dict[str, SkillCallable] = {}
        for name in skill_names:
            entry = self._registry.get(name)
            if entry is None:
                raise SkillError(f"Skill '{name}' not found in registry")
            fn = entry.callable
            if fn is None:
                raise SkillError(
                    f"Skill '{name}' has no implementation (missing skill.py)"
                )
            result[name] = fn
        return result
