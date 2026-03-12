"""Atlas skills — declare, discover, inject, invoke."""

from atlas.skills.platform import PlatformToolProvider
from atlas.skills.registry import RegisteredSkill, SkillRegistry
from atlas.skills.resolver import SkillResolver
from atlas.skills.types import SkillCallable, SkillError, SkillSpec

__all__ = [
    "SkillCallable",
    "SkillError",
    "SkillSpec",
    "SkillRegistry",
    "RegisteredSkill",
    "SkillResolver",
    "PlatformToolProvider",
]
