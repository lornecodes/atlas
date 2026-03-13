"""E2E integration tests — skill registry, resolver, and schema wired together.

These tests exercise the full skill stack: discover skills from directories,
register and resolve them, lazy-load callables, and handle error cases.
No mocking — real skill.yaml and skill.py files created in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.skills.registry import RegisteredSkill, SkillRegistry
from atlas.skills.resolver import SkillResolver
from atlas.skills.schema import load_skill
from atlas.skills.types import SkillCallable, SkillError, SkillSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SKILL_YAML_TEMPLATE = """\
skill:
  name: {name}
  version: "{version}"
  description: "{description}"
  input:
    schema:
      type: object
      properties:
        message:
          type: string
  output:
    schema:
      type: object
      properties:
        result:
          type: string
"""

SKILL_PY_TEMPLATE = """\
async def execute(input_data: dict) -> dict:
    return {"result": input_data.get("message", "default")}
"""


def _create_skill(
    base: Path,
    name: str,
    *,
    version: str = "1.0.0",
    description: str = "Test skill",
    include_py: bool = True,
    yaml_content: str | None = None,
) -> Path:
    """Create a skill directory with skill.yaml and optionally skill.py.

    Returns the path to skill.yaml.
    """
    skill_dir = base / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = skill_dir / "skill.yaml"
    if yaml_content is not None:
        yaml_path.write_text(yaml_content, encoding="utf-8")
    else:
        yaml_path.write_text(
            SKILL_YAML_TEMPLATE.format(
                name=name, version=version, description=description,
            ),
            encoding="utf-8",
        )

    if include_py:
        py_path = skill_dir / "skill.py"
        py_path.write_text(SKILL_PY_TEMPLATE, encoding="utf-8")

    return yaml_path


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Root directory for skill discovery."""
    return tmp_path / "skills"


@pytest.fixture
def registry(skill_dir: Path) -> SkillRegistry:
    """A SkillRegistry with a single search path."""
    return SkillRegistry(search_paths=[skill_dir])


@pytest.fixture
def resolver(registry: SkillRegistry) -> SkillResolver:
    return SkillResolver(registry)


# ---------------------------------------------------------------------------
# Discovery & registration
# ---------------------------------------------------------------------------


class TestSkillDiscovery:
    """Discover and register skills from the filesystem."""

    def test_discover_skills_from_directory(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """Create two skills, discover() finds both."""
        _create_skill(skill_dir, "alpha")
        _create_skill(skill_dir, "beta", version="2.0.0")

        count = registry.discover()

        assert count == 2
        assert "alpha" in registry
        assert "beta" in registry
        assert len(registry) == 2

    def test_register_single_skill(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """register(path) returns a SkillSpec with correct metadata."""
        yaml_path = _create_skill(skill_dir, "gamma", version="3.1.0")

        spec = registry.register(yaml_path)

        assert spec.name == "gamma"
        assert spec.version == "3.1.0"
        assert registry.get("gamma") is not None
        assert registry.get("gamma").spec is spec

    def test_register_callable_programmatic(self, registry: SkillRegistry):
        """register_callable() makes a skill immediately available."""
        spec = SkillSpec(
            name="inline-skill",
            version="0.1.0",
            description="Registered via code",
        )

        async def my_fn(input_data: dict) -> dict:
            return {"ok": True}

        registry.register_callable(spec, my_fn)

        assert "inline-skill" in registry
        entry = registry.get("inline-skill")
        assert entry is not None
        assert entry.callable is my_fn

    def test_skill_lazy_loading(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """After register(path), _callable is None until first .callable access."""
        yaml_path = _create_skill(skill_dir, "lazy-one")

        registry.register(yaml_path)
        entry = registry.get("lazy-one")

        # Internal _callable not yet loaded
        assert entry._callable is None

        # Accessing .callable triggers lazy load
        fn = entry.callable
        assert fn is not None
        assert callable(fn)

        # Subsequent access returns the same object (cached)
        assert entry.callable is fn


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestSkillExecution:
    """Register a skill with a callable and invoke it."""

    @pytest.mark.asyncio
    async def test_skill_execution(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """Register a real skill, invoke its callable, verify output."""
        _create_skill(skill_dir, "echo-skill")
        registry.discover()

        entry = registry.get("echo-skill")
        fn = entry.callable
        assert fn is not None

        result = await fn({"message": "hello world"})
        assert result == {"result": "hello world"}

    @pytest.mark.asyncio
    async def test_skill_execution_default(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """Callable returns default when message key is absent."""
        _create_skill(skill_dir, "echo-default")
        registry.discover()

        fn = registry.get("echo-default").callable
        result = await fn({})
        assert result == {"result": "default"}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestSkillErrorHandling:
    """Graceful handling of invalid or incomplete skills."""

    def test_invalid_yaml_graceful(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """Garbage YAML content is skipped, discover() returns 0."""
        _create_skill(
            skill_dir,
            "broken",
            yaml_content=":::not valid yaml at all [[[",
        )

        count = registry.discover()

        assert count == 0
        assert "broken" not in registry
        assert len(registry) == 0

    def test_missing_skill_py(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """skill.yaml present but no skill.py means callable is None."""
        _create_skill(skill_dir, "no-impl", include_py=False)
        registry.discover()

        entry = registry.get("no-impl")
        assert entry is not None
        assert entry.module_path is None
        assert entry.callable is None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class TestSkillResolver:
    """SkillResolver wires registry lookups to callables."""

    @pytest.mark.asyncio
    async def test_skill_resolver_resolves(
        self,
        registry: SkillRegistry,
        resolver: SkillResolver,
        skill_dir: Path,
    ):
        """resolve() returns a dict mapping name to callable."""
        _create_skill(skill_dir, "resolvable")
        registry.discover()

        result = await resolver.resolve(["resolvable"])

        assert "resolvable" in result
        assert callable(result["resolvable"])

        # The resolved callable should work
        output = await result["resolvable"]({"message": "ping"})
        assert output == {"result": "ping"}

    @pytest.mark.asyncio
    async def test_skill_resolver_missing(self, resolver: SkillResolver):
        """resolve() raises SkillError for a skill not in the registry."""
        with pytest.raises(SkillError, match="not found"):
            await resolver.resolve(["nonexistent"])

    @pytest.mark.asyncio
    async def test_skill_resolver_no_callable(
        self,
        registry: SkillRegistry,
        resolver: SkillResolver,
        skill_dir: Path,
    ):
        """Skill registered without callable raises SkillError on resolve."""
        _create_skill(skill_dir, "no-code", include_py=False)
        registry.discover()

        with pytest.raises(SkillError, match="no implementation"):
            await resolver.resolve(["no-code"])


# ---------------------------------------------------------------------------
# Container protocol (__contains__, __len__, list_all)
# ---------------------------------------------------------------------------


class TestSkillContainerProtocol:
    """Registry supports __contains__, __len__, and list_all()."""

    def test_skill_contains_and_len(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """'name' in registry and len(registry) work correctly."""
        assert len(registry) == 0
        assert "delta" not in registry

        _create_skill(skill_dir, "delta")
        _create_skill(skill_dir, "epsilon")
        registry.discover()

        assert "delta" in registry
        assert "epsilon" in registry
        assert "zeta" not in registry
        assert len(registry) == 2

    def test_skill_list_all(
        self, registry: SkillRegistry, skill_dir: Path,
    ):
        """list_all() returns all registered skills as RegisteredSkill instances."""
        _create_skill(skill_dir, "first")
        _create_skill(skill_dir, "second")
        _create_skill(skill_dir, "third")
        registry.discover()

        all_skills = registry.list_all()

        assert len(all_skills) == 3
        assert all(isinstance(s, RegisteredSkill) for s in all_skills)
        names = {s.spec.name for s in all_skills}
        assert names == {"first", "second", "third"}
