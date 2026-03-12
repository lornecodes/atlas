"""Tests for the Atlas skills system — types, schema, registry, resolver, context, pool."""

from __future__ import annotations

import asyncio
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from atlas.contract.registry import AgentRegistry, RegisteredAgent
from atlas.contract.types import AgentContract, RequiresSpec, SchemaSpec
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.runtime.context import AgentContext, SkillInvocationError
from atlas.skills.registry import SkillRegistry, RegisteredSkill
from atlas.skills.resolver import SkillResolver
from atlas.skills.schema import load_skill
from atlas.skills.types import SkillCallable, SkillError, SkillSpec

from conftest import AGENTS_DIR

SKILLS_DIR = Path(__file__).parent.parent / "skills"


# === SkillSpec ===

class TestSkillSpec:
    def test_from_dict_minimal(self):
        spec = SkillSpec.from_dict({"skill": {"name": "test", "version": "1.0.0"}})
        assert spec.name == "test"
        assert spec.version == "1.0.0"
        assert spec.description == ""

    def test_from_dict_full(self):
        spec = SkillSpec.from_dict({
            "skill": {
                "name": "reverse",
                "version": "1.0.0",
                "description": "Reverses a string",
                "input": {"schema": {"type": "object", "properties": {"text": {"type": "string"}}}},
                "output": {"schema": {"type": "object", "properties": {"text": {"type": "string"}}}},
            }
        })
        assert spec.name == "reverse"
        assert spec.description == "Reverses a string"
        assert spec.input_schema.properties == {"text": {"type": "string"}}

    def test_frozen(self):
        spec = SkillSpec(name="test", version="1.0.0")
        with pytest.raises(AttributeError):
            spec.name = "changed"

    def test_from_dict_flat(self):
        """Support flat format (no 'skill' wrapper)."""
        spec = SkillSpec.from_dict({"name": "flat", "version": "2.0.0"})
        assert spec.name == "flat"


# === Schema Loading ===

class TestSkillSchema:
    def test_load_valid(self, tmp_path):
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text(textwrap.dedent("""\
            skill:
              name: "test-skill"
              version: "1.0.0"
              description: "A test skill"
        """))
        spec = load_skill(skill_yaml)
        assert spec.name == "test-skill"
        assert spec.version == "1.0.0"

    def test_load_invalid_yaml(self, tmp_path):
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text("{{invalid yaml")
        with pytest.raises(SkillError, match="Invalid YAML"):
            load_skill(skill_yaml)

    def test_load_missing_name(self, tmp_path):
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text("skill:\n  version: '1.0.0'\n")
        with pytest.raises(SkillError, match="validation failed"):
            load_skill(skill_yaml)

    def test_load_missing_version(self, tmp_path):
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text("skill:\n  name: test\n")
        with pytest.raises(SkillError, match="validation failed"):
            load_skill(skill_yaml)

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(SkillError, match="not found"):
            load_skill(tmp_path / "nonexistent.yaml")

    def test_load_with_schemas(self, tmp_path):
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text(textwrap.dedent("""\
            skill:
              name: "schema-skill"
              version: "1.0.0"
              input:
                schema:
                  type: object
                  properties:
                    x:
                      type: integer
                  required: [x]
              output:
                schema:
                  type: object
                  properties:
                    y:
                      type: integer
        """))
        spec = load_skill(skill_yaml)
        assert spec.input_schema.required == ["x"]
        assert "y" in spec.output_schema.properties

    def test_load_invalid_version_format(self, tmp_path):
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text("skill:\n  name: test\n  version: 'bad'\n")
        with pytest.raises(SkillError, match="validation failed"):
            load_skill(skill_yaml)

    def test_load_non_mapping(self, tmp_path):
        skill_yaml = tmp_path / "skill.yaml"
        skill_yaml.write_text("- list\n- item\n")
        with pytest.raises(SkillError, match="YAML mapping"):
            load_skill(skill_yaml)


# === Skill Registry ===

class TestSkillRegistry:
    def test_register_from_yaml(self):
        registry = SkillRegistry()
        spec = registry.register(SKILLS_DIR / "reverse" / "skill.yaml")
        assert spec.name == "reverse"
        assert "reverse" in registry

    def test_discover(self):
        registry = SkillRegistry(search_paths=[SKILLS_DIR])
        count = registry.discover()
        assert count >= 1
        assert "reverse" in registry

    def test_get_existing(self):
        registry = SkillRegistry(search_paths=[SKILLS_DIR])
        registry.discover()
        entry = registry.get("reverse")
        assert entry is not None
        assert entry.spec.name == "reverse"

    def test_get_missing_returns_none(self):
        registry = SkillRegistry()
        assert registry.get("nonexistent") is None

    def test_register_callable(self):
        registry = SkillRegistry()
        spec = SkillSpec(name="custom", version="1.0.0")

        async def my_skill(input_data: dict) -> dict:
            return {"result": "ok"}

        registry.register_callable(spec, my_skill)
        assert "custom" in registry
        entry = registry.get("custom")
        assert entry.callable is my_skill

    def test_list_all(self):
        registry = SkillRegistry(search_paths=[SKILLS_DIR])
        registry.discover()
        skills = registry.list_all()
        assert len(skills) >= 1
        names = [s.spec.name for s in skills]
        assert "reverse" in names

    def test_contains(self):
        registry = SkillRegistry()
        assert "foo" not in registry

    def test_len(self):
        registry = SkillRegistry()
        assert len(registry) == 0
        spec = SkillSpec(name="a", version="1.0.0")
        registry.register_callable(spec, AsyncMock(return_value={}))
        assert len(registry) == 1

    def test_discover_skips_invalid(self, tmp_path):
        # Create a bad skill.yaml
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "skill.yaml").write_text("skill:\n  name: bad\n")  # missing version
        registry = SkillRegistry(search_paths=[tmp_path])
        count = registry.discover()
        assert count == 0

    def test_discover_nonexistent_path(self, tmp_path):
        registry = SkillRegistry(search_paths=[tmp_path / "does_not_exist"])
        count = registry.discover()
        assert count == 0


# === Skill Callable Loading ===

class TestSkillCallable:
    def test_lazy_load_reverse(self):
        registry = SkillRegistry(search_paths=[SKILLS_DIR])
        registry.discover()
        entry = registry.get("reverse")
        fn = entry.callable
        assert fn is not None
        assert callable(fn)

    @pytest.mark.asyncio
    async def test_execute_reverse(self):
        registry = SkillRegistry(search_paths=[SKILLS_DIR])
        registry.discover()
        entry = registry.get("reverse")
        result = await entry.callable({"text": "hello"})
        assert result == {"text": "olleh"}


# === Skill Resolver ===

class TestSkillResolver:
    @pytest.mark.asyncio
    async def test_resolve_all(self):
        registry = SkillRegistry(search_paths=[SKILLS_DIR])
        registry.discover()
        resolver = SkillResolver(registry)
        result = await resolver.resolve(["reverse"])
        assert "reverse" in result
        assert callable(result["reverse"])

    @pytest.mark.asyncio
    async def test_resolve_empty_list(self):
        registry = SkillRegistry()
        resolver = SkillResolver(registry)
        result = await resolver.resolve([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_skill_not_found_in_registry(self):
        registry = SkillRegistry()
        resolver = SkillResolver(registry)
        with pytest.raises(SkillError, match="not found"):
            await resolver.resolve(["nonexistent"])

    @pytest.mark.asyncio
    async def test_skill_no_callable(self, tmp_path):
        """Skill with yaml but no skill.py should fail."""
        skill_dir = tmp_path / "nocode"
        skill_dir.mkdir()
        (skill_dir / "skill.yaml").write_text(
            "skill:\n  name: nocode\n  version: '1.0.0'\n"
        )
        registry = SkillRegistry(search_paths=[tmp_path])
        registry.discover()
        resolver = SkillResolver(registry)
        with pytest.raises(SkillError, match="no implementation"):
            await resolver.resolve(["nocode"])


# === Context skill() Method ===

class TestContextSkill:
    @pytest.mark.asyncio
    async def test_invoke_skill(self):
        async def reverse(input_data: dict) -> dict:
            return {"text": input_data["text"][::-1]}

        ctx = AgentContext()
        ctx._skills = {"reverse": reverse}
        result = await ctx.skill("reverse", {"text": "hello"})
        assert result == {"text": "olleh"}

    @pytest.mark.asyncio
    async def test_invoke_unknown_skill_raises(self):
        ctx = AgentContext()
        with pytest.raises(SkillInvocationError, match="not available"):
            await ctx.skill("nonexistent", {})

    @pytest.mark.asyncio
    async def test_no_skills_configured(self):
        ctx = AgentContext()
        assert ctx._skills == {}
        with pytest.raises(SkillInvocationError):
            await ctx.skill("anything", {})

    @pytest.mark.asyncio
    async def test_skill_error_propagates(self):
        """If a skill callable raises, the exception propagates."""
        async def broken(input_data: dict) -> dict:
            raise ValueError("skill broke")

        ctx = AgentContext()
        ctx._skills = {"broken": broken}
        with pytest.raises(ValueError, match="skill broke"):
            await ctx.skill("broken", {})


# === Pool Skill Wiring ===

@pytest.fixture
def skill_registry():
    registry = SkillRegistry(search_paths=[SKILLS_DIR])
    registry.discover()
    return registry


class TestPoolSkillWiring:
    @pytest.mark.asyncio
    async def test_pool_injects_skills_into_context(self, registry, skill_registry):
        """Agent requiring skills gets them injected into context."""
        queue = JobQueue()
        resolver = SkillResolver(skill_registry)

        # Mock the echo agent to require the reverse skill
        original_get = registry.get

        def mock_get(name, version=None):
            entry = original_get(name, version)
            if entry and name == "echo":
                contract = AgentContract(
                    name=entry.contract.name,
                    version=entry.contract.version,
                    requires=RequiresSpec(skills=["reverse"]),
                    input_schema=entry.contract.input_schema,
                    output_schema=entry.contract.output_schema,
                )
                return RegisteredAgent(
                    contract=contract,
                    source_path=entry.source_path,
                    module_path=entry.module_path,
                    _agent_class=entry.agent_class,
                )
            return entry

        registry.get = mock_get

        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=1,
            skill_resolver=resolver,
        )
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hello"})
            await pool.submit(job)
            await asyncio.sleep(0.3)
            result = queue.get(job.id)
            assert result.status == "completed"
        finally:
            await pool.stop()
            registry.get = original_get

    @pytest.mark.asyncio
    async def test_pool_fails_job_when_skill_missing(self, registry):
        """Agent requiring a skill not in registry should fail."""
        queue = JobQueue()
        empty_skill_reg = SkillRegistry()
        resolver = SkillResolver(empty_skill_reg)

        original_get = registry.get

        def mock_get(name, version=None):
            entry = original_get(name, version)
            if entry and name == "echo":
                contract = AgentContract(
                    name=entry.contract.name,
                    version=entry.contract.version,
                    requires=RequiresSpec(skills=["nonexistent"]),
                    input_schema=entry.contract.input_schema,
                    output_schema=entry.contract.output_schema,
                )
                return RegisteredAgent(
                    contract=contract,
                    source_path=entry.source_path,
                    module_path=entry.module_path,
                    _agent_class=entry.agent_class,
                )
            return entry

        registry.get = mock_get

        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=1,
            skill_resolver=resolver,
        )
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hello"})
            await pool.submit(job)
            await asyncio.sleep(0.3)
            result = queue.get(job.id)
            assert result.status == "failed"
            assert "Skill resolution failed" in result.error
        finally:
            await pool.stop()
            registry.get = original_get

    @pytest.mark.asyncio
    async def test_pool_fails_job_when_no_resolver(self, registry):
        """Agent requiring skills without a resolver configured should fail."""
        queue = JobQueue()

        original_get = registry.get

        def mock_get(name, version=None):
            entry = original_get(name, version)
            if entry and name == "echo":
                contract = AgentContract(
                    name=entry.contract.name,
                    version=entry.contract.version,
                    requires=RequiresSpec(skills=["reverse"]),
                    input_schema=entry.contract.input_schema,
                    output_schema=entry.contract.output_schema,
                )
                return RegisteredAgent(
                    contract=contract,
                    source_path=entry.source_path,
                    module_path=entry.module_path,
                    _agent_class=entry.agent_class,
                )
            return entry

        registry.get = mock_get

        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=1,
            # No skill_resolver!
        )
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hello"})
            await pool.submit(job)
            await asyncio.sleep(0.3)
            result = queue.get(job.id)
            assert result.status == "failed"
            assert "no SkillResolver configured" in result.error
        finally:
            await pool.stop()
            registry.get = original_get

    @pytest.mark.asyncio
    async def test_pool_without_skills_works_normally(self, registry):
        """Pool without skill requirements should work as before."""
        queue = JobQueue()
        pool = ExecutionPool(
            registry, queue, max_concurrent=1, warm_pool_size=1,
        )
        await pool.start()
        try:
            job = JobData(agent_name="echo", input_data={"message": "hello"})
            await pool.submit(job)
            await asyncio.sleep(0.3)
            result = queue.get(job.id)
            assert result.status == "completed"
        finally:
            await pool.stop()
