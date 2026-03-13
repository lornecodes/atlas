"""Tests for atlas.registry.resolver — dependency resolution."""

import pytest
from pathlib import Path

from atlas.contract.registry import AgentRegistry
from atlas.contract.types import (
    AgentContract,
    AgentDependency,
    RequiresSpec,
)
from atlas.registry.file_provider import FileRegistryProvider
from atlas.registry.package import pack
from atlas.registry.provider import PackageMetadata
from atlas.registry.resolver import DependencyResolver, _version_matches


class TestVersionMatches:
    def test_wildcard(self):
        assert _version_matches("*", "1.0.0")
        assert _version_matches("*", "99.99.99")

    def test_exact(self):
        assert _version_matches("1.2.0", "1.2.0")
        assert not _version_matches("1.2.0", "1.3.0")

    def test_gte(self):
        assert _version_matches(">=1.0.0", "1.0.0")
        assert _version_matches(">=1.0.0", "2.0.0")
        assert not _version_matches(">=2.0.0", "1.0.0")

    def test_gt(self):
        assert _version_matches(">1.0.0", "1.0.1")
        assert _version_matches(">1.0.0", "2.0.0")
        assert not _version_matches(">1.0.0", "1.0.0")


class TestAgentDependency:
    def test_from_string(self):
        dep = AgentDependency.from_dict("translator")
        assert dep.name == "translator"
        assert dep.version == "*"

    def test_from_dict_name_only(self):
        dep = AgentDependency.from_dict({"name": "translator"})
        assert dep.name == "translator"
        assert dep.version == "*"

    def test_from_dict_with_version(self):
        dep = AgentDependency.from_dict({"name": "translator", "version": ">=1.0.0"})
        assert dep.name == "translator"
        assert dep.version == ">=1.0.0"

    def test_from_dict_invalid(self):
        with pytest.raises(ValueError):
            AgentDependency.from_dict(42)


class TestRequiresSpecAgents:
    def test_empty_by_default(self):
        spec = RequiresSpec()
        assert spec.agents == []

    def test_from_dict_string_list(self):
        spec = RequiresSpec.from_dict({"agents": ["translator", "summarizer"]})
        assert len(spec.agents) == 2
        assert spec.agents[0].name == "translator"
        assert spec.agents[0].version == "*"

    def test_from_dict_detailed(self):
        spec = RequiresSpec.from_dict({
            "agents": [
                {"name": "translator", "version": ">=1.0.0"},
                "summarizer",
            ]
        })
        assert len(spec.agents) == 2
        assert spec.agents[0].version == ">=1.0.0"
        assert spec.agents[1].version == "*"


def _make_contract(name, version="1.0.0", agents=None) -> AgentContract:
    """Create a minimal contract for testing."""
    requires = RequiresSpec(
        agents=[AgentDependency.from_dict(a) for a in (agents or [])]
    )
    return AgentContract(name=name, version=version, requires=requires)


def _create_agent_dir(tmp_path, name, version="1.0.0", capabilities=None):
    """Create a valid agent directory and return path."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    caps = capabilities or []
    caps_yaml = "\n".join(f"    - {c}" for c in caps) if caps else ""
    yaml_content = f"agent:\n  name: {name}\n  version: '{version}'\n"
    if caps:
        yaml_content += f"  capabilities:\n{caps_yaml}\n"
    (d / "agent.yaml").write_text(yaml_content, encoding="utf-8")
    (d / "agent.py").write_text("# agent\n", encoding="utf-8")
    return d


class TestDependencyResolver:
    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    @pytest.fixture
    def file_reg(self, tmp_path):
        return FileRegistryProvider(tmp_path / "pkg-registry")

    @pytest.fixture
    def resolver(self, registry, file_reg):
        return DependencyResolver(registry, [file_reg])

    @pytest.mark.asyncio
    async def test_check_no_deps(self, resolver):
        contract = _make_contract("my-agent")
        missing = await resolver.check(contract)
        assert missing == []

    @pytest.mark.asyncio
    async def test_check_finds_missing(self, resolver):
        contract = _make_contract("my-agent", agents=["translator"])
        missing = await resolver.check(contract)
        assert len(missing) == 1
        assert missing[0].name == "translator"

    @pytest.mark.asyncio
    async def test_check_all_present(self, resolver, registry, tmp_path):
        # Register "translator" in the agent registry
        agent_dir = _create_agent_dir(tmp_path, "translator")
        registry.register(agent_dir / "agent.yaml")

        contract = _make_contract("my-agent", agents=["translator"])
        missing = await resolver.check(contract)
        assert missing == []

    @pytest.mark.asyncio
    async def test_check_version_mismatch(self, resolver, registry, tmp_path):
        agent_dir = _create_agent_dir(tmp_path, "translator", version="0.9.0")
        registry.register(agent_dir / "agent.yaml")

        contract = _make_contract(
            "my-agent", agents=[{"name": "translator", "version": ">=1.0.0"}]
        )
        missing = await resolver.check(contract)
        assert len(missing) == 1

    @pytest.mark.asyncio
    async def test_resolve_pulls_from_registry(
        self, resolver, registry, file_reg, tmp_path
    ):
        # Publish translator to file registry
        translator_dir = _create_agent_dir(
            tmp_path / "sources", "translator", version="1.0.0"
        )
        meta, data = pack(translator_dir)
        await file_reg.publish(meta, data)

        contract = _make_contract("my-agent", agents=["translator"])
        install_dir = tmp_path / "installed"
        install_dir.mkdir()

        installed = await resolver.resolve(contract, install_dir)
        assert installed == 1
        assert "translator" in registry

    @pytest.mark.asyncio
    async def test_resolve_version_matching(
        self, resolver, registry, file_reg, tmp_path
    ):
        # Publish two versions
        for ver in ["0.9.0", "1.2.0"]:
            d = _create_agent_dir(tmp_path / f"t-{ver}", "translator", version=ver)
            meta, data = pack(d)
            await file_reg.publish(meta, data)

        contract = _make_contract(
            "my-agent", agents=[{"name": "translator", "version": ">=1.0.0"}]
        )
        install_dir = tmp_path / "installed"
        install_dir.mkdir()

        installed = await resolver.resolve(contract, install_dir)
        assert installed == 1
        entry = registry.get("translator")
        assert entry is not None
        assert entry.contract.version == "1.2.0"  # Got the highest matching

    @pytest.mark.asyncio
    async def test_resolve_not_found(self, resolver, tmp_path):
        contract = _make_contract("my-agent", agents=["nonexistent"])
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        installed = await resolver.resolve(contract, install_dir)
        assert installed == 0

    @pytest.mark.asyncio
    async def test_resolve_skips_already_present(
        self, resolver, registry, file_reg, tmp_path
    ):
        # Already registered
        agent_dir = _create_agent_dir(tmp_path, "translator")
        registry.register(agent_dir / "agent.yaml")

        contract = _make_contract("my-agent", agents=["translator"])
        install_dir = tmp_path / "installed"
        install_dir.mkdir()

        installed = await resolver.resolve(contract, install_dir)
        assert installed == 0  # Nothing to install

    @pytest.mark.asyncio
    async def test_resolve_warns_transitive(
        self, resolver, registry, file_reg, tmp_path, caplog
    ):
        # Create agent with its own dependencies
        d = tmp_path / "sources" / "translator"
        d.mkdir(parents=True)
        (d / "agent.yaml").write_text(
            "agent:\n  name: translator\n  version: '1.0.0'\n"
            "  requires:\n    agents: [language-detector]\n",
            encoding="utf-8",
        )
        (d / "agent.py").write_text("# agent\n", encoding="utf-8")
        meta, data = pack(d)
        await file_reg.publish(meta, data)

        contract = _make_contract("my-agent", agents=["translator"])
        install_dir = tmp_path / "installed"
        install_dir.mkdir()

        installed = await resolver.resolve(contract, install_dir)
        assert installed == 1
        assert "transitive dependencies" in caplog.text.lower()
