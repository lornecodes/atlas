"""Tests for agent registry discovery and querying."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atlas.contract.registry import AgentRegistry

from conftest import AGENTS_DIR


class TestRegistryDiscovery:
    def test_discover_finds_all_agents(self, registry: AgentRegistry):
        assert len(registry) >= 5  # echo, summarizer, translator, formatter, slow-starter

    def test_discover_returns_count(self, agents_dir: Path):
        reg = AgentRegistry(search_paths=[agents_dir])
        count = reg.discover()
        assert count >= 5

    def test_discover_skips_invalid(self, tmp_path, agents_dir: Path):
        # Create an invalid agent.yaml alongside valid ones
        bad_dir = tmp_path / "bad_agent"
        bad_dir.mkdir()
        (bad_dir / "agent.yaml").write_text("not: valid: agent")

        reg = AgentRegistry(search_paths=[agents_dir, tmp_path])
        count = reg.discover()
        assert count >= 5  # Valid ones still found

    def test_discover_empty_dir(self, tmp_path):
        reg = AgentRegistry(search_paths=[tmp_path])
        assert reg.discover() == 0

    def test_discover_nonexistent_dir(self):
        reg = AgentRegistry(search_paths=[Path("/nonexistent")])
        assert reg.discover() == 0


class TestRegistryLookup:
    def test_get_by_name(self, registry: AgentRegistry):
        agent = registry.get("echo")
        assert agent is not None
        assert agent.contract.name == "echo"

    def test_get_missing(self, registry: AgentRegistry):
        assert registry.get("nonexistent") is None

    def test_contains(self, registry: AgentRegistry):
        assert "echo" in registry
        assert "nonexistent" not in registry

    def test_list_all(self, registry: AgentRegistry):
        agents = registry.list_all()
        names = {a.contract.name for a in agents}
        assert "echo" in names
        assert "summarizer" in names


class TestCapabilitySearch:
    def test_search_exact(self, registry: AgentRegistry):
        results = registry.search("summarization")
        assert len(results) >= 1
        names = [r.contract.name for r in results]
        assert "summarizer" in names

    def test_search_shared_capability(self, registry: AgentRegistry):
        results = registry.search("text-processing")
        names = {r.contract.name for r in results}
        assert "summarizer" in names
        assert "translator" in names
        assert "formatter" in names

    def test_search_no_match(self, registry: AgentRegistry):
        assert registry.search("quantum-entanglement") == []


class TestVersioning:
    def test_multiple_versions(self, tmp_path):
        for ver in ["1.0.0", "1.1.0", "2.0.0"]:
            d = tmp_path / f"agent-{ver}"
            d.mkdir()
            (d / "agent.yaml").write_text(yaml.dump({
                "agent": {"name": "multi", "version": ver}
            }))

        reg = AgentRegistry(search_paths=[tmp_path])
        reg.discover()

        # Default gets latest
        latest = reg.get("multi")
        assert latest is not None
        assert latest.contract.version == "2.0.0"

        # Specific version
        v1 = reg.get("multi", "1.0.0")
        assert v1 is not None
        assert v1.contract.version == "1.0.0"

    def test_missing_version(self, registry: AgentRegistry):
        assert registry.get("echo", "99.99.99") is None


class TestAgentClassLoading:
    def test_load_echo_class(self, registry: AgentRegistry):
        agent = registry.get("echo")
        assert agent is not None
        cls = agent.agent_class
        assert cls is not None
        assert cls.__name__ == "EchoAgent"

    def test_load_summarizer_class(self, registry: AgentRegistry):
        agent = registry.get("summarizer")
        assert agent is not None
        cls = agent.agent_class
        assert cls is not None
        assert cls.__name__ == "SummarizerAgent"
