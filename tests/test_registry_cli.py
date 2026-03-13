"""Tests for CLI registry commands and RegistryConfig."""

import asyncio
import json
import pytest
from pathlib import Path

from atlas.registry.config import RegistryConfig, RegistryEntry, _expand_env
from atlas.registry.file_provider import FileRegistryProvider


class TestExpandEnv:
    def test_expand_existing(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        assert _expand_env("${MY_TOKEN}") == "secret123"

    def test_expand_missing(self):
        assert _expand_env("${DEFINITELY_NOT_SET_XYZ}") == ""

    def test_no_expansion(self):
        assert _expand_env("plain-text") == "plain-text"


class TestRegistryEntry:
    def test_to_dict_file(self):
        entry = RegistryEntry(name="local", type="file", path="./registry")
        d = entry.to_dict()
        assert d["name"] == "local"
        assert d["type"] == "file"
        assert d["path"] == "./registry"
        assert "url" not in d
        assert "auth_token" not in d

    def test_to_dict_http(self):
        entry = RegistryEntry(
            name="team", type="http",
            url="https://example.com", auth_token="${TOKEN}"
        )
        d = entry.to_dict()
        assert d["url"] == "https://example.com"
        assert d["auth_token"] == "${TOKEN}"

    def test_from_dict(self):
        entry = RegistryEntry.from_dict({
            "name": "test", "type": "http", "url": "http://x.com"
        })
        assert entry.name == "test"
        assert entry.type == "http"
        assert entry.url == "http://x.com"


class TestRegistryConfig:
    @pytest.fixture
    def config_path(self, tmp_path):
        return tmp_path / "registries.yaml"

    def test_add_and_list(self, config_path):
        cfg = RegistryConfig(config_path)
        cfg.add_registry("local", "file", path="./reg")
        entries = cfg.list_registries()
        assert len(entries) == 1
        assert entries[0]["name"] == "local"

    def test_add_replaces_same_name(self, config_path):
        cfg = RegistryConfig(config_path)
        cfg.add_registry("local", "file", path="./old")
        cfg.add_registry("local", "file", path="./new")
        entries = cfg.list_registries()
        assert len(entries) == 1
        assert entries[0]["path"] == "./new"

    def test_remove(self, config_path):
        cfg = RegistryConfig(config_path)
        cfg.add_registry("local", "file", path="./reg")
        assert cfg.remove_registry("local")
        assert cfg.list_registries() == []

    def test_remove_nonexistent(self, config_path):
        cfg = RegistryConfig(config_path)
        assert not cfg.remove_registry("missing")

    def test_save_and_reload(self, config_path):
        cfg = RegistryConfig(config_path)
        cfg.add_registry("local", "file", path="./reg")
        cfg.add_registry("team", "http", url="https://x.com", auth_token="t")
        cfg.save()

        cfg2 = RegistryConfig(config_path)
        entries = cfg2.list_registries()
        assert len(entries) == 2
        names = {e["name"] for e in entries}
        assert "local" in names
        assert "team" in names

    def test_get_provider_file(self, config_path, tmp_path):
        cfg = RegistryConfig(config_path)
        reg_dir = str(tmp_path / "file-reg")
        cfg.add_registry("local", "file", path=reg_dir)
        provider = cfg.get_provider("local")
        assert provider is not None
        assert isinstance(provider, FileRegistryProvider)

    def test_get_provider_http(self, config_path):
        from atlas.registry.http_provider import HttpRegistryProvider
        cfg = RegistryConfig(config_path)
        cfg.add_registry("team", "http", url="https://x.com")
        provider = cfg.get_provider("team")
        assert provider is not None
        assert isinstance(provider, HttpRegistryProvider)

    def test_get_provider_not_found(self, config_path):
        cfg = RegistryConfig(config_path)
        assert cfg.get_provider("missing") is None

    def test_get_all_providers(self, config_path, tmp_path):
        cfg = RegistryConfig(config_path)
        cfg.add_registry("a", "file", path=str(tmp_path / "a"))
        cfg.add_registry("b", "file", path=str(tmp_path / "b"))
        providers = cfg.get_all_providers()
        assert len(providers) == 2

    def test_load_corrupt_yaml(self, config_path):
        config_path.write_text("not: [valid: yaml: here", encoding="utf-8")
        cfg = RegistryConfig(config_path)
        assert cfg.list_registries() == []

    def test_load_empty_file(self, config_path):
        config_path.write_text("", encoding="utf-8")
        cfg = RegistryConfig(config_path)
        assert cfg.list_registries() == []


class TestContractAgentDependencies:
    """Test that AgentDependency integrates correctly with contract schema."""

    def test_schema_accepts_string_agents(self, tmp_path):
        from atlas.contract.schema import load_contract
        yaml = tmp_path / "agent.yaml"
        yaml.write_text(
            "agent:\n  name: test\n  version: '1.0.0'\n"
            "  requires:\n    agents:\n      - translator\n      - summarizer\n",
            encoding="utf-8",
        )
        contract = load_contract(yaml)
        assert len(contract.requires.agents) == 2
        assert contract.requires.agents[0].name == "translator"

    def test_schema_accepts_object_agents(self, tmp_path):
        from atlas.contract.schema import load_contract
        yaml = tmp_path / "agent.yaml"
        yaml.write_text(
            "agent:\n  name: test\n  version: '1.0.0'\n"
            "  requires:\n    agents:\n"
            "      - name: translator\n        version: '>=1.0.0'\n"
            "      - name: summarizer\n",
            encoding="utf-8",
        )
        contract = load_contract(yaml)
        assert len(contract.requires.agents) == 2
        assert contract.requires.agents[0].version == ">=1.0.0"
        assert contract.requires.agents[1].version == "*"

    def test_schema_no_agents_field(self, tmp_path):
        from atlas.contract.schema import load_contract
        yaml = tmp_path / "agent.yaml"
        yaml.write_text(
            "agent:\n  name: test\n  version: '1.0.0'\n",
            encoding="utf-8",
        )
        contract = load_contract(yaml)
        assert contract.requires.agents == []
