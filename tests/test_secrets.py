"""Tests for secret providers and resolver."""

import json
import os
from pathlib import Path

import pytest

from atlas.security.secrets import (
    EnvSecretProvider,
    FileSecretProvider,
    SecretError,
    SecretProvider,
    SecretResolver,
)


class TestEnvSecretProvider:
    @pytest.mark.asyncio
    async def test_get_existing(self, monkeypatch):
        monkeypatch.setenv("ATLAS_SECRET_API_KEY", "test-key-123")
        provider = EnvSecretProvider()
        value = await provider.get("API_KEY")
        assert value == "test-key-123"

    @pytest.mark.asyncio
    async def test_get_missing(self):
        provider = EnvSecretProvider()
        value = await provider.get("NONEXISTENT_KEY_XYZ")
        assert value is None

    @pytest.mark.asyncio
    async def test_custom_prefix(self, monkeypatch):
        monkeypatch.setenv("MY_APP_DB_URL", "postgres://localhost")
        provider = EnvSecretProvider(prefix="MY_APP_")
        value = await provider.get("DB_URL")
        assert value == "postgres://localhost"

    @pytest.mark.asyncio
    async def test_empty_prefix(self, monkeypatch):
        monkeypatch.setenv("RAW_SECRET", "raw-value")
        provider = EnvSecretProvider(prefix="")
        value = await provider.get("RAW_SECRET")
        assert value == "raw-value"

    def test_implements_protocol(self):
        assert isinstance(EnvSecretProvider(), SecretProvider)


class TestFileSecretProvider:
    @pytest.mark.asyncio
    async def test_json_file(self, tmp_path):
        secrets_file = tmp_path / "secrets.json"
        secrets_file.write_text(json.dumps({"API_KEY": "json-key", "DB_URL": "pg://db"}))
        provider = FileSecretProvider(secrets_file)
        assert await provider.get("API_KEY") == "json-key"
        assert await provider.get("DB_URL") == "pg://db"

    @pytest.mark.asyncio
    async def test_yaml_file(self, tmp_path):
        secrets_file = tmp_path / "secrets.yaml"
        secrets_file.write_text("API_KEY: yaml-key\nDB_URL: pg://yaml-db\n")
        provider = FileSecretProvider(secrets_file)
        assert await provider.get("API_KEY") == "yaml-key"
        assert await provider.get("DB_URL") == "pg://yaml-db"

    @pytest.mark.asyncio
    async def test_missing_key(self, tmp_path):
        secrets_file = tmp_path / "secrets.json"
        secrets_file.write_text(json.dumps({"KEY_A": "a"}))
        provider = FileSecretProvider(secrets_file)
        assert await provider.get("KEY_B") is None

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        provider = FileSecretProvider("/nonexistent/secrets.json")
        with pytest.raises(SecretError, match="not found"):
            await provider.get("KEY")

    @pytest.mark.asyncio
    async def test_invalid_json(self, tmp_path):
        secrets_file = tmp_path / "bad.json"
        secrets_file.write_text("[1, 2, 3]")  # array, not object
        provider = FileSecretProvider(secrets_file)
        with pytest.raises(SecretError, match="must contain"):
            await provider.get("KEY")

    @pytest.mark.asyncio
    async def test_caches_after_first_load(self, tmp_path):
        secrets_file = tmp_path / "secrets.json"
        secrets_file.write_text(json.dumps({"KEY": "v1"}))
        provider = FileSecretProvider(secrets_file)
        assert await provider.get("KEY") == "v1"
        # Overwrite file — provider should use cache
        secrets_file.write_text(json.dumps({"KEY": "v2"}))
        assert await provider.get("KEY") == "v1"

    def test_implements_protocol(self, tmp_path):
        f = tmp_path / "s.json"
        f.write_text("{}")
        assert isinstance(FileSecretProvider(f), SecretProvider)


class TestSecretResolver:
    @pytest.mark.asyncio
    async def test_resolve_all(self, monkeypatch):
        monkeypatch.setenv("ATLAS_SECRET_A", "val-a")
        monkeypatch.setenv("ATLAS_SECRET_B", "val-b")
        resolver = SecretResolver(
            EnvSecretProvider(),
            allowed_secrets={"A", "B"},
        )
        result = await resolver.resolve(["A", "B"])
        assert result == {"A": "val-a", "B": "val-b"}

    @pytest.mark.asyncio
    async def test_resolve_empty_list(self):
        resolver = SecretResolver(EnvSecretProvider())
        result = await resolver.resolve([])
        assert result == {}

    @pytest.mark.asyncio
    async def test_not_in_whitelist(self, monkeypatch):
        monkeypatch.setenv("ATLAS_SECRET_BLOCKED", "value")
        resolver = SecretResolver(
            EnvSecretProvider(),
            allowed_secrets={"ALLOWED_ONLY"},
        )
        with pytest.raises(SecretError, match="not in the allowed"):
            await resolver.resolve(["BLOCKED"])

    @pytest.mark.asyncio
    async def test_secret_not_found(self, monkeypatch):
        resolver = SecretResolver(
            EnvSecretProvider(),
            allowed_secrets={"MISSING"},
        )
        with pytest.raises(SecretError, match="not found"):
            await resolver.resolve(["MISSING"])

    @pytest.mark.asyncio
    async def test_no_whitelist_allows_all(self, monkeypatch):
        monkeypatch.setenv("ATLAS_SECRET_ANY", "open-value")
        resolver = SecretResolver(EnvSecretProvider(), allowed_secrets=None)
        result = await resolver.resolve(["ANY"])
        assert result == {"ANY": "open-value"}
