"""Tests for FileRegistryProvider and HttpRegistryProvider."""

import asyncio
import json
import pytest
from pathlib import Path

from atlas.registry.provider import PackageMetadata, RegistryProvider
from atlas.registry.file_provider import FileRegistryProvider
from atlas.registry.http_provider import HttpRegistryProvider, HttpRegistryError


def _make_metadata(name="echo", version="1.0.0", **kwargs) -> PackageMetadata:
    return PackageMetadata(
        name=name,
        version=version,
        description=kwargs.get("description", f"The {name} agent"),
        capabilities=kwargs.get("capabilities", ["text-processing"]),
        sha256=kwargs.get("sha256", "abc123"),
        published_at=kwargs.get("published_at", "2026-01-01T00:00:00"),
        size_bytes=kwargs.get("size_bytes", 1024),
    )


class TestPackageMetadata:
    def test_to_dict_roundtrip(self):
        meta = _make_metadata()
        d = meta.to_dict()
        restored = PackageMetadata.from_dict(d)
        assert restored.name == meta.name
        assert restored.version == meta.version
        assert restored.description == meta.description
        assert restored.capabilities == meta.capabilities
        assert restored.sha256 == meta.sha256

    def test_from_dict_defaults(self):
        meta = PackageMetadata.from_dict({})
        assert meta.name == ""
        assert meta.version == ""
        assert meta.capabilities == []
        assert meta.size_bytes == 0

    def test_frozen(self):
        meta = _make_metadata()
        with pytest.raises(AttributeError):
            meta.name = "changed"


class TestFileRegistryProvider:
    @pytest.fixture
    def registry_dir(self, tmp_path):
        return tmp_path / "registry"

    @pytest.fixture
    def provider(self, registry_dir):
        return FileRegistryProvider(registry_dir)

    @pytest.mark.asyncio
    async def test_publish_and_download(self, provider):
        meta = _make_metadata()
        data = b"fake-tar-gz-data"
        ok = await provider.publish(meta, data)
        assert ok

        downloaded = await provider.download("echo", "1.0.0")
        assert downloaded == data

    @pytest.mark.asyncio
    async def test_publish_creates_manifest(self, provider, registry_dir):
        meta = _make_metadata()
        await provider.publish(meta, b"data")

        manifest_path = registry_dir / "echo" / "1.0.0" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["name"] == "echo"

    @pytest.mark.asyncio
    async def test_get_metadata(self, provider):
        meta = _make_metadata()
        await provider.publish(meta, b"data")

        result = await provider.get_metadata("echo", "1.0.0")
        assert result is not None
        assert result.name == "echo"
        assert result.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_metadata_not_found(self, provider):
        result = await provider.get_metadata("missing", "1.0.0")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_not_found(self, provider):
        result = await provider.download("missing", "1.0.0")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_versions(self, provider):
        await provider.publish(_make_metadata(version="1.0.0"), b"v1")
        await provider.publish(_make_metadata(version="1.1.0"), b"v11")
        await provider.publish(_make_metadata(version="2.0.0"), b"v2")

        versions = await provider.list_versions("echo")
        assert len(versions) == 3
        version_strs = [v.version for v in versions]
        assert "1.0.0" in version_strs
        assert "1.1.0" in version_strs
        assert "2.0.0" in version_strs

    @pytest.mark.asyncio
    async def test_list_versions_empty(self, provider):
        versions = await provider.list_versions("nonexistent")
        assert versions == []

    @pytest.mark.asyncio
    async def test_search_by_name(self, provider):
        await provider.publish(_make_metadata(name="echo"), b"d1")
        await provider.publish(_make_metadata(name="summarizer"), b"d2")

        results = await provider.search("echo")
        assert len(results) == 1
        assert results[0].name == "echo"

    @pytest.mark.asyncio
    async def test_search_by_description(self, provider):
        await provider.publish(
            _make_metadata(name="my-agent", description="A text classifier"), b"d"
        )
        results = await provider.search("classifier")
        assert len(results) == 1
        assert results[0].name == "my-agent"

    @pytest.mark.asyncio
    async def test_search_by_capability(self, provider):
        await provider.publish(
            _make_metadata(name="a1", capabilities=["summarization"]), b"d"
        )
        await provider.publish(
            _make_metadata(name="a2", capabilities=["classification"]), b"d"
        )
        results = await provider.search("summarization")
        assert len(results) == 1
        assert results[0].name == "a1"

    @pytest.mark.asyncio
    async def test_search_limit(self, provider):
        for i in range(5):
            await provider.publish(_make_metadata(name=f"agent-{i}"), b"d")
        results = await provider.search("agent", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty(self, provider):
        results = await provider.search("nothing")
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_agents(self, provider):
        await provider.publish(_make_metadata(name="echo"), b"d1")
        await provider.publish(_make_metadata(name="summarizer"), b"d2")
        await provider.publish(_make_metadata(name="translator"), b"d3")

        # All agents searchable
        results = await provider.search("agent")
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_protocol_compliance(self, provider):
        """FileRegistryProvider satisfies the RegistryProvider protocol."""
        assert isinstance(provider, RegistryProvider)

    @pytest.mark.asyncio
    async def test_publish_overwrite(self, provider):
        meta = _make_metadata()
        await provider.publish(meta, b"version-1")
        await provider.publish(meta, b"version-2")

        data = await provider.download("echo", "1.0.0")
        assert data == b"version-2"

    @pytest.mark.asyncio
    async def test_corrupt_manifest_ignored(self, provider, registry_dir):
        # Manually create a corrupt manifest
        d = registry_dir / "broken" / "1.0.0"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text("not json", encoding="utf-8")
        (d / "package.tar.gz").write_bytes(b"data")

        results = await provider.search("broken")
        assert results == []  # Corrupt manifest is skipped


class TestHttpRegistryProvider:
    def test_protocol_compliance(self):
        provider = HttpRegistryProvider("http://example.com")
        assert isinstance(provider, RegistryProvider)

    def test_error_type_exported(self):
        from atlas.registry.http_provider import HttpRegistryError
        assert issubclass(HttpRegistryError, Exception)

    def test_auth_header(self):
        provider = HttpRegistryProvider("http://example.com", auth_token="test-token")
        assert provider._headers["Authorization"] == "Bearer test-token"

    def test_url_trailing_slash_stripped(self):
        provider = HttpRegistryProvider("http://example.com/api/")
        assert provider._url == "http://example.com/api"
