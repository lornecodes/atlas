"""Tests for atlas.registry.package — pack/unpack agent archives."""

import os
import tarfile
import io
import pytest
from pathlib import Path

from atlas.registry.package import pack, unpack, PackageError, MAX_PACKAGE_BYTES


@pytest.fixture
def agent_dir(tmp_path):
    """Create a minimal valid agent directory."""
    d = tmp_path / "my-agent"
    d.mkdir()
    (d / "agent.yaml").write_text(
        "agent:\n  name: test-agent\n  version: '1.0.0'\n  description: A test agent\n"
        "  capabilities: [testing]\n",
        encoding="utf-8",
    )
    (d / "agent.py").write_text("# Agent implementation\n", encoding="utf-8")
    return d


@pytest.fixture
def agent_dir_with_extras(agent_dir):
    """Agent dir with extra files."""
    (agent_dir / "config.json").write_text('{"key": "value"}', encoding="utf-8")
    sub = agent_dir / "templates"
    sub.mkdir()
    (sub / "prompt.txt").write_text("Hello {{name}}", encoding="utf-8")
    return agent_dir


class TestPack:
    def test_pack_basic(self, agent_dir):
        meta, data = pack(agent_dir)
        assert meta.name == "test-agent"
        assert meta.version == "1.0.0"
        assert meta.description == "A test agent"
        assert "testing" in meta.capabilities
        assert meta.sha256
        assert meta.size_bytes == len(data)
        assert meta.published_at
        assert len(data) > 0

    def test_pack_preserves_all_files(self, agent_dir_with_extras):
        meta, data = pack(agent_dir_with_extras)
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
        assert "agent.yaml" in names
        assert "agent.py" in names
        assert "config.json" in names
        assert "templates/prompt.txt" in names

    def test_pack_skips_pycache(self, agent_dir):
        cache = agent_dir / "__pycache__"
        cache.mkdir()
        (cache / "agent.cpython-311.pyc").write_bytes(b"\x00" * 100)

        meta, data = pack(agent_dir)
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
        assert not any("__pycache__" in n for n in names)

    def test_pack_skips_hidden_files(self, agent_dir):
        (agent_dir / ".env").write_text("SECRET=123", encoding="utf-8")
        meta, data = pack(agent_dir)
        buf = io.BytesIO(data)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            names = tar.getnames()
        assert ".env" not in names

    def test_pack_no_agent_yaml(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(PackageError, match="No agent.yaml"):
            pack(empty)

    def test_pack_invalid_contract(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "agent.yaml").write_text("not: valid", encoding="utf-8")
        with pytest.raises(PackageError, match="Invalid agent contract"):
            pack(d)

    def test_pack_oversized(self, agent_dir, monkeypatch):
        import atlas.registry.package as pkg
        monkeypatch.setattr(pkg, "MAX_PACKAGE_BYTES", 100)
        # agent.yaml alone is >100 bytes
        with pytest.raises(PackageError, match="exceeds"):
            pack(agent_dir)

    def test_pack_metadata_sha256_deterministic(self, agent_dir):
        meta1, data1 = pack(agent_dir)
        meta2, data2 = pack(agent_dir)
        # Same content = same sha
        assert meta1.sha256 == meta2.sha256


class TestUnpack:
    def test_unpack_roundtrip(self, agent_dir, tmp_path):
        meta, data = pack(agent_dir)
        target = tmp_path / "unpacked"
        target.mkdir()
        contract = unpack(data, target)
        assert contract.name == "test-agent"
        assert contract.version == "1.0.0"
        assert (target / "agent.yaml").exists()
        assert (target / "agent.py").exists()

    def test_unpack_with_extras(self, agent_dir_with_extras, tmp_path):
        meta, data = pack(agent_dir_with_extras)
        target = tmp_path / "unpacked"
        target.mkdir()
        contract = unpack(data, target)
        assert (target / "config.json").exists()
        assert (target / "templates" / "prompt.txt").exists()

    def test_unpack_validates_contract(self, tmp_path):
        # Create a tar.gz with invalid agent.yaml
        target = tmp_path / "target"
        target.mkdir()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            yaml_content = b"not: valid"
            info = tarfile.TarInfo(name="agent.yaml")
            info.size = len(yaml_content)
            tar.addfile(info, io.BytesIO(yaml_content))
        with pytest.raises(PackageError, match="Invalid agent contract"):
            unpack(buf.getvalue(), target)

    def test_unpack_missing_agent_yaml(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            content = b"just a file"
            info = tarfile.TarInfo(name="readme.txt")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        with pytest.raises(PackageError, match="does not contain agent.yaml"):
            unpack(buf.getvalue(), target)

    def test_unpack_path_traversal(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            content = b"evil"
            info = tarfile.TarInfo(name="../../etc/passwd")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
        with pytest.raises(PackageError, match="Path traversal"):
            unpack(buf.getvalue(), target)

    def test_unpack_oversized(self, tmp_path, monkeypatch):
        import atlas.registry.package as pkg
        monkeypatch.setattr(pkg, "MAX_PACKAGE_BYTES", 10)
        target = tmp_path / "target"
        target.mkdir()
        with pytest.raises(PackageError, match="exceeds"):
            unpack(b"x" * 20, target)

    def test_unpack_invalid_archive(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        with pytest.raises(PackageError, match="Failed to extract"):
            unpack(b"not a tar.gz", target)
