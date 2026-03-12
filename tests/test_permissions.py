"""Tests for PermissionsSpec — declarative agent permissions."""

import pytest

from atlas.contract.permissions import PermissionsSpec
from atlas.contract.types import AgentContract


class TestPermissionsDefaults:
    def test_default_filesystem(self):
        p = PermissionsSpec()
        assert p.filesystem == ["read"]

    def test_default_network(self):
        p = PermissionsSpec()
        assert p.network == ["outbound"]

    def test_default_spawn(self):
        p = PermissionsSpec()
        assert p.spawn is False

    def test_default_memory(self):
        p = PermissionsSpec()
        assert p.max_memory_mb == 512

    def test_default_cpu(self):
        p = PermissionsSpec()
        assert p.max_cpu_seconds == 60

    def test_default_secrets_empty(self):
        p = PermissionsSpec()
        assert p.secrets == []

    def test_default_isolation(self):
        p = PermissionsSpec()
        assert p.isolation == "process"

    def test_default_container_image_empty(self):
        p = PermissionsSpec()
        assert p.container_image == ""


class TestPermissionsFromDict:
    def test_from_none(self):
        p = PermissionsSpec.from_dict(None)
        assert p == PermissionsSpec()

    def test_from_empty(self):
        p = PermissionsSpec.from_dict({})
        assert p == PermissionsSpec()

    def test_full_dict(self):
        p = PermissionsSpec.from_dict({
            "filesystem": ["read", "write"],
            "network": ["inbound", "outbound"],
            "spawn": True,
            "max_memory_mb": 1024,
            "max_cpu_seconds": 120,
            "secrets": ["API_KEY", "DB_URL"],
            "isolation": "container",
            "container_image": "python:3.12-slim",
        })
        assert p.filesystem == ["read", "write"]
        assert p.network == ["inbound", "outbound"]
        assert p.spawn is True
        assert p.max_memory_mb == 1024
        assert p.max_cpu_seconds == 120
        assert p.secrets == ["API_KEY", "DB_URL"]
        assert p.isolation == "container"
        assert p.container_image == "python:3.12-slim"

    def test_partial_dict(self):
        p = PermissionsSpec.from_dict({"spawn": True, "secrets": ["KEY"]})
        assert p.spawn is True
        assert p.secrets == ["KEY"]
        assert p.filesystem == ["read"]  # default


class TestPermissionsToDict:
    def test_roundtrip(self):
        original = PermissionsSpec(
            filesystem=["read", "write"],
            network=["outbound"],
            spawn=True,
            max_memory_mb=256,
            max_cpu_seconds=30,
            secrets=["SECRET_A"],
            isolation="container",
            container_image="my-image:latest",
        )
        d = original.to_dict()
        restored = PermissionsSpec.from_dict(d)
        assert restored == original

    def test_default_roundtrip(self):
        original = PermissionsSpec()
        d = original.to_dict()
        restored = PermissionsSpec.from_dict(d)
        assert restored == original


class TestPermissionsImmutable:
    def test_frozen(self):
        p = PermissionsSpec()
        with pytest.raises(AttributeError):
            p.spawn = True


class TestAgentContractPermissions:
    def test_contract_default_permissions(self):
        contract = AgentContract.from_dict({
            "name": "test-agent",
            "version": "1.0.0",
        })
        assert contract.permissions == PermissionsSpec()

    def test_contract_with_permissions(self):
        contract = AgentContract.from_dict({
            "name": "secure-agent",
            "version": "1.0.0",
            "permissions": {
                "filesystem": ["read", "write"],
                "network": [],
                "spawn": True,
                "max_memory_mb": 256,
                "secrets": ["API_KEY"],
                "isolation": "container",
            },
        })
        assert contract.permissions.filesystem == ["read", "write"]
        assert contract.permissions.network == []
        assert contract.permissions.spawn is True
        assert contract.permissions.max_memory_mb == 256
        assert contract.permissions.secrets == ["API_KEY"]
        assert contract.permissions.isolation == "container"

    def test_contract_nested_agent_key(self):
        contract = AgentContract.from_dict({
            "agent": {
                "name": "nested-agent",
                "version": "2.0.0",
                "permissions": {"isolation": "container"},
            }
        })
        assert contract.permissions.isolation == "container"
