"""Tests for SecurityPolicy — runtime security configuration."""

import tempfile
from pathlib import Path

import pytest

from atlas.contract.permissions import PermissionsSpec
from atlas.security.policy import SecurityPolicy


class TestPolicyDefaults:
    def test_default_container_image(self):
        p = SecurityPolicy()
        assert p.container_image == "python:3.12-slim"

    def test_default_container_network(self):
        p = SecurityPolicy()
        assert p.container_network == "none"

    def test_default_secret_provider(self):
        p = SecurityPolicy()
        assert p.secret_provider == "env"

    def test_default_max_memory(self):
        p = SecurityPolicy()
        assert p.max_memory_mb == 1024

    def test_default_max_cpu(self):
        p = SecurityPolicy()
        assert p.max_cpu_seconds == 300

    def test_default_allowed_secrets_empty(self):
        p = SecurityPolicy()
        assert p.allowed_secrets == set()

    def test_default_permissions(self):
        p = SecurityPolicy()
        assert p.default_permissions == PermissionsSpec()


class TestPolicyFromDict:
    def test_from_none(self):
        p = SecurityPolicy.from_dict(None)
        assert p.container_image == "python:3.12-slim"

    def test_from_empty(self):
        p = SecurityPolicy.from_dict({})
        assert p == SecurityPolicy()

    def test_full_dict(self):
        p = SecurityPolicy.from_dict({
            "container_image": "custom:v1",
            "container_network": "bridge",
            "secret_provider": "file",
            "secret_file_path": "/etc/secrets.json",
            "secret_env_prefix": "MY_",
            "allowed_secrets": ["KEY_A", "KEY_B"],
            "max_memory_mb": 2048,
            "max_cpu_seconds": 600,
            "default_permissions": {
                "filesystem": [],
                "network": [],
                "spawn": False,
            },
        })
        assert p.container_image == "custom:v1"
        assert p.container_network == "bridge"
        assert p.secret_provider == "file"
        assert p.secret_file_path == "/etc/secrets.json"
        assert p.secret_env_prefix == "MY_"
        assert p.allowed_secrets == {"KEY_A", "KEY_B"}
        assert p.max_memory_mb == 2048
        assert p.max_cpu_seconds == 600
        assert p.default_permissions.filesystem == []


class TestPolicyToDict:
    def test_roundtrip(self):
        original = SecurityPolicy(
            container_image="img:v2",
            allowed_secrets={"A", "B"},
            max_memory_mb=512,
        )
        d = original.to_dict()
        restored = SecurityPolicy.from_dict(d)
        assert restored.container_image == original.container_image
        assert restored.allowed_secrets == original.allowed_secrets
        assert restored.max_memory_mb == original.max_memory_mb


class TestPolicyResolvePermissions:
    def test_resolve_with_contract_permissions(self):
        policy = SecurityPolicy(max_memory_mb=1024, max_cpu_seconds=300)
        contract_perms = PermissionsSpec(max_memory_mb=512, max_cpu_seconds=60)
        resolved = policy.resolve_permissions(contract_perms)
        assert resolved.max_memory_mb == 512  # contract is lower
        assert resolved.max_cpu_seconds == 60

    def test_resolve_caps_at_policy(self):
        policy = SecurityPolicy(max_memory_mb=256, max_cpu_seconds=30)
        contract_perms = PermissionsSpec(max_memory_mb=1024, max_cpu_seconds=120)
        resolved = policy.resolve_permissions(contract_perms)
        assert resolved.max_memory_mb == 256  # capped by policy
        assert resolved.max_cpu_seconds == 30

    def test_resolve_none_uses_defaults(self):
        policy = SecurityPolicy(
            default_permissions=PermissionsSpec(filesystem=[], network=[]),
        )
        resolved = policy.resolve_permissions(None)
        assert resolved.filesystem == []
        assert resolved.network == []

    def test_resolve_container_image_fallback(self):
        policy = SecurityPolicy(container_image="default:latest")
        contract_perms = PermissionsSpec(isolation="container", container_image="")
        resolved = policy.resolve_permissions(contract_perms)
        assert resolved.container_image == "default:latest"

    def test_resolve_container_image_contract_wins(self):
        policy = SecurityPolicy(container_image="default:latest")
        contract_perms = PermissionsSpec(
            isolation="container", container_image="custom:v1"
        )
        resolved = policy.resolve_permissions(contract_perms)
        assert resolved.container_image == "custom:v1"


class TestPolicyFromYaml:
    def test_load_yaml(self, tmp_path):
        yaml_file = tmp_path / "policy.yaml"
        yaml_file.write_text(
            "container_image: test:v1\n"
            "max_memory_mb: 256\n"
            "allowed_secrets:\n"
            "  - SECRET_A\n"
            "  - SECRET_B\n"
        )
        policy = SecurityPolicy.from_yaml(yaml_file)
        assert policy.container_image == "test:v1"
        assert policy.max_memory_mb == 256
        assert policy.allowed_secrets == {"SECRET_A", "SECRET_B"}

    def test_load_empty_yaml(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        policy = SecurityPolicy.from_yaml(yaml_file)
        assert policy == SecurityPolicy()
