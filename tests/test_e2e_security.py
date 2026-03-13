"""E2E security tests — policy resolution, permissions, container command generation.

These tests exercise the security stack (phases 9-10): SecurityPolicy caps and
merges permissions, PermissionsSpec enforces immutability and defaults,
SchemaSpec blocks mutation, and ContainerSlot builds correct docker run commands.
No mocking of internal components.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.contract.permissions import PermissionsSpec
from atlas.contract.types import SchemaSpec
from atlas.security.container import ContainerError, ContainerSlot
from atlas.security.policy import SecurityPolicy

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------


class TestPolicyResolvesPermissions:
    """SecurityPolicy merges contract permissions with defaults and caps."""

    def test_policy_resolves_permissions(self):
        """Contract permissions pass through with policy caps applied."""
        policy = SecurityPolicy(max_memory_mb=1024, max_cpu_seconds=300)
        contract_perms = PermissionsSpec(
            filesystem=["read", "write"],
            network=["outbound", "inbound"],
            spawn=True,
            max_memory_mb=768,
            max_cpu_seconds=120,
            secrets=["API_KEY"],
            isolation="container",
            container_image="node:20-slim",
        )

        resolved = policy.resolve_permissions(contract_perms)

        assert resolved.filesystem == ["read", "write"]
        assert resolved.network == ["outbound", "inbound"]
        assert resolved.spawn is True
        assert resolved.max_memory_mb == 768  # under cap, kept
        assert resolved.max_cpu_seconds == 120  # under cap, kept
        assert resolved.secrets == ["API_KEY"]
        assert resolved.isolation == "container"
        assert resolved.container_image == "node:20-slim"

    def test_policy_caps_memory(self):
        """Agent requests 8192MB, policy caps at 1024MB."""
        policy = SecurityPolicy(max_memory_mb=1024)
        contract_perms = PermissionsSpec(max_memory_mb=8192)

        resolved = policy.resolve_permissions(contract_perms)

        assert resolved.max_memory_mb == 1024

    def test_policy_caps_cpu(self):
        """Agent requests 600s CPU, policy caps at 300."""
        policy = SecurityPolicy(max_cpu_seconds=300)
        contract_perms = PermissionsSpec(max_cpu_seconds=600)

        resolved = policy.resolve_permissions(contract_perms)

        assert resolved.max_cpu_seconds == 300

    def test_policy_fills_container_image_from_default(self):
        """When contract has no container_image, policy default is used."""
        policy = SecurityPolicy(container_image="python:3.12-slim")
        contract_perms = PermissionsSpec(container_image="")

        resolved = policy.resolve_permissions(contract_perms)

        assert resolved.container_image == "python:3.12-slim"

    def test_policy_uses_defaults_when_no_contract_perms(self):
        """When contract has no permissions, policy defaults are used."""
        default_perms = PermissionsSpec(
            filesystem=["read"],
            max_memory_mb=256,
            max_cpu_seconds=30,
        )
        policy = SecurityPolicy(
            default_permissions=default_perms,
            max_memory_mb=1024,
            max_cpu_seconds=300,
        )

        resolved = policy.resolve_permissions(None)

        assert resolved.filesystem == ["read"]
        assert resolved.max_memory_mb == 256  # default under cap
        assert resolved.max_cpu_seconds == 30


# ---------------------------------------------------------------------------
# PermissionsSpec
# ---------------------------------------------------------------------------


class TestPermissionsSpec:
    """PermissionsSpec defaults, serialization, and aliasing safety."""

    def test_permissions_from_dict_roundtrip(self):
        """from_dict with all fields roundtrips via to_dict."""
        d = {
            "filesystem": ["read", "write"],
            "network": ["outbound", "inbound"],
            "spawn": True,
            "max_memory_mb": 2048,
            "max_cpu_seconds": 180,
            "secrets": ["DB_PASSWORD", "API_KEY"],
            "isolation": "container",
            "container_image": "ruby:3.2",
        }

        spec = PermissionsSpec.from_dict(d)
        result = spec.to_dict()

        assert result["filesystem"] == ["read", "write"]
        assert result["network"] == ["outbound", "inbound"]
        assert result["spawn"] is True
        assert result["max_memory_mb"] == 2048
        assert result["max_cpu_seconds"] == 180
        assert result["secrets"] == ["DB_PASSWORD", "API_KEY"]
        assert result["isolation"] == "container"
        assert result["container_image"] == "ruby:3.2"

    def test_permissions_defaults(self):
        """PermissionsSpec() has correct conservative defaults."""
        spec = PermissionsSpec()

        assert spec.filesystem == ["read"]
        assert spec.network == ["outbound"]
        assert spec.spawn is False
        assert spec.max_memory_mb == 512
        assert spec.max_cpu_seconds == 60
        assert spec.secrets == []
        assert spec.isolation == "process"
        assert spec.container_image == ""

    def test_permissions_list_aliasing(self):
        """Mutating the source dict after from_dict does not affect the spec."""
        source = {
            "filesystem": ["read"],
            "network": ["outbound"],
            "secrets": ["KEY_A"],
        }

        spec = PermissionsSpec.from_dict(source)

        # Mutate the source
        source["filesystem"].append("write")
        source["network"].append("inbound")
        source["secrets"].append("KEY_B")

        # Spec should be unaffected (frozen dataclass + list() copies in from_dict)
        assert spec.filesystem == ["read"]
        assert spec.network == ["outbound"]
        assert spec.secrets == ["KEY_A"]


# ---------------------------------------------------------------------------
# SchemaSpec immutability
# ---------------------------------------------------------------------------


class TestSchemaSpecImmutability:
    """SchemaSpec protects its raw dict from external mutation."""

    def test_schema_spec_immutability(self):
        """Create SchemaSpec with raw dict, mutate original dict — spec not affected."""
        original = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        spec = SchemaSpec(raw=original)

        # Mutate the original
        original["properties"]["age"] = {"type": "integer"}
        original["required"].append("age")

        # Spec should be unaffected (deep copy on init)
        assert "age" not in spec.properties
        assert spec.required == ["name"]

    def test_schema_spec_setattr_blocked(self):
        """SchemaSpec.__setattr__ raises AttributeError."""
        spec = SchemaSpec(raw={"type": "object"})

        with pytest.raises(AttributeError, match="immutable"):
            spec.foo = "bar"


# ---------------------------------------------------------------------------
# ContainerSlot command generation
# ---------------------------------------------------------------------------


class TestContainerSlotBuildCommand:
    """ContainerSlot._build_command() generates correct docker run args."""

    def _make_slot(self, **kwargs) -> ContainerSlot:
        """Create a ContainerSlot with _docker_cmd pre-set to skip on_startup."""
        slot = ContainerSlot(**kwargs)
        # Bypass on_startup Docker check by setting the internal command path
        slot._docker_cmd = "/usr/bin/docker"
        return slot

    def test_container_slot_builds_command(self):
        """Full command includes --rm, -i, --network, --memory, --cpus, -e, -v, -w."""
        perms = PermissionsSpec(
            filesystem=["read", "write"],
            max_memory_mb=256,
            max_cpu_seconds=30,
        )
        slot = self._make_slot(
            image="python:3.12-slim",
            permissions=perms,
            secrets={"DB_HOST": "localhost", "DB_PORT": "5432"},
            network="bridge",
            working_dir="/tmp/workdir",
        )

        cmd = slot._build_command()

        assert cmd[0] == "/usr/bin/docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        assert "-i" in cmd

        # Network
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "bridge"

        # Memory limit
        mem_idx = cmd.index("--memory")
        assert cmd[mem_idx + 1] == "256m"

        # CPU limit
        assert "--cpus" in cmd

        # Secrets as -e flags
        env_indices = [i for i, v in enumerate(cmd) if v == "-e"]
        env_values = [cmd[i + 1] for i in env_indices]
        assert "DB_HOST=localhost" in env_values
        assert "DB_PORT=5432" in env_values

        # Working directory mount (write permission → no :ro)
        v_idx = cmd.index("-v")
        assert cmd[v_idx + 1] == "/tmp/workdir:/workspace"

        # Working directory set
        w_idx = cmd.index("-w")
        assert cmd[w_idx + 1] == "/workspace"

        # Image is last
        assert cmd[-1] == "python:3.12-slim"

    def test_container_network_isolation(self):
        """network='none' produces '--network none' in command."""
        slot = self._make_slot(image="alpine:latest", network="none")

        cmd = slot._build_command()

        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "none"

    def test_container_working_dir_validation(self):
        """working_dir containing '..' raises ContainerError."""
        slot = self._make_slot(
            image="python:3.12-slim",
            working_dir="/tmp/../etc/shadow",
        )

        with pytest.raises(ContainerError, match="must not contain"):
            slot._build_command()

    def test_container_readonly_mount(self):
        """Read-only filesystem permission produces :ro volume mount."""
        perms = PermissionsSpec(filesystem=["read"])
        slot = self._make_slot(
            image="python:3.12-slim",
            permissions=perms,
            working_dir="/tmp/readonly",
        )

        cmd = slot._build_command()

        v_idx = cmd.index("-v")
        assert cmd[v_idx + 1] == "/tmp/readonly:/workspace:ro"


# ---------------------------------------------------------------------------
# Policy serialization
# ---------------------------------------------------------------------------


class TestPolicySerialization:
    """SecurityPolicy roundtrips through to_dict/from_dict."""

    def test_policy_serialization_roundtrip(self):
        """to_dict() -> from_dict() preserves all fields."""
        original = SecurityPolicy(
            default_permissions=PermissionsSpec(
                filesystem=["read", "write"],
                network=["outbound"],
                spawn=True,
                max_memory_mb=2048,
                max_cpu_seconds=120,
                secrets=["SECRET_A"],
                isolation="container",
                container_image="node:20",
            ),
            container_image="python:3.12-slim",
            container_network="bridge",
            secret_provider="file",
            secret_file_path="/run/secrets",
            secret_env_prefix="MY_SECRET_",
            allowed_secrets={"KEY_1", "KEY_2"},
            max_memory_mb=4096,
            max_cpu_seconds=600,
        )

        d = original.to_dict()
        restored = SecurityPolicy.from_dict(d)

        assert restored.container_image == original.container_image
        assert restored.container_network == original.container_network
        assert restored.secret_provider == original.secret_provider
        assert restored.secret_file_path == original.secret_file_path
        assert restored.secret_env_prefix == original.secret_env_prefix
        assert restored.allowed_secrets == original.allowed_secrets
        assert restored.max_memory_mb == original.max_memory_mb
        assert restored.max_cpu_seconds == original.max_cpu_seconds

        # Default permissions roundtrip
        assert restored.default_permissions.filesystem == ["read", "write"]
        assert restored.default_permissions.spawn is True
        assert restored.default_permissions.max_memory_mb == 2048
        assert restored.default_permissions.isolation == "container"
        assert restored.default_permissions.container_image == "node:20"
