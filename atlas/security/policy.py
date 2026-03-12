"""SecurityPolicy — runtime security configuration for the Atlas pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.contract.permissions import PermissionsSpec


@dataclass
class SecurityPolicy:
    """Operator-level security configuration.

    Controls default permissions, resource caps, container settings,
    and secret injection. Loaded from a YAML file or set programmatically.
    """

    # Default permissions applied when a contract doesn't declare its own
    default_permissions: PermissionsSpec = field(default_factory=PermissionsSpec)

    # Container defaults
    container_image: str = "python:3.12-slim"
    container_network: str = "none"

    # Secret provider configuration
    secret_provider: str = "env"  # env | file
    secret_file_path: str = ""
    secret_env_prefix: str = "ATLAS_SECRET_"
    allowed_secrets: set[str] = field(default_factory=set)

    # Global resource caps (policy can restrict but not expand)
    max_memory_mb: int = 1024
    max_cpu_seconds: int = 300

    def resolve_permissions(self, contract_perms: PermissionsSpec | None) -> PermissionsSpec:
        """Merge contract permissions with policy defaults and caps.

        Order: contract declares → policy defaults fill gaps → policy caps enforce maximums.
        """
        perms = contract_perms or self.default_permissions

        # Apply global caps
        return PermissionsSpec(
            filesystem=perms.filesystem,
            network=perms.network,
            spawn=perms.spawn,
            max_memory_mb=min(perms.max_memory_mb, self.max_memory_mb),
            max_cpu_seconds=min(perms.max_cpu_seconds, self.max_cpu_seconds),
            secrets=perms.secrets,
            isolation=perms.isolation,
            container_image=perms.container_image or self.container_image,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "default_permissions": self.default_permissions.to_dict(),
            "container_image": self.container_image,
            "container_network": self.container_network,
            "secret_provider": self.secret_provider,
            "secret_file_path": self.secret_file_path,
            "secret_env_prefix": self.secret_env_prefix,
            "allowed_secrets": sorted(self.allowed_secrets),
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
        }

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> SecurityPolicy:
        """Parse a policy dict (from YAML or JSON)."""
        if not d:
            return SecurityPolicy()
        return SecurityPolicy(
            default_permissions=PermissionsSpec.from_dict(d.get("default_permissions")),
            container_image=d.get("container_image", "python:3.12-slim"),
            container_network=d.get("container_network", "none"),
            secret_provider=d.get("secret_provider", "env"),
            secret_file_path=d.get("secret_file_path", ""),
            secret_env_prefix=d.get("secret_env_prefix", "ATLAS_SECRET_"),
            allowed_secrets=set(d.get("allowed_secrets", [])),
            max_memory_mb=int(d.get("max_memory_mb", 1024)),
            max_cpu_seconds=int(d.get("max_cpu_seconds", 300)),
        )

    @staticmethod
    def from_yaml(path: Path | str) -> SecurityPolicy:
        """Load a SecurityPolicy from a YAML file."""
        import yaml
        p = Path(path)
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        return SecurityPolicy.from_dict(data)
