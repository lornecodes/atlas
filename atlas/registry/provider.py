"""RegistryProvider protocol and PackageMetadata dataclass.

Agent marketplace types — analogous to knowledge/provider.py but for
distributing agent code (tar.gz archives) rather than knowledge entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PackageMetadata:
    """Metadata for a published agent package."""

    name: str
    version: str
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    sha256: str = ""
    published_at: str = ""  # ISO 8601
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "sha256": self.sha256,
            "published_at": self.published_at,
            "size_bytes": self.size_bytes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> PackageMetadata:
        return PackageMetadata(
            name=d.get("name", ""),
            version=d.get("version", ""),
            description=d.get("description", ""),
            capabilities=d.get("capabilities", []),
            sha256=d.get("sha256", ""),
            published_at=d.get("published_at", ""),
            size_bytes=d.get("size_bytes", 0),
        )


@runtime_checkable
class RegistryProvider(Protocol):
    """Read/write agent packages to a registry.

    Implementations: FileRegistryProvider (default), HttpRegistryProvider.
    """

    async def search(
        self, query: str, *, limit: int = 20
    ) -> list[PackageMetadata]: ...

    async def list_versions(self, name: str) -> list[PackageMetadata]: ...

    async def get_metadata(
        self, name: str, version: str
    ) -> PackageMetadata | None: ...

    async def download(self, name: str, version: str) -> bytes | None: ...

    async def publish(self, metadata: PackageMetadata, data: bytes) -> bool: ...
