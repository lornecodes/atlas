"""FileRegistryProvider — directory-based agent registry.

Storage layout:
    registry/
    ├── echo/
    │   ├── 1.0.0/
    │   │   ├── manifest.json
    │   │   └── package.tar.gz
    │   └── 1.1.0/
    │       ├── manifest.json
    │       └── package.tar.gz
    └── summarizer/
        └── 1.0.0/
            ├── manifest.json
            └── package.tar.gz
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from atlas.registry.provider import PackageMetadata


class FileRegistryProvider:
    """File-backed agent registry using directory structure."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._lock = asyncio.Lock()

    async def search(
        self, query: str, *, limit: int = 20
    ) -> list[PackageMetadata]:
        """Search agents by name, description, or capabilities."""
        results: list[PackageMetadata] = []
        query_lower = query.lower()

        for agent_dir in self._iter_agent_dirs():
            for version_dir in self._iter_version_dirs(agent_dir):
                meta = self._read_manifest(version_dir)
                if not meta:
                    continue
                searchable = (
                    f"{meta.name} {meta.description} "
                    f"{' '.join(meta.capabilities)}"
                ).lower()
                if query_lower in searchable:
                    results.append(meta)
                    if len(results) >= limit:
                        return results

        return results

    async def list_versions(self, name: str) -> list[PackageMetadata]:
        """List all versions of a named agent."""
        agent_dir = self._root / name
        if not agent_dir.is_dir():
            return []
        results = []
        for version_dir in self._iter_version_dirs(agent_dir):
            meta = self._read_manifest(version_dir)
            if meta:
                results.append(meta)
        return results

    async def get_metadata(
        self, name: str, version: str
    ) -> PackageMetadata | None:
        """Get metadata for a specific agent version."""
        version_dir = self._root / name / version
        if not version_dir.is_dir():
            return None
        return self._read_manifest(version_dir)

    async def download(self, name: str, version: str) -> bytes | None:
        """Download the package archive for a specific agent version."""
        pkg_path = self._root / name / version / "package.tar.gz"
        if not pkg_path.is_file():
            return None
        return pkg_path.read_bytes()

    async def publish(self, metadata: PackageMetadata, data: bytes) -> bool:
        """Publish an agent package to the registry."""
        async with self._lock:
            version_dir = self._root / metadata.name / metadata.version
            version_dir.mkdir(parents=True, exist_ok=True)

            # Write manifest
            manifest_path = version_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(metadata.to_dict(), indent=2),
                encoding="utf-8",
            )

            # Write package archive
            pkg_path = version_dir / "package.tar.gz"
            pkg_path.write_bytes(data)

            return True

    def _iter_agent_dirs(self) -> list[Path]:
        """List agent name directories."""
        if not self._root.is_dir():
            return []
        return [d for d in sorted(self._root.iterdir()) if d.is_dir()]

    def _iter_version_dirs(self, agent_dir: Path) -> list[Path]:
        """List version directories under an agent."""
        if not agent_dir.is_dir():
            return []
        return [d for d in sorted(agent_dir.iterdir()) if d.is_dir()]

    def _read_manifest(self, version_dir: Path) -> PackageMetadata | None:
        """Read manifest.json from a version directory."""
        manifest_path = version_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            return PackageMetadata.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None
