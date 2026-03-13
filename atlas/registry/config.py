"""RegistryConfig — persistent registry list.

Stored in .atlas/registries.yaml (project-local) or ~/.atlas/registries.yaml.
Supports environment variable expansion in auth_token values.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import logging as _logging

import yaml

from atlas.registry.file_provider import FileRegistryProvider
from atlas.registry.http_provider import HttpRegistryProvider
from atlas.registry.provider import RegistryProvider

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")
_config_logger = _logging.getLogger(__name__)


def _expand_env(value: str) -> str:
    """Expand ${VAR} references in a string."""
    def _replace(m: re.Match) -> str:
        var = m.group(1)
        val = os.environ.get(var)
        if val is None:
            _config_logger.warning("Environment variable %s is not set", var)
            return ""
        return val
    return _ENV_VAR_RE.sub(_replace, value)


@dataclass
class RegistryEntry:
    """A single registry configuration entry."""

    name: str
    type: str  # "file" or "http"
    path: str = ""  # file registries
    url: str = ""  # http registries
    auth_token: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.path:
            d["path"] = self.path
        if self.url:
            d["url"] = self.url
        if self.auth_token:
            d["auth_token"] = self.auth_token
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RegistryEntry:
        return RegistryEntry(
            name=d.get("name", ""),
            type=d.get("type", "file"),
            path=d.get("path", ""),
            url=d.get("url", ""),
            auth_token=d.get("auth_token", ""),
        )


class RegistryConfig:
    """Manages the persistent list of configured registries."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._entries: list[RegistryEntry] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        """Load registries from YAML file."""
        try:
            data = yaml.safe_load(self._path.read_text(encoding="utf-8"))
            if data and isinstance(data, dict):
                for entry_data in data.get("registries", []):
                    self._entries.append(RegistryEntry.from_dict(entry_data))
        except (yaml.YAMLError, OSError):
            self._entries = []

    def save(self) -> None:
        """Save registries to YAML file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"registries": [e.to_dict() for e in self._entries]}
        self._path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    def add_registry(
        self,
        name: str,
        type: str,
        *,
        path: str = "",
        url: str = "",
        auth_token: str = "",
    ) -> None:
        """Add a registry. Replaces any existing entry with the same name."""
        self._entries = [e for e in self._entries if e.name != name]
        self._entries.append(
            RegistryEntry(
                name=name, type=type, path=path, url=url, auth_token=auth_token
            )
        )

    def remove_registry(self, name: str) -> bool:
        """Remove a registry by name. Returns True if found."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.name != name]
        return len(self._entries) < before

    def list_registries(self) -> list[dict[str, Any]]:
        """List all configured registries."""
        return [e.to_dict() for e in self._entries]

    def get_provider(self, name: str) -> RegistryProvider | None:
        """Create a RegistryProvider for a named registry."""
        entry = next((e for e in self._entries if e.name == name), None)
        if not entry:
            return None

        if entry.type == "file":
            return FileRegistryProvider(entry.path)
        elif entry.type == "http":
            token = _expand_env(entry.auth_token) if entry.auth_token else None
            return HttpRegistryProvider(entry.url, auth_token=token)

        return None

    def get_all_providers(self) -> list[RegistryProvider]:
        """Create providers for all configured registries."""
        providers = []
        for entry in self._entries:
            p = self.get_provider(entry.name)
            if p:
                providers.append(p)
        return providers
