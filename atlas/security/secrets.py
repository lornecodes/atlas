"""Secret providers — resolve secret names to values at execution time."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class SecretError(Exception):
    """Raised when a required secret cannot be resolved."""


@runtime_checkable
class SecretProvider(Protocol):
    """Protocol for secret backends."""

    async def get(self, name: str) -> str | None:
        """Get a secret value by name. Returns None if not found."""
        ...


class EnvSecretProvider:
    """Reads secrets from environment variables.

    Looks up ``{prefix}{name}`` in os.environ. Default prefix: ``ATLAS_SECRET_``.
    """

    def __init__(self, prefix: str = "ATLAS_SECRET_") -> None:
        self._prefix = prefix

    async def get(self, name: str) -> str | None:
        return os.environ.get(f"{self._prefix}{name}")


class FileSecretProvider:
    """Reads secrets from a JSON or YAML file.

    File format: ``{"SECRET_NAME": "value", ...}``
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cache: dict[str, str] | None = None

    def _load(self) -> dict[str, str]:
        if self._cache is not None:
            return self._cache

        if not self._path.exists():
            raise SecretError(f"Secrets file not found: {self._path}")

        text = self._path.read_text()
        suffix = self._path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)

        if not isinstance(data, dict):
            raise SecretError(f"Secrets file must contain a JSON/YAML object: {self._path}")

        self._cache = {str(k): str(v) for k, v in data.items()}
        return self._cache

    async def get(self, name: str) -> str | None:
        return self._load().get(name)


class SecretResolver:
    """Resolves a list of secret names using a provider + whitelist.

    Only secrets in ``allowed_secrets`` can be resolved. If a required
    secret is not found, raises SecretError.
    """

    def __init__(
        self,
        provider: SecretProvider,
        allowed_secrets: set[str] | None = None,
    ) -> None:
        self._provider = provider
        self._allowed = allowed_secrets

    async def resolve(self, secret_names: list[str]) -> dict[str, str]:
        """Resolve secret names → values.

        Returns a dict of {name: value} for all resolved secrets.
        Raises SecretError if a secret is not in the whitelist or not found.
        """
        result: dict[str, str] = {}
        for name in secret_names:
            if self._allowed is not None and name not in self._allowed:
                raise SecretError(
                    f"Secret '{name}' is not in the allowed secrets whitelist"
                )
            value = await self._provider.get(name)
            if value is None:
                raise SecretError(f"Secret '{name}' not found")
            result[name] = value
        return result
