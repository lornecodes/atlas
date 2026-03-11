"""Agent registry — discover, register, and query agents."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas.contract.schema import ContractError, load_contract
from atlas.contract.types import AgentContract
from atlas.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RegisteredAgent:
    """An agent known to the registry."""

    contract: AgentContract
    source_path: Path
    module_path: Path | None = None
    _agent_class: type | None = field(default=None, repr=False)

    @property
    def agent_class(self) -> type | None:
        """Lazy-load the agent's Python class."""
        if self._agent_class is not None:
            return self._agent_class
        if self.module_path and self.module_path.exists():
            self._agent_class = _load_agent_class(self.module_path)
        return self._agent_class


class AgentRegistry:
    """Discover, register, and query agents.

    Agents are discovered by scanning directories for agent.yaml files.
    Each agent.yaml must sit next to an agent.py implementing AgentBase.
    """

    def __init__(self, search_paths: list[Path | str] | None = None):
        self._agents: dict[str, dict[str, RegisteredAgent]] = {}  # name -> version -> agent
        self._search_paths: list[Path] = [Path(p) for p in (search_paths or [])]

    def register(self, path: Path | str) -> AgentContract:
        """Register a single agent from its agent.yaml path."""
        path = Path(path)
        contract = load_contract(path)
        agent_dir = path.parent
        module_path = agent_dir / "agent.py"

        entry = RegisteredAgent(
            contract=contract,
            source_path=path,
            module_path=module_path if module_path.exists() else None,
        )

        if contract.name not in self._agents:
            self._agents[contract.name] = {}
        self._agents[contract.name][contract.version] = entry
        return contract

    def discover(self) -> int:
        """Scan search_paths for agent.yaml files, register all found.

        Returns the number of agents discovered.
        """
        count = 0
        for search_path in self._search_paths:
            search_path = Path(search_path)
            if not search_path.is_dir():
                logger.debug("Search path does not exist: %s", search_path)
                continue
            for yaml_path in search_path.rglob("agent.yaml"):
                try:
                    self.register(yaml_path)
                    count += 1
                except ContractError as e:
                    logger.warning("Skipping invalid contract %s: %s", yaml_path, e)
        logger.info("Discovered %d agents from %d search paths", count, len(self._search_paths))
        return count

    def get(self, name: str, version: str | None = None) -> RegisteredAgent | None:
        """Get a registered agent by name and optional version.

        If version is None, returns the latest (highest semver).
        """
        versions = self._agents.get(name)
        if not versions:
            return None
        if version:
            return versions.get(version)
        # Return latest version (simple string sort works for semver with same digit counts)
        latest = sorted(versions.keys(), key=_semver_key)[-1]
        return versions[latest]

    def search(self, capability: str) -> list[RegisteredAgent]:
        """Find agents that declare a given capability."""
        results = []
        for versions in self._agents.values():
            for agent in versions.values():
                if capability in agent.contract.capabilities:
                    results.append(agent)
        return results

    def get_orchestrator(self, name: str) -> RegisteredAgent | None:
        """Get a registered orchestrator by name.

        Returns the agent only if its contract type is 'orchestrator'.
        """
        entry = self.get(name)
        if entry and entry.contract.type == "orchestrator":
            return entry
        return None

    def list_orchestrators(self) -> list[RegisteredAgent]:
        """List all registered orchestrators (latest version of each)."""
        return [
            a for a in self.list_all()
            if a.contract.type == "orchestrator"
        ]

    def list_all(self) -> list[RegisteredAgent]:
        """List all registered agents (latest version of each)."""
        results = []
        for name in self._agents:
            agent = self.get(name)
            if agent:
                results.append(agent)
        return results

    def list_all_versions(self) -> list[RegisteredAgent]:
        """List all registered agents across all versions."""
        results = []
        for versions in self._agents.values():
            results.extend(versions.values())
        return results

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents


def _semver_key(version: str) -> tuple[int, ...]:
    """Parse semver string to tuple for sorting."""
    try:
        return tuple(int(x) for x in version.split("."))
    except ValueError:
        return (0, 0, 0)


def _load_agent_class(module_path: Path) -> type | None:
    """Dynamically load an agent class from a Python module.

    Looks for a class with __agent__ = True or the first AgentBase subclass.
    """
    spec = importlib.util.spec_from_file_location(
        f"atlas.agents.{module_path.parent.name}",
        module_path,
    )
    if not spec or not spec.loader:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None

    # Find the agent class — prefer classes defined in this module
    from atlas.runtime.base import AgentBase

    candidates = []
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, AgentBase)
            and attr is not AgentBase
        ):
            candidates.append(attr)

    if not candidates:
        return None

    # Prefer classes defined in this module over imported base classes
    local = [c for c in candidates if c.__module__ == module.__name__]
    return local[0] if local else candidates[0]
