"""DependencyResolver — resolve requires.agents from registries.

Checks which agent dependencies are missing from the local AgentRegistry,
then pulls them from configured RegistryProviders.
"""

from __future__ import annotations

from pathlib import Path

from atlas.contract.registry import AgentRegistry, _semver_key
from atlas.contract.types import AgentContract, AgentDependency
from atlas.logging import get_logger
from atlas.registry.package import unpack
from atlas.registry.provider import PackageMetadata, RegistryProvider

logger = get_logger(__name__)


def _version_matches(required: str, available: str) -> bool:
    """Check if an available version satisfies a version requirement.

    Supports: "*" (any), exact ("1.2.0"), range (">=1.0.0").
    """
    if required == "*":
        return True

    if required.startswith(">="):
        min_ver = required[2:]
        return _semver_key(available) >= _semver_key(min_ver)

    if required.startswith(">"):
        min_ver = required[1:]
        return _semver_key(available) > _semver_key(min_ver)

    # Exact match
    return available == required


class DependencyResolver:
    """Resolve agent dependencies from remote registries."""

    def __init__(
        self,
        registry: AgentRegistry,
        providers: list[RegistryProvider],
    ) -> None:
        self._registry = registry
        self._providers = providers

    async def check(self, contract: AgentContract) -> list[AgentDependency]:
        """Return list of missing dependencies for a contract."""
        missing = []
        for dep in contract.requires.agents:
            agent = self._registry.get(dep.name)
            if not agent:
                missing.append(dep)
                continue
            if not _version_matches(dep.version, agent.contract.version):
                missing.append(dep)
        return missing

    async def resolve(
        self, contract: AgentContract, install_dir: Path
    ) -> int:
        """Pull missing deps from registries, unpack, register.

        Returns count of agents installed.
        """
        missing = await self.check(contract)
        if not missing:
            return 0

        installed = 0
        for dep in missing:
            found = await self._find_and_download(dep)
            if not found:
                logger.warning(
                    "Dependency '%s' (version %s) not found in any registry",
                    dep.name,
                    dep.version,
                )
                continue

            metadata, data = found
            agent_dir = install_dir / metadata.name / metadata.version
            agent_dir.mkdir(parents=True, exist_ok=True)

            dep_contract = unpack(data, agent_dir)
            yaml_path = agent_dir / "agent.yaml"
            self._registry.register(yaml_path)
            installed += 1

            logger.info(
                "Installed %s@%s from registry",
                metadata.name,
                metadata.version,
            )

            # Warn about transitive deps (no auto-resolution in v1)
            if dep_contract.requires.agents:
                transitive = [d.name for d in dep_contract.requires.agents]
                logger.warning(
                    "Agent '%s' has transitive dependencies %s — "
                    "run 'atlas pull' for each if needed",
                    dep.name,
                    transitive,
                )

        return installed

    async def _find_and_download(
        self, dep: AgentDependency
    ) -> tuple[PackageMetadata, bytes] | None:
        """Search providers in order for a matching package."""
        for provider in self._providers:
            try:
                versions = await provider.list_versions(dep.name)
                if not versions:
                    continue

                # Find best matching version
                matching = [
                    v for v in versions
                    if _version_matches(dep.version, v.version)
                ]
                if not matching:
                    continue

                # Pick highest matching version
                best = sorted(matching, key=lambda m: _semver_key(m.version))[-1]
                data = await provider.download(best.name, best.version)
                if data:
                    return best, data
            except Exception as e:
                logger.warning(
                    "Error checking registry for '%s': %s", dep.name, e
                )
                continue

        return None
