"""CLI commands for the agent registry (publish, pull, search)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

registry_app = typer.Typer(
    name="registry",
    help="Manage agent registries — add, remove, publish, pull, search.",
    no_args_is_help=True,
)


def _default_config_path() -> Path:
    """Return the default registries.yaml path."""
    local = Path(".atlas/registries.yaml")
    if local.exists():
        return local
    return Path.home() / ".atlas" / "registries.yaml"


def _get_config(config_path: str | None = None):
    from atlas.registry.config import RegistryConfig
    path = Path(config_path) if config_path else _default_config_path()
    return RegistryConfig(path), path


@registry_app.command("add")
def add(
    name: str = typer.Argument(help="Registry name"),
    file: str | None = typer.Option(None, "--file", help="Path for file-based registry"),
    http: str | None = typer.Option(None, "--http", help="URL for HTTP registry"),
    token: str | None = typer.Option(None, "--token", help="Auth token for HTTP registry"),
    config: str | None = typer.Option(None, "--config", help="Path to registries.yaml"),
) -> None:
    """Add a registry."""
    if not file and not http:
        typer.echo("Must specify --file or --http", err=True)
        raise typer.Exit(1)

    cfg, cfg_path = _get_config(config)
    if file:
        cfg.add_registry(name, "file", path=file)
    else:
        cfg.add_registry(name, "http", url=http or "", auth_token=token or "")
    cfg.save()
    typer.echo(f"Added registry '{name}' -> {cfg_path}")


@registry_app.command("remove")
def remove(
    name: str = typer.Argument(help="Registry name to remove"),
    config: str | None = typer.Option(None, "--config", help="Path to registries.yaml"),
) -> None:
    """Remove a registry."""
    cfg, _ = _get_config(config)
    if cfg.remove_registry(name):
        cfg.save()
        typer.echo(f"Removed registry '{name}'")
    else:
        typer.echo(f"Registry '{name}' not found", err=True)
        raise typer.Exit(1)


@registry_app.command("list")
def list_registries(
    config: str | None = typer.Option(None, "--config", help="Path to registries.yaml"),
) -> None:
    """List configured registries."""
    cfg, _ = _get_config(config)
    entries = cfg.list_registries()
    if not entries:
        typer.echo("No registries configured.")
        return
    for e in entries:
        loc = e.get("path") or e.get("url", "")
        typer.echo(f"  {e['name']} ({e['type']}) -> {loc}")


@registry_app.command("publish")
def publish(
    agent_dir: str = typer.Argument(help="Path to agent directory"),
    registry_name: str = typer.Option("local", "--registry", "-r", help="Target registry name"),
    config: str | None = typer.Option(None, "--config", help="Path to registries.yaml"),
) -> None:
    """Publish an agent to a registry."""
    from atlas.registry.package import pack, PackageError

    try:
        metadata, data = pack(agent_dir)
    except PackageError as e:
        typer.echo(f"Pack failed: {e}", err=True)
        raise typer.Exit(1)

    cfg, _ = _get_config(config)
    provider = cfg.get_provider(registry_name)
    if not provider:
        typer.echo(f"Registry '{registry_name}' not found", err=True)
        raise typer.Exit(1)

    ok = asyncio.run(provider.publish(metadata, data))
    if ok:
        typer.echo(
            f"Published {metadata.name}@{metadata.version} "
            f"({metadata.size_bytes} bytes, sha256={metadata.sha256[:12]}...)"
        )
    else:
        typer.echo("Publish failed", err=True)
        raise typer.Exit(1)


@registry_app.command("pull")
def pull(
    agent_name: str = typer.Argument(help="Agent name to pull"),
    version: str = typer.Option("*", "--version", "-v", help="Version requirement"),
    registry_name: str | None = typer.Option(None, "--registry", "-r", help="Registry to pull from (searches all if omitted)"),
    install_dir: str = typer.Option("./agents", "--install-dir", help="Where to install the agent"),
    config: str | None = typer.Option(None, "--config", help="Path to registries.yaml"),
) -> None:
    """Pull an agent from a registry."""
    from atlas.contract.types import AgentDependency
    from atlas.registry.resolver import DependencyResolver, _version_matches
    from atlas.contract.registry import AgentRegistry, _semver_key

    cfg, _ = _get_config(config)

    if registry_name:
        providers = [cfg.get_provider(registry_name)]
        providers = [p for p in providers if p]
    else:
        providers = cfg.get_all_providers()

    if not providers:
        typer.echo("No registries available", err=True)
        raise typer.Exit(1)

    async def _pull():
        for provider in providers:
            versions = await provider.list_versions(agent_name)
            matching = [v for v in versions if _version_matches(version, v.version)]
            if not matching:
                continue
            best = sorted(matching, key=lambda m: _semver_key(m.version))[-1]
            data = await provider.download(best.name, best.version)
            if data:
                from atlas.registry.package import unpack
                target = Path(install_dir) / best.name
                target.mkdir(parents=True, exist_ok=True)
                contract = unpack(data, target)
                typer.echo(
                    f"Installed {contract.name}@{contract.version} -> {target}"
                )
                return
        typer.echo(f"Agent '{agent_name}' (version {version}) not found in any registry", err=True)
        raise typer.Exit(1)

    asyncio.run(_pull())


@registry_app.command("search")
def search(
    query: str = typer.Argument(help="Search query"),
    registry_name: str | None = typer.Option(None, "--registry", "-r", help="Registry to search"),
    config: str | None = typer.Option(None, "--config", help="Path to registries.yaml"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Search for agents across registries."""
    cfg, _ = _get_config(config)

    if registry_name:
        providers = [cfg.get_provider(registry_name)]
        providers = [p for p in providers if p]
    else:
        providers = cfg.get_all_providers()

    if not providers:
        typer.echo("No registries available", err=True)
        raise typer.Exit(1)

    async def _search():
        all_results = []
        for provider in providers:
            results = await provider.search(query)
            all_results.extend(results)
        return all_results

    results = asyncio.run(_search())

    if json_output:
        typer.echo(json.dumps([r.to_dict() for r in results], indent=2))
    elif not results:
        typer.echo("No agents found.")
    else:
        for r in results:
            caps = f" [{', '.join(r.capabilities)}]" if r.capabilities else ""
            typer.echo(f"  {r.name}@{r.version} — {r.description}{caps}")
