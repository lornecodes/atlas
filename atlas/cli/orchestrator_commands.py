"""CLI commands for orchestrator management."""

from __future__ import annotations

import json

import typer

orch_app = typer.Typer(
    name="orchestrator",
    help="Manage orchestrators -- list, inspect, set, reset.",
    no_args_is_help=True,
)


@orch_app.command("list")
def list_orchestrators(
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """List all discovered orchestrators."""
    from atlas.contract.registry import AgentRegistry

    registry = AgentRegistry(search_paths=[agents_path])
    registry.discover()

    orchestrators = registry.list_orchestrators()
    if not orchestrators:
        typer.echo("No orchestrators found.")
        return

    if json_output:
        data = [
            {
                "name": e.contract.name,
                "version": e.contract.version,
                "description": e.contract.description,
                "capabilities": list(e.contract.capabilities),
            }
            for e in orchestrators
        ]
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"Found {len(orchestrators)} orchestrator(s):\n")
        for e in orchestrators:
            caps = ", ".join(e.contract.capabilities) if e.contract.capabilities else ""
            typer.echo(f"  {e.contract.name}  v{e.contract.version}  [{caps}]")
            if e.contract.description:
                typer.echo(f"    {e.contract.description}")


@orch_app.command("inspect")
def inspect_orchestrator(
    name: str = typer.Argument(help="Orchestrator name"),
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Show detailed info about an orchestrator."""
    from atlas.contract.registry import AgentRegistry

    registry = AgentRegistry(search_paths=[agents_path])
    registry.discover()

    entry = registry.get_orchestrator(name)
    if not entry:
        typer.echo(f"Orchestrator not found: {name}", err=True)
        raise typer.Exit(1)

    contract = entry.contract
    if json_output:
        typer.echo(json.dumps({
            "name": contract.name,
            "version": contract.version,
            "type": contract.type,
            "description": contract.description,
            "capabilities": list(contract.capabilities),
            "has_implementation": entry.agent_class is not None,
        }, indent=2))
    else:
        typer.echo(f"Orchestrator: {contract.name}")
        typer.echo(f"Version: {contract.version}")
        typer.echo(f"Description: {contract.description}")
        typer.echo(f"Capabilities: {', '.join(contract.capabilities)}")
        typer.echo(f"Implementation: {'yes' if entry.agent_class else 'no'}")


@orch_app.command("set")
def set_orchestrator(
    name: str = typer.Argument(help="Orchestrator agent name"),
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", help="Server port"),
) -> None:
    """Set the orchestrator on a running Atlas server."""
    import httpx

    url = f"http://{host}:{port}/api/orchestrator"
    try:
        resp = httpx.post(url, json={"name": name}, timeout=5.0)
        data = resp.json()
        if resp.status_code == 200:
            typer.echo(f"Orchestrator set to: {data['orchestrator']}")
        else:
            typer.echo(f"Error: {data.get('error', 'Unknown error')}", err=True)
            raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"Could not connect to Atlas server at {host}:{port}", err=True)
        raise typer.Exit(1)


@orch_app.command("reset")
def reset_orchestrator(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8080, "--port", help="Server port"),
) -> None:
    """Reset to the default orchestrator on a running Atlas server."""
    import httpx

    url = f"http://{host}:{port}/api/orchestrator"
    try:
        resp = httpx.post(url, json={"name": None}, timeout=5.0)
        data = resp.json()
        if resp.status_code == 200:
            typer.echo(f"Orchestrator reset to: {data['orchestrator']}")
        else:
            typer.echo(f"Error: {data.get('error', 'Unknown error')}", err=True)
            raise typer.Exit(1)
    except httpx.ConnectError:
        typer.echo(f"Could not connect to Atlas server at {host}:{port}", err=True)
        raise typer.Exit(1)
