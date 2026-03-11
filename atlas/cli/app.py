"""Atlas CLI — run agents, inspect contracts, manage chains.

Usage:
    atlas run <agent> -i '{"key": "value"}'
    atlas run-chain <file> -i '{"key": "value"}'
    atlas list
    atlas inspect <agent>
    atlas validate <path>
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from atlas.cli.formatting import (
    format_agent_list,
    format_contract,
    format_result,
)
from atlas.cli.orchestrator_commands import orch_app
from atlas.cli.pool_commands import pool_app

app = typer.Typer(
    name="atlas",
    help="Atlas — agent runtime, registry, and composition engine.",
    no_args_is_help=True,
)
app.add_typer(pool_app, name="pool")
app.add_typer(orch_app, name="orchestrator")


def _get_registry(agents_path: str):
    """Create and discover an agent registry."""
    from atlas.contract.registry import AgentRegistry
    registry = AgentRegistry(search_paths=[agents_path])
    registry.discover()
    return registry


@app.command()
def run(
    agent: str = typer.Argument(help="Agent name to run"),
    input_data: str = typer.Option(..., "--input", "-i", help="JSON input data"),
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
    db: str | None = typer.Option(None, "--db", help="SQLite database path for persistence"),
) -> None:
    """Run a single agent with the given input."""
    try:
        data = json.loads(input_data)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON input: {e}", err=True)
        raise typer.Exit(1)

    registry = _get_registry(agents_path)

    from atlas.runtime.runner import run_agent, RunError

    try:
        result = asyncio.run(run_agent(registry, agent, data))
    except RunError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    if not result.success:
        typer.echo(f"Agent failed: {result.error}", err=True)
        raise typer.Exit(1)

    typer.echo(format_result(result.data, json_output=json_output))


@app.command("run-chain")
def run_chain(
    chain_file: str = typer.Argument(help="Path to chain YAML file"),
    input_data: str = typer.Option(..., "--input", "-i", help="JSON trigger input"),
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Run a chain definition from a YAML file."""
    chain_path = Path(chain_file)
    if not chain_path.exists():
        typer.echo(f"Chain file not found: {chain_file}", err=True)
        raise typer.Exit(1)

    try:
        data = json.loads(input_data)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON input: {e}", err=True)
        raise typer.Exit(1)

    registry = _get_registry(agents_path)

    from atlas.chains.definition import ChainDefinition
    from atlas.chains.runner import ChainRunner

    from atlas.mediation.engine import MediationEngine

    chain = ChainDefinition.from_yaml(chain_path)
    runner = ChainRunner(registry, MediationEngine())
    result = asyncio.run(runner.execute(chain, data))

    if not result.success:
        typer.echo(f"Chain failed at step {result.failed_at}: {result.error}", err=True)
        if result.partial_outputs:
            typer.echo(f"Partial outputs: {json.dumps(result.partial_outputs, indent=2)}")
        raise typer.Exit(1)

    typer.echo(format_result(result.output, json_output=json_output))


@app.command("list")
def list_agents(
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """List all discovered agents."""
    registry = _get_registry(agents_path)
    agents = []
    for entry in registry.list_all_versions():
        agents.append({
            "name": entry.contract.name,
            "version": entry.contract.version,
            "description": entry.contract.description,
            "capabilities": list(entry.contract.capabilities),
        })

    if json_output:
        typer.echo(json.dumps(agents, indent=2))
    else:
        typer.echo(format_agent_list(agents))


@app.command()
def inspect(
    agent: str = typer.Argument(help="Agent name to inspect"),
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Show detailed contract information for an agent."""
    registry = _get_registry(agents_path)
    entry = registry.get(agent)

    if not entry:
        typer.echo(f"Agent not found: {agent}", err=True)
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps({
            "name": entry.contract.name,
            "version": entry.contract.version,
            "description": entry.contract.description,
            "input_schema": entry.contract.input_schema.to_json_schema(),
            "output_schema": entry.contract.output_schema.to_json_schema(),
            "capabilities": list(entry.contract.capabilities),
            "execution_timeout": entry.contract.execution_timeout,
        }, indent=2))
    else:
        typer.echo(format_contract(entry.contract))


@app.command()
def validate(
    path: str = typer.Argument(help="Path to agent.yaml file or directory"),
) -> None:
    """Validate an agent.yaml contract."""
    target = Path(path)

    if target.is_dir():
        target = target / "agent.yaml"

    if not target.exists():
        typer.echo(f"File not found: {target}", err=True)
        raise typer.Exit(1)

    from atlas.contract.schema import load_contract, ContractError

    try:
        contract = load_contract(target)
        typer.echo(f"Valid: {contract.name} v{contract.version}")
    except (ContractError, Exception) as e:
        typer.echo(f"Invalid: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def serve(
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", help="Port to listen on"),
    db: str | None = typer.Option(None, "--db", help="SQLite database path for persistence"),
    max_concurrent: int = typer.Option(4, "--max-concurrent", help="Max concurrent jobs"),
    warm_pool_size: int = typer.Option(2, "--warm-pool-size", help="Warm pool size"),
    orchestrator_name: str | None = typer.Option(None, "--orchestrator", help="Orchestrator agent name"),
) -> None:
    """Start the HTTP API server with an execution pool."""
    try:
        from aiohttp import web
    except ImportError:
        typer.echo("aiohttp is required for serve. Install with: pip install atlas[serve]", err=True)
        raise typer.Exit(1)

    async def _run_server():
        from atlas.events import EventBus
        from atlas.pool.executor import ExecutionPool
        from atlas.pool.queue import JobQueue
        from atlas.serve import create_app

        registry = _get_registry(agents_path)
        bus = EventBus()
        store = None

        if db:
            from atlas.store.job_store import JobStore
            store = JobStore(db)
            await store.init()

            async def persist(job, old, new):
                await store.save(job)
            bus.subscribe(persist)

        orch = None
        if orchestrator_name:
            from atlas.orchestrator import Orchestrator
            entry = registry.get_orchestrator(orchestrator_name)
            if not entry:
                typer.echo(f"Orchestrator not found: {orchestrator_name}", err=True)
            elif entry.agent_class:
                orch_instance = entry.agent_class()
                if isinstance(orch_instance, Orchestrator):
                    orch = orch_instance
                    typer.echo(f"Using orchestrator: {orchestrator_name}")
                else:
                    typer.echo(f"Warning: {orchestrator_name} does not implement Orchestrator protocol", err=True)

        queue = JobQueue(store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=max_concurrent,
            warm_pool_size=warm_pool_size,
            orchestrator=orch,
        )

        from atlas.chains.executor import ChainExecutor
        chain_executor = ChainExecutor(registry)

        app = create_app(registry, queue, pool, store, event_bus=bus, chain_executor=chain_executor)
        await pool.start()

        typer.echo(f"Atlas serving on http://{host}:{port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        try:
            await asyncio.Event().wait()  # Run until cancelled
        finally:
            await pool.stop()
            if store:
                await store.close()
            await runner.cleanup()

    try:
        asyncio.run(_run_server())
    except KeyboardInterrupt:
        typer.echo("\nShutting down.")


if __name__ == "__main__":
    app()
