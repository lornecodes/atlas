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
from atlas.cli.security_commands import security_app
from atlas.cli.skill_commands import skill_app
from atlas.cli.trigger_commands import trigger_app

app = typer.Typer(
    name="atlas",
    help="Atlas — agent runtime, registry, and composition engine.",
    no_args_is_help=True,
)
app.add_typer(pool_app, name="pool")
app.add_typer(orch_app, name="orchestrator")
app.add_typer(trigger_app, name="trigger")
app.add_typer(security_app, name="security")
app.add_typer(skill_app, name="skill")


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
    triggers_path: str | None = typer.Option(None, "--triggers-path", help="Path to trigger YAML directory"),
    trigger_poll: float = typer.Option(10.0, "--trigger-poll", help="Trigger poll interval in seconds"),
    security_policy_path: str | None = typer.Option(None, "--security-policy", help="Path to security policy YAML"),
    skills_path: str | None = typer.Option(None, "--skills-path", help="Path to skills directory"),
    mcp_port: int | None = typer.Option(None, "--mcp-port", help="MCP HTTP server port (enables MCP alongside REST)"),
    auth_token: str | None = typer.Option(None, "--auth-token", help="Bearer token for MCP auth"),
    remote: list[str] | None = typer.Option(None, "--remote", help="Remote MCP server (name=url or name=url@token)"),
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

        # Load security policy if specified
        sec_policy = None
        secret_resolver = None
        if security_policy_path:
            from atlas.security.policy import SecurityPolicy
            from atlas.security.secrets import EnvSecretProvider, FileSecretProvider, SecretResolver
            sec_policy = SecurityPolicy.from_yaml(security_policy_path)
            typer.echo(f"Security policy loaded from {security_policy_path}")

            if sec_policy.secret_provider == "file" and sec_policy.secret_file_path:
                provider = FileSecretProvider(sec_policy.secret_file_path)
            else:
                provider = EnvSecretProvider(prefix=sec_policy.secret_env_prefix)
            secret_resolver = SecretResolver(
                provider,
                allowed_secrets=sec_policy.allowed_secrets or None,
            )

        # Skills — always create registry (platform tools need it even without --skills-path)
        from atlas.skills.registry import SkillRegistry
        from atlas.skills.resolver import SkillResolver
        skill_registry = SkillRegistry(
            search_paths=[skills_path] if skills_path else [],
        )
        if skills_path:
            count = skill_registry.discover()
            if count:
                typer.echo(f"Loaded {count} skill(s) from {skills_path}")

        queue = JobQueue(store=store, event_bus=bus)
        pool = ExecutionPool(
            registry, queue,
            max_concurrent=max_concurrent,
            warm_pool_size=warm_pool_size,
            orchestrator=orch,
            security_policy=sec_policy,
            secret_resolver=secret_resolver,
        )

        # Register platform tools (needs pool to exist first)
        from atlas.skills.platform import PlatformToolProvider
        platform_provider = PlatformToolProvider(registry, queue, pool)
        pt_count = platform_provider.register_all(skill_registry)
        typer.echo(f"Registered {pt_count} platform tool(s)")

        # Connect to remote MCP servers (federation)
        remote_provider = None
        remote_agent_provider = None
        if remote:
            from atlas.mcp.client import RemoteToolProvider, parse_remote_spec
            from atlas.mcp.remote_agents import RemoteAgentProvider
            remote_provider = RemoteToolProvider()
            remote_agent_provider = RemoteAgentProvider()
            for spec in remote:
                try:
                    server = parse_remote_spec(spec)
                    rt_count = await remote_provider.connect(server, skill_registry)
                    typer.echo(f"Connected to '{server.name}': {rt_count} remote tool(s)")
                    ra_count = await remote_agent_provider.connect(server, registry, skill_registry)
                    if ra_count:
                        typer.echo(f"  -> {ra_count} remote agent(s) registered")
                except Exception as e:
                    typer.echo(f"Warning: failed to connect remote '{spec}': {e}", err=True)

        skill_resolver = SkillResolver(skill_registry)
        pool._skill_resolver = skill_resolver

        from atlas.chains.executor import ChainExecutor
        chain_executor = ChainExecutor(registry)

        # Set up trigger system
        trigger_store = None
        trigger_scheduler = None
        db_path = db or "atlas_jobs.db"

        from atlas.store.trigger_store import TriggerStore
        from atlas.triggers.scheduler import TriggerScheduler

        trigger_store = TriggerStore(db_path)
        await trigger_store.init()

        trigger_scheduler = TriggerScheduler(
            store=trigger_store,
            pool=pool,
            chain_executor=chain_executor,
            event_bus=bus,
            poll_interval=trigger_poll,
        )

        # Load trigger definitions from YAML files
        if triggers_path:
            triggers_dir = Path(triggers_path)
            if triggers_dir.is_dir():
                from atlas.triggers.models import TriggerDefinition
                count = 0
                for yaml_file in sorted(triggers_dir.glob("*.yaml")):
                    try:
                        trigger = TriggerDefinition.from_yaml(yaml_file)
                        trigger.validate()
                        if trigger.trigger_type != "webhook":
                            trigger.next_fire = trigger.compute_next_fire()
                        await trigger_store.save(trigger)
                        count += 1
                    except Exception as e:
                        typer.echo(f"Warning: failed to load {yaml_file.name}: {e}", err=True)
                if count:
                    typer.echo(f"Loaded {count} trigger(s) from {triggers_path}")

        app = create_app(
            registry, queue, pool, store, event_bus=bus,
            chain_executor=chain_executor,
            trigger_store=trigger_store,
            trigger_scheduler=trigger_scheduler,
            security_policy=sec_policy,
            skill_registry=skill_registry,
        )
        await pool.start()
        await trigger_scheduler.start()

        typer.echo(f"Atlas serving on http://{host}:{port}")
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()

        # Optionally start MCP HTTP server alongside REST
        mcp_task = None
        if mcp_port:
            from atlas.mcp.transport import run_mcp_http
            mcp_task = asyncio.create_task(
                run_mcp_http(skill_registry, host=host, port=mcp_port, auth_token=auth_token)
            )
            typer.echo(f"Atlas MCP HTTP server on http://{host}:{mcp_port}")

        try:
            await asyncio.Event().wait()  # Run until cancelled
        finally:
            if mcp_task:
                mcp_task.cancel()
            if remote_agent_provider:
                remote_agent_provider.disconnect_all(registry)
            if remote_provider:
                await remote_provider.disconnect_all()
            await trigger_scheduler.stop()
            await pool.stop()
            if trigger_store:
                await trigger_store.close()
            if store:
                await store.close()
            await runner.cleanup()

    try:
        asyncio.run(_run_server())
    except KeyboardInterrupt:
        typer.echo("\nShutting down.")


@app.command("mcp")
def mcp_server(
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    skills_path: str | None = typer.Option(None, "--skills-path", help="Path to skills directory"),
    http: bool = typer.Option(False, "--serve", help="Run as HTTP server instead of stdio"),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP server host"),
    port: int = typer.Option(8400, "--port", help="HTTP server port"),
    auth_token: str | None = typer.Option(None, "--auth-token", help="Bearer token for auth"),
    remote: list[str] | None = typer.Option(None, "--remote", help="Remote MCP server (name=url or name=url@token)"),
) -> None:
    """Start the MCP server (stdio by default, or HTTP with --serve)."""
    try:
        from mcp.server import Server  # noqa: F401
    except ImportError:
        typer.echo("mcp is required. Install with: pip install atlas[mcp]", err=True)
        raise typer.Exit(1)

    from atlas.skills.registry import SkillRegistry
    from atlas.skills.platform import PlatformToolProvider
    from atlas.pool.executor import ExecutionPool
    from atlas.pool.queue import JobQueue

    registry = _get_registry(agents_path)
    skill_registry = SkillRegistry(
        search_paths=[skills_path] if skills_path else [],
    )
    if skills_path:
        skill_registry.discover()

    # Register platform tools in read-only mode — pool isn't running,
    # so exec tools (spawn/status/cancel) return clear errors.
    queue = JobQueue()
    pool = ExecutionPool(registry, queue, max_concurrent=1, warm_pool_size=0)
    provider = PlatformToolProvider(registry, queue, pool, read_only=True)
    provider.register_all(skill_registry)

    async def _connect_remotes():
        """Connect to remote MCP servers if specified."""
        if not remote:
            return None, None
        from atlas.mcp.client import RemoteToolProvider, parse_remote_spec
        from atlas.mcp.remote_agents import RemoteAgentProvider
        remote_prov = RemoteToolProvider()
        agent_prov = RemoteAgentProvider()
        for spec in remote:
            try:
                server = parse_remote_spec(spec)
                rt_count = await remote_prov.connect(server, skill_registry)
                typer.echo(f"Connected to '{server.name}': {rt_count} remote tool(s)")
                ra_count = await agent_prov.connect(server, registry, skill_registry)
                if ra_count:
                    typer.echo(f"  -> {ra_count} remote agent(s) registered")
            except Exception as e:
                typer.echo(f"Warning: failed to connect remote '{spec}': {e}", err=True)
        return remote_prov, agent_prov

    if http:
        async def _run_http():
            remote_provider, agent_provider = await _connect_remotes()
            try:
                from atlas.mcp.transport import run_mcp_http
                await run_mcp_http(skill_registry, host=host, port=port, auth_token=auth_token)
            finally:
                if agent_provider:
                    agent_provider.disconnect_all(registry)
                if remote_provider:
                    await remote_provider.disconnect_all()

        typer.echo(f"Atlas MCP HTTP server on http://{host}:{port}")
        try:
            asyncio.run(_run_http())
        except KeyboardInterrupt:
            typer.echo("\nShutting down.")
    else:
        async def _run():
            remote_provider, agent_provider = await _connect_remotes()
            try:
                from atlas.mcp.stdio import run_stdio
                await run_stdio(skill_registry)
            finally:
                if agent_provider:
                    agent_provider.disconnect_all(registry)
                if remote_provider:
                    await remote_provider.disconnect_all()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    app()
