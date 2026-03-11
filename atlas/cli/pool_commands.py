"""Pool CLI commands — run jobs through the execution pool."""

from __future__ import annotations

import asyncio
import json

import typer

from atlas.cli.formatting import format_job, format_job_list, format_result

pool_app = typer.Typer(
    name="pool",
    help="Execution pool — run agents with warm slots and concurrency control.",
    no_args_is_help=True,
)


def _job_to_dict(job) -> dict:
    """Convert JobData to a serializable dict."""
    return {
        "id": job.id,
        "agent_name": job.agent_name,
        "status": job.status,
        "input_data": job.input_data,
        "output_data": job.output_data,
        "error": job.error,
        "priority": job.priority,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "warmup_ms": job.warmup_ms,
        "execution_ms": job.execution_ms,
    }


async def _pool_run(
    agent_name: str,
    input_data: dict,
    agents_path: str,
    priority: int,
    db: str | None,
    max_concurrent: int,
    warm_pool_size: int,
    json_output: bool,
) -> None:
    """Async implementation of pool run."""
    from atlas.contract.registry import AgentRegistry
    from atlas.events import EventBus
    from atlas.pool.executor import ExecutionPool
    from atlas.pool.job import JobData
    from atlas.pool.queue import JobQueue

    registry = AgentRegistry(search_paths=[agents_path])
    registry.discover()

    bus = EventBus()
    store = None

    if db:
        from atlas.store.job_store import JobStore
        store = JobStore(db)
        await store.init()

        async def persist(job, old_status, new_status):
            await store.save(job)

        bus.subscribe(persist)

    queue = JobQueue(store=store, event_bus=bus)
    pool = ExecutionPool(
        registry,
        queue,
        max_concurrent=max_concurrent,
        warm_pool_size=warm_pool_size,
    )

    await pool.start()
    try:
        job = JobData(
            agent_name=agent_name,
            input_data=input_data,
            priority=priority,
        )
        await pool.submit(job)
        result = await queue.wait_for_terminal(job.id, timeout=120.0)

        if result is None:
            typer.echo("Job timed out waiting for completion.", err=True)
            raise typer.Exit(1)

        if result.status == "completed":
            if json_output:
                typer.echo(json.dumps(_job_to_dict(result), indent=2))
            else:
                typer.echo(format_result(result.output_data or {}))
                timing = f"warmup={result.warmup_ms:.1f}ms, exec={result.execution_ms:.1f}ms"
                typer.echo(f"\n[{result.id}] {timing}")
        else:
            typer.echo(f"Job failed: {result.error}", err=True)
            if json_output:
                typer.echo(json.dumps(_job_to_dict(result), indent=2))
            raise typer.Exit(1)
    finally:
        await pool.stop()
        if store:
            await store.close()


@pool_app.command("run")
def run(
    agent: str = typer.Argument(help="Agent name to run"),
    input_data: str = typer.Option(..., "--input", "-i", help="JSON input data"),
    agents_path: str = typer.Option("./agents", "--agents-path", help="Path to agents directory"),
    priority: int = typer.Option(0, "--priority", "-p", help="Job priority (higher = first)"),
    db: str | None = typer.Option(None, "--db", help="SQLite database path for persistence"),
    max_concurrent: int = typer.Option(4, "--max-concurrent", help="Max concurrent jobs"),
    warm_pool_size: int = typer.Option(2, "--warm-pool-size", help="Warm slot pool size"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Run a job through the execution pool (one-shot)."""
    try:
        data = json.loads(input_data)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON input: {e}", err=True)
        raise typer.Exit(1)

    asyncio.run(_pool_run(
        agent, data, agents_path, priority, db,
        max_concurrent, warm_pool_size, json_output,
    ))


async def _pool_status(job_id: str, db: str, json_output: bool) -> None:
    """Async implementation of pool status."""
    from atlas.store.job_store import JobStore

    store = JobStore(db)
    await store.init()
    try:
        job = await store.get(job_id)
        if not job:
            typer.echo(f"Job not found: {job_id}", err=True)
            raise typer.Exit(1)
        typer.echo(format_job(_job_to_dict(job), json_output=json_output))
    finally:
        await store.close()


@pool_app.command("status")
def status(
    job_id: str = typer.Argument(help="Job ID to look up"),
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Show status of a persisted job."""
    asyncio.run(_pool_status(job_id, db, json_output))


async def _pool_list(
    db: str,
    filter_status: str | None,
    filter_agent: str | None,
    limit: int,
    json_output: bool,
) -> None:
    """Async implementation of pool list."""
    from atlas.store.job_store import JobStore

    store = JobStore(db)
    await store.init()
    try:
        jobs = await store.list(
            status=filter_status,
            agent_name=filter_agent,
            limit=limit,
        )
        job_dicts = [_job_to_dict(j) for j in jobs]
        typer.echo(format_job_list(job_dicts, json_output=json_output))
    finally:
        await store.close()


@pool_app.command("list")
def list_jobs(
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
    filter_status: str | None = typer.Option(None, "--status", help="Filter by status"),
    filter_agent: str | None = typer.Option(None, "--agent", help="Filter by agent name"),
    limit: int = typer.Option(50, "--limit", help="Max results"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """List persisted jobs from the store."""
    asyncio.run(_pool_list(db, filter_status, filter_agent, limit, json_output))
