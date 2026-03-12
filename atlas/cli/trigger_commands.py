"""CLI commands for trigger management."""

from __future__ import annotations

import asyncio
import json
import time

import typer

trigger_app = typer.Typer(
    name="trigger",
    help="Manage triggers — create, list, delete, enable/disable.",
    no_args_is_help=True,
)


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


async def _get_store(db: str):
    from atlas.store.trigger_store import TriggerStore
    store = TriggerStore(db)
    await store.init()
    return store


@trigger_app.command()
def create(
    trigger_type: str = typer.Option(..., "--type", "-t", help="Trigger type: cron, interval, one_shot, webhook"),
    agent: str = typer.Option(..., "--agent", "-a", help="Target agent name"),
    name: str = typer.Option("", "--name", "-n", help="Trigger name"),
    cron: str = typer.Option("", "--cron", help="Cron expression (for cron type)"),
    interval: float = typer.Option(0, "--interval", help="Interval in seconds (for interval type)"),
    fire_at: float = typer.Option(0, "--fire-at", help="Unix timestamp (for one_shot type)"),
    secret: str = typer.Option("", "--secret", help="Webhook HMAC secret"),
    input_data: str = typer.Option("{}", "--input", "-i", help="JSON input data"),
    priority: int = typer.Option(0, "--priority", "-p", help="Job priority"),
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
) -> None:
    """Create a new trigger."""
    try:
        data = json.loads(input_data)
    except json.JSONDecodeError as e:
        typer.echo(f"Invalid JSON input: {e}", err=True)
        raise typer.Exit(1)

    from atlas.triggers.models import TriggerDefinition

    trigger = TriggerDefinition(
        name=name,
        trigger_type=trigger_type,
        agent_name=agent,
        cron_expr=cron,
        interval_seconds=interval,
        fire_at=fire_at,
        webhook_secret=secret,
        input_data=data,
        priority=priority,
    )

    try:
        trigger.validate()
    except ValueError as e:
        typer.echo(f"Invalid trigger: {e}", err=True)
        raise typer.Exit(1)

    if trigger.trigger_type != "webhook":
        trigger.next_fire = trigger.compute_next_fire()

    async def _create():
        store = await _get_store(db)
        try:
            await store.save(trigger)
        finally:
            await store.close()

    _run(_create())
    typer.echo(f"Created trigger {trigger.id} ({trigger_type} → {agent})")


@trigger_app.command("list")
def list_triggers(
    trigger_type: str | None = typer.Option(None, "--type", "-t", help="Filter by type"),
    enabled: bool | None = typer.Option(None, "--enabled", help="Filter by enabled status"),
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """List all triggers."""
    async def _list():
        store = await _get_store(db)
        try:
            return await store.list(trigger_type=trigger_type, enabled=enabled)
        finally:
            await store.close()

    triggers = _run(_list())

    if json_output:
        typer.echo(json.dumps([t.to_dict() for t in triggers], indent=2))
        return

    if not triggers:
        typer.echo("No triggers found.")
        return

    # Simple table output
    typer.echo(f"{'ID':<20} {'Name':<15} {'Type':<10} {'Target':<15} {'Enabled':<8} {'Fires':<6}")
    typer.echo("-" * 80)
    for t in triggers:
        typer.echo(
            f"{t.id:<20} {(t.name or '-'):<15} {t.trigger_type:<10} "
            f"{t.target:<15} {'yes' if t.enabled else 'no':<8} {t.fire_count:<6}"
        )


@trigger_app.command()
def get(
    trigger_id: str = typer.Argument(help="Trigger ID"),
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
) -> None:
    """Get details for a trigger."""
    async def _get():
        store = await _get_store(db)
        try:
            return await store.get(trigger_id)
        finally:
            await store.close()

    trigger = _run(_get())
    if not trigger:
        typer.echo(f"Trigger not found: {trigger_id}", err=True)
        raise typer.Exit(1)

    typer.echo(json.dumps(trigger.to_dict(), indent=2))


@trigger_app.command()
def delete(
    trigger_id: str = typer.Argument(help="Trigger ID"),
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
) -> None:
    """Delete a trigger."""
    async def _delete():
        store = await _get_store(db)
        try:
            return await store.delete(trigger_id)
        finally:
            await store.close()

    deleted = _run(_delete())
    if deleted:
        typer.echo(f"Deleted trigger {trigger_id}")
    else:
        typer.echo(f"Trigger not found: {trigger_id}", err=True)
        raise typer.Exit(1)


@trigger_app.command()
def enable(
    trigger_id: str = typer.Argument(help="Trigger ID"),
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
) -> None:
    """Enable a trigger."""
    _set_enabled(trigger_id, True, db)


@trigger_app.command()
def disable(
    trigger_id: str = typer.Argument(help="Trigger ID"),
    db: str = typer.Option("atlas_jobs.db", "--db", help="SQLite database path"),
) -> None:
    """Disable a trigger."""
    _set_enabled(trigger_id, False, db)


def _set_enabled(trigger_id: str, enabled: bool, db: str) -> None:
    async def _update():
        store = await _get_store(db)
        try:
            trigger = await store.get(trigger_id)
            if not trigger:
                return None
            trigger.enabled = enabled
            if enabled and trigger.trigger_type != "webhook":
                trigger.next_fire = trigger.compute_next_fire()
            await store.save(trigger)
            return trigger
        finally:
            await store.close()

    trigger = _run(_update())
    if not trigger:
        typer.echo(f"Trigger not found: {trigger_id}", err=True)
        raise typer.Exit(1)

    state = "enabled" if enabled else "disabled"
    typer.echo(f"Trigger {trigger_id} {state}")
