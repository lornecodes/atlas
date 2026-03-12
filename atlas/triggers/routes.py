"""HTTP routes for trigger management and webhook endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import TYPE_CHECKING

from aiohttp import web

from atlas.logging import get_logger
from atlas.triggers.models import TriggerDefinition

if TYPE_CHECKING:
    from atlas.store.trigger_store import TriggerStore
    from atlas.triggers.scheduler import TriggerScheduler

from atlas.app_keys import TRIGGER_SCHEDULER, TRIGGER_STORE

logger = get_logger(__name__)


# --- Trigger CRUD ---

async def _handle_create_trigger(request: web.Request) -> web.Response:
    """POST /api/triggers — create a new trigger."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    try:
        trigger = TriggerDefinition.from_dict(body)
        trigger.validate()
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Compute initial next_fire for scheduled triggers
    if trigger.trigger_type != "webhook":
        trigger.next_fire = trigger.compute_next_fire()

    store: TriggerStore = request.app[TRIGGER_STORE]
    await store.save(trigger)

    logger.info("Created trigger %s (%s)", trigger.id, trigger.name or trigger.target)
    return web.json_response(trigger.to_dict(), status=201)


async def _handle_list_triggers(request: web.Request) -> web.Response:
    """GET /api/triggers — list triggers with optional filters."""
    store: TriggerStore = request.app[TRIGGER_STORE]

    trigger_type = request.query.get("type")
    enabled_str = request.query.get("enabled")
    enabled = None
    if enabled_str is not None:
        enabled = enabled_str.lower() in ("true", "1", "yes")
    limit = int(request.query.get("limit", 100))

    triggers = await store.list(trigger_type=trigger_type, enabled=enabled, limit=limit)
    return web.json_response([t.to_dict() for t in triggers])


async def _handle_get_trigger(request: web.Request) -> web.Response:
    """GET /api/triggers/{id} — get a single trigger."""
    store: TriggerStore = request.app[TRIGGER_STORE]
    trigger_id = request.match_info["id"]
    trigger = await store.get(trigger_id)
    if not trigger:
        return web.json_response({"error": "Trigger not found"}, status=404)
    return web.json_response(trigger.to_dict())


async def _handle_update_trigger(request: web.Request) -> web.Response:
    """PUT /api/triggers/{id} — update a trigger."""
    store: TriggerStore = request.app[TRIGGER_STORE]
    trigger_id = request.match_info["id"]

    existing = await store.get(trigger_id)
    if not existing:
        return web.json_response({"error": "Trigger not found"}, status=404)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Apply updates to existing trigger
    updatable = [
        "name", "enabled", "agent_name", "chain_name", "input_data",
        "priority", "metadata", "cron_expr", "interval_seconds", "fire_at",
        "webhook_secret",
    ]
    for key in updatable:
        if key in body:
            setattr(existing, key, body[key])

    try:
        existing.validate()
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    # Recompute next_fire if schedule changed
    if existing.trigger_type != "webhook":
        existing.next_fire = existing.compute_next_fire()

    await store.save(existing)
    return web.json_response(existing.to_dict())


async def _handle_delete_trigger(request: web.Request) -> web.Response:
    """DELETE /api/triggers/{id} — delete a trigger."""
    store: TriggerStore = request.app[TRIGGER_STORE]
    trigger_id = request.match_info["id"]
    deleted = await store.delete(trigger_id)
    if not deleted:
        return web.json_response({"error": "Trigger not found"}, status=404)
    return web.json_response({"deleted": True})


async def _handle_fire_trigger(request: web.Request) -> web.Response:
    """POST /api/triggers/{id}/fire — manually fire a trigger."""
    scheduler: TriggerScheduler | None = request.app.get(TRIGGER_SCHEDULER)
    if not scheduler:
        return web.json_response({"error": "Scheduler not available"}, status=503)

    trigger_id = request.match_info["id"]
    try:
        job_id = await scheduler.fire_manual(trigger_id)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=404)

    return web.json_response({"job_id": job_id}, status=201)


# --- Webhook Endpoint ---

async def _handle_webhook(request: web.Request) -> web.Response:
    """POST /api/hooks/{id} — webhook receiver."""
    store: TriggerStore = request.app[TRIGGER_STORE]
    scheduler: TriggerScheduler | None = request.app.get(TRIGGER_SCHEDULER)
    if not scheduler:
        return web.json_response({"error": "Scheduler not available"}, status=503)

    trigger_id = request.match_info["id"]
    trigger = await store.get(trigger_id)
    if not trigger:
        return web.json_response({"error": "Webhook not found"}, status=404)

    # HMAC validation
    if trigger.webhook_secret:
        body_bytes = await request.read()
        signature = request.headers.get("X-Atlas-Signature", "")
        if not _validate_hmac(trigger.webhook_secret, body_bytes, signature):
            return web.json_response({"error": "Invalid signature"}, status=403)
        try:
            payload = json.loads(body_bytes) if body_bytes else {}
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
    else:
        try:
            payload = await request.json() if request.content_length else {}
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

    try:
        job_id = await scheduler.fire_webhook(trigger_id, payload=payload)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)

    return web.json_response({"job_id": job_id}, status=201)


def _validate_hmac(secret: str, body: bytes, signature: str) -> bool:
    """Validate HMAC-SHA256 signature."""
    expected = hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    # Accept with or without "sha256=" prefix
    if signature.startswith("sha256="):
        signature = signature[7:]
    return hmac.compare_digest(expected, signature)


# --- Route Registration ---

def setup_trigger_routes(app: web.Application) -> None:
    """Register all trigger-related routes on the app."""
    app.router.add_post("/api/triggers", _handle_create_trigger)
    app.router.add_get("/api/triggers", _handle_list_triggers)
    app.router.add_get("/api/triggers/{id}", _handle_get_trigger)
    app.router.add_put("/api/triggers/{id}", _handle_update_trigger)
    app.router.add_delete("/api/triggers/{id}", _handle_delete_trigger)
    app.router.add_post("/api/triggers/{id}/fire", _handle_fire_trigger)
    app.router.add_post("/api/hooks/{id}", _handle_webhook)
