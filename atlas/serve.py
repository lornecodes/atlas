"""Lightweight HTTP API for the execution pool — aiohttp-based."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from aiohttp import web

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.logging import get_logger
from atlas.pool.executor import ExecutionPool
from atlas.pool.job import JobData
from atlas.pool.queue import JobQueue
from atlas.store.job_store import JobStore

if TYPE_CHECKING:
    from atlas.chains.executor import ChainExecutor

from atlas.app_keys import (
    CHAIN_EXECUTOR as _CHAIN_EXECUTOR_KEY,
    EVENT_BUS as _EVENT_BUS_KEY,
    JOB_TO_DICT as _JOB_TO_DICT_KEY,
    METRICS as _METRICS_KEY,
    POOL as _POOL_KEY,
    QUEUE as _QUEUE_KEY,
    REGISTRY as _REGISTRY_KEY,
    STORE as _STORE_KEY,
)

logger = get_logger(__name__)


async def _handle_submit(request: web.Request) -> web.Response:
    """POST /api/jobs — submit a new job."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    agent = body.get("agent")
    if not agent:
        return web.json_response({"error": "Missing 'agent' field"}, status=400)

    input_data = body.get("input", {})
    priority = int(body.get("priority", 0))

    pool: ExecutionPool = request.app[_POOL_KEY]
    job = JobData(agent_name=agent, input_data=input_data, priority=priority)
    job_id = await pool.submit(job)

    return web.json_response({"id": job_id}, status=201)


async def _handle_get_job(request: web.Request) -> web.Response:
    """GET /api/jobs/{id} — get a job by ID."""
    job_id = request.match_info["id"]
    queue: JobQueue = request.app[_QUEUE_KEY]
    job = queue.get(job_id)

    if not job:
        # Try the store if we have one
        store: JobStore | None = request.app.get(_STORE_KEY)
        if store:
            job = await store.get(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)

    return web.json_response(_job_to_dict(job))


async def _handle_list_jobs(request: web.Request) -> web.Response:
    """GET /api/jobs — list jobs with optional filters."""
    store: JobStore | None = request.app.get(_STORE_KEY)
    queue: JobQueue = request.app[_QUEUE_KEY]

    status_filter = request.query.get("status")
    agent_filter = request.query.get("agent")
    limit = int(request.query.get("limit", 50))

    if store:
        jobs = await store.list(
            status=status_filter,
            agent_name=agent_filter,
            limit=limit,
        )
    else:
        # In-memory only — list from queue
        all_jobs = list(queue._jobs.values())
        if status_filter:
            all_jobs = [j for j in all_jobs if j.status == status_filter]
        if agent_filter:
            all_jobs = [j for j in all_jobs if j.agent_name == agent_filter]
        jobs = all_jobs[:limit]

    return web.json_response([_job_to_dict(j) for j in jobs])


async def _handle_cancel_job(request: web.Request) -> web.Response:
    """DELETE /api/jobs/{id} — cancel a pending job."""
    job_id = request.match_info["id"]
    queue: JobQueue = request.app[_QUEUE_KEY]
    cancelled = await queue.cancel(job_id)
    return web.json_response({"cancelled": cancelled})


async def _handle_health(request: web.Request) -> web.Response:
    """GET /api/health — pool health stats."""
    queue: JobQueue = request.app[_QUEUE_KEY]
    return web.json_response({
        "status": "ok",
        "pending": queue.pending_count,
        "running": queue.running_count,
        "capacity_remaining": queue.capacity_remaining,
    })


def _job_to_dict(job: JobData) -> dict[str, Any]:
    """Convert a JobData to a JSON-serializable dict."""
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
        "retry_count": job.retry_count,
        "original_job_id": job.original_job_id,
        "metadata": job.metadata,
    }


async def _handle_submit_chain(request: web.Request) -> web.Response:
    """POST /api/chains — submit a chain for async execution."""
    executor = request.app.get(_CHAIN_EXECUTOR_KEY)
    if not executor:
        return web.json_response({"error": "Chain execution not available"}, status=503)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    chain_def = body.get("chain")
    if not chain_def:
        return web.json_response({"error": "Missing 'chain' field"}, status=400)

    input_data = body.get("input", {})

    from atlas.chains.definition import ChainDefinition
    try:
        chain = ChainDefinition.from_dict(chain_def)
    except (KeyError, TypeError) as e:
        return web.json_response({"error": f"Invalid chain definition: {e}"}, status=400)

    if not chain.steps:
        return web.json_response({"error": "Chain must have at least one step"}, status=400)

    execution_id = executor.submit(chain, input_data)
    return web.json_response({"id": execution_id}, status=201)


async def _handle_get_chain(request: web.Request) -> web.Response:
    """GET /api/chains/{id} — get chain execution status."""
    executor = request.app.get(_CHAIN_EXECUTOR_KEY)
    if not executor:
        return web.json_response({"error": "Chain execution not available"}, status=503)

    execution_id = request.match_info["id"]
    execution = executor.get(execution_id)
    if not execution:
        return web.json_response({"error": "Chain execution not found"}, status=404)
    return web.json_response(execution.to_dict())


async def _handle_list_chains(request: web.Request) -> web.Response:
    """GET /api/chains — list chain executions."""
    executor = request.app.get(_CHAIN_EXECUTOR_KEY)
    if not executor:
        return web.json_response({"error": "Chain execution not available"}, status=503)

    status_filter = request.query.get("status")
    limit = int(request.query.get("limit", 50))
    executions = executor.list(status=status_filter, limit=limit)
    return web.json_response([e.to_dict() for e in executions])


async def _handle_metrics(request: web.Request) -> web.Response:
    """GET /api/metrics — global + per-agent metrics."""
    collector = request.app.get(_METRICS_KEY)
    if not collector:
        return web.json_response({"error": "Metrics not available"}, status=503)
    return web.json_response(collector.get_all_metrics())


async def _handle_agent_metrics(request: web.Request) -> web.Response:
    """GET /api/metrics/{agent} — metrics for a specific agent."""
    collector = request.app.get(_METRICS_KEY)
    if not collector:
        return web.json_response({"error": "Metrics not available"}, status=503)
    agent_name = request.match_info["agent"]
    data = collector.get_agent_metrics(agent_name)
    if data is None:
        return web.json_response({"error": f"No metrics for agent: {agent_name}"}, status=404)
    return web.json_response(data)


def create_app(
    registry: AgentRegistry,
    queue: JobQueue,
    pool: ExecutionPool,
    store: JobStore | None = None,
    event_bus: EventBus | None = None,
    chain_executor: "ChainExecutor | None" = None,
) -> web.Application:
    """Create the aiohttp application with pool routes."""
    app = web.Application()

    app[_REGISTRY_KEY] = registry
    app[_QUEUE_KEY] = queue
    app[_POOL_KEY] = pool
    if store:
        app[_STORE_KEY] = store

    # Wire EventBus, WebSocket streaming, and metrics if available
    if event_bus:
        app[_EVENT_BUS_KEY] = event_bus
        app[_JOB_TO_DICT_KEY] = _job_to_dict

        from atlas.metrics import MetricsCollector
        app[_METRICS_KEY] = MetricsCollector(event_bus)

    if chain_executor:
        app[_CHAIN_EXECUTOR_KEY] = chain_executor

    # REST routes
    app.router.add_post("/api/jobs", _handle_submit)
    app.router.add_get("/api/jobs", _handle_list_jobs)
    app.router.add_get("/api/jobs/{id}", _handle_get_job)
    app.router.add_delete("/api/jobs/{id}", _handle_cancel_job)
    app.router.add_get("/api/health", _handle_health)

    # Chain routes
    app.router.add_post("/api/chains", _handle_submit_chain)
    app.router.add_get("/api/chains", _handle_list_chains)
    app.router.add_get("/api/chains/{id}", _handle_get_chain)

    # WebSocket routes
    from atlas.ws import handle_events_ws, handle_job_events_ws
    app.router.add_get("/api/events", handle_events_ws)
    app.router.add_get("/api/jobs/{id}/events", handle_job_events_ws)

    # Metrics routes
    app.router.add_get("/api/metrics", _handle_metrics)
    app.router.add_get("/api/metrics/{agent}", _handle_agent_metrics)

    return app
