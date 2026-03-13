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
    from atlas.security.policy import SecurityPolicy
    from atlas.store.trigger_store import TriggerStore
    from atlas.triggers.scheduler import TriggerScheduler

from atlas.app_keys import (
    CHAIN_EXECUTOR as _CHAIN_EXECUTOR_KEY,
    EVENT_BUS as _EVENT_BUS_KEY,
    JOB_TO_DICT as _JOB_TO_DICT_KEY,
    METRICS as _METRICS_KEY,
    POOL as _POOL_KEY,
    QUEUE as _QUEUE_KEY,
    REGISTRY as _REGISTRY_KEY,
    STORE as _STORE_KEY,
    TRACE_COLLECTOR as _TRACE_KEY,
    TRIGGER_SCHEDULER as _TRIGGER_SCHEDULER_KEY,
    TRIGGER_STORE as _TRIGGER_STORE_KEY,
    SECURITY_POLICY as _SECURITY_POLICY_KEY,
    SKILL_REGISTRY as _SKILL_REGISTRY_KEY,
    FILE_REGISTRY_PROVIDER as _FILE_REGISTRY_KEY,
)

logger = get_logger(__name__)


def _clamp_limit(request: web.Request, default: int = 50, maximum: int = 1000) -> int:
    """Parse and clamp the 'limit' query parameter."""
    try:
        return max(1, min(int(request.query.get("limit", default)), maximum))
    except (ValueError, TypeError):
        return default


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
    limit = _clamp_limit(request)

    if store:
        jobs = await store.list(
            status=status_filter,
            agent_name=agent_filter,
            limit=limit,
        )
    else:
        # In-memory only — list from queue
        all_jobs = queue.list_all()
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
    response: dict[str, Any] = {
        "status": "ok",
        "pending": queue.pending_count,
        "running": queue.running_count,
        "capacity_remaining": queue.capacity_remaining,
    }
    pool: ExecutionPool = request.app[_POOL_KEY]
    if pool._hardware:
        response["hardware"] = pool._hardware.status()
    return web.json_response(response)


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
    limit = _clamp_limit(request)
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


async def _handle_list_traces(request: web.Request) -> web.Response:
    """GET /api/traces — list execution traces."""
    collector = request.app.get(_TRACE_KEY)
    if not collector:
        return web.json_response({"error": "Traces not available"}, status=503)

    agent_filter = request.query.get("agent")
    limit = _clamp_limit(request)
    traces = collector.list(limit=limit, agent_name=agent_filter)
    return web.json_response([t.to_dict() for t in traces])


async def _handle_get_trace(request: web.Request) -> web.Response:
    """GET /api/traces/{id} — get a single trace."""
    collector = request.app.get(_TRACE_KEY)
    if not collector:
        return web.json_response({"error": "Traces not available"}, status=503)

    trace_id = request.match_info["id"]
    trace = collector.get(trace_id)
    if not trace:
        return web.json_response({"error": "Trace not found"}, status=404)
    return web.json_response(trace.to_dict())


async def _handle_get_orchestrator(request: web.Request) -> web.Response:
    """GET /api/orchestrator — current orchestrator info."""
    pool: ExecutionPool = request.app[_POOL_KEY]
    orch = pool.orchestrator
    return web.json_response({
        "name": type(orch).__name__,
        "type": getattr(orch, "contract", {}) and "custom" or "default",
    })


async def _handle_set_orchestrator(request: web.Request) -> web.Response:
    """POST /api/orchestrator — set or reset orchestrator at runtime."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    name = body.get("name")
    pool: ExecutionPool = request.app[_POOL_KEY]
    registry: AgentRegistry = request.app[_REGISTRY_KEY]

    if not name:
        from atlas.orchestrator.default import DefaultOrchestrator
        pool.set_orchestrator(DefaultOrchestrator())
        return web.json_response({"orchestrator": "DefaultOrchestrator"})

    entry = registry.get_orchestrator(name)
    if not entry or not entry.agent_class:
        return web.json_response(
            {"error": f"Orchestrator not found: {name}"}, status=404
        )

    from atlas.orchestrator.protocol import Orchestrator
    instance = entry.agent_class()
    if not isinstance(instance, Orchestrator):
        return web.json_response(
            {"error": f"{name} does not implement Orchestrator protocol"}, status=400
        )

    pool.set_orchestrator(instance)
    return web.json_response({"orchestrator": name})


async def _handle_security_policy(request: web.Request) -> web.Response:
    """GET /api/security/policy — view the active security policy."""
    policy = request.app.get(_SECURITY_POLICY_KEY)
    if not policy:
        return web.json_response({"error": "No security policy configured"}, status=404)
    return web.json_response(policy.to_dict())


async def _handle_list_skills(request: web.Request) -> web.Response:
    """GET /api/skills — list all registered skills."""
    registry = request.app.get(_SKILL_REGISTRY_KEY)
    if not registry:
        return web.json_response({"error": "Skills not configured"}, status=404)
    skills = [
        {
            "name": s.spec.name,
            "version": s.spec.version,
            "description": s.spec.description,
            "input_schema": s.spec.input_schema.to_json_schema(),
            "output_schema": s.spec.output_schema.to_json_schema(),
        }
        for s in registry.list_all()
    ]
    return web.json_response(skills)


async def _handle_registry_search(request: web.Request) -> web.Response:
    """GET /api/registry/search — search published agents."""
    provider = request.app.get(_FILE_REGISTRY_KEY)
    if not provider:
        return web.json_response({"error": "Registry not configured"}, status=503)
    query = request.query.get("q", "")
    limit = _clamp_limit(request, default=20)
    results = await provider.search(query, limit=limit)
    return web.json_response([m.to_dict() for m in results])


async def _handle_registry_versions(request: web.Request) -> web.Response:
    """GET /api/registry/agents/{name}/versions — list agent versions."""
    provider = request.app.get(_FILE_REGISTRY_KEY)
    if not provider:
        return web.json_response({"error": "Registry not configured"}, status=503)
    name = request.match_info["name"]
    versions = await provider.list_versions(name)
    return web.json_response([m.to_dict() for m in versions])


async def _handle_registry_metadata(request: web.Request) -> web.Response:
    """GET /api/registry/agents/{name}/{version}/metadata"""
    provider = request.app.get(_FILE_REGISTRY_KEY)
    if not provider:
        return web.json_response({"error": "Registry not configured"}, status=503)
    name = request.match_info["name"]
    version = request.match_info["version"]
    meta = await provider.get_metadata(name, version)
    if not meta:
        return web.json_response({"error": "Not found"}, status=404)
    return web.json_response(meta.to_dict())


async def _handle_registry_download(request: web.Request) -> web.Response:
    """GET /api/registry/agents/{name}/{version}/download"""
    provider = request.app.get(_FILE_REGISTRY_KEY)
    if not provider:
        return web.json_response({"error": "Registry not configured"}, status=503)
    name = request.match_info["name"]
    version = request.match_info["version"]
    data = await provider.download(name, version)
    if not data:
        return web.json_response({"error": "Not found"}, status=404)
    return web.Response(body=data, content_type="application/gzip")


async def _handle_registry_publish(request: web.Request) -> web.Response:
    """POST /api/registry/agents/{name}/{version} — publish a package."""
    provider = request.app.get(_FILE_REGISTRY_KEY)
    if not provider:
        return web.json_response({"error": "Registry not configured"}, status=503)

    name = request.match_info["name"]
    version = request.match_info["version"]

    reader = await request.multipart()
    metadata_raw = None
    package_data = None

    async for part in reader:
        if part.name == "metadata":
            import ast
            metadata_raw = ast.literal_eval(await part.text())
        elif part.name == "package":
            package_data = await part.read()

    if not metadata_raw or not package_data:
        return web.json_response(
            {"error": "Missing metadata or package fields"}, status=400
        )

    from atlas.registry.provider import PackageMetadata
    metadata = PackageMetadata.from_dict(metadata_raw)

    ok = await provider.publish(metadata, package_data)
    return web.json_response({"published": ok})


def create_app(
    registry: AgentRegistry,
    queue: JobQueue,
    pool: ExecutionPool,
    store: JobStore | None = None,
    event_bus: EventBus | None = None,
    chain_executor: "ChainExecutor | None" = None,
    trigger_store: "TriggerStore | None" = None,
    trigger_scheduler: "TriggerScheduler | None" = None,
    security_policy: "SecurityPolicy | None" = None,
    skill_registry: Any = None,
    file_registry_provider: Any = None,
) -> web.Application:
    """Create the aiohttp application with pool routes."""
    app = web.Application()

    app[_REGISTRY_KEY] = registry
    app[_QUEUE_KEY] = queue
    app[_POOL_KEY] = pool
    if store:
        app[_STORE_KEY] = store

    # Wire EventBus, WebSocket streaming, metrics, and traces if available
    if event_bus:
        app[_EVENT_BUS_KEY] = event_bus
        app[_JOB_TO_DICT_KEY] = _job_to_dict

        from atlas.metrics import MetricsCollector
        app[_METRICS_KEY] = MetricsCollector(event_bus)

        from atlas.trace import TraceCollector
        app[_TRACE_KEY] = TraceCollector(event_bus)

    if security_policy:
        app[_SECURITY_POLICY_KEY] = security_policy

    if skill_registry:
        app[_SKILL_REGISTRY_KEY] = skill_registry

    if file_registry_provider:
        app[_FILE_REGISTRY_KEY] = file_registry_provider

    if chain_executor:
        app[_CHAIN_EXECUTOR_KEY] = chain_executor

    # Wire trigger system if available
    if trigger_store:
        app[_TRIGGER_STORE_KEY] = trigger_store
        if trigger_scheduler:
            app[_TRIGGER_SCHEDULER_KEY] = trigger_scheduler
        from atlas.triggers.routes import setup_trigger_routes
        setup_trigger_routes(app)

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

    # Trace routes
    app.router.add_get("/api/traces", _handle_list_traces)
    app.router.add_get("/api/traces/{id}", _handle_get_trace)

    # Orchestrator routes
    app.router.add_get("/api/orchestrator", _handle_get_orchestrator)
    app.router.add_post("/api/orchestrator", _handle_set_orchestrator)

    # Security routes
    app.router.add_get("/api/security/policy", _handle_security_policy)

    # Skill routes
    app.router.add_get("/api/skills", _handle_list_skills)

    # Registry routes (only if file_registry_provider is configured)
    if file_registry_provider:
        app.router.add_get("/api/registry/search", _handle_registry_search)
        app.router.add_get(
            "/api/registry/agents/{name}/versions", _handle_registry_versions
        )
        app.router.add_get(
            "/api/registry/agents/{name}/{version}/metadata",
            _handle_registry_metadata,
        )
        app.router.add_get(
            "/api/registry/agents/{name}/{version}/download",
            _handle_registry_download,
        )
        app.router.add_post(
            "/api/registry/agents/{name}/{version}",
            _handle_registry_publish,
        )

    return app
