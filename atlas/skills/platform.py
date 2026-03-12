"""Platform tools — expose Atlas internals as skills for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atlas.logging import get_logger
from atlas.skills.types import SkillCallable, SkillSpec

if TYPE_CHECKING:
    from atlas.contract.registry import AgentRegistry
    from atlas.pool.executor import ExecutionPool
    from atlas.pool.queue import JobQueue
    from atlas.skills.registry import SkillRegistry

logger = get_logger(__name__)

# All platform tool names share this prefix.
PLATFORM_PREFIX = "atlas."


class PlatformToolProvider:
    """Registers Atlas internals as skills on a SkillRegistry.

    Each platform tool is a closure capturing the relevant Atlas component
    (registry, queue, pool, collectors). Agents with ``requires.platform_tools: true``
    get all ``atlas.*`` skills auto-injected via the normal skill resolution path.
    """

    def __init__(
        self,
        registry: "AgentRegistry",
        queue: "JobQueue",
        pool: "ExecutionPool",
        *,
        metrics_collector: Any | None = None,
        trace_collector: Any | None = None,
        read_only: bool = False,
    ) -> None:
        self._registry = registry
        self._queue = queue
        self._pool = pool
        self._metrics = metrics_collector
        self._traces = trace_collector
        self._read_only = read_only

    def register_all(self, skill_registry: "SkillRegistry") -> int:
        """Register all platform tools on *skill_registry*. Returns count."""
        tools: list[tuple[SkillSpec, SkillCallable]] = [
            # --- Registry group ---
            (
                SkillSpec(name="atlas.registry.list", version="1.0.0",
                          description="List registered agents"),
                self._make_registry_list(),
            ),
            (
                SkillSpec(name="atlas.registry.describe", version="1.0.0",
                          description="Describe an agent's contract"),
                self._make_registry_describe(),
            ),
            (
                SkillSpec(name="atlas.registry.search", version="1.0.0",
                          description="Search agents by capability"),
                self._make_registry_search(),
            ),
            # --- Exec group ---
            (
                SkillSpec(name="atlas.exec.spawn", version="1.0.0",
                          description="Submit a job to the execution pool"),
                self._make_exec_spawn(),
            ),
            (
                SkillSpec(name="atlas.exec.status", version="1.0.0",
                          description="Get a job's current status"),
                self._make_exec_status(),
            ),
            (
                SkillSpec(name="atlas.exec.cancel", version="1.0.0",
                          description="Cancel a pending job"),
                self._make_exec_cancel(),
            ),
            # --- Queue group ---
            (
                SkillSpec(name="atlas.queue.inspect", version="1.0.0",
                          description="Inspect the job queue"),
                self._make_queue_inspect(),
            ),
            # --- Monitor group ---
            (
                SkillSpec(name="atlas.monitor.health", version="1.0.0",
                          description="Pool health stats"),
                self._make_monitor_health(),
            ),
            (
                SkillSpec(name="atlas.monitor.metrics", version="1.0.0",
                          description="Global or per-agent metrics"),
                self._make_monitor_metrics(),
            ),
            (
                SkillSpec(name="atlas.monitor.trace", version="1.0.0",
                          description="Get a single execution trace"),
                self._make_monitor_trace(),
            ),
            (
                SkillSpec(name="atlas.monitor.traces", version="1.0.0",
                          description="List execution traces"),
                self._make_monitor_traces(),
            ),
            # --- Exec: synchronous run ---
            (
                SkillSpec(name="atlas.exec.run", version="1.0.0",
                          description="Execute an agent synchronously and return result"),
                self._make_exec_run(),
            ),
        ]

        for spec, fn in tools:
            skill_registry.register_callable(spec, fn)

        logger.info("Registered %d platform tools", len(tools))
        return len(tools)

    # ------------------------------------------------------------------
    # Registry tools
    # ------------------------------------------------------------------

    def _make_registry_list(self) -> SkillCallable:
        registry = self._registry

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            type_filter = input_data.get("type")
            entries = registry.list_all()
            if type_filter:
                entries = [e for e in entries if e.contract.type == type_filter]
            return {
                "agents": [
                    {
                        "name": e.contract.name,
                        "version": e.contract.version,
                        "type": e.contract.type,
                        "description": e.contract.description,
                    }
                    for e in entries
                ],
            }

        return _fn

    def _make_registry_describe(self) -> SkillCallable:
        registry = self._registry

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            name = input_data.get("name", "")
            entry = registry.get(name)
            if not entry:
                return {"error": f"Agent '{name}' not found"}
            c = entry.contract
            return {
                "name": c.name,
                "version": c.version,
                "type": c.type,
                "description": c.description,
                "capabilities": c.capabilities,
                "input_schema": c.input_schema.to_json_schema(),
                "output_schema": c.output_schema.to_json_schema(),
                "execution_timeout": c.execution_timeout,
            }

        return _fn

    def _make_registry_search(self) -> SkillCallable:
        registry = self._registry

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            capabilities = input_data.get("capabilities", [])
            matches: list[dict[str, Any]] = []
            for cap in capabilities:
                for entry in registry.search(cap):
                    info = {
                        "name": entry.contract.name,
                        "version": entry.contract.version,
                        "capabilities": entry.contract.capabilities,
                    }
                    if info not in matches:
                        matches.append(info)
            return {"agents": matches}

        return _fn

    # ------------------------------------------------------------------
    # Exec tools
    # ------------------------------------------------------------------

    _READ_ONLY_ERROR = {"error": "Pool is not running — exec tools require a live pool (use 'atlas serve')"}

    def _make_exec_run(self) -> SkillCallable:
        registry = self._registry

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            from atlas.runtime.runner import RunError, run_agent

            agent_name = input_data.get("agent", "")
            agent_input = input_data.get("input", {})
            if not agent_name:
                return {"success": False, "data": {}, "error": "Missing 'agent' parameter", "agent_name": ""}
            try:
                result = await run_agent(registry, agent_name, agent_input)
                return {
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                    "agent_name": result.agent_name,
                }
            except RunError as e:
                return {"success": False, "data": {}, "error": str(e), "agent_name": agent_name}

        return _fn

    def _make_exec_spawn(self) -> SkillCallable:
        pool = self._pool
        read_only = self._read_only

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            if read_only:
                return PlatformToolProvider._READ_ONLY_ERROR
            from atlas.pool.job import JobData

            agent = input_data.get("agent", "")
            inp = input_data.get("input", {})
            priority = int(input_data.get("priority", 0))
            job = JobData(agent_name=agent, input_data=inp, priority=priority)
            job_id = await pool.submit(job)
            return {"job_id": job_id}

        return _fn

    def _make_exec_status(self) -> SkillCallable:
        queue = self._queue
        read_only = self._read_only

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            if read_only:
                return PlatformToolProvider._READ_ONLY_ERROR
            job_id = input_data.get("job_id", "")
            job = queue.get(job_id)
            if not job:
                return {"error": f"Job '{job_id}' not found"}
            return {
                "id": job.id,
                "agent_name": job.agent_name,
                "status": job.status,
                "error": job.error,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "completed_at": job.completed_at,
                "execution_ms": job.execution_ms,
            }

        return _fn

    def _make_exec_cancel(self) -> SkillCallable:
        queue = self._queue
        read_only = self._read_only

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            if read_only:
                return PlatformToolProvider._READ_ONLY_ERROR
            job_id = input_data.get("job_id", "")
            cancelled = await queue.cancel(job_id)
            return {"cancelled": cancelled}

        return _fn

    # ------------------------------------------------------------------
    # Queue tools
    # ------------------------------------------------------------------

    def _make_queue_inspect(self) -> SkillCallable:
        queue = self._queue

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            status_filter = input_data.get("status")
            limit = int(input_data.get("limit", 50))
            jobs = queue.list_all()
            if status_filter:
                jobs = [j for j in jobs if j.status == status_filter]
            jobs = jobs[:limit]
            return {
                "jobs": [
                    {
                        "id": j.id,
                        "agent_name": j.agent_name,
                        "status": j.status,
                        "priority": j.priority,
                        "created_at": j.created_at,
                    }
                    for j in jobs
                ],
            }

        return _fn

    # ------------------------------------------------------------------
    # Monitor tools
    # ------------------------------------------------------------------

    def _make_monitor_health(self) -> SkillCallable:
        queue = self._queue

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            return {
                "pending": queue.pending_count,
                "running": queue.running_count,
                "capacity_remaining": queue.capacity_remaining,
            }

        return _fn

    def _make_monitor_metrics(self) -> SkillCallable:
        metrics = self._metrics

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            if not metrics:
                return {"error": "Metrics not available"}
            agent = input_data.get("agent")
            if agent:
                data = metrics.get_agent_metrics(agent)
                if data is None:
                    return {"error": f"No metrics for agent: {agent}"}
                return data
            return metrics.get_all_metrics()

        return _fn

    def _make_monitor_trace(self) -> SkillCallable:
        traces = self._traces

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            if not traces:
                return {"error": "Traces not available"}
            trace_id = input_data.get("trace_id", "")
            trace = traces.get(trace_id)
            if not trace:
                return {"error": f"Trace '{trace_id}' not found"}
            return trace.to_dict()

        return _fn

    def _make_monitor_traces(self) -> SkillCallable:
        traces = self._traces

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            if not traces:
                return {"error": "Traces not available"}
            agent_filter = input_data.get("agent")
            limit = int(input_data.get("limit", 50))
            items = traces.list(limit=limit, agent_name=agent_filter)
            return {"traces": [t.to_dict() for t in items]}

        return _fn
