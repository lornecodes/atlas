"""ChainExecutor — async chain execution with status tracking."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from atlas.chains.definition import ChainDefinition
from atlas.chains.runner import ChainResult, ChainRunner
from atlas.contract.registry import AgentRegistry
from atlas.logging import get_logger
from atlas.mediation.engine import MediationEngine

logger = get_logger(__name__)


@dataclass
class ChainExecution:
    """Tracks the state of a chain execution."""

    id: str = ""
    chain_name: str = ""
    status: str = "pending"  # pending | running | completed | failed
    input_data: dict[str, Any] = field(default_factory=dict)
    result: ChainResult | None = None
    created_at: float = 0.0
    completed_at: float = 0.0
    current_step: int = -1
    total_steps: int = 0

    def __post_init__(self):
        if not self.id:
            self.id = f"chain-{uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        d: dict[str, Any] = {
            "id": self.id,
            "chain_name": self.chain_name,
            "status": self.status,
            "input_data": self.input_data,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
        }
        if self.result:
            d["result"] = {
                "success": self.result.success,
                "output": self.result.output,
                "error": self.result.error,
                "failed_at": self.result.failed_at,
                "steps": [
                    {
                        "agent_name": s.agent_name,
                        "success": s.agent_result.success,
                        "output": s.agent_result.data,
                        "error": s.agent_result.error,
                    }
                    for s in self.result.steps
                ],
                "mediation_summary": self.result.mediation_summary,
            }
        else:
            d["result"] = None
        return d


_MAX_COMPLETED = 1000  # Evict oldest terminal executions beyond this cap


class ChainExecutor:
    """Submit and track async chain executions."""

    def __init__(self, registry: AgentRegistry, max_completed: int = _MAX_COMPLETED) -> None:
        self._registry = registry
        self._mediation = MediationEngine()
        self._executions: dict[str, ChainExecution] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._max_completed = max_completed

    def submit(self, chain: ChainDefinition, input_data: dict) -> str:
        """Start async chain execution. Returns execution ID."""
        execution = ChainExecution(
            chain_name=chain.name,
            input_data=input_data,
            total_steps=len(chain.steps),
        )
        self._executions[execution.id] = execution

        task = asyncio.create_task(self._run_chain(execution, chain, input_data))
        self._tasks[execution.id] = task
        return execution.id

    def get(self, execution_id: str) -> ChainExecution | None:
        """Get a chain execution by ID."""
        return self._executions.get(execution_id)

    def list(
        self,
        *,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ChainExecution]:
        """List chain executions, optionally filtered by status."""
        executions = list(self._executions.values())
        if status:
            executions = [e for e in executions if e.status == status]
        executions.sort(key=lambda e: e.created_at, reverse=True)
        return executions[:limit]

    async def _run_chain(
        self,
        execution: ChainExecution,
        chain: ChainDefinition,
        input_data: dict,
    ) -> None:
        """Execute the chain and update execution state."""
        execution.status = "running"
        runner = ChainRunner(self._registry, self._mediation)

        try:
            result = await runner.execute(chain, input_data)
            execution.current_step = len(result.steps) - 1
            execution.result = result
            execution.status = "completed" if result.success else "failed"
        except Exception as e:
            logger.error("Chain '%s' execution error: %s", chain.name, e)
            execution.result = ChainResult(
                success=False,
                error=str(e),
            )
            execution.status = "failed"
        finally:
            execution.completed_at = time.time()
            self._tasks.pop(execution.id, None)
            self._evict_old()

    def _evict_old(self) -> None:
        """Remove oldest terminal executions when over the cap."""
        terminal = [
            e for e in self._executions.values()
            if e.status in ("completed", "failed")
        ]
        if len(terminal) <= self._max_completed:
            return
        terminal.sort(key=lambda e: e.completed_at)
        to_remove = len(terminal) - self._max_completed
        for e in terminal[:to_remove]:
            del self._executions[e.id]

