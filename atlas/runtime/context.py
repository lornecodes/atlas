"""AgentContext — runtime context passed to agents during execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


class SpawnError(Exception):
    """Raised when an agent spawn fails."""


@dataclass
class SpawnResult:
    """Result of spawning a child agent."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# Type for the spawn callback injected by the pool
SpawnCallback = Callable[[str, dict[str, Any], int, int], Awaitable[SpawnResult]]


@dataclass
class AgentContext:
    """Context provided to an agent at execution time.

    This is the agent's window into the platform. Currently minimal —
    will grow as spikes prove out what agents actually need.
    """

    # Unique ID for this execution
    job_id: str = ""

    # Chain context (if executing as part of a chain)
    chain_name: str = ""
    step_index: int = -1

    # Accumulated data from prior chain steps (read-only view)
    chain_data: dict[str, Any] = field(default_factory=dict)

    # Spawn depth (0 = top-level, incremented on recursive spawn)
    depth: int = 0
    max_depth: int = 3

    # Whether this agent is allowed to spawn (set from contract.requires.spawn_agents)
    spawn_allowed: bool = False

    # Arbitrary metadata from the trigger
    metadata: dict[str, Any] = field(default_factory=dict)

    # Spawn callback — injected by the pool, not set by agents
    _spawn_callback: SpawnCallback | None = field(default=None, repr=False)

    async def spawn(
        self,
        agent_name: str,
        input_data: dict[str, Any],
        priority: int = 0,
        timeout: float = 60.0,
    ) -> SpawnResult:
        """Spawn a child agent and wait for its result.

        Args:
            agent_name: Name of the agent to spawn.
            input_data: Input data for the child agent.
            priority: Priority for the child job.
            timeout: Max seconds to wait for the child to complete.

        Returns:
            SpawnResult with the child's output or error.

        Raises:
            SpawnError: If spawning is not allowed or depth exceeded.
        """
        if not self.spawn_allowed:
            raise SpawnError("Agent does not have spawn_agents permission")

        if self.depth >= self.max_depth:
            raise SpawnError(
                f"Max spawn depth ({self.max_depth}) exceeded at depth {self.depth}"
            )

        if self._spawn_callback is None:
            raise SpawnError("No spawn callback configured — agent not running in pool")

        return await self._spawn_callback(agent_name, input_data, priority, self.depth)
