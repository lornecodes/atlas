"""AgentContext — runtime context passed to agents during execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Awaitable

if TYPE_CHECKING:
    from atlas.contract.permissions import PermissionsSpec


class SpawnError(Exception):
    """Raised when an agent spawn fails."""


class SkillInvocationError(Exception):
    """Raised when a skill invocation fails."""


@dataclass
class SpawnResult:
    """Result of spawning a child agent."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# Type for the spawn callback injected by the pool
# Args: agent_name, input_data, priority, parent_depth, parent_job_id
SpawnCallback = Callable[[str, dict[str, Any], int, int, str], Awaitable[SpawnResult]]


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

    # Dependency injection — agents check here before creating their own
    # Keys are agent-defined (e.g. "llm_provider", "langchain_chain", "anthropic_client")
    providers: dict[str, Any] = field(default_factory=dict)

    # Resolved permissions for this execution (set by pool)
    permissions: "PermissionsSpec | None" = None

    # Resolved secrets for this execution (set by pool before execute)
    secrets: dict[str, str] = field(default_factory=dict)

    # Execution metadata — agents write here during execute().
    # The pool reads this after execute() to build traces.
    # Keys: input_tokens, output_tokens, model, etc.
    execution_metadata: dict[str, Any] = field(default_factory=dict)

    # Spawn callback — injected by the pool, not set by agents
    _spawn_callback: SpawnCallback | None = field(default=None, repr=False)

    # Resolved skills — injected by the pool, not set by agents
    # Maps skill name → async callable(dict) -> dict
    _skills: dict[str, Any] = field(default_factory=dict, repr=False)

    # Resolved skill specs — injected by the pool alongside _skills
    # Maps skill name → SkillSpec (used by DynamicLLMAgent for tool definitions)
    _skill_specs: dict[str, Any] = field(default_factory=dict, repr=False)

    # Shared memory provider — injected by pool if contract requires.memory is True
    _memory_provider: Any = field(default=None, repr=False)

    # Knowledge provider — injected by pool if contract requires.knowledge is enabled
    _knowledge_provider: Any = field(default=None, repr=False)

    # Knowledge ACL — built from contract requirements + operator policy
    _knowledge_acl: Any = field(default=None, repr=False)

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

        return await self._spawn_callback(
            agent_name, input_data, priority, self.depth, self.job_id
        )

    async def skill(
        self,
        name: str,
        input_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke a platform skill by name.

        Args:
            name: Skill name (must be declared in requires.skills).
            input_data: Input data for the skill.

        Returns:
            The skill's output dict.

        Raises:
            SkillInvocationError: If the skill is not available.
        """
        if name not in self._skills:
            raise SkillInvocationError(
                f"Skill '{name}' not available — declare it in requires.skills"
            )
        return await self._skills[name](input_data)

    async def memory_read(self) -> str:
        """Read shared memory. Returns empty string if memory not enabled."""
        if not self._memory_provider:
            return ""
        return await self._memory_provider.read()

    async def memory_write(self, content: str) -> None:
        """Overwrite shared memory."""
        if not self._memory_provider:
            raise RuntimeError("Memory not enabled — set requires.memory: true in contract")
        await self._memory_provider.write(content)

    async def memory_append(self, entry: str) -> None:
        """Append to shared memory."""
        if not self._memory_provider:
            raise RuntimeError("Memory not enabled — set requires.memory: true in contract")
        await self._memory_provider.append(entry)

    # -- Knowledge methods (ACL-enforced) --

    async def knowledge_search(
        self,
        query: str,
        *,
        domain: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[Any]:
        """Search knowledge entries. ACL-filtered."""
        if not self._knowledge_provider:
            return []
        results = await self._knowledge_provider.search(
            query, domain=domain, tags=tags, limit=limit
        )
        if self._knowledge_acl:
            results = [e for e in results if self._knowledge_acl.can_read(e.domain)]
        return results

    async def knowledge_get(self, entry_id: str) -> Any:
        """Get a knowledge entry by ID. Returns None if not found or ACL-denied."""
        if not self._knowledge_provider:
            return None
        entry = await self._knowledge_provider.get(entry_id)
        if entry and self._knowledge_acl and not self._knowledge_acl.can_read(entry.domain):
            return None
        return entry

    async def knowledge_store(
        self,
        content: str,
        *,
        domain: str = "general",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a knowledge entry. Raises PermissionError if ACL denies write."""
        if not self._knowledge_provider:
            raise RuntimeError(
                "Knowledge not enabled — set requires.knowledge in contract"
            )
        if self._knowledge_acl and not self._knowledge_acl.can_write(domain):
            raise PermissionError(
                f"Agent not allowed to write to knowledge domain '{domain}'"
            )
        from atlas.knowledge.provider import KnowledgeEntry

        entry = KnowledgeEntry(
            id="",
            content=content,
            domain=domain,
            tags=tags or [],
            metadata=metadata or {},
        )
        return await self._knowledge_provider.create(entry)

    async def knowledge_update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Update a knowledge entry. Raises PermissionError if ACL denies write."""
        if not self._knowledge_provider:
            raise RuntimeError(
                "Knowledge not enabled — set requires.knowledge in contract"
            )
        existing = await self._knowledge_provider.get(entry_id)
        if not existing:
            return None
        if self._knowledge_acl and not self._knowledge_acl.can_write(existing.domain):
            raise PermissionError(
                f"Agent not allowed to write to knowledge domain '{existing.domain}'"
            )
        return await self._knowledge_provider.update(
            entry_id, content=content, tags=tags, metadata=metadata
        )
