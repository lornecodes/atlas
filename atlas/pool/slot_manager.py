"""SlotManager — warm slot pool for agent lifecycle management."""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from atlas.contract.registry import AgentRegistry
from atlas.logging import get_logger
from atlas.runtime.base import AgentBase
from atlas.runtime.context import AgentContext

if TYPE_CHECKING:
    from atlas.contract.permissions import PermissionsSpec
    from atlas.security.policy import SecurityPolicy

logger = get_logger(__name__)


class SlotState(enum.Enum):
    """Valid states for an agent slot."""

    WARMING = "warming"
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"


@dataclass
class AgentSlot:
    """A warm agent instance ready to accept work."""

    agent_name: str
    instance: AgentBase
    created_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)
    jobs_completed: int = 0
    state: SlotState = SlotState.IDLE


class SlotManager:
    """Manages warm agent slots with lifecycle awareness.

    Handles acquire (cold start or warm reuse), release (return to pool or
    destroy), idle reaping, and graceful shutdown.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        warm_pool_size: int = 2,
        warmup_timeout: float = 30.0,
        security_policy: "SecurityPolicy | None" = None,
    ) -> None:
        self._registry = registry
        self._warm_pool_size = warm_pool_size
        self._warmup_timeout = warmup_timeout
        self._security_policy = security_policy
        self._warm_slots: dict[str, list[AgentSlot]] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        agent_name: str,
        *,
        resolved_permissions: "PermissionsSpec | None" = None,
        secrets: dict[str, str] | None = None,
    ) -> tuple[AgentSlot, float]:
        """Get a warm slot or create a cold one. Returns (slot, warmup_ms).

        If resolved_permissions has isolation="container", creates a
        ContainerSlot instead of an in-process AgentSlot.
        """
        # Container isolation path
        if resolved_permissions and resolved_permissions.isolation == "container":
            return await self._acquire_container(
                agent_name, resolved_permissions, secrets or {}
            )

        # Standard in-process path
        async with self._lock:
            slots = self._warm_slots.get(agent_name, [])
            for slot in slots:
                if slot.state == SlotState.IDLE:
                    slot.state = SlotState.BUSY
                    slot.last_used = time.monotonic()
                    logger.debug("Reusing warm slot for %s", agent_name)
                    return slot, 0.0

        # Cold start outside lock
        entry = self._registry.get(agent_name)
        if not entry or not entry.agent_class:
            raise RuntimeError(f"Agent not found or no implementation: {agent_name}")

        ctx = AgentContext()
        instance = entry.agent_class(entry.contract, ctx)

        start = time.monotonic()
        try:
            await asyncio.wait_for(
                instance.on_startup(),
                timeout=self._warmup_timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Agent {agent_name} startup timed out ({self._warmup_timeout}s)")

        warmup_ms = (time.monotonic() - start) * 1000
        logger.debug("Cold start for %s took %.1fms", agent_name, warmup_ms)

        slot = AgentSlot(agent_name=agent_name, instance=instance, state=SlotState.BUSY)
        return slot, warmup_ms

    async def _acquire_container(
        self,
        agent_name: str,
        permissions: "PermissionsSpec",
        secrets: dict[str, str],
    ) -> tuple[AgentSlot, float]:
        """Create a ContainerSlot for isolated execution."""
        from atlas.security.container import ContainerSlot as DockerSlot

        image = permissions.container_image
        if not image:
            raise RuntimeError(
                f"Agent '{agent_name}' requires container isolation but no "
                f"container_image is set (check agent permissions or security policy)"
            )

        network = "none"
        if self._security_policy:
            network = self._security_policy.container_network

        # Use contract timeout if available, otherwise fall back to permissions
        entry = self._registry.get(agent_name)
        timeout = entry.contract.execution_timeout if entry else 60.0

        container = DockerSlot(
            image=image,
            permissions=permissions,
            secrets=secrets,
            network=network,
            timeout=timeout + 5.0,  # slightly longer than executor timeout
        )

        start = time.monotonic()
        await container.on_startup()
        warmup_ms = (time.monotonic() - start) * 1000
        logger.debug("Container slot for %s ready (%.1fms)", agent_name, warmup_ms)

        # Wrap in adapter — carry the real contract for traceability
        contract = entry.contract if entry else None
        adapter = _ContainerAdapter(container, contract)
        slot = AgentSlot(
            agent_name=agent_name,
            instance=adapter,
            state=SlotState.BUSY,
        )
        return slot, warmup_ms

    async def release(self, slot: AgentSlot) -> None:
        """Return a slot to the warm pool or destroy it."""
        async with self._lock:
            slots = self._warm_slots.setdefault(slot.agent_name, [])
            idle_count = sum(1 for s in slots if s.state == SlotState.IDLE)

            if idle_count < self._warm_pool_size:
                slot.state = SlotState.IDLE
                slot.last_used = time.monotonic()
                if slot not in slots:
                    slots.append(slot)
                return

        await self.destroy(slot)

    async def destroy(self, slot: AgentSlot) -> None:
        """Shutdown and remove a slot."""
        await self._shutdown_slot(slot)
        async with self._lock:
            slots = self._warm_slots.get(slot.agent_name, [])
            if slot in slots:
                slots.remove(slot)

    async def reap_idle(self, idle_timeout: float) -> int:
        """Remove idle slots past the timeout. Returns count reaped."""
        expired: list[AgentSlot] = []
        now = time.monotonic()
        async with self._lock:
            for agent_name, slots in list(self._warm_slots.items()):
                for slot in list(slots):
                    if slot.state == SlotState.IDLE and (now - slot.last_used) > idle_timeout:
                        slots.remove(slot)
                        expired.append(slot)
        # Shutdown outside the lock so on_shutdown() doesn't block acquire/release
        for slot in expired:
            await self._shutdown_slot(slot)
        return len(expired)

    async def shutdown_all(self) -> None:
        """Shutdown all warm slots."""
        all_slots: list[AgentSlot] = []
        async with self._lock:
            for slots in self._warm_slots.values():
                all_slots.extend(slots)
            self._warm_slots.clear()
        # Shutdown outside the lock
        for slot in all_slots:
            await self._shutdown_slot(slot)

    async def _shutdown_slot(self, slot: AgentSlot) -> None:
        """Call on_shutdown for an agent slot."""
        slot.state = SlotState.DRAINING
        try:
            await asyncio.wait_for(slot.instance.on_shutdown(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Slot shutdown timed out for %s", slot.agent_name)
        except Exception as e:
            logger.warning("Slot shutdown error for %s: %s", slot.agent_name, e)


class _ContainerAdapter(AgentBase):
    """Adapts a ContainerSlot to the AgentBase interface.

    This lets the executor treat container-backed agents identically
    to in-process agents — same execute/on_shutdown lifecycle.
    """

    def __init__(self, container, contract=None) -> None:
        from atlas.contract.types import AgentContract
        if contract is None:
            contract = AgentContract(name="_container", version="0.0.0")
        context = AgentContext()
        super().__init__(contract, context)
        self._container = container

    async def execute(self, input_data: dict) -> dict:
        context_data = {
            "job_id": self.context.job_id,
            "chain_name": self.context.chain_name,
            "step_index": self.context.step_index,
        }
        return await self._container.execute(input_data, context=context_data)

    async def on_shutdown(self) -> None:
        await self._container.on_shutdown()
