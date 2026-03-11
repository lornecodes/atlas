"""AgentBase — the thin base class all Atlas agents implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas.contract.types import AgentContract
    from atlas.runtime.context import AgentContext


class AgentBase(ABC):
    """Base class for Atlas agents.

    Agents implement execute(). That's it.
    The runtime handles loading, I/O validation, and lifecycle.
    """

    def __init__(self, contract: AgentContract, context: AgentContext) -> None:
        self.contract = contract
        self.context = context

    @abstractmethod
    async def execute(self, input_data: dict) -> dict:
        """The agent's core logic.

        Input has already been validated against the contract's input schema.
        Output will be validated against the contract's output schema.
        """
        ...

    async def on_startup(self) -> None:
        """Called once when the agent instance is created.

        Use for: loading models, warming caches, establishing connections.
        """

    async def on_shutdown(self) -> None:
        """Called when the agent instance is being destroyed.

        Use for: cleanup, releasing resources, flushing buffers.
        """
