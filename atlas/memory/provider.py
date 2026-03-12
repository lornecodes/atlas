"""MemoryProvider protocol — the interface for shared agent memory."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryProvider(Protocol):
    """Read/write shared memory for agents.

    Implementations: FileMemoryProvider (default), HttpMemoryProvider (hook).
    Agents opt in via ``requires.memory: true`` in their contract.
    """

    async def read(self) -> str:
        """Return current memory contents."""
        ...

    async def write(self, content: str) -> None:
        """Overwrite memory with new content."""
        ...

    async def append(self, entry: str) -> None:
        """Append an entry to memory."""
        ...
