"""FileMemoryProvider — file-backed shared memory (default, zero config)."""

from __future__ import annotations

import asyncio
from pathlib import Path


class FileMemoryProvider:
    """Read/write shared memory to a markdown file.

    Thread-safe via asyncio.Lock. Creates the file on first write.

    Usage::

        provider = FileMemoryProvider(Path("memory.md"))
        content = await provider.read()
        await provider.append("Agent learned: rate limit is 100/min")
    """

    def __init__(self, path: Path | str = "memory.md") -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def read(self) -> str:
        """Return current memory contents, or empty string if file doesn't exist."""
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8")

    async def write(self, content: str) -> None:
        """Overwrite memory with new content."""
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(content, encoding="utf-8")

    async def append(self, entry: str) -> None:
        """Append an entry to memory with a newline separator."""
        async with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            current = ""
            if self._path.exists():
                current = self._path.read_text(encoding="utf-8")
            separator = "\n" if current and not current.endswith("\n") else ""
            self._path.write_text(current + separator + entry + "\n", encoding="utf-8")
