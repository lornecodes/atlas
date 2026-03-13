"""HttpMemoryProvider — HTTP hook for external memory systems."""

from __future__ import annotations

from typing import Any


class HttpMemoryProvider:
    """Read/write shared memory via HTTP GET/POST.

    Hook for external memory backends (Redis, vector DB, Kronos, etc.).
    GET returns memory content as text. POST sends JSON with action.

    Usage::

        provider = HttpMemoryProvider("http://localhost:9000/memory", auth_token="secret")
        content = await provider.read()
        await provider.append("Agent learned: rate limit is 100/min")
    """

    def __init__(self, url: str, *, auth_token: str | None = None) -> None:
        self._url = url
        self._auth_token = auth_token

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        return headers

    async def read(self) -> str:
        """GET memory content from the remote endpoint."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(self._url, headers=self._headers()) as resp:
                resp.raise_for_status()
                return await resp.text()

    async def write(self, content: str) -> None:
        """POST a write action to the remote endpoint."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                self._url,
                headers=self._headers(),
                json={"content": content, "action": "write"},
            )
            resp.raise_for_status()

    async def append(self, entry: str) -> None:
        """POST an append action to the remote endpoint."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                self._url,
                headers=self._headers(),
                json={"content": entry, "action": "append"},
            )
            resp.raise_for_status()
