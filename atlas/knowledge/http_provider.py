"""HttpKnowledgeProvider — REST hook for external knowledge systems."""

from __future__ import annotations

from typing import Any

from atlas.knowledge.provider import KnowledgeEntry


class HttpKnowledgeError(Exception):
    """Raised when an HTTP knowledge request fails."""


class HttpKnowledgeProvider:
    """HTTP-based knowledge provider delegating to an external REST API.

    Endpoints:
        GET  {url}/search?q=...&domain=...&tags=...&limit=...
        GET  {url}/entries/{id}
        POST {url}/entries
        PATCH {url}/entries/{id}
        DELETE {url}/entries/{id}
        GET  {url}/entries?domain=...&limit=...&offset=...
    """

    def __init__(self, url: str, *, auth_token: str | None = None) -> None:
        self._url = url.rstrip("/")
        self._headers: dict[str, str] = {}
        if auth_token:
            self._headers["Authorization"] = f"Bearer {auth_token}"

    async def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[KnowledgeEntry]:
        import aiohttp

        params: dict[str, str] = {"q": query, "limit": str(limit)}
        if domain:
            params["domain"] = domain
        if tags:
            params["tags"] = ",".join(tags)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/search", params=params, headers=self._headers
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return [KnowledgeEntry.from_dict(d) for d in data]
        except aiohttp.ClientError as e:
            raise HttpKnowledgeError(f"Knowledge search failed: {e}") from e

    async def get(self, entry_id: str) -> KnowledgeEntry | None:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/entries/{entry_id}", headers=self._headers
                ) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    data = await resp.json()
                    return KnowledgeEntry.from_dict(data)
        except aiohttp.ClientError as e:
            raise HttpKnowledgeError(f"Knowledge get failed: {e}") from e

    async def create(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._url}/entries",
                    json=entry.to_dict(),
                    headers=self._headers,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return KnowledgeEntry.from_dict(data)
        except aiohttp.ClientError as e:
            raise HttpKnowledgeError(f"Knowledge create failed: {e}") from e

    async def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEntry | None:
        import aiohttp

        body: dict[str, Any] = {}
        if content is not None:
            body["content"] = content
        if tags is not None:
            body["tags"] = tags
        if metadata is not None:
            body["metadata"] = metadata

        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self._url}/entries/{entry_id}",
                    json=body,
                    headers=self._headers,
                ) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    data = await resp.json()
                    return KnowledgeEntry.from_dict(data)
        except aiohttp.ClientError as e:
            raise HttpKnowledgeError(f"Knowledge update failed: {e}") from e

    async def list_entries(
        self,
        *,
        domain: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeEntry]:
        import aiohttp

        params: dict[str, str] = {"limit": str(limit), "offset": str(offset)}
        if domain:
            params["domain"] = domain

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._url}/entries", params=params, headers=self._headers
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return [KnowledgeEntry.from_dict(d) for d in data]
        except aiohttp.ClientError as e:
            raise HttpKnowledgeError(f"Knowledge list failed: {e}") from e

    async def delete(self, entry_id: str) -> bool:
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self._url}/entries/{entry_id}", headers=self._headers
                ) as resp:
                    return resp.status == 200
        except aiohttp.ClientError as e:
            raise HttpKnowledgeError(f"Knowledge delete failed: {e}") from e
