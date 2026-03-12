"""MCPKnowledgeProvider — delegates to an MCP server exposing knowledge tools."""

from __future__ import annotations

import json
from typing import Any

from atlas.knowledge.provider import KnowledgeEntry


class MCPKnowledgeProvider:
    """Knowledge provider that delegates to an MCP server.

    Expects the MCP server to expose tools:
        knowledge_search, knowledge_get, knowledge_create,
        knowledge_update, knowledge_list, knowledge_delete
    """

    def __init__(self, session: Any) -> None:
        self._session = session

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool and return parsed JSON result."""
        result = await self._session.call_tool(name, arguments=arguments)
        if hasattr(result, "content") and result.content:
            text = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
            return json.loads(text)
        return None

    async def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[KnowledgeEntry]:
        args: dict[str, Any] = {"query": query, "limit": limit}
        if domain:
            args["domain"] = domain
        if tags:
            args["tags"] = tags
        data = await self._call_tool("knowledge_search", args)
        if not data or not isinstance(data, list):
            return []
        return [KnowledgeEntry.from_dict(d) for d in data]

    async def get(self, entry_id: str) -> KnowledgeEntry | None:
        data = await self._call_tool("knowledge_get", {"entry_id": entry_id})
        if not data:
            return None
        return KnowledgeEntry.from_dict(data)

    async def create(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        data = await self._call_tool("knowledge_create", {"entry": entry.to_dict()})
        return KnowledgeEntry.from_dict(data)

    async def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEntry | None:
        args: dict[str, Any] = {"entry_id": entry_id}
        if content is not None:
            args["content"] = content
        if tags is not None:
            args["tags"] = tags
        if metadata is not None:
            args["metadata"] = metadata
        data = await self._call_tool("knowledge_update", args)
        if not data:
            return None
        return KnowledgeEntry.from_dict(data)

    async def list_entries(
        self,
        *,
        domain: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeEntry]:
        args: dict[str, Any] = {"limit": limit, "offset": offset}
        if domain:
            args["domain"] = domain
        data = await self._call_tool("knowledge_list", args)
        if not data or not isinstance(data, list):
            return []
        return [KnowledgeEntry.from_dict(d) for d in data]

    async def delete(self, entry_id: str) -> bool:
        data = await self._call_tool("knowledge_delete", {"entry_id": entry_id})
        return bool(data)
