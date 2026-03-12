"""KnowledgeProvider protocol and KnowledgeEntry dataclass.

Structured, searchable, domain-scoped knowledge for Atlas agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class KnowledgeEntry:
    """A single knowledge entry with metadata."""

    id: str
    content: str
    domain: str = "general"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""  # ISO 8601
    updated_at: str = ""  # ISO 8601

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "domain": self.domain,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> KnowledgeEntry:
        return KnowledgeEntry(
            id=d.get("id", ""),
            content=d.get("content", ""),
            domain=d.get("domain", "general"),
            tags=d.get("tags", []),
            metadata=d.get("metadata", {}),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@runtime_checkable
class KnowledgeProvider(Protocol):
    """Read/write structured knowledge entries.

    Implementations: FileKnowledgeProvider (default), HttpKnowledgeProvider,
    MCPKnowledgeProvider.
    """

    async def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[KnowledgeEntry]: ...

    async def get(self, entry_id: str) -> KnowledgeEntry | None: ...

    async def create(self, entry: KnowledgeEntry) -> KnowledgeEntry: ...

    async def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEntry | None: ...

    async def list_entries(
        self,
        *,
        domain: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeEntry]: ...

    async def delete(self, entry_id: str) -> bool: ...
