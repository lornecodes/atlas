"""FileKnowledgeProvider — markdown files with YAML frontmatter.

Storage layout (mini-Kronos):
    knowledge/
    ├── general/
    │   └── entry-abc123.md
    ├── physics/
    │   └── pac-theory.md
    └── ai-systems/
        └── grim-patterns.md
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from atlas.knowledge.provider import KnowledgeEntry


import re

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


def _validate_id(entry_id: str) -> str:
    """Validate entry ID is safe for use as a filename."""
    if not _SAFE_ID_RE.match(entry_id):
        raise ValueError(
            f"Invalid knowledge entry ID: {entry_id!r} — "
            "must be alphanumeric with .-_ (max 128 chars)"
        )
    return entry_id


def _validate_domain(domain: str) -> str:
    """Validate domain name is safe for use as a directory name."""
    if not _SAFE_ID_RE.match(domain):
        raise ValueError(
            f"Invalid knowledge domain: {domain!r} — "
            "must be alphanumeric with .-_ (max 128 chars)"
        )
    return domain


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter and body from markdown text."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    import yaml

    try:
        frontmatter = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        return {}, text
    body = text[end + 3 :].strip()
    return frontmatter, body


def _render_entry(entry: KnowledgeEntry) -> str:
    """Render a KnowledgeEntry to markdown with YAML frontmatter."""
    import yaml

    fm: dict[str, Any] = {
        "id": entry.id,
        "domain": entry.domain,
    }
    if entry.tags:
        fm["tags"] = list(entry.tags)
    if entry.metadata:
        fm["metadata"] = dict(entry.metadata)
    if entry.created_at:
        fm["created_at"] = entry.created_at
    if entry.updated_at:
        fm["updated_at"] = entry.updated_at

    header = yaml.dump(fm, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{header}\n---\n{entry.content}\n"


def _entry_from_file(fm: dict[str, Any], body: str) -> KnowledgeEntry:
    """Build a KnowledgeEntry from parsed frontmatter + body."""
    return KnowledgeEntry(
        id=fm.get("id", ""),
        content=body,
        domain=fm.get("domain", "general"),
        tags=fm.get("tags", []),
        metadata=fm.get("metadata", {}),
        created_at=str(fm.get("created_at", "")),
        updated_at=str(fm.get("updated_at", "")),
    )


class FileKnowledgeProvider:
    """File-backed knowledge provider using markdown + YAML frontmatter."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._lock = asyncio.Lock()

    async def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        tags: list[str] | None = None,
        limit: int = 10,
    ) -> list[KnowledgeEntry]:
        """Case-insensitive substring search on content + tags."""
        results: list[KnowledgeEntry] = []
        query_lower = query.lower()

        dirs = [self._root / domain] if domain else self._iter_domain_dirs()
        for d in dirs:
            if not d.is_dir():
                continue
            for f in d.glob("*.md"):
                fm, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
                entry = _entry_from_file(fm, body)

                # Tag filter
                if tags and not any(t in entry.tags for t in tags):
                    continue

                # Text match on content + tags
                searchable = (entry.content + " " + " ".join(entry.tags)).lower()
                if query_lower in searchable:
                    results.append(entry)
                    if len(results) >= limit:
                        return results

        return results

    async def get(self, entry_id: str) -> KnowledgeEntry | None:
        """Retrieve entry by ID (scans all domains)."""
        _validate_id(entry_id)
        for d in self._iter_domain_dirs():
            path = d / f"{entry_id}.md"
            if path.is_file():
                fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
                return _entry_from_file(fm, body)
        return None

    async def create(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Create a new entry. Generates ID if empty."""
        async with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            _validate_domain(entry.domain)
            entry_id = entry.id or f"{entry.domain}-{uuid4().hex[:8]}"
            _validate_id(entry_id)
            created = KnowledgeEntry(
                id=entry_id,
                content=entry.content,
                domain=entry.domain,
                tags=list(entry.tags),
                metadata=dict(entry.metadata),
                created_at=now,
                updated_at=now,
            )
            domain_dir = self._root / created.domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            path = domain_dir / f"{created.id}.md"
            path.write_text(_render_entry(created), encoding="utf-8")
            return created

    async def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeEntry | None:
        """Update fields on an existing entry."""
        _validate_id(entry_id)
        async with self._lock:
            for d in self._iter_domain_dirs():
                path = d / f"{entry_id}.md"
                if not path.is_file():
                    continue
                fm, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
                existing = _entry_from_file(fm, body)
                now = datetime.now(timezone.utc).isoformat()
                updated = KnowledgeEntry(
                    id=existing.id,
                    content=content if content is not None else existing.content,
                    domain=existing.domain,
                    tags=list(tags) if tags is not None else list(existing.tags),
                    metadata=(
                        dict(metadata)
                        if metadata is not None
                        else dict(existing.metadata)
                    ),
                    created_at=existing.created_at,
                    updated_at=now,
                )
                path.write_text(_render_entry(updated), encoding="utf-8")
                return updated
        return None

    async def list_entries(
        self,
        *,
        domain: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeEntry]:
        """List entries, optionally filtered by domain."""
        results: list[KnowledgeEntry] = []
        dirs = [self._root / domain] if domain else self._iter_domain_dirs()
        skipped = 0
        for d in dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.md")):
                if skipped < offset:
                    skipped += 1
                    continue
                fm, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
                results.append(_entry_from_file(fm, body))
                if len(results) >= limit:
                    return results
        return results

    async def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        _validate_id(entry_id)
        async with self._lock:
            for d in self._iter_domain_dirs():
                path = d / f"{entry_id}.md"
                if path.is_file():
                    path.unlink()
                    return True
        return False

    def _iter_domain_dirs(self) -> list[Path]:
        """List domain subdirectories."""
        if not self._root.is_dir():
            return []
        return [d for d in sorted(self._root.iterdir()) if d.is_dir()]
