"""E2E integration tests — knowledge + memory providers wired together.

These tests exercise the full stack: create/search/update/delete knowledge
entries through FileKnowledgeProvider, read/write/append through
FileMemoryProvider, and error propagation through HttpMemoryProvider.
No mocking of internal components.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas.knowledge.file_provider import FileKnowledgeProvider
from atlas.knowledge.provider import KnowledgeEntry
from atlas.memory.file_provider import FileMemoryProvider
from atlas.memory.http_provider import HttpMemoryProvider


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def knowledge_root(tmp_path: Path) -> Path:
    return tmp_path / "knowledge"


@pytest.fixture
def knowledge(knowledge_root: Path) -> FileKnowledgeProvider:
    return FileKnowledgeProvider(knowledge_root)


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.md"


@pytest.fixture
def memory(memory_path: Path) -> FileMemoryProvider:
    return FileMemoryProvider(memory_path)


# ---------------------------------------------------------------------------
# Knowledge CRUD
# ---------------------------------------------------------------------------


class TestKnowledgeCRUD:
    """Create, read, update, delete through FileKnowledgeProvider."""

    @pytest.mark.asyncio
    async def test_knowledge_create_and_get(self, knowledge: FileKnowledgeProvider):
        entry = KnowledgeEntry(
            id="test-entry-001",
            content="PAC theory describes recursive entropy.",
            domain="physics",
            tags=["pac", "entropy"],
            metadata={"source": "dft-paper"},
        )
        created = await knowledge.create(entry)

        assert created.id == "test-entry-001"
        assert created.content == "PAC theory describes recursive entropy."
        assert created.domain == "physics"
        assert created.tags == ["pac", "entropy"]
        assert created.metadata == {"source": "dft-paper"}
        assert created.created_at != ""
        assert created.updated_at != ""

        fetched = await knowledge.get("test-entry-001")
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.content == created.content
        assert fetched.domain == created.domain
        assert fetched.tags == created.tags
        assert fetched.metadata == created.metadata

    @pytest.mark.asyncio
    async def test_knowledge_update(self, knowledge: FileKnowledgeProvider):
        entry = KnowledgeEntry(
            id="update-me",
            content="Original content.",
            domain="general",
            tags=["draft"],
        )
        created = await knowledge.create(entry)
        original_updated_at = created.updated_at

        updated = await knowledge.update(
            "update-me", content="Revised content with new insights."
        )
        assert updated is not None
        assert updated.content == "Revised content with new insights."
        assert updated.id == "update-me"
        assert updated.domain == "general"
        # updated_at should change (or at least not be empty)
        assert updated.updated_at != ""

        # Verify via get
        fetched = await knowledge.get("update-me")
        assert fetched is not None
        assert fetched.content == "Revised content with new insights."

    @pytest.mark.asyncio
    async def test_knowledge_delete(self, knowledge: FileKnowledgeProvider):
        entry = KnowledgeEntry(
            id="delete-me",
            content="Ephemeral knowledge.",
            domain="general",
        )
        await knowledge.create(entry)

        # Verify it exists
        assert await knowledge.get("delete-me") is not None

        # Delete
        deleted = await knowledge.delete("delete-me")
        assert deleted is True

        # Verify it is gone
        assert await knowledge.get("delete-me") is None

        # Deleting again returns False
        assert await knowledge.delete("delete-me") is False


# ---------------------------------------------------------------------------
# Knowledge search
# ---------------------------------------------------------------------------


class TestKnowledgeSearch:
    """Search by query, domain, and tags."""

    @pytest.mark.asyncio
    async def test_knowledge_search_by_query(
        self, knowledge: FileKnowledgeProvider
    ):
        entries = [
            KnowledgeEntry(id="e1", content="Entropy flows downhill.", domain="physics"),
            KnowledgeEntry(id="e2", content="Agent routing logic.", domain="ai"),
            KnowledgeEntry(id="e3", content="Docker container setup.", domain="ops"),
        ]
        for e in entries:
            await knowledge.create(e)

        results = await knowledge.search("entropy")
        assert len(results) == 1
        assert results[0].id == "e1"

    @pytest.mark.asyncio
    async def test_knowledge_search_by_domain(
        self, knowledge: FileKnowledgeProvider
    ):
        entries = [
            KnowledgeEntry(id="p1", content="Recursive field.", domain="physics"),
            KnowledgeEntry(id="p2", content="Recursive agent.", domain="ai"),
            KnowledgeEntry(id="p3", content="Recursive build.", domain="physics"),
        ]
        for e in entries:
            await knowledge.create(e)

        results = await knowledge.search("recursive", domain="physics")
        assert len(results) == 2
        assert all(r.domain == "physics" for r in results)
        result_ids = {r.id for r in results}
        assert result_ids == {"p1", "p3"}

    @pytest.mark.asyncio
    async def test_knowledge_search_by_tags(
        self, knowledge: FileKnowledgeProvider
    ):
        entries = [
            KnowledgeEntry(
                id="t1", content="PAC math.", domain="physics", tags=["pac", "math"]
            ),
            KnowledgeEntry(
                id="t2", content="SEC math.", domain="physics", tags=["sec", "math"]
            ),
            KnowledgeEntry(
                id="t3", content="PAC code.", domain="ai", tags=["pac", "code"]
            ),
        ]
        for e in entries:
            await knowledge.create(e)

        # Search with tag filter — only entries with "sec" tag
        results = await knowledge.search("math", tags=["sec"])
        assert len(results) == 1
        assert results[0].id == "t2"


# ---------------------------------------------------------------------------
# Knowledge listing + pagination
# ---------------------------------------------------------------------------


class TestKnowledgeListing:

    @pytest.mark.asyncio
    async def test_knowledge_list_pagination(
        self, knowledge: FileKnowledgeProvider
    ):
        for i in range(5):
            await knowledge.create(
                KnowledgeEntry(
                    id=f"page-{i:02d}",
                    content=f"Entry number {i}.",
                    domain="general",
                )
            )

        page = await knowledge.list_entries(limit=2, offset=2)
        assert len(page) == 2

        # Verify these are the 3rd and 4th entries (sorted by filename)
        all_entries = await knowledge.list_entries(limit=50)
        assert len(all_entries) == 5
        expected_ids = [e.id for e in all_entries[2:4]]
        actual_ids = [e.id for e in page]
        assert actual_ids == expected_ids


# ---------------------------------------------------------------------------
# Knowledge persistence
# ---------------------------------------------------------------------------


class TestKnowledgePersistence:

    @pytest.mark.asyncio
    async def test_knowledge_persistence(self, knowledge_root: Path):
        """Entries survive provider recreation on the same directory."""
        provider1 = FileKnowledgeProvider(knowledge_root)
        await provider1.create(
            KnowledgeEntry(id="persist-1", content="Survives restart.", domain="general")
        )
        await provider1.create(
            KnowledgeEntry(id="persist-2", content="Also survives.", domain="physics")
        )

        # New provider instance on the same root
        provider2 = FileKnowledgeProvider(knowledge_root)

        entry1 = await provider2.get("persist-1")
        assert entry1 is not None
        assert entry1.content == "Survives restart."

        entry2 = await provider2.get("persist-2")
        assert entry2 is not None
        assert entry2.content == "Also survives."
        assert entry2.domain == "physics"


# ---------------------------------------------------------------------------
# File memory provider
# ---------------------------------------------------------------------------


class TestFileMemory:
    """FileMemoryProvider read/write/append lifecycle."""

    @pytest.mark.asyncio
    async def test_memory_write_and_read(self, memory: FileMemoryProvider):
        await memory.write("hello")
        content = await memory.read()
        assert content == "hello"

    @pytest.mark.asyncio
    async def test_memory_append(self, memory: FileMemoryProvider):
        await memory.write("line1")
        await memory.append("line2")
        content = await memory.read()
        assert "line1" in content
        assert "line2" in content
        # Append adds newline separator between entries
        lines = content.strip().splitlines()
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_memory_persistence(self, memory_path: Path):
        """Memory content survives provider recreation on the same file."""
        provider1 = FileMemoryProvider(memory_path)
        await provider1.write("persistent state")

        provider2 = FileMemoryProvider(memory_path)
        content = await provider2.read()
        assert content == "persistent state"


# ---------------------------------------------------------------------------
# HTTP memory error propagation
# ---------------------------------------------------------------------------


class TestHttpMemoryErrors:

    @pytest.mark.asyncio
    async def test_http_memory_error_propagation(self):
        """HttpMemoryProvider propagates server errors as ClientResponseError."""
        import aiohttp
        from aiohttp import web
        from aiohttp.test_utils import AioHTTPTestCase, TestServer

        async def handle_post(request: web.Request) -> web.Response:
            return web.Response(status=500, text="Internal Server Error")

        app = web.Application()
        app.router.add_post("/memory", handle_post)

        server = TestServer(app)
        await server.start_server()

        try:
            url = f"http://localhost:{server.port}/memory"
            provider = HttpMemoryProvider(url)

            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await provider.write("this should fail")
            assert exc_info.value.status == 500

            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await provider.append("this should also fail")
            assert exc_info.value.status == 500
        finally:
            await server.close()
