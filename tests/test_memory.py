"""Tests for shared memory system — FileMemoryProvider and context integration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from atlas.memory.file_provider import FileMemoryProvider
from atlas.memory.provider import MemoryProvider
from atlas.runtime.context import AgentContext


class TestFileMemoryProvider:
    """FileMemoryProvider basic operations."""

    @pytest.mark.asyncio
    async def test_read_nonexistent_returns_empty(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")
        content = await provider.read()
        assert content == ""

    @pytest.mark.asyncio
    async def test_write_then_read(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")
        await provider.write("Hello world")
        content = await provider.read()
        assert content == "Hello world"

    @pytest.mark.asyncio
    async def test_write_overwrites(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")
        await provider.write("first")
        await provider.write("second")
        content = await provider.read()
        assert content == "second"

    @pytest.mark.asyncio
    async def test_append_to_empty(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")
        await provider.append("entry one")
        content = await provider.read()
        assert "entry one" in content

    @pytest.mark.asyncio
    async def test_append_multiple(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")
        await provider.append("first learning")
        await provider.append("second learning")
        content = await provider.read()
        assert "first learning" in content
        assert "second learning" in content

    @pytest.mark.asyncio
    async def test_append_adds_newline_separator(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")
        await provider.write("existing content")
        await provider.append("new entry")
        content = await provider.read()
        assert "existing content\nnew entry" in content

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "deep" / "nested" / "memory.md")
        await provider.write("test")
        assert (tmp_path / "deep" / "nested" / "memory.md").exists()

    @pytest.mark.asyncio
    async def test_string_path_accepted(self, tmp_path):
        provider = FileMemoryProvider(str(tmp_path / "memory.md"))
        await provider.write("works")
        assert await provider.read() == "works"

    @pytest.mark.asyncio
    async def test_protocol_compliance(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")
        assert isinstance(provider, MemoryProvider)


class TestFileMemoryConcurrency:
    """Concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_appends(self, tmp_path):
        provider = FileMemoryProvider(tmp_path / "memory.md")

        async def append_n(n):
            for i in range(5):
                await provider.append(f"agent-{n}-entry-{i}")

        await asyncio.gather(append_n(1), append_n(2), append_n(3))
        content = await provider.read()

        # All 15 entries should be present
        for n in range(1, 4):
            for i in range(5):
                assert f"agent-{n}-entry-{i}" in content


class TestAgentContextMemory:
    """Memory methods on AgentContext."""

    @pytest.mark.asyncio
    async def test_memory_read_with_provider(self):
        mock = AsyncMock()
        mock.read = AsyncMock(return_value="saved knowledge")
        ctx = AgentContext()
        ctx._memory_provider = mock
        result = await ctx.memory_read()
        assert result == "saved knowledge"

    @pytest.mark.asyncio
    async def test_memory_read_without_provider(self):
        ctx = AgentContext()
        result = await ctx.memory_read()
        assert result == ""

    @pytest.mark.asyncio
    async def test_memory_write_with_provider(self):
        mock = AsyncMock()
        ctx = AgentContext()
        ctx._memory_provider = mock
        await ctx.memory_write("new content")
        mock.write.assert_called_once_with("new content")

    @pytest.mark.asyncio
    async def test_memory_write_without_provider_raises(self):
        ctx = AgentContext()
        with pytest.raises(RuntimeError, match="Memory not enabled"):
            await ctx.memory_write("content")

    @pytest.mark.asyncio
    async def test_memory_append_with_provider(self):
        mock = AsyncMock()
        ctx = AgentContext()
        ctx._memory_provider = mock
        await ctx.memory_append("new entry")
        mock.append.assert_called_once_with("new entry")

    @pytest.mark.asyncio
    async def test_memory_append_without_provider_raises(self):
        ctx = AgentContext()
        with pytest.raises(RuntimeError, match="Memory not enabled"):
            await ctx.memory_append("entry")


class TestMemoryAccumulation:
    """Test that multiple agents build on shared memory."""

    @pytest.mark.asyncio
    async def test_agents_share_memory(self, tmp_path):
        """Simulate two agents using the same memory provider."""
        provider = FileMemoryProvider(tmp_path / "memory.md")

        # Agent 1 writes
        ctx1 = AgentContext()
        ctx1._memory_provider = provider
        await ctx1.memory_append("Agent 1 learned: API rate limit is 100/min")

        # Agent 2 reads what agent 1 wrote
        ctx2 = AgentContext()
        ctx2._memory_provider = provider
        content = await ctx2.memory_read()
        assert "API rate limit is 100/min" in content

        # Agent 2 adds its own learning
        await ctx2.memory_append("Agent 2 learned: retry after 60 seconds")

        # Agent 3 reads both
        ctx3 = AgentContext()
        ctx3._memory_provider = provider
        content = await ctx3.memory_read()
        assert "API rate limit is 100/min" in content
        assert "retry after 60 seconds" in content
