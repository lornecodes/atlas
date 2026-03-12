"""Tests for Phase 12: Knowledge Base & Access Control."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.knowledge.acl import KnowledgeACL
from atlas.knowledge.file_provider import FileKnowledgeProvider
from atlas.knowledge.provider import KnowledgeEntry, KnowledgeProvider
from atlas.runtime.context import AgentContext


# ── KnowledgeEntry ──────────────────────────────────────────────────

class TestKnowledgeEntry:
    def test_frozen(self):
        e = KnowledgeEntry(id="a", content="hello")
        with pytest.raises(AttributeError):
            e.content = "bye"

    def test_defaults(self):
        e = KnowledgeEntry(id="a", content="hello")
        assert e.domain == "general"
        assert e.tags == []
        assert e.metadata == {}
        assert e.created_at == ""
        assert e.updated_at == ""

    def test_to_dict(self):
        e = KnowledgeEntry(id="a", content="hello", domain="physics", tags=["pac"])
        d = e.to_dict()
        assert d["id"] == "a"
        assert d["domain"] == "physics"
        assert d["tags"] == ["pac"]

    def test_from_dict(self):
        d = {"id": "x", "content": "world", "domain": "ai", "tags": ["llm"]}
        e = KnowledgeEntry.from_dict(d)
        assert e.id == "x"
        assert e.domain == "ai"
        assert e.tags == ["llm"]

    def test_roundtrip(self):
        original = KnowledgeEntry(
            id="rt", content="roundtrip", domain="test",
            tags=["a", "b"], metadata={"key": "val"},
        )
        restored = KnowledgeEntry.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.domain == original.domain
        assert restored.tags == original.tags
        assert restored.metadata == original.metadata


# ── KnowledgeACL ────────────────────────────────────────────────────

class TestKnowledgeACL:
    def test_default_read_all(self):
        acl = KnowledgeACL()
        assert acl.can_read("anything") is True
        assert acl.can_read("physics") is True

    def test_default_write_none(self):
        acl = KnowledgeACL()
        assert acl.can_write("general") is False

    def test_explicit_read_domains(self):
        acl = KnowledgeACL(read_domains=["physics", "ai"])
        assert acl.can_read("physics") is True
        assert acl.can_read("ai") is True
        assert acl.can_read("personal") is False

    def test_explicit_write_domains(self):
        acl = KnowledgeACL(write_domains=["ai", "general"])
        assert acl.can_write("ai") is True
        assert acl.can_write("general") is True
        assert acl.can_write("physics") is False

    def test_wildcard_write(self):
        acl = KnowledgeACL(write_domains=["*"])
        assert acl.can_write("anything") is True

    def test_wildcard_write_blocked_by_protected(self):
        acl = KnowledgeACL(
            write_domains=["*"],
            protected_domains=frozenset(["physics", "personal"]),
        )
        assert acl.can_write("general") is True
        assert acl.can_write("ai") is True
        assert acl.can_write("physics") is False
        assert acl.can_write("personal") is False

    def test_explicit_write_overrides_protected(self):
        acl = KnowledgeACL(
            write_domains=["physics"],
            protected_domains=frozenset(["physics"]),
        )
        # Explicit listing bypasses protected check
        assert acl.can_write("physics") is True

    def test_from_dict_empty(self):
        acl = KnowledgeACL.from_dict(None)
        assert acl.read_domains == ["*"]
        assert acl.write_domains == []

    def test_from_dict_full(self):
        acl = KnowledgeACL.from_dict({
            "read_domains": ["physics"],
            "write_domains": ["ai"],
            "protected_domains": ["physics"],
        })
        assert acl.read_domains == ["physics"]
        assert acl.write_domains == ["ai"]
        assert "physics" in acl.protected_domains

    def test_to_dict(self):
        acl = KnowledgeACL(
            read_domains=["*"],
            write_domains=["ai"],
            protected_domains=frozenset(["physics"]),
        )
        d = acl.to_dict()
        assert d["read_domains"] == ["*"]
        assert d["write_domains"] == ["ai"]
        assert d["protected_domains"] == ["physics"]


# ── FileKnowledgeProvider ───────────────────────────────────────────

class TestFileKnowledgeProvider:
    @pytest.mark.asyncio
    async def test_create_and_get(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        entry = KnowledgeEntry(id="", content="Hello world", domain="general")
        created = await provider.create(entry)
        assert created.id.startswith("general-")
        assert created.content == "Hello world"
        assert created.created_at != ""

        retrieved = await provider.get(created.id)
        assert retrieved is not None
        assert retrieved.content == "Hello world"

    @pytest.mark.asyncio
    async def test_create_with_explicit_id(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        entry = KnowledgeEntry(id="my-entry", content="test", domain="ai")
        created = await provider.create(entry)
        assert created.id == "my-entry"
        assert (tmp_path / "kb" / "ai" / "my-entry.md").exists()

    @pytest.mark.asyncio
    async def test_search_by_content(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="", content="API rate limit is 100/min", domain="general"))
        await provider.create(KnowledgeEntry(id="", content="Database uses PostgreSQL", domain="general"))

        results = await provider.search("rate limit")
        assert len(results) == 1
        assert "rate limit" in results[0].content

    @pytest.mark.asyncio
    async def test_search_by_tags(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="", content="entry1", tags=["api", "auth"]))
        await provider.create(KnowledgeEntry(id="", content="entry2", tags=["db"]))

        results = await provider.search("entry", tags=["api"])
        assert len(results) == 1
        assert results[0].tags == ["api", "auth"]

    @pytest.mark.asyncio
    async def test_search_by_domain(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="", content="physics fact", domain="physics"))
        await provider.create(KnowledgeEntry(id="", content="ai fact", domain="ai"))

        results = await provider.search("fact", domain="physics")
        assert len(results) == 1
        assert results[0].domain == "physics"

    @pytest.mark.asyncio
    async def test_search_limit(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        for i in range(5):
            await provider.create(KnowledgeEntry(id="", content=f"fact {i}"))

        results = await provider.search("fact", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        results = await provider.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        result = await provider.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        created = await provider.create(KnowledgeEntry(id="upd", content="old", tags=["a"]))
        updated = await provider.update("upd", content="new", tags=["b"])
        assert updated is not None
        assert updated.content == "new"
        assert updated.tags == ["b"]
        assert updated.updated_at != created.created_at

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        result = await provider.update("nope", content="x")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_partial(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="p", content="original", tags=["keep"]))
        updated = await provider.update("p", content="changed")
        assert updated.content == "changed"
        assert updated.tags == ["keep"]  # tags unchanged

    @pytest.mark.asyncio
    async def test_delete(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="del", content="bye"))
        assert await provider.delete("del") is True
        assert await provider.get("del") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        assert await provider.delete("nope") is False

    @pytest.mark.asyncio
    async def test_list_entries(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="a1", content="one", domain="ai"))
        await provider.create(KnowledgeEntry(id="a2", content="two", domain="ai"))
        await provider.create(KnowledgeEntry(id="p1", content="three", domain="physics"))

        all_entries = await provider.list_entries()
        assert len(all_entries) == 3

        ai_only = await provider.list_entries(domain="ai")
        assert len(ai_only) == 2

    @pytest.mark.asyncio
    async def test_list_with_offset(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        for i in range(5):
            await provider.create(KnowledgeEntry(id=f"e{i}", content=f"entry {i}"))
        results = await provider.list_entries(offset=2, limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_concurrent_creates(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")

        async def create_n(n):
            for i in range(3):
                await provider.create(
                    KnowledgeEntry(id=f"c{n}-{i}", content=f"agent {n} entry {i}")
                )

        await asyncio.gather(create_n(1), create_n(2), create_n(3))
        all_entries = await provider.list_entries()
        assert len(all_entries) == 9

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        created = await provider.create(
            KnowledgeEntry(id="m", content="test", metadata={"source": "agent-1"})
        )
        retrieved = await provider.get("m")
        assert retrieved.metadata == {"source": "agent-1"}

    @pytest.mark.asyncio
    async def test_string_path(self, tmp_path):
        provider = FileKnowledgeProvider(str(tmp_path / "kb"))
        created = await provider.create(KnowledgeEntry(id="s", content="str path"))
        assert created.id == "s"

    @pytest.mark.asyncio
    async def test_protocol_compliance(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        assert isinstance(provider, KnowledgeProvider)


# ── KnowledgeRequirement ────────────────────────────────────────────

class TestKnowledgeRequirement:
    def test_from_dict_none(self):
        from atlas.contract.types import KnowledgeRequirement
        kr = KnowledgeRequirement.from_dict(None)
        assert kr.enabled is False

    def test_from_dict_bool_true(self):
        from atlas.contract.types import KnowledgeRequirement
        kr = KnowledgeRequirement.from_dict(True)
        assert kr.enabled is True
        assert kr.read_domains == ["*"]

    def test_from_dict_bool_false(self):
        from atlas.contract.types import KnowledgeRequirement
        kr = KnowledgeRequirement.from_dict(False)
        assert kr.enabled is False

    def test_from_dict_object(self):
        from atlas.contract.types import KnowledgeRequirement
        kr = KnowledgeRequirement.from_dict({
            "domains": ["physics"],
            "read_domains": ["physics", "ai"],
            "write_domains": ["ai"],
        })
        assert kr.enabled is True
        assert kr.domains == ["physics"]
        assert kr.read_domains == ["physics", "ai"]
        assert kr.write_domains == ["ai"]

    def test_defaults(self):
        from atlas.contract.types import KnowledgeRequirement
        kr = KnowledgeRequirement()
        assert kr.enabled is False
        assert kr.domains == []
        assert kr.read_domains == ["*"]
        assert kr.write_domains == []


# ── RequiresSpec with knowledge ─────────────────────────────────────

class TestRequiresSpecKnowledge:
    def test_from_dict_no_knowledge(self):
        from atlas.contract.types import RequiresSpec
        rs = RequiresSpec.from_dict({"platform_tools": True})
        assert rs.knowledge.enabled is False

    def test_from_dict_knowledge_true(self):
        from atlas.contract.types import RequiresSpec
        rs = RequiresSpec.from_dict({"knowledge": True})
        assert rs.knowledge.enabled is True

    def test_from_dict_knowledge_object(self):
        from atlas.contract.types import RequiresSpec
        rs = RequiresSpec.from_dict({
            "knowledge": {"domains": ["ai"], "write_domains": ["ai"]},
        })
        assert rs.knowledge.enabled is True
        assert rs.knowledge.write_domains == ["ai"]

    def test_backward_compat(self):
        from atlas.contract.types import RequiresSpec
        rs = RequiresSpec.from_dict(None)
        assert rs.knowledge.enabled is False
        assert rs.memory is False


# ── Contract Schema Validation ──────────────────────────────────────

class TestContractSchemaKnowledge:
    def test_knowledge_boolean_valid(self):
        from atlas.contract.schema import load_contract
        import tempfile
        import yaml
        contract_data = {
            "agent": {
                "name": "test-agent",
                "version": "1.0.0",
                "requires": {"knowledge": True},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(contract_data, f)
            f.flush()
            contract = load_contract(f.name)
        assert contract.requires.knowledge.enabled is True

    def test_knowledge_object_valid(self):
        from atlas.contract.schema import load_contract
        import tempfile
        import yaml
        contract_data = {
            "agent": {
                "name": "test-agent",
                "version": "1.0.0",
                "requires": {
                    "knowledge": {
                        "domains": ["physics"],
                        "read_domains": ["*"],
                        "write_domains": ["physics"],
                    }
                },
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(contract_data, f)
            f.flush()
            contract = load_contract(f.name)
        assert contract.requires.knowledge.enabled is True
        assert contract.requires.knowledge.write_domains == ["physics"]

    def test_no_knowledge_backward_compat(self):
        from atlas.contract.schema import load_contract
        import tempfile
        import yaml
        contract_data = {
            "agent": {
                "name": "test-agent",
                "version": "1.0.0",
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(contract_data, f)
            f.flush()
            contract = load_contract(f.name)
        assert contract.requires.knowledge.enabled is False


# ── AgentContext Knowledge Methods ──────────────────────────────────

class TestAgentContextKnowledge:
    @pytest.mark.asyncio
    async def test_search_without_provider(self):
        ctx = AgentContext()
        results = await ctx.knowledge_search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_without_provider(self):
        ctx = AgentContext()
        result = await ctx.knowledge_get("id")
        assert result is None

    @pytest.mark.asyncio
    async def test_store_without_provider_raises(self):
        ctx = AgentContext()
        with pytest.raises(RuntimeError, match="Knowledge not enabled"):
            await ctx.knowledge_store("content")

    @pytest.mark.asyncio
    async def test_update_without_provider_raises(self):
        ctx = AgentContext()
        with pytest.raises(RuntimeError, match="Knowledge not enabled"):
            await ctx.knowledge_update("id", content="x")

    @pytest.mark.asyncio
    async def test_search_with_provider(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="s1", content="hello world"))

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        results = await ctx.knowledge_search("hello")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_store_with_provider(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        ctx = AgentContext()
        ctx._knowledge_provider = provider

        created = await ctx.knowledge_store("test content", domain="ai", tags=["test"])
        assert created.content == "test content"
        assert created.domain == "ai"

    @pytest.mark.asyncio
    async def test_search_acl_filtered(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="pub", content="public fact", domain="general"))
        await provider.create(KnowledgeEntry(id="priv", content="private fact", domain="personal"))

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(read_domains=["general"])

        results = await ctx.knowledge_search("fact")
        assert len(results) == 1
        assert results[0].domain == "general"

    @pytest.mark.asyncio
    async def test_get_acl_denied(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="secret", content="top secret", domain="personal"))

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(read_domains=["general"])

        result = await ctx.knowledge_get("secret")
        assert result is None  # ACL blocks read

    @pytest.mark.asyncio
    async def test_store_acl_denied(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(write_domains=["general"])

        with pytest.raises(PermissionError, match="not allowed to write"):
            await ctx.knowledge_store("content", domain="physics")

    @pytest.mark.asyncio
    async def test_store_acl_allowed(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(write_domains=["ai"])

        created = await ctx.knowledge_store("content", domain="ai")
        assert created.domain == "ai"

    @pytest.mark.asyncio
    async def test_update_acl_denied(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="u1", content="original", domain="physics"))

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(write_domains=["general"])

        with pytest.raises(PermissionError, match="not allowed to write"):
            await ctx.knowledge_update("u1", content="modified")


# ── Knowledge Accumulation ──────────────────────────────────────────

class TestKnowledgeAccumulation:
    @pytest.mark.asyncio
    async def test_agent_1_stores_agent_2_finds(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")

        # Agent 1 stores knowledge
        ctx1 = AgentContext()
        ctx1._knowledge_provider = provider
        ctx1._knowledge_acl = KnowledgeACL(write_domains=["general"])
        await ctx1.knowledge_store("API rate limit is 100/min", domain="general", tags=["api"])

        # Agent 2 searches and finds it
        ctx2 = AgentContext()
        ctx2._knowledge_provider = provider
        results = await ctx2.knowledge_search("rate limit")
        assert len(results) == 1
        assert "100/min" in results[0].content

    @pytest.mark.asyncio
    async def test_multi_domain_accumulation(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(write_domains=["*"])

        await ctx.knowledge_store("physics fact", domain="physics")
        await ctx.knowledge_store("ai fact", domain="ai")
        await ctx.knowledge_store("general fact", domain="general")

        all_entries = await provider.list_entries()
        assert len(all_entries) == 3

        physics = await provider.list_entries(domain="physics")
        assert len(physics) == 1

    @pytest.mark.asyncio
    async def test_domain_isolation(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")

        # Reader agent can only see "general"
        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(
            read_domains=["general"],
            write_domains=["general"],
        )

        await ctx.knowledge_store("visible fact", domain="general")

        # Store something in physics directly (bypassing ACL)
        await provider.create(KnowledgeEntry(id="p1", content="hidden fact", domain="physics"))

        # Reader shouldn't see physics
        results = await ctx.knowledge_search("fact")
        assert len(results) == 1
        assert results[0].domain == "general"


# ── Pool Knowledge Injection ────────────────────────────────────────

class TestPoolKnowledgeInjection:
    @pytest.mark.asyncio
    async def test_injects_when_opted_in(self, tmp_path):
        """Verify the pool injection logic pattern (unit-level, no full pool)."""
        from atlas.contract.types import AgentContract, RequiresSpec, KnowledgeRequirement
        from atlas.knowledge.acl import KnowledgeACL

        provider = FileKnowledgeProvider(tmp_path / "kb")
        policy = KnowledgeACL(protected_domains=frozenset(["physics"]))

        contract = AgentContract(
            name="test", version="1.0.0",
            requires=RequiresSpec(
                knowledge=KnowledgeRequirement(
                    enabled=True,
                    read_domains=["*"],
                    write_domains=["ai"],
                ),
            ),
        )

        ctx = AgentContext()

        # Simulate pool injection logic
        if contract.requires.knowledge.enabled and provider:
            ctx._knowledge_provider = provider
            ctx._knowledge_acl = KnowledgeACL(
                read_domains=contract.requires.knowledge.read_domains,
                write_domains=contract.requires.knowledge.write_domains,
                protected_domains=policy.protected_domains,
            )

        assert ctx._knowledge_provider is provider
        assert ctx._knowledge_acl.can_read("anything") is True
        assert ctx._knowledge_acl.can_write("ai") is True
        assert ctx._knowledge_acl.can_write("physics") is False

    @pytest.mark.asyncio
    async def test_skips_when_not_opted_in(self):
        from atlas.contract.types import AgentContract, RequiresSpec

        contract = AgentContract(
            name="test", version="1.0.0",
            requires=RequiresSpec(),
        )
        ctx = AgentContext()

        # Simulate pool injection — should skip
        if contract.requires.knowledge.enabled:
            ctx._knowledge_provider = "should not be set"

        assert ctx._knowledge_provider is None


# ── DynamicLLMAgent Knowledge ───────────────────────────────────────

class TestDynamicLLMKnowledge:
    @pytest.mark.asyncio
    async def test_knowledge_injected_into_system_prompt(self, tmp_path):
        """Verify knowledge search results appear in system prompt."""
        from atlas.contract.types import AgentContract, ProviderSpec, RequiresSpec, KnowledgeRequirement

        provider = FileKnowledgeProvider(tmp_path / "kb")
        # Auto-search extracts string values from input via _extract_search_text()
        input_data = {"query": "hello"}
        await provider.create(KnowledgeEntry(
            id="k1",
            content="Knowledge about hello world usage",
            domain="general",
        ))

        contract = AgentContract(
            name="test-llm",
            version="1.0.0",
            provider=ProviderSpec(type="llm", system_prompt="You are helpful.", output_format="json"),
            requires=RequiresSpec(knowledge=KnowledgeRequirement(enabled=True)),
        )

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(read_domains=["*"])

        # Mock the anthropic client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"result": "ok"}'
        mock_response.content = [mock_text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        ctx.providers["anthropic_client"] = mock_client

        from atlas.runtime.dynamic_llm_agent import DynamicLLMAgent
        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()

        result = await agent.execute(input_data)

        # Verify system prompt included knowledge
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Knowledge about" in call_kwargs["system"]
        assert "Relevant Knowledge" in call_kwargs["system"]

    @pytest.mark.asyncio
    async def test_knowledge_tools_exposed(self, tmp_path):
        """Verify knowledge_store and knowledge_search tools appear."""
        from atlas.contract.types import AgentContract, ProviderSpec, RequiresSpec, KnowledgeRequirement

        provider = FileKnowledgeProvider(tmp_path / "kb")
        contract = AgentContract(
            name="test-llm",
            version="1.0.0",
            provider=ProviderSpec(type="llm", system_prompt="test", output_format="json"),
            requires=RequiresSpec(knowledge=KnowledgeRequirement(enabled=True)),
        )

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(read_domains=["*"], write_domains=["*"])

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_text_block = MagicMock()
        mock_text_block.type = "text"
        mock_text_block.text = '{"result": "ok"}'
        mock_response.content = [mock_text_block]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        ctx.providers["anthropic_client"] = mock_client

        from atlas.runtime.dynamic_llm_agent import DynamicLLMAgent
        agent = DynamicLLMAgent(contract, ctx)
        await agent.on_startup()
        await agent.execute({"query": "test"})

        call_kwargs = mock_client.messages.create.call_args[1]
        tool_names = [t["name"] for t in call_kwargs.get("tools", [])]
        assert "knowledge_store" in tool_names
        assert "knowledge_search" in tool_names


# ── ExecAgent Knowledge ─────────────────────────────────────────────

class TestExecAgentKnowledge:
    @pytest.mark.asyncio
    async def test_knowledge_in_envelope(self, tmp_path):
        """Verify knowledge entries appear in the stdin envelope."""
        from atlas.contract.types import AgentContract, ProviderSpec

        provider = FileKnowledgeProvider(tmp_path / "kb")
        # Auto-search extracts string values from input via _extract_search_text()
        input_data = {"message": "test"}
        await provider.create(KnowledgeEntry(
            id="k1",
            content="facts about test processing",
            domain="general",
        ))

        # Create a simple exec script that echoes the envelope
        script_dir = tmp_path / "agent"
        script_dir.mkdir()
        script = script_dir / "run.py"
        script.write_text(
            'import json, sys\n'
            'envelope = json.loads(sys.stdin.read())\n'
            'print(json.dumps({"knowledge_count": len(envelope.get("knowledge", []))}))\n'
        )

        contract = AgentContract(
            name="test-exec", version="1.0.0",
            provider=ProviderSpec(type="exec", command=["python", "run.py"]),
        )

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(read_domains=["*"])
        ctx.metadata["_agent_dir"] = str(script_dir)

        from atlas.runtime.exec_agent import ExecAgent
        agent = ExecAgent(contract, ctx)
        result = await agent.execute(input_data)
        assert result["knowledge_count"] == 1

    @pytest.mark.asyncio
    async def test_knowledge_store_from_exec(self, tmp_path):
        """Verify _knowledge_store in output creates entries."""
        from atlas.contract.types import AgentContract, ProviderSpec

        provider = FileKnowledgeProvider(tmp_path / "kb")

        script_dir = tmp_path / "agent"
        script_dir.mkdir()
        script = script_dir / "run.py"
        script.write_text(
            'import json, sys\n'
            'envelope = json.loads(sys.stdin.read())\n'
            'print(json.dumps({"result": "ok", "_knowledge_store": {"content": "learned something", "domain": "ai"}}))\n'
        )

        contract = AgentContract(
            name="test-exec", version="1.0.0",
            provider=ProviderSpec(type="exec", command=["python", "run.py"]),
        )

        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(write_domains=["ai"])
        ctx.metadata["_agent_dir"] = str(script_dir)

        from atlas.runtime.exec_agent import ExecAgent
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({"message": "test"})

        # _knowledge_store should be stripped from output
        assert "_knowledge_store" not in result
        assert result["result"] == "ok"

        # Entry should be in the knowledge base
        entries = await provider.list_entries(domain="ai")
        assert len(entries) == 1
        assert entries[0].content == "learned something"


# ── FileKnowledgeProvider Validation ──────────────────────────────

class TestFileKnowledgeProviderValidation:
    @pytest.mark.asyncio
    async def test_create_rejects_path_traversal_id(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        entry = KnowledgeEntry(id="../../etc/passwd", content="hack")
        with pytest.raises(ValueError, match="Invalid knowledge entry ID"):
            await provider.create(entry)

    @pytest.mark.asyncio
    async def test_create_rejects_path_traversal_domain(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        entry = KnowledgeEntry(id="ok", content="test", domain="../etc")
        with pytest.raises(ValueError, match="Invalid knowledge domain"):
            await provider.create(entry)

    @pytest.mark.asyncio
    async def test_get_rejects_unsafe_id(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        with pytest.raises(ValueError, match="Invalid knowledge entry ID"):
            await provider.get("../../secret")

    @pytest.mark.asyncio
    async def test_delete_rejects_unsafe_id(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        with pytest.raises(ValueError, match="Invalid knowledge entry ID"):
            await provider.delete("../bad")

    @pytest.mark.asyncio
    async def test_update_rejects_unsafe_id(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        with pytest.raises(ValueError, match="Invalid knowledge entry ID"):
            await provider.update("../../bad", content="x")

    @pytest.mark.asyncio
    async def test_create_accepts_valid_ids(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        for valid_id in ["my-entry", "entry_123", "CamelCase", "a.b.c"]:
            entry = KnowledgeEntry(id=valid_id, content="test")
            created = await provider.create(entry)
            assert created.id == valid_id


# ── Frontmatter Edge Cases ────────────────────────────────────────

class TestFrontmatterEdgeCases:
    @pytest.mark.asyncio
    async def test_corrupted_yaml_gracefully_handled(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        domain_dir = tmp_path / "kb" / "general"
        domain_dir.mkdir(parents=True)
        # Write a file with invalid YAML frontmatter
        bad_file = domain_dir / "bad.md"
        bad_file.write_text("---\n: [invalid yaml\n---\nBody content", encoding="utf-8")

        results = await provider.search("Body")
        # Should not crash — parses body only since YAML is invalid
        assert len(results) == 1
        assert "Body content" in results[0].content

    @pytest.mark.asyncio
    async def test_no_frontmatter(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        domain_dir = tmp_path / "kb" / "general"
        domain_dir.mkdir(parents=True)
        plain_file = domain_dir / "plain.md"
        plain_file.write_text("Just plain markdown content", encoding="utf-8")

        results = await provider.search("plain markdown")
        assert len(results) == 1
        assert results[0].content == "Just plain markdown content"

    @pytest.mark.asyncio
    async def test_empty_frontmatter(self, tmp_path):
        provider = FileKnowledgeProvider(tmp_path / "kb")
        domain_dir = tmp_path / "kb" / "general"
        domain_dir.mkdir(parents=True)
        f = domain_dir / "empty-fm.md"
        f.write_text("---\n---\nContent only", encoding="utf-8")

        results = await provider.search("Content only")
        assert len(results) == 1


# ── _extract_search_text ──────────────────────────────────────────

class TestExtractSearchText:
    def test_priority_keys(self):
        from atlas.runtime.dynamic_llm_agent import _extract_search_text
        data = {"query": "hello world", "other": "ignored first"}
        result = _extract_search_text(data)
        assert result.startswith("hello world")

    def test_fallback_keys(self):
        from atlas.runtime.dynamic_llm_agent import _extract_search_text
        data = {"custom_field": "fallback value"}
        assert "fallback value" in _extract_search_text(data)

    def test_max_length(self):
        from atlas.runtime.dynamic_llm_agent import _extract_search_text
        data = {"text": "x" * 300}
        result = _extract_search_text(data, max_len=50)
        assert len(result) <= 50

    def test_empty_data(self):
        from atlas.runtime.dynamic_llm_agent import _extract_search_text
        assert _extract_search_text({}) == ""

    def test_non_string_values_ignored(self):
        from atlas.runtime.dynamic_llm_agent import _extract_search_text
        data = {"count": 42, "flag": True, "text": "actual text"}
        result = _extract_search_text(data)
        assert "actual text" in result
        assert "42" not in result


# ── HttpKnowledgeProvider Error Handling ──────────────────────────

class TestHttpKnowledgeProviderErrors:
    @pytest.mark.asyncio
    async def test_error_type_exported(self):
        from atlas.knowledge.http_provider import HttpKnowledgeError
        assert issubclass(HttpKnowledgeError, Exception)


# ── ACL Edge Cases ────────────────────────────────────────────────

class TestKnowledgeACLEdgeCases:
    def test_from_dict_with_string_protected(self):
        """Ensure protected_domains handles a single string gracefully."""
        acl = KnowledgeACL.from_dict({
            "protected_domains": ["physics", "personal"],
        })
        assert "physics" in acl.protected_domains
        assert "personal" in acl.protected_domains

    def test_wildcard_read_and_write(self):
        acl = KnowledgeACL(read_domains=["*"], write_domains=["*"])
        assert acl.can_read("anything") is True
        assert acl.can_write("anything") is True

    def test_empty_write_domains(self):
        acl = KnowledgeACL(write_domains=[])
        assert acl.can_write("general") is False

    def test_protected_with_explicit_write(self):
        """Explicit domain listing should bypass protected check."""
        acl = KnowledgeACL(
            write_domains=["physics"],
            protected_domains=frozenset(["physics"]),
        )
        assert acl.can_write("physics") is True


# ── Full Lifecycle E2E ────────────────────────────────────────────

class TestKnowledgeLifecycleE2E:
    @pytest.mark.asyncio
    async def test_full_crud_lifecycle(self, tmp_path):
        """Create → Get → Update → Search → Delete lifecycle."""
        provider = FileKnowledgeProvider(tmp_path / "kb")

        # Create
        entry = KnowledgeEntry(id="lifecycle-1", content="initial content", domain="general", tags=["test"])
        created = await provider.create(entry)
        assert created.id == "lifecycle-1"
        assert created.created_at != ""

        # Get
        fetched = await provider.get("lifecycle-1")
        assert fetched is not None
        assert fetched.content == "initial content"

        # Update
        updated = await provider.update("lifecycle-1", content="updated content", tags=["test", "updated"])
        assert updated.content == "updated content"
        assert updated.tags == ["test", "updated"]
        assert updated.created_at == created.created_at
        assert updated.updated_at != created.updated_at

        # Search
        results = await provider.search("updated content")
        assert len(results) == 1
        assert results[0].id == "lifecycle-1"

        # List
        all_entries = await provider.list_entries()
        assert len(all_entries) == 1

        # Delete
        assert await provider.delete("lifecycle-1") is True
        assert await provider.get("lifecycle-1") is None
        assert await provider.list_entries() == []

    @pytest.mark.asyncio
    async def test_multi_agent_knowledge_sharing(self, tmp_path):
        """Simulate multiple agents with different ACLs sharing knowledge."""
        provider = FileKnowledgeProvider(tmp_path / "kb")

        # Writer agent (can write to ai domain)
        writer_ctx = AgentContext()
        writer_ctx._knowledge_provider = provider
        writer_ctx._knowledge_acl = KnowledgeACL(
            read_domains=["*"],
            write_domains=["ai"],
        )

        # Reader agent (can only read general)
        reader_ctx = AgentContext()
        reader_ctx._knowledge_provider = provider
        reader_ctx._knowledge_acl = KnowledgeACL(
            read_domains=["general"],
            write_domains=[],
        )

        # Admin agent (wildcard write, but physics is protected)
        admin_ctx = AgentContext()
        admin_ctx._knowledge_provider = provider
        admin_ctx._knowledge_acl = KnowledgeACL(
            read_domains=["*"],
            write_domains=["*"],
            protected_domains=frozenset(["physics"]),
        )

        # Writer stores in ai domain
        await writer_ctx.knowledge_store("LLM best practices", domain="ai", tags=["llm"])

        # Writer cannot write to physics
        with pytest.raises(PermissionError):
            await writer_ctx.knowledge_store("physics note", domain="physics")

        # Reader can't see ai domain
        results = await reader_ctx.knowledge_search("LLM")
        assert len(results) == 0

        # Reader can't write at all
        with pytest.raises(PermissionError):
            await reader_ctx.knowledge_store("test", domain="general")

        # Admin can write to ai (via wildcard)
        await admin_ctx.knowledge_store("admin note", domain="ai")

        # Admin cannot write to physics (protected)
        with pytest.raises(PermissionError):
            await admin_ctx.knowledge_store("physics hack", domain="physics")

        # Admin can read everything
        results = await admin_ctx.knowledge_search("LLM")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_knowledge_survives_persistence(self, tmp_path):
        """Data persists across provider instances (same directory)."""
        root = tmp_path / "kb"

        # First provider instance writes
        p1 = FileKnowledgeProvider(root)
        await p1.create(KnowledgeEntry(id="persist-1", content="survived", domain="general"))

        # Second provider instance reads
        p2 = FileKnowledgeProvider(root)
        result = await p2.get("persist-1")
        assert result is not None
        assert result.content == "survived"

        # Search also works
        results = await p2.search("survived")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_cross_domain_search(self, tmp_path):
        """Search without domain filter finds entries across all domains."""
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="d1", content="quantum physics fact", domain="physics"))
        await provider.create(KnowledgeEntry(id="d2", content="quantum computing fact", domain="computing"))
        await provider.create(KnowledgeEntry(id="d3", content="classical music fact", domain="general"))

        results = await provider.search("quantum")
        assert len(results) == 2
        domains = {r.domain for r in results}
        assert domains == {"physics", "computing"}

    @pytest.mark.asyncio
    async def test_tag_search_across_domains(self, tmp_path):
        """Tag filtering works across domains."""
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="t1", content="api entry one", domain="ai", tags=["api", "auth"]))
        await provider.create(KnowledgeEntry(id="t2", content="api entry two", domain="general", tags=["api"]))
        await provider.create(KnowledgeEntry(id="t3", content="db entry", domain="general", tags=["db"]))

        results = await provider.search("entry", tags=["api"])
        assert len(results) == 2

        results = await provider.search("entry", tags=["auth"])
        assert len(results) == 1
        assert results[0].id == "t1"

    @pytest.mark.asyncio
    async def test_update_preserves_domain(self, tmp_path):
        """Updating an entry should not change its domain or file location."""
        provider = FileKnowledgeProvider(tmp_path / "kb")
        await provider.create(KnowledgeEntry(id="dom-test", content="original", domain="physics"))

        updated = await provider.update("dom-test", content="modified")
        assert updated.domain == "physics"

        # File should still be in physics dir
        assert (tmp_path / "kb" / "physics" / "dom-test.md").exists()
        retrieved = await provider.get("dom-test")
        assert retrieved.content == "modified"
        assert retrieved.domain == "physics"

    @pytest.mark.asyncio
    async def test_knowledge_context_update_then_search(self, tmp_path):
        """Store → update → search reflects updated content."""
        provider = FileKnowledgeProvider(tmp_path / "kb")
        ctx = AgentContext()
        ctx._knowledge_provider = provider
        ctx._knowledge_acl = KnowledgeACL(read_domains=["*"], write_domains=["*"])

        created = await ctx.knowledge_store("rate limit is 100/min", domain="general")
        await ctx.knowledge_update(created.id, content="rate limit is 200/min")

        results = await ctx.knowledge_search("200/min")
        assert len(results) == 1
        assert "200/min" in results[0].content

        # Old content should not match
        results = await ctx.knowledge_search("100/min")
        assert len(results) == 0
