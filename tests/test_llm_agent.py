"""Tests for LLMAgent base class, LLM agents, provider config, and store retry fields."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from atlas.contract.types import AgentContract, ModelSpec
from atlas.llm.anthropic import MODEL_TIERS, AnthropicProvider, model_for_preference
from atlas.llm.provider import LLMProvider, LLMResponse
from atlas.pool.job import JobData
from atlas.runtime.context import AgentContext
from atlas.runtime.llm_agent import LLMAgent
from atlas.store.job_store import JobStore


# === Mock Provider ===


class MockProvider:
    """A mock LLM provider for testing."""

    def __init__(self, response_text: str = "mock response"):
        self.response_text = response_text
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> LLMResponse:
        self.calls.append(prompt)
        return LLMResponse(
            text=self.response_text,
            input_tokens=10,
            output_tokens=5,
            model="mock-model",
        )


# === Concrete test agent using LLMAgent ===


class _SampleAgent(LLMAgent):
    """A minimal LLMAgent for testing the base class."""

    def build_prompt(self, input_data: dict[str, Any]) -> str:
        return f"Process: {input_data['text']}"

    def parse_response(
        self, response: LLMResponse, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        return {"result": response.text, "model": response.model}


# === LLMAgent Base Tests ===


class TestLLMAgentBase:
    def _make_agent(self, provider: MockProvider | None = None) -> _SampleAgent:
        contract = AgentContract(
            name="test-llm",
            version="1.0.0",
            model=ModelSpec(preference="fast"),
        )
        ctx = AgentContext()
        agent = _SampleAgent(contract, ctx)
        if provider:
            agent._provider = provider
        return agent

    async def test_execute_calls_provider(self):
        provider = MockProvider("hello world")
        agent = self._make_agent(provider)
        result = await agent.execute({"text": "test input"})
        assert result == {"result": "hello world", "model": "mock-model"}
        assert len(provider.calls) == 1
        assert "test input" in provider.calls[0]

    async def test_execute_without_startup_raises(self):
        agent = self._make_agent()
        with pytest.raises(RuntimeError, match="not started"):
            await agent.execute({"text": "test"})

    async def test_build_prompt_called(self):
        provider = MockProvider()
        agent = self._make_agent(provider)
        await agent.execute({"text": "specific input"})
        assert "specific input" in provider.calls[0]

    async def test_multiple_executions_reuse_provider(self):
        """Warm slot model: provider created once, reused across calls."""
        provider = MockProvider("ok")
        agent = self._make_agent(provider)
        await agent.execute({"text": "first"})
        await agent.execute({"text": "second"})
        assert len(provider.calls) == 2


# === LLM Summarizer Tests ===


class TestLLMSummarizer:
    def _make_agent(self, provider: MockProvider) -> Any:
        import importlib.util
        from pathlib import Path

        agent_dir = Path(__file__).parent.parent / "agents" / "llm-summarizer"
        spec = importlib.util.spec_from_file_location(
            "llm_summarizer_agent", agent_dir / "agent.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        contract = AgentContract(
            name="llm-summarizer",
            version="1.0.0",
            model=ModelSpec(preference="fast"),
        )
        agent = mod.LLMSummarizerAgent(contract, AgentContext())
        agent._provider = provider
        return agent

    async def test_summarizer_prompt(self):
        provider = MockProvider("A concise summary.")
        agent = self._make_agent(provider)
        result = await agent.execute({"text": "Long text here..."})
        assert result["summary"] == "A concise summary."
        assert result["model"] == "mock-model"
        assert "Summarize" in provider.calls[0]
        assert "Long text here..." in provider.calls[0]

    async def test_summarizer_max_sentences(self):
        provider = MockProvider("Short.")
        agent = self._make_agent(provider)
        await agent.execute({"text": "Some text", "max_sentences": 1})
        assert "1 sentences" in provider.calls[0]

    async def test_summarizer_default_max_sentences(self):
        provider = MockProvider("Default summary.")
        agent = self._make_agent(provider)
        await agent.execute({"text": "Some text"})
        assert "3 sentences" in provider.calls[0]


# === LLM Classifier Tests ===


class TestLLMClassifier:
    def _make_agent(self, provider: MockProvider) -> Any:
        import importlib.util
        from pathlib import Path

        agent_dir = Path(__file__).parent.parent / "agents" / "llm-classifier"
        spec = importlib.util.spec_from_file_location(
            "llm_classifier_agent", agent_dir / "agent.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        contract = AgentContract(
            name="llm-classifier",
            version="1.0.0",
            model=ModelSpec(preference="fast"),
        )
        agent = mod.LLMClassifierAgent(contract, AgentContext())
        agent._provider = provider
        return agent

    async def test_classifier_json_response(self):
        provider = MockProvider('{"category": "science", "confidence": "high"}')
        agent = self._make_agent(provider)
        result = await agent.execute({
            "text": "Photosynthesis is...",
            "categories": ["science", "sports", "politics"],
        })
        assert result["category"] == "science"
        assert result["confidence"] == "high"
        assert result["model"] == "mock-model"

    async def test_classifier_prompt_contains_categories(self):
        provider = MockProvider('{"category": "sports", "confidence": "medium"}')
        agent = self._make_agent(provider)
        await agent.execute({
            "text": "The match was exciting",
            "categories": ["science", "sports"],
        })
        assert "science" in provider.calls[0]
        assert "sports" in provider.calls[0]

    async def test_classifier_invalid_json_fallback(self):
        provider = MockProvider("The category is science.")
        agent = self._make_agent(provider)
        result = await agent.execute({
            "text": "DNA replication",
            "categories": ["science", "sports"],
        })
        # Should fallback gracefully
        assert result["category"] in ["science", "sports"]
        assert result["confidence"] == "low"

    async def test_classifier_wrong_category_corrected(self):
        """If LLM returns a category not in the list, find closest match."""
        provider = MockProvider('{"category": "Science", "confidence": "high"}')
        agent = self._make_agent(provider)
        result = await agent.execute({
            "text": "DNA",
            "categories": ["science", "sports"],
        })
        assert result["category"] == "science"  # Case-corrected


# === Provider Configuration Tests ===


class TestProviderConfig:
    def test_model_for_preference_fast(self):
        model = model_for_preference("fast")
        assert "haiku" in model

    def test_model_for_preference_balanced(self):
        model = model_for_preference("balanced")
        assert "sonnet" in model

    def test_model_for_preference_powerful(self):
        model = model_for_preference("powerful")
        assert model  # Just check it returns something

    def test_model_for_preference_unknown_defaults_balanced(self):
        model = model_for_preference("unknown")
        assert model == model_for_preference("balanced")

    def test_model_tiers_populated(self):
        assert "fast" in MODEL_TIERS
        assert "balanced" in MODEL_TIERS
        assert "powerful" in MODEL_TIERS


# === Store Retry Field Persistence Tests ===


class TestStoreRetryFields:
    async def test_save_and_get_retry_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "test.db"))
            await store.init()

            job = JobData(
                agent_name="echo",
                input_data={"msg": "hi"},
                retry_count=2,
                original_job_id="job-original",
            )
            await store.save(job)

            loaded = await store.get(job.id)
            assert loaded is not None
            assert loaded.retry_count == 2
            assert loaded.original_job_id == "job-original"

            await store.close()

    async def test_retry_fields_default_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "test.db"))
            await store.init()

            job = JobData(agent_name="echo", input_data={"msg": "hi"})
            await store.save(job)

            loaded = await store.get(job.id)
            assert loaded.retry_count == 0
            assert loaded.original_job_id == ""

            await store.close()

    async def test_retry_fields_updated_on_upsert(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "test.db"))
            await store.init()

            job = JobData(agent_name="echo", input_data={"msg": "hi"})
            await store.save(job)

            # Update retry fields
            job.retry_count = 1
            job.original_job_id = "job-root"
            await store.save(job)

            loaded = await store.get(job.id)
            assert loaded.retry_count == 1
            assert loaded.original_job_id == "job-root"

            await store.close()

    async def test_list_preserves_retry_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(os.path.join(tmp, "test.db"))
            await store.init()

            job = JobData(
                agent_name="echo",
                input_data={"msg": "hi"},
                retry_count=3,
                original_job_id="job-first",
            )
            await store.save(job)

            jobs = await store.list()
            assert len(jobs) == 1
            assert jobs[0].retry_count == 3
            assert jobs[0].original_job_id == "job-first"

            await store.close()


# === Agent Discovery Tests ===


class TestLLMAgentDiscovery:
    def test_llm_summarizer_discovered(self, registry):
        entry = registry.get("llm-summarizer")
        assert entry is not None
        assert entry.contract.name == "llm-summarizer"
        assert "llm" in entry.contract.capabilities

    def test_llm_classifier_discovered(self, registry):
        entry = registry.get("llm-classifier")
        assert entry is not None
        assert entry.contract.name == "llm-classifier"
        assert "classification" in entry.contract.capabilities

    def test_llm_agents_have_fast_preference(self, registry):
        for name in ["llm-summarizer", "llm-classifier"]:
            entry = registry.get(name)
            assert entry.contract.model.preference == "fast"
