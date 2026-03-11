"""Tests for LangChain summarizer agent, DI via context.providers, and 3-vendor chain.

No monkey-patching or unittest.mock.patch. All test dependencies are
injected through AgentContext.providers — the same DI path production
code uses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atlas.chains.definition import ChainDefinition, ChainStep
from atlas.chains.runner import ChainRunner
from atlas.contract.registry import AgentRegistry
from atlas.llm.provider import LLMResponse
from atlas.mediation.engine import MediationEngine
from atlas.runtime.context import AgentContext

AGENTS_DIR = Path(__file__).parent.parent / "agents"
CHAINS_DIR = Path(__file__).parent.parent / "chains"


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


# ─── Test Doubles (no mock library) ─────────────────────────────────


class FakeProvider:
    """A fake LLM provider that returns canned responses in order."""

    def __init__(self, *responses: LLMResponse) -> None:
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[str] = []

    async def complete(self, prompt: str) -> LLMResponse:
        self.calls.append(prompt)
        resp = self._responses[self._idx]
        self._idx = min(self._idx + 1, len(self._responses) - 1)
        return resp


class FakeLangChainChain:
    """A fake LCEL chain with an async ainvoke."""

    def __init__(self, output: str) -> None:
        self._output = output
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, input_data: dict[str, Any]) -> str:
        self.calls.append(input_data)
        return self._output


# ─── Discovery ───────────────────────────────────────────────────────


class TestLangChainDiscovery:
    def test_langchain_summarizer_discovered(self, registry: AgentRegistry):
        entry = registry.get("langchain-summarizer")
        assert entry is not None
        assert entry.contract.name == "langchain-summarizer"

    def test_has_langchain_capability(self, registry: AgentRegistry):
        entry = registry.get("langchain-summarizer")
        assert "langchain" in entry.contract.capabilities
        assert "summarization" in entry.contract.capabilities
        assert "llm" in entry.contract.capabilities

    def test_capability_search(self, registry: AgentRegistry):
        results = registry.search("langchain")
        names = [a.contract.name for a in results]
        assert "langchain-summarizer" in names

    def test_schema_has_required_fields(self, registry: AgentRegistry):
        entry = registry.get("langchain-summarizer")
        assert "text" in entry.contract.input_schema.properties
        assert "summary" in entry.contract.output_schema.properties
        assert "key_points" in entry.contract.output_schema.properties
        assert "model" in entry.contract.output_schema.properties


# ─── Agent Logic (DI via context.providers) ──────────────────────────


def _load_agent_class():
    """Dynamically load the LangChainSummarizerAgent class."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "langchain_summarizer_agent",
        AGENTS_DIR / "langchain-summarizer" / "agent.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LangChainSummarizerAgent


class TestLangChainSummarizerAgent:
    def _make_agent(self, raw_output: str) -> tuple:
        """Create agent with a fake chain injected via context.providers."""
        from atlas.contract.types import AgentContract, ModelSpec

        cls = _load_agent_class()
        contract = AgentContract(
            name="langchain-summarizer",
            version="1.0.0",
            model=ModelSpec(preference="fast"),
        )
        fake_chain = FakeLangChainChain(raw_output)
        ctx = AgentContext(
            providers={
                "langchain_chain": (fake_chain, "claude-haiku-4-5-20251001"),
            }
        )
        agent = cls(contract, ctx)
        return agent, fake_chain

    async def test_execute_json_response(self):
        """Agent parses structured JSON from LangChain chain."""
        raw = json.dumps({
            "summary": "Recursion is self-reference in action.",
            "key_points": ["Base case needed", "Stack grows per call"],
        })
        agent, _ = self._make_agent(raw)
        await agent.on_startup()
        result = await agent.execute({"text": "Long article about recursion..."})

        assert result["summary"] == "Recursion is self-reference in action."
        assert result["key_points"] == ["Base case needed", "Stack grows per call"]
        assert result["model"] == "claude-haiku-4-5-20251001"

    async def test_execute_passes_max_points(self):
        """max_points is forwarded to the chain."""
        raw = json.dumps({"summary": "Brief.", "key_points": ["One"]})
        agent, fake_chain = self._make_agent(raw)
        await agent.on_startup()
        await agent.execute({"text": "Some text", "max_points": 3})

        assert fake_chain.calls[0]["max_points"] == 3

    async def test_execute_default_max_points(self):
        """Default max_points is 5."""
        raw = json.dumps({"summary": "Ok.", "key_points": []})
        agent, fake_chain = self._make_agent(raw)
        await agent.on_startup()
        await agent.execute({"text": "Text"})

        assert fake_chain.calls[0]["max_points"] == 5

    async def test_parse_strips_markdown_fences(self):
        """Agent handles markdown-fenced JSON from LLM."""
        raw = '```json\n{"summary": "Clean.", "key_points": ["A"]}\n```'
        agent, _ = self._make_agent(raw)
        await agent.on_startup()
        result = await agent.execute({"text": "Input"})
        assert result["summary"] == "Clean."
        assert result["key_points"] == ["A"]

    async def test_parse_fallback_on_invalid_json(self):
        """Non-JSON output falls back gracefully."""
        agent, _ = self._make_agent("This is just plain text, no JSON.")
        await agent.on_startup()
        result = await agent.execute({"text": "Input"})
        assert result["summary"] == "This is just plain text, no JSON."
        assert result["key_points"] == []

    async def test_execute_without_startup_raises(self):
        """Calling execute before on_startup raises RuntimeError."""
        from atlas.contract.types import AgentContract, ModelSpec

        cls = _load_agent_class()
        contract = AgentContract(
            name="langchain-summarizer",
            version="1.0.0",
            model=ModelSpec(preference="fast"),
        )
        agent = cls(contract, AgentContext())
        with pytest.raises(RuntimeError, match="not started"):
            await agent.execute({"text": "test"})

    async def test_di_skips_langchain_imports(self):
        """When providers are injected, on_startup doesn't import langchain."""
        raw = json.dumps({"summary": "DI works.", "key_points": []})
        agent, _ = self._make_agent(raw)
        # on_startup should succeed without langchain installed
        await agent.on_startup()
        assert agent._chain is not None
        assert agent._model_name == "claude-haiku-4-5-20251001"


# ─── 3-Vendor Chain (DI, no mocking) ────────────────────────────────


class TestThreeVendorChain:
    async def test_write_review_summarize_chain(self, registry: AgentRegistry):
        """Claude writes -> GPT reviews -> LangChain summarizes. Full 3-vendor chain."""
        chain = ChainDefinition(
            name="write-review-summarize",
            steps=[
                ChainStep(
                    agent_name="claude-writer",
                    name="write",
                    input_map={
                        "topic": "$.trigger.topic",
                        "style": "$.trigger.style",
                    },
                ),
                ChainStep(
                    agent_name="openai-reviewer",
                    name="review",
                    input_map={
                        "content": "$.steps.write.output.content",
                        "criteria": "$.trigger.criteria",
                    },
                ),
                ChainStep(
                    agent_name="langchain-summarizer",
                    name="summarize",
                    input_map={
                        "text": "$.steps.review.output.review",
                    },
                ),
            ],
        )

        # Fake providers for each agent — injected via chain runner DI
        writer_provider = FakeProvider(
            LLMResponse(
                text="Entropy measures disorder in a system.",
                model="claude-haiku-4-5-20251001",
            )
        )
        reviewer_provider = FakeProvider(
            LLMResponse(
                text=json.dumps({
                    "review": "Accurate but could elaborate on thermodynamic context.",
                    "rating": 7,
                    "suggestions": ["Add entropy formula", "Mention second law"],
                }),
                model="gpt-4o-mini",
            )
        )
        summarizer_chain = FakeLangChainChain(
            json.dumps({
                "summary": "Review is positive with suggestions for depth.",
                "key_points": ["Add formula", "Reference second law"],
            })
        )

        runner = ChainRunner(registry, MediationEngine())
        result = await runner.execute(
            chain,
            {
                "topic": "entropy",
                "style": "concise",
                "criteria": "scientific accuracy",
            },
            providers={
                "claude-writer": {"llm_provider": writer_provider},
                "openai-reviewer": {"llm_provider": reviewer_provider},
                "langchain-summarizer": {
                    "langchain_chain": (summarizer_chain, "claude-haiku-4-5-20251001"),
                },
            },
        )

        assert result.success, f"Chain failed: {result.error}"
        assert len(result.steps) == 3

        # Verify step agents
        assert result.steps[0].agent_name == "claude-writer"
        assert result.steps[1].agent_name == "openai-reviewer"
        assert result.steps[2].agent_name == "langchain-summarizer"

        # Final output is from the langchain summarizer
        final = result.output
        assert final["summary"] == "Review is positive with suggestions for depth."
        assert final["key_points"] == ["Add formula", "Reference second law"]
        assert final["model"] == "claude-haiku-4-5-20251001"

    async def test_chain_yaml_loads(self):
        """The write-review-summarize.yaml loads correctly."""
        chain = ChainDefinition.from_yaml(CHAINS_DIR / "write-review-summarize.yaml")
        assert chain.name == "write-review-summarize"
        assert len(chain.steps) == 3
        assert chain.steps[0].agent_name == "claude-writer"
        assert chain.steps[0].name == "write"
        assert chain.steps[1].agent_name == "openai-reviewer"
        assert chain.steps[1].name == "review"
        assert chain.steps[2].agent_name == "langchain-summarizer"
        assert chain.steps[2].name == "summarize"

    async def test_chain_yaml_input_maps(self):
        """Input maps in YAML correctly reference prior steps."""
        chain = ChainDefinition.from_yaml(CHAINS_DIR / "write-review-summarize.yaml")

        assert chain.steps[1].input_map["content"] == "$.steps.write.output.content"
        assert chain.steps[2].input_map["text"] == "$.steps.review.output.review"

    async def test_mediation_flows_data_between_all_three_steps(
        self, registry: AgentRegistry
    ):
        """Verify data flows correctly through mediation at each step boundary."""
        chain = ChainDefinition(
            name="flow-test",
            steps=[
                ChainStep(
                    agent_name="claude-writer",
                    name="write",
                    input_map={"topic": "$.trigger.topic"},
                ),
                ChainStep(
                    agent_name="openai-reviewer",
                    name="review",
                    input_map={"content": "$.steps.write.output.content"},
                ),
                ChainStep(
                    agent_name="langchain-summarizer",
                    name="summarize",
                    input_map={"text": "$.steps.review.output.review"},
                ),
            ],
        )

        writer_provider = FakeProvider(
            LLMResponse(text="Written content here.", model="claude")
        )
        reviewer_provider = FakeProvider(
            LLMResponse(
                text=json.dumps({
                    "review": "The review of written content.",
                    "rating": 8,
                    "suggestions": [],
                }),
                model="gpt",
            )
        )
        summarizer_chain = FakeLangChainChain(
            json.dumps({
                "summary": "Summary of the review.",
                "key_points": ["Point A"],
            })
        )

        runner = ChainRunner(registry, MediationEngine())
        result = await runner.execute(
            chain,
            {"topic": "flow test"},
            providers={
                "claude-writer": {"llm_provider": writer_provider},
                "openai-reviewer": {"llm_provider": reviewer_provider},
                "langchain-summarizer": {
                    "langchain_chain": (summarizer_chain, "test-model"),
                },
            },
        )

        assert result.success, f"Chain failed: {result.error}"

        # Verify reviewer received writer's content via mediation
        assert len(reviewer_provider.calls) == 1
        assert "Written content here." in reviewer_provider.calls[0]

        # Verify summarizer received reviewer's review via mediation
        assert len(summarizer_chain.calls) == 1
        assert summarizer_chain.calls[0]["text"] == "The review of written content."

        # Verify final output
        assert result.output["summary"] == "Summary of the review."


# ─── All 3 Frameworks in Registry ────────────────────────────────────


class TestThreeFrameworkRegistry:
    def test_all_three_frameworks_registered(self, registry: AgentRegistry):
        """Anthropic SDK, OpenAI SDK, and LangChain agents all coexist."""
        assert registry.get("claude-writer") is not None
        assert registry.get("claude-tools") is not None
        assert registry.get("openai-reviewer") is not None
        assert registry.get("langchain-summarizer") is not None

    def test_llm_capability_finds_all(self, registry: AgentRegistry):
        """All LLM-backed agents (regardless of framework) are discoverable."""
        llm_agents = registry.search("llm")
        names = [a.contract.name for a in llm_agents]
        assert "claude-writer" in names
        assert "openai-reviewer" in names
        assert "langchain-summarizer" in names
