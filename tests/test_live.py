"""Live integration tests -- real API calls, no mocks.

Run with:
    python -m pytest tests/test_live.py -v -s

Requires ANTHROPIC_API_KEY in environment (or GRIM/.env).
Skips automatically if no key is available.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest


def _safe_print(text: str) -> None:
    """Print text safely on Windows (cp1252 can't handle all Unicode)."""
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

from atlas.chains.definition import ChainDefinition, ChainStep
from atlas.chains.runner import ChainRunner
from atlas.contract.registry import AgentRegistry
from atlas.mediation.engine import MediationEngine
from atlas.runtime.context import AgentContext
from atlas.runtime.runner import run_agent

AGENTS_DIR = Path(__file__).parent.parent / "agents"


def _load_api_key() -> str | None:
    """Try env var first, then GRIM/.env."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    env_file = Path(__file__).parent.parent.parent / "GRIM" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


API_KEY = _load_api_key()
skip_no_key = pytest.mark.skipif(not API_KEY, reason="ANTHROPIC_API_KEY not available")


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch):
    """Ensure the API key is in env for all tests."""
    if API_KEY:
        monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry(search_paths=[AGENTS_DIR])
    reg.discover()
    return reg


# ─── Single Agent: Claude Writer (Anthropic SDK) ────────────────────


@skip_no_key
class TestLiveClaudeWriter:
    async def test_writes_real_content(self, registry: AgentRegistry):
        """claude-writer produces real content from Claude."""
        t0 = time.time()
        result = await run_agent(
            registry,
            "claude-writer",
            {"topic": "why recursion appears in nature", "style": "concise"},
        )
        elapsed = time.time() - t0

        assert result.success, f"Agent failed: {result.error}"
        assert len(result.data["content"]) > 50, "Content too short"
        assert result.data["word_count"] > 10
        assert "claude" in result.data["model"].lower()

        _safe_print(f"\n{'='*60}")
        _safe_print(f"CLAUDE-WRITER ({elapsed:.1f}s, model: {result.data['model']})")
        _safe_print(f"{'='*60}")
        _safe_print(result.data["content"])
        _safe_print(f"Word count: {result.data['word_count']}")


# ─── Single Agent: Claude Tools (Anthropic SDK + tool use) ──────────


@skip_no_key
class TestLiveClaudeTools:
    async def test_tool_use_with_calculate(self, registry: AgentRegistry):
        """claude-tools uses the calculate tool to answer a math question."""
        t0 = time.time()
        result = await run_agent(
            registry,
            "claude-tools",
            {"question": "What is the square root of 144 multiplied by 7?"},
        )
        elapsed = time.time() - t0

        assert result.success, f"Agent failed: {result.error}"
        assert len(result.data["tools_used"]) > 0, "Expected tool usage"
        assert "calculate" in result.data["tools_used"], "Expected calculate tool"

        _safe_print(f"\n{'='*60}")
        _safe_print(f"CLAUDE-TOOLS ({elapsed:.1f}s, {result.data['steps']} steps)")
        _safe_print(f"{'='*60}")
        _safe_print(f"Tools used: {result.data['tools_used']}")
        _safe_print(f"Answer: {result.data['answer']}")

    async def test_tool_use_with_lookup(self, registry: AgentRegistry):
        """claude-tools uses the lookup tool to answer a knowledge question."""
        t0 = time.time()
        result = await run_agent(
            registry,
            "claude-tools",
            {"question": "What is entropy and what is its relationship to information theory?"},
        )
        elapsed = time.time() - t0

        assert result.success, f"Agent failed: {result.error}"

        _safe_print(f"\n{'='*60}")
        _safe_print(f"CLAUDE-TOOLS LOOKUP ({elapsed:.1f}s, {result.data['steps']} steps)")
        _safe_print(f"{'='*60}")
        _safe_print(f"Tools used: {result.data['tools_used']}")
        _safe_print(f"Answer: {result.data['answer']}")


# ─── Single Agent: LangChain Summarizer (LCEL + Anthropic) ──────────


@skip_no_key
class TestLiveLangChainSummarizer:
    async def test_summarizes_real_text(self, registry: AgentRegistry):
        """langchain-summarizer uses LCEL + ChatAnthropic to summarize text."""
        long_text = (
            "The Fibonacci sequence appears throughout nature in surprising ways. "
            "Sunflower heads arrange their seeds in spirals that follow consecutive "
            "Fibonacci numbers. Pinecone scales spiral in patterns of 8 and 13. "
            "The nautilus shell curves in a logarithmic spiral approximating the "
            "golden ratio, which is the limit of consecutive Fibonacci ratios. "
            "Even the branching patterns of trees and the arrangement of leaves "
            "around stems follow Fibonacci-like patterns, optimizing sunlight "
            "exposure through phyllotaxis."
        )

        t0 = time.time()
        result = await run_agent(
            registry,
            "langchain-summarizer",
            {"text": long_text, "max_points": 3},
        )
        elapsed = time.time() - t0

        assert result.success, f"Agent failed: {result.error}"
        assert len(result.data["summary"]) > 20
        assert len(result.data["key_points"]) > 0

        _safe_print(f"\n{'='*60}")
        _safe_print(f"LANGCHAIN-SUMMARIZER ({elapsed:.1f}s, model: {result.data['model']})")
        _safe_print(f"{'='*60}")
        _safe_print(f"Summary: {result.data['summary']}")
        _safe_print(f"Key points:")
        for i, pt in enumerate(result.data["key_points"], 1):
            _safe_print(f"  {i}. {pt}")


# ─── 2-Step Chain: Write -> Summarize (Anthropic + LangChain) ────────


@skip_no_key
class TestLiveWriteThenSummarize:
    async def test_write_then_summarize(self, registry: AgentRegistry):
        """Real 2-framework chain: Claude writes, LangChain summarizes."""
        chain = ChainDefinition(
            name="write-then-summarize",
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
                    agent_name="langchain-summarizer",
                    name="summarize",
                    input_map={
                        "text": "$.steps.write.output.content",
                    },
                ),
            ],
        )

        runner = ChainRunner(registry, MediationEngine())

        t0 = time.time()
        result = await runner.execute(
            chain,
            {"topic": "how emergence creates complexity from simple rules", "style": "detailed"},
        )
        elapsed = time.time() - t0

        assert result.success, f"Chain failed: {result.error}"
        assert len(result.steps) == 2

        writer_output = result.steps[0].agent_result.data
        summarizer_output = result.output

        _safe_print(f"\n{'='*60}")
        _safe_print(f"CHAIN: WRITE -> SUMMARIZE ({elapsed:.1f}s)")
        _safe_print(f"{'='*60}")
        _safe_print(f"\n--- Step 1: claude-writer ({writer_output['model']}) ---")
        _safe_print(writer_output["content"])
        _safe_print(f"[{writer_output['word_count']} words]")
        _safe_print(f"\n--- Step 2: langchain-summarizer ({summarizer_output['model']}) ---")
        _safe_print(f"Summary: {summarizer_output['summary']}")
        _safe_print(f"Key points:")
        for i, pt in enumerate(summarizer_output["key_points"], 1):
            _safe_print(f"  {i}. {pt}")


# ─── 3-Step Chain: Write -> Review -> Summarize ───────────────────────


@skip_no_key
class TestLiveThreeVendorChain:
    async def test_write_review_summarize(self, registry: AgentRegistry):
        """Real 3-step chain: Claude writes, Claude reviews (via Anthropic provider
        injected into openai-reviewer), LangChain summarizes.

        This proves the DI system works in production -- we inject an Anthropic
        provider into the openai-reviewer agent so it works without an OpenAI key.
        """
        from atlas.llm.anthropic import AnthropicProvider

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

        runner = ChainRunner(registry, MediationEngine())

        # DI: inject Anthropic provider into openai-reviewer (no OpenAI key needed)
        anthropic_provider = AnthropicProvider(model="claude-haiku-4-5-20251001")

        t0 = time.time()
        result = await runner.execute(
            chain,
            {
                "topic": "entropy and the arrow of time",
                "style": "scientific",
                "criteria": "scientific accuracy, depth of explanation, clarity",
            },
            providers={
                "openai-reviewer": {"llm_provider": anthropic_provider},
            },
        )
        elapsed = time.time() - t0

        assert result.success, f"Chain failed: {result.error}"
        assert len(result.steps) == 3

        writer_out = result.steps[0].agent_result.data
        reviewer_out = result.steps[1].agent_result.data
        summarizer_out = result.output

        _safe_print(f"\n{'='*60}")
        _safe_print(f"CHAIN: WRITE -> REVIEW -> SUMMARIZE ({elapsed:.1f}s)")
        _safe_print(f"{'='*60}")
        _safe_print(f"\n--- Step 1: claude-writer ({writer_out['model']}) ---")
        _safe_print(writer_out["content"])
        _safe_print(f"[{writer_out['word_count']} words]")
        _safe_print(f"\n--- Step 2: openai-reviewer (injected: {reviewer_out['model']}) ---")
        _safe_print(f"Rating: {reviewer_out['rating']}/10")
        _safe_print(f"Review: {reviewer_out['review']}")
        if reviewer_out["suggestions"]:
            _safe_print(f"Suggestions:")
            for s in reviewer_out["suggestions"]:
                _safe_print(f"  - {s}")
        _safe_print(f"\n--- Step 3: langchain-summarizer ({summarizer_out['model']}) ---")
        _safe_print(f"Summary: {summarizer_out['summary']}")
        _safe_print(f"Key points:")
        for i, pt in enumerate(summarizer_out["key_points"], 1):
            _safe_print(f"  {i}. {pt}")

        # Verify mediation summary
        _safe_print(f"\n--- Mediation ---")
        for ms in result.mediation_summary:
            _safe_print(f"  Step {ms['step']} ({ms['agent']}): {ms['strategy']}")
