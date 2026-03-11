"""Tests for the LLM bridge — provider protocol, strategy, token tracking."""

from __future__ import annotations

import json

import pytest

from atlas.llm.provider import LLMProvider, LLMResponse
from atlas.mediation.analyzer import CompatLevel, CompatibilityReport
from atlas.mediation.engine import MediationEngine
from atlas.mediation.strategies import (
    LLMBridgeStrategy,
    MediationContext,
    _build_bridge_prompt,
    _extract_json,
)

from conftest import make_schema


class MockProvider:
    """Mock LLM provider implementing the protocol."""

    def __init__(self, responses: list[str] | None = None):
        self.calls: list[str] = []
        self._responses = list(responses or ['{}'])
        self._call_count = 0

    async def complete(self, prompt: str) -> LLMResponse:
        self.calls.append(prompt)
        idx = min(self._call_count, len(self._responses) - 1)
        text = self._responses[idx]
        self._call_count += 1
        return LLMResponse(
            text=text,
            input_tokens=len(prompt.split()),
            output_tokens=len(text.split()),
            model="mock-model",
        )


def _semantic_report(required: list[str] | None = None) -> CompatibilityReport:
    return CompatibilityReport(
        level=CompatLevel.SEMANTIC,
        unmapped_required=required or [],
        confidence=0.5,
    )


FLAT_SOURCE = make_schema({"text": {"type": "string"}})
FLAT_TARGET = make_schema({"summary": {"type": "string"}})


# ============================================================================
# TestLLMProviderProtocol
# ============================================================================

class TestLLMProviderProtocol:
    """LLMProvider protocol compliance."""

    def test_mock_implements_protocol(self):
        """MockProvider satisfies LLMProvider protocol."""
        provider = MockProvider()
        assert isinstance(provider, LLMProvider)

    async def test_provider_returns_response(self):
        """Provider returns LLMResponse with all fields."""
        provider = MockProvider(['{"summary": "hello"}'])
        resp = await provider.complete("test prompt")
        assert isinstance(resp, LLMResponse)
        assert resp.text == '{"summary": "hello"}'
        assert resp.input_tokens > 0
        assert resp.model == "mock-model"

    def test_response_is_frozen(self):
        """LLMResponse is immutable."""
        resp = LLMResponse(text="hi", input_tokens=5, output_tokens=3, model="m")
        with pytest.raises(AttributeError):
            resp.text = "changed"


# ============================================================================
# TestCallableBackwardCompat
# ============================================================================

class TestCallableBackwardCompat:
    """Backward compatibility with bare async callables."""

    async def test_callable_provider(self):
        """Bare async callable works as LLM provider."""
        async def fake_llm(prompt: str) -> str:
            return '{"summary": "from callable"}'

        strategy = LLMBridgeStrategy(fake_llm, max_retries=1)
        report = _semantic_report(["summary"])
        ctx = MediationContext(
            source_schema=FLAT_SOURCE,
            target_schema=FLAT_TARGET,
        )
        result = await strategy.transform({}, report, ctx)
        assert result["summary"] == "from callable"

    async def test_callable_no_token_tracking(self):
        """Bare callable doesn't provide token tracking (stays at 0)."""
        async def fake_llm(prompt: str) -> str:
            return '{"summary": "ok"}'

        strategy = LLMBridgeStrategy(fake_llm, max_retries=1)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)
        await strategy.transform({}, report, ctx)
        assert strategy.total_tokens == 0


# ============================================================================
# TestLLMBridgeTransform
# ============================================================================

class TestLLMBridgeTransform:
    """LLMBridgeStrategy with mock provider."""

    async def test_successful_transform(self):
        """Basic transform returns parsed JSON."""
        provider = MockProvider(['{"summary": "transformed text"}'])
        strategy = LLMBridgeStrategy(provider, max_retries=1)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)

        result = await strategy.transform({"text": "hello"}, report, ctx)
        assert result == {"summary": "transformed text"}

    async def test_token_tracking(self):
        """Provider-based calls track tokens."""
        provider = MockProvider(['{"summary": "ok"}'])
        strategy = LLMBridgeStrategy(provider, max_retries=1)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)

        await strategy.transform({"text": "hello"}, report, ctx)
        assert strategy.total_input_tokens > 0
        assert strategy.total_output_tokens > 0
        assert strategy.total_tokens == strategy.total_input_tokens + strategy.total_output_tokens

    async def test_retry_on_bad_json(self):
        """Retries when first response is not valid JSON."""
        provider = MockProvider([
            "not json at all",
            '{"summary": "retry worked"}',
        ])
        strategy = LLMBridgeStrategy(provider, max_retries=3, retry_base_delay=0.0)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)

        result = await strategy.transform({"text": "hello"}, report, ctx)
        assert result["summary"] == "retry worked"
        assert len(provider.calls) == 2

    async def test_fails_after_max_retries(self):
        """Raises RuntimeError after exhausting retries."""
        provider = MockProvider(["not json", "still not json", "nope"])
        strategy = LLMBridgeStrategy(provider, max_retries=3, retry_base_delay=0.0)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)

        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            await strategy.transform({"text": "hello"}, report, ctx)

    async def test_validates_required_fields(self):
        """Rejects response missing required fields."""
        provider = MockProvider(['{"wrong_field": "value"}'])
        strategy = LLMBridgeStrategy(provider, max_retries=1, retry_base_delay=0.0)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)

        with pytest.raises(RuntimeError, match="failed after 1 attempts"):
            await strategy.transform({"text": "hello"}, report, ctx)

    async def test_handles_markdown_fenced_json(self):
        """Extracts JSON from markdown code fences."""
        provider = MockProvider(['```json\n{"summary": "fenced"}\n```'])
        strategy = LLMBridgeStrategy(provider, max_retries=1)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)

        result = await strategy.transform({"text": "hello"}, report, ctx)
        assert result["summary"] == "fenced"

    async def test_prompt_contains_schemas(self):
        """Bridge prompt includes source and target schema info."""
        provider = MockProvider(['{"summary": "ok"}'])
        strategy = LLMBridgeStrategy(provider, max_retries=1)
        report = _semantic_report(["summary"])
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)

        await strategy.transform({"text": "hello"}, report, ctx)
        prompt = provider.calls[0]
        assert "SOURCE SCHEMA" in prompt
        assert "TARGET SCHEMA" in prompt
        assert "hello" in prompt

    async def test_nested_schema_transform(self):
        """LLM bridge handles nested object schemas."""
        source = make_schema({"name": {"type": "string"}, "city": {"type": "string"}})
        target = make_schema({
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "address": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
        })
        provider = MockProvider([json.dumps({
            "user": {"name": "Alice", "address": {"city": "NYC"}}
        })])
        strategy = LLMBridgeStrategy(provider, max_retries=1)
        report = _semantic_report(["user"])
        ctx = MediationContext(source_schema=source, target_schema=target)

        result = await strategy.transform({"name": "Alice", "city": "NYC"}, report, ctx)
        assert result["user"]["name"] == "Alice"
        assert result["user"]["address"]["city"] == "NYC"

    async def test_not_configured_raises(self):
        """Raises RuntimeError when no provider configured."""
        strategy = LLMBridgeStrategy(None)
        report = _semantic_report()
        ctx = MediationContext(source_schema=FLAT_SOURCE, target_schema=FLAT_TARGET)
        with pytest.raises(RuntimeError, match="not configured"):
            await strategy.transform({}, report, ctx)


# ============================================================================
# TestEngineTokenTracking
# ============================================================================

class TestEngineTokenTracking:
    """MediationEngine exposes LLM token totals."""

    def test_no_provider_zero_tokens(self):
        """Engine without LLM provider reports 0 tokens."""
        engine = MediationEngine()
        assert engine.total_llm_tokens == 0

    async def test_provider_token_tracking(self):
        """Engine with provider tracks tokens through mediation."""
        provider = MockProvider(['{"summary": "ok"}'])
        engine = MediationEngine(llm_provider=provider)

        source = make_schema({"items": {"type": "array"}})
        target = make_schema({"summary": {"type": "string"}})

        result = await engine.mediate(
            {"items": []}, source, target
        )
        assert result.success
        assert engine.total_llm_tokens > 0


# ============================================================================
# TestExtractJson
# ============================================================================

class TestExtractJson:
    """JSON extraction from LLM responses."""

    def test_raw_json(self):
        assert json.loads(_extract_json('{"a": 1}')) == {"a": 1}

    def test_fenced_json(self):
        assert json.loads(_extract_json('```json\n{"a": 1}\n```')) == {"a": 1}

    def test_fenced_no_lang(self):
        assert json.loads(_extract_json('```\n{"a": 1}\n```')) == {"a": 1}

    def test_non_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("This is just text with no JSON")
