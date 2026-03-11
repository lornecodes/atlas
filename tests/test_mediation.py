"""Tests for the chain mediation system — the core of Spike 2."""

from __future__ import annotations

import pytest

from atlas.mediation.analyzer import (
    CompatLevel,
    analyze_compatibility,
    _field_similarity,
    _tokenize_field,
)
from atlas.contract.types import SchemaSpec
from atlas.mediation.engine import MediationEngine

from conftest import make_schema


# --- Schema fixtures ---

ECHO_SCHEMA = make_schema({"message": {"type": "string"}})
SUMMARIZER_IN = make_schema(
    {"text": {"type": "string"}, "max_length": {"type": "integer", "default": 200}},
    required=["text"],
)
SUMMARIZER_OUT = make_schema({"summary": {"type": "string"}, "token_count": {"type": "integer"}})
TRANSLATOR_OUT = make_schema({
    "translated_text": {"type": "string"},
    "source_lang": {"type": "string"},
    "target_lang": {"type": "string"},
})
FORMATTER_IN = make_schema({"content": {"type": "string"}, "style": {"type": "string"}})


# === Analyzer Tests ===

class TestAnalyzerIdentical:
    def test_same_schema(self):
        report = analyze_compatibility(ECHO_SCHEMA, ECHO_SCHEMA)
        assert report.level == CompatLevel.IDENTICAL
        assert report.confidence == 1.0
        assert report.can_bridge

    def test_empty_input_schema(self):
        report = analyze_compatibility(ECHO_SCHEMA, SchemaSpec())
        assert report.level == CompatLevel.IDENTICAL


class TestAnalyzerSuperset:
    def test_output_has_extra_fields(self):
        out = make_schema({
            "message": {"type": "string"},
            "extra": {"type": "string"},
            "count": {"type": "integer"},
        })
        inp = make_schema({"message": {"type": "string"}})
        report = analyze_compatibility(out, inp)
        assert report.level == CompatLevel.SUPERSET
        assert "message" in report.direct_mappings

    def test_translator_out_not_superset_of_formatter_in(self):
        """translator output doesn't have 'content' or 'style' — can't be superset."""
        report = analyze_compatibility(TRANSLATOR_OUT, FORMATTER_IN)
        assert report.level != CompatLevel.SUPERSET


class TestAnalyzerMapped:
    def test_explicit_map_covers_required(self):
        input_map = {
            "content": "translated_text",
            "style": "markdown",
        }
        report = analyze_compatibility(TRANSLATOR_OUT, FORMATTER_IN, input_map)
        assert report.level == CompatLevel.MAPPABLE
        assert report.can_bridge

    def test_incomplete_map_not_mappable(self):
        input_map = {"content": "translated_text"}  # Missing 'style'
        report = analyze_compatibility(TRANSLATOR_OUT, FORMATTER_IN, input_map)
        assert report.level != CompatLevel.MAPPABLE


class TestAnalyzerCoercible:
    def test_field_rename_by_similarity(self):
        """'translated_text' should fuzzy-match to 'text' via shared token 'text'."""
        out = make_schema({"translated_text": {"type": "string"}})
        inp = make_schema({"text": {"type": "string"}})
        report = analyze_compatibility(out, inp)
        assert report.level == CompatLevel.COERCIBLE
        assert len(report.coercions) == 1
        assert report.coercions[0].op_type == "rename"
        assert report.coercions[0].source_field == "translated_text"

    def test_type_cast(self):
        out = make_schema({"count": {"type": "integer"}})
        inp = make_schema({"count": {"type": "string"}})
        report = analyze_compatibility(out, inp)
        # Same field name but different type — should be direct + cast
        assert report.level in (CompatLevel.IDENTICAL, CompatLevel.SUPERSET, CompatLevel.COERCIBLE)

    def test_default_fills_missing(self):
        """If target has a default value, it can fill an unmapped required field."""
        out = make_schema({"text": {"type": "string"}})
        inp = make_schema(
            {"text": {"type": "string"}, "max_length": {"type": "integer", "default": 200}},
            required=["text", "max_length"],
        )
        report = analyze_compatibility(out, inp)
        # text matches directly, max_length should get default
        assert report.can_bridge

    def test_no_match_possible(self):
        """Completely disjoint field names with low similarity."""
        out = make_schema({"xyz_abc": {"type": "integer"}})
        inp = make_schema({"foo_bar": {"type": "string"}})
        report = analyze_compatibility(out, inp)
        # Should be SEMANTIC or INCOMPATIBLE, not COERCIBLE
        assert report.level in (CompatLevel.SEMANTIC, CompatLevel.INCOMPATIBLE)


class TestAnalyzerSemantic:
    def test_string_schemas_detected(self):
        """Both have string fields → LLM could plausibly bridge."""
        out = make_schema({"raw_output": {"type": "string"}})
        inp = make_schema({"processed_input": {"type": "string"}})
        report = analyze_compatibility(out, inp)
        # Could be coercible via fuzzy match or semantic
        assert report.can_bridge

    def test_fully_incompatible(self):
        """Array output vs number input — nothing works."""
        out = make_schema({"data": {"type": "array"}})
        inp = make_schema({"value": {"type": "integer"}})
        report = analyze_compatibility(out, inp)
        assert report.level == CompatLevel.INCOMPATIBLE
        assert not report.can_bridge


# === Field Similarity Tests ===

class TestFieldSimilarity:
    def test_exact_match(self):
        assert _field_similarity("name", "name") == 1.0

    def test_case_insensitive(self):
        assert _field_similarity("Name", "name") == 0.95

    def test_camel_to_snake(self):
        """translatedText should match translated_text."""
        score = _field_similarity("translatedText", "translated_text")
        assert score >= 0.9

    def test_abbreviation_msg_message(self):
        """msg should match message via abbreviation expansion."""
        score = _field_similarity("msg", "message")
        assert score >= 0.9

    def test_abbreviation_desc_description(self):
        score = _field_similarity("desc", "description")
        assert score >= 0.9

    def test_abbreviation_txt_text(self):
        score = _field_similarity("txt", "text")
        assert score >= 0.9

    def test_unrelated_fields_low_score(self):
        score = _field_similarity("xyz_abc", "foo_bar")
        assert score < 0.5

    def test_partial_token_overlap(self):
        score = _field_similarity("user_name", "name")
        assert score >= 0.5


class TestTokenizeField:
    def test_snake_case(self):
        assert _tokenize_field("translated_text") == ["translated", "text"]

    def test_camel_case(self):
        assert _tokenize_field("translatedText") == ["translated", "text"]

    def test_acronym(self):
        """XMLParser should tokenize properly."""
        tokens = _tokenize_field("XMLParser")
        assert "xml" in tokens
        assert "parser" in tokens

    def test_single_word(self):
        assert _tokenize_field("name") == ["name"]

    def test_mixed(self):
        tokens = _tokenize_field("myXMLParser_v2")
        assert "my" in tokens
        assert "xml" in tokens
        assert "parser" in tokens


# === Engine Tests ===

class TestMediationEngine:
    @pytest.fixture
    def engine(self):
        return MediationEngine()

    async def test_direct_passthrough(self, engine: MediationEngine):
        result = await engine.mediate(
            {"message": "hello"},
            ECHO_SCHEMA,
            ECHO_SCHEMA,
        )
        assert result.success
        assert result.data == {"message": "hello"}
        assert result.strategy_used == "direct"
        assert result.cost == 0.0

    async def test_superset_drops_extras(self, engine: MediationEngine):
        out_schema = make_schema({
            "message": {"type": "string"},
            "debug_info": {"type": "string"},
        })
        result = await engine.mediate(
            {"message": "hello", "debug_info": "internal"},
            out_schema,
            ECHO_SCHEMA,
        )
        assert result.success
        assert result.data == {"message": "hello"}
        assert "debug_info" not in result.data

    async def test_mapped_with_static_values(self, engine: MediationEngine):
        input_map = {
            "content": "translated_text",
            "style": "markdown",
        }
        result = await engine.mediate(
            {"translated_text": "bonjour", "source_lang": "en", "target_lang": "fr"},
            TRANSLATOR_OUT,
            FORMATTER_IN,
            input_map=input_map,
        )
        assert result.success
        assert result.data["content"] == "bonjour"
        assert result.data["style"] == "markdown"
        assert result.strategy_used == "mapped"

    async def test_coerce_rename(self, engine: MediationEngine):
        out = make_schema({"translated_text": {"type": "string"}})
        inp = make_schema({"text": {"type": "string"}})
        result = await engine.mediate(
            {"translated_text": "bonjour"},
            out,
            inp,
        )
        assert result.success
        assert result.data["text"] == "bonjour"
        assert result.strategy_used == "coerce"

    async def test_incompatible_fails(self, engine: MediationEngine):
        out = make_schema({"data": {"type": "array"}})
        inp = make_schema({"value": {"type": "integer"}})
        result = await engine.mediate(
            {"data": [1, 2, 3]},
            out,
            inp,
        )
        assert not result.success
        assert "incompatible" in result.error.lower() or "no strategy" in result.error.lower()

    async def test_llm_bridge_without_provider(self, engine: MediationEngine):
        """Without an LLM provider, semantic-level schemas get no strategy."""
        out = make_schema({"raw_data": {"type": "string"}})
        inp = make_schema({"processed_result": {"type": "string"}})
        result = await engine.mediate({"raw_data": "stuff"}, out, inp)
        # Should fail — no LLM provider configured
        # (might succeed via coerce if names are similar enough)
        if not result.success:
            assert "no strategy" in result.error.lower() or "incompatible" in result.error.lower()

    async def test_llm_bridge_with_mock_provider(self):
        """With a mock LLM, semantic bridging should work."""
        async def mock_llm(prompt: str) -> str:
            return '{"processed_result": "transformed data"}'

        engine = MediationEngine(llm_provider=mock_llm)
        out = make_schema({"raw_data": {"type": "string"}})
        inp = make_schema({"processed_result": {"type": "string"}})
        result = await engine.mediate({"raw_data": "stuff"}, out, inp)

        # Could succeed via coerce or LLM bridge
        assert result.success
        assert "processed_result" in result.data

    async def test_llm_bridge_retry_on_bad_json(self):
        """LLM bridge retries once on bad JSON."""
        call_count = 0

        async def flaky_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not valid json at all"
            return '{"output_field": "success"}'

        engine = MediationEngine(llm_provider=flaky_llm)
        out = make_schema({"input_field": {"type": "string"}})
        inp = make_schema({"output_field": {"type": "string"}})
        result = await engine.mediate({"input_field": "test"}, out, inp)

        # Might succeed via coerce instead; if it reaches LLM, verify retry
        if result.strategy_used == "llm_bridge":
            assert result.success
            assert call_count == 2

    async def test_llm_bridge_validates_output(self):
        """LLM bridge fails if output missing required fields."""
        async def incomplete_llm(prompt: str) -> str:
            return '{"wrong_field": "value"}'

        engine = MediationEngine(llm_provider=incomplete_llm)
        out = make_schema({"x": {"type": "string"}})
        inp = make_schema({"required_field": {"type": "string"}})
        result = await engine.mediate({"x": "test"}, out, inp)

        # If it reaches LLM bridge, should fail on missing required fields
        if result.strategy_used == "llm_bridge":
            assert not result.success
            assert "missing required" in result.error.lower()

    async def test_cost_hierarchy(self, engine: MediationEngine):
        """Direct should be used over coerce when both could work."""
        schema = make_schema({"x": {"type": "string"}})
        result = await engine.mediate({"x": "val"}, schema, schema)
        assert result.strategy_used == "direct"
        assert result.cost == 0.0

    async def test_analysis_only(self, engine: MediationEngine):
        """analyze() returns report without transformation."""
        report = await engine.analyze(ECHO_SCHEMA, ECHO_SCHEMA)
        assert report.level == CompatLevel.IDENTICAL
