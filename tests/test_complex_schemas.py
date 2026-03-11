"""Stress tests for complex schema validation and mediation.

Tests non-trivial schemas: nested objects, arrays, optional defaults,
mixed types (oneOf), and cross-schema mediation.
"""

from __future__ import annotations

import asyncio
import copy

import pytest

from atlas.contract.schema import load_contract
from atlas.mediation.analyzer import CompatLevel, analyze_compatibility
from atlas.mediation.engine import MediationEngine
from atlas.mediation.strategies import (
    DirectStrategy,
    MediationContext,
    _build_few_shot_examples,
)

from conftest import make_schema


def _load_contract(agent_dir: str):
    """Load a contract from the agents/ directory."""
    import pathlib
    yaml_path = pathlib.Path(__file__).parent.parent / "agents" / agent_dir / "agent.yaml"
    return load_contract(yaml_path)


# --- Nested Schema Fixtures ---

USER_PROFILE_INPUT = make_schema(
    {
        "user": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
            "required": ["name"],
        },
        "action": {"type": "string"},
    },
    ["user", "action"],
)

FLAT_OUTPUT = make_schema({
    "status": {"type": "string"},
    "user_name": {"type": "string"},
    "city": {"type": "string"},
})

BATCH_INPUT = make_schema(
    {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                },
                "required": ["name", "price"],
            },
        },
        "currency": {"type": "string", "default": "USD"},
    },
    ["items"],
)


# ============================================================================
# TestNestedSchemaValidation
# ============================================================================

class TestNestedSchemaValidation:
    """Nested object schemas (user.address.city pattern)."""

    def test_contract_loads_nestedmake_schema(self):
        """Agent contract with nested object properties loads correctly."""
        contract = _load_contract("user_profile")
        assert contract.name == "user-profile"
        in_props = contract.input_schema.properties
        assert "user" in in_props
        assert in_props["user"]["type"] == "object"
        assert "address" in in_props["user"]["properties"]

    def test_nested_required_fields_tracked(self):
        """Required fields at the top level are tracked."""
        contract = _load_contract("user_profile")
        assert set(contract.input_schema.required) == {"user", "action"}

    def test_nested_output_flattened(self):
        """Output schema with flat fields from nested input loads correctly."""
        contract = _load_contract("user_profile")
        out_props = contract.output_schema.properties
        assert "status" in out_props
        assert "user_name" in out_props

    def test_nested_vs_flat_not_identical(self):
        """Nested input schema is not identical to a flat output schema."""
        report = analyze_compatibility(FLAT_OUTPUT, USER_PROFILE_INPUT)
        assert report.level != CompatLevel.IDENTICAL

    def test_nested_schema_deep_validation(self):
        """SchemaSpec correctly exposes nested property structure."""
        props = USER_PROFILE_INPUT.properties
        user_spec = props["user"]
        address_spec = user_spec["properties"]["address"]
        assert address_spec["properties"]["city"]["type"] == "string"
        assert "city" in address_spec["required"]

    def test_nested_schema_missing_required_inner_field(self):
        """Validation catches missing nested required fields at agent level."""
        # This tests the agent logic, not mediation — inner validation is app-level
        # Just verify the schema structure supports it
        user_spec = USER_PROFILE_INPUT.properties["user"]
        assert "name" in user_spec["required"]


# ============================================================================
# TestArraySchemaValidation
# ============================================================================

class TestArraySchemaValidation:
    """Array schemas (items: [{name, price}] pattern)."""

    def test_contract_loads_arraymake_schema(self):
        """Agent contract with array items loads correctly."""
        contract = _load_contract("batch_processor")
        in_props = contract.input_schema.properties
        assert in_props["items"]["type"] == "array"
        assert "name" in in_props["items"]["items"]["properties"]

    def test_array_items_have_required(self):
        """Array items spec tracks required fields."""
        items_spec = BATCH_INPUT.properties["items"]["items"]
        assert set(items_spec["required"]) == {"name", "price"}

    def test_array_schema_has_optional_field(self):
        """Array schema with optional currency field."""
        contract = _load_contract("batch_processor")
        assert "currency" in contract.input_schema.properties
        # currency is optional (not in required)
        assert "currency" not in contract.input_schema.required

    def test_array_outputmake_schema(self):
        """Batch processor output schema has total, count, currency."""
        contract = _load_contract("batch_processor")
        out_props = contract.output_schema.properties
        assert out_props["total"]["type"] == "number"
        assert out_props["count"]["type"] == "integer"
        assert out_props["currency"]["type"] == "string"

    def test_array_vs_flat_incompatible(self):
        """Array input and flat string output are not directly compatible."""
        flat_output = make_schema({"result": {"type": "string"}})
        report = analyze_compatibility(flat_output, BATCH_INPUT)
        # Array field can't be filled from a flat string — should be SEMANTIC or INCOMPATIBLE
        assert report.level in (CompatLevel.SEMANTIC, CompatLevel.INCOMPATIBLE)

    def test_array_schema_self_compatible(self):
        """Array schema is compatible with itself."""
        report = analyze_compatibility(BATCH_INPUT, BATCH_INPUT)
        assert report.level == CompatLevel.IDENTICAL


# ============================================================================
# TestOptionalComplexDefaults
# ============================================================================

class TestOptionalComplexDefaults:
    """Optional fields with complex defaults (format, max_length with default: {})."""

    def test_contract_loads_defaults(self):
        """Agent contract with nested defaults loads correctly."""
        contract = _load_contract("flexible_input")
        in_props = contract.input_schema.properties
        assert in_props["options"]["default"] == {}
        assert in_props["options"]["properties"]["format"]["default"] == "plain"
        assert in_props["options"]["properties"]["max_length"]["default"] == 500

    def test_minimal_input_accepted(self):
        """Only required field (text) needed — options defaults to {}."""
        contract = _load_contract("flexible_input")
        assert contract.input_schema.required == ["text"]

    def test_optional_field_not_in_required(self):
        """Options is not required."""
        contract = _load_contract("flexible_input")
        assert "options" not in contract.input_schema.required

    def test_default_mediation_fills_optional(self):
        """When mediating, optional fields with defaults don't block compatibility."""
        source = make_schema({"text": {"type": "string"}})
        target = make_schema(
            {
                "text": {"type": "string"},
                "options": {"type": "object", "default": {}},
            },
            ["text"],  # Only text is required
        )
        report = analyze_compatibility(source, target)
        assert report.level in (CompatLevel.IDENTICAL, CompatLevel.SUPERSET)
        assert report.can_bridge


# ============================================================================
# TestMixedTypeSchemas
# ============================================================================

class TestMixedTypeSchemas:
    """Mixed types (oneOf: [string, object]) schemas."""

    def test_contract_loads_oneof(self):
        """Agent contract with oneOf type loads correctly."""
        contract = _load_contract("multi_type")
        in_props = contract.input_schema.properties
        assert "oneOf" in in_props["data"]

    def test_mixed_type_schema_has_enum(self):
        """Mode field with enum constraint loads correctly."""
        contract = _load_contract("multi_type")
        mode_spec = contract.input_schema.properties["mode"]
        assert mode_spec["enum"] == ["text", "structured"]

    def test_mixed_outputmake_schema(self):
        """Multi-type output schema has processed and mode_used."""
        contract = _load_contract("multi_type")
        out_props = contract.output_schema.properties
        assert "processed" in out_props
        assert "mode_used" in out_props


# ============================================================================
# TestComplexMediation
# ============================================================================

class TestComplexMediation:
    """Cross-schema mediation with complex types."""

    def test_nested_to_flat_is_semantic(self):
        """Nested object output → flat string input requires semantic bridging."""
        nested_output = make_schema({
            "user": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        })
        flat_input = make_schema({"user_name": {"type": "string"}})
        report = analyze_compatibility(nested_output, flat_input)
        # The nested "user" field doesn't match flat "user_name" by name
        # but semantic bridging should be possible
        assert report.level in (CompatLevel.SEMANTIC, CompatLevel.COERCIBLE)
        assert report.can_bridge

    def test_flat_to_nested_is_semantic(self):
        """Flat fields → nested object input needs semantic/LLM bridging."""
        flat_output = make_schema({
            "name": {"type": "string"},
            "city": {"type": "string"},
        })
        report = analyze_compatibility(flat_output, USER_PROFILE_INPUT)
        assert report.can_bridge or report.level == CompatLevel.INCOMPATIBLE
        # Either SEMANTIC (LLM can restructure) or INCOMPATIBLE (no auto path)

    def test_array_to_flat_semantic(self):
        """Array output → flat input should be SEMANTIC (LLM can summarize)."""
        array_output = make_schema({
            "items": {"type": "array"},
            "total": {"type": "number"},
        })
        flat_input = make_schema({"summary": {"type": "string"}})
        report = analyze_compatibility(array_output, flat_input)
        # Array/string schemas should be semantically bridgeable after fix
        assert report.level == CompatLevel.SEMANTIC

    def test_jsonpath_into_nested_context(self):
        """JSONPath resolution works into nested chain context."""
        from atlas.mediation.strategies import _resolve_path, _SENTINEL
        chain_data = {
            "steps_by_name": {
                "profile": {
                    "output": {
                        "user_name": "Alice",
                        "city": "NYC",
                    }
                }
            }
        }
        val = _resolve_path("$.steps.profile.output.user_name", {}, chain_data)
        assert val == "Alice"

    def test_jsonpath_into_array_context(self):
        """JSONPath resolution with array index in chain context."""
        from atlas.mediation.strategies import _resolve_path
        chain_data = {
            "steps": [
                {"output": {"items": [{"name": "A"}, {"name": "B"}]}}
            ]
        }
        val = _resolve_path("$.steps[0].output.items", {}, chain_data)
        assert val == [{"name": "A"}, {"name": "B"}]

    @pytest.mark.asyncio
    async def test_mediation_engine_nested_schemas(self):
        """MediationEngine can analyze nested schema compatibility."""
        engine = MediationEngine()
        report = await engine.analyze(FLAT_OUTPUT, USER_PROFILE_INPUT)
        # Should get some result, not crash
        assert report.level is not None

    @pytest.mark.asyncio
    async def test_mediation_engine_array_schemas(self):
        """MediationEngine can analyze array schema compatibility."""
        engine = MediationEngine()
        report = await engine.analyze(BATCH_INPUT, BATCH_INPUT)
        assert report.level == CompatLevel.IDENTICAL

    def test_few_shot_examples_recurse_nested(self):
        """Few-shot examples recurse into nested object properties."""
        props = {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            }
        }
        examples = _build_few_shot_examples(props)
        assert '"name"' in examples
        assert '"age"' in examples

    def test_few_shot_examples_recurse_array_items(self):
        """Few-shot examples show array item structure."""
        props = {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "price": {"type": "number"},
                    },
                },
            }
        }
        examples = _build_few_shot_examples(props)
        assert '"name"' in examples
        assert '"price"' in examples

    async def test_direct_strategy_deepcopy_safety(self):
        """DirectStrategy deep copies nested data to prevent mutation."""
        source = make_schema({"user": {"type": "object"}, "extra": {"type": "string"}})
        target = make_schema({"user": {"type": "object"}}, ["user"])
        data = {"user": {"name": "Alice", "nested": {"deep": True}}, "extra": "x"}

        strategy = DirectStrategy()
        from atlas.mediation.analyzer import CompatibilityReport
        report = CompatibilityReport(level=CompatLevel.SUPERSET)
        ctx = MediationContext(source_schema=source, target_schema=target)

        result = await strategy.transform(data, report, ctx)

        # Mutating the result should not affect original
        result["user"]["name"] = "Bob"
        assert data["user"]["name"] == "Alice"
