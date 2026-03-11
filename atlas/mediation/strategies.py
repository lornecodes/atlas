"""Mediation strategies — the transformation implementations.

Each strategy handles a different compatibility level, from zero-cost
passthrough to expensive LLM-based transformation.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING

from atlas.contract.types import SchemaSpec
from atlas.mediation.analyzer import CompatLevel, CompatibilityReport, CoercionOp

if TYPE_CHECKING:
    from atlas.llm.provider import LLMProvider


@dataclass
class MediationContext:
    """Context for a mediation operation."""

    source_schema: SchemaSpec
    target_schema: SchemaSpec
    chain_context: dict[str, Any] = field(default_factory=dict)


class MediationStrategy(ABC):
    """Base class for mediation strategies."""

    name: str = "base"
    cost: float = 0.0  # Relative cost (0 = free, 1 = expensive)

    @abstractmethod
    def can_handle(self, report: CompatibilityReport) -> bool:
        """Whether this strategy can handle the given compatibility report."""
        ...

    @abstractmethod
    async def transform(
        self,
        data: dict,
        report: CompatibilityReport,
        context: MediationContext,
    ) -> dict:
        """Transform data from source schema to target schema."""
        ...


class DirectStrategy(MediationStrategy):
    """Pass through, keeping only fields the target schema expects."""

    name = "direct"
    cost = 0.0

    def can_handle(self, report: CompatibilityReport) -> bool:
        return report.level in (CompatLevel.IDENTICAL, CompatLevel.SUPERSET)

    async def transform(
        self,
        data: dict,
        report: CompatibilityReport,
        context: MediationContext,
    ) -> dict:
        target_fields = set(context.target_schema.properties.keys())
        if not target_fields:
            return dict(data)

        # Keep matching fields, deep copy for nested data safety
        result = {}
        for key, value in data.items():
            if key in target_fields:
                result[key] = copy.deepcopy(value)
        return result


class MappedStrategy(MediationStrategy):
    """Apply explicit input_map — JSONPath or static values."""

    name = "mapped"
    cost = 0.0

    def can_handle(self, report: CompatibilityReport) -> bool:
        return report.level == CompatLevel.MAPPABLE

    async def transform(
        self,
        data: dict,
        report: CompatibilityReport,
        context: MediationContext,
    ) -> dict:
        result = {}
        chain_data = context.chain_context

        for target_field, source_expr in report.direct_mappings.items():
            if source_expr.startswith("$."):
                # JSONPath-like resolution
                value = _resolve_path(source_expr, data, chain_data)
                if value is not _SENTINEL:
                    result[target_field] = value
            elif source_expr in data:
                # Direct field reference
                result[target_field] = data[source_expr]
            else:
                # Static value
                result[target_field] = source_expr

        return result


class CoerceStrategy(MediationStrategy):
    """Deterministic transforms: rename fields, cast types, apply defaults."""

    name = "coerce"
    cost = 0.0

    def can_handle(self, report: CompatibilityReport) -> bool:
        return report.level == CompatLevel.COERCIBLE

    async def transform(
        self,
        data: dict,
        report: CompatibilityReport,
        context: MediationContext,
    ) -> dict:
        result = {}

        # Apply direct mappings first (exact name matches)
        for target_field, source_field in report.direct_mappings.items():
            if source_field in data:
                result[target_field] = data[source_field]

        # Apply coercion operations
        for op in report.coercions:
            if op.op_type == "rename":
                if op.source_field in data:
                    result[op.target_field] = data[op.source_field]

            elif op.op_type == "cast":
                if op.source_field in data:
                    result[op.target_field] = _cast_value(
                        data[op.source_field],
                        op.details.get("to_type", "string"),
                    )

            elif op.op_type == "default":
                if op.target_field not in result:
                    result[op.target_field] = op.details.get("value")

        return result


# Type for legacy LLM provider callback (backward compat)
LLMCallable = Callable[[str], Awaitable[str]]


class LLMBridgeStrategy(MediationStrategy):
    """Ask an LLM to transform the data when no deterministic path exists.

    Accepts either:
    - An LLMProvider protocol instance (preferred — gives token tracking)
    - A bare async callable (str → str) for backward compat
    """

    name = "llm_bridge"
    cost = 1.0

    def __init__(
        self,
        llm_provider: LLMProvider | LLMCallable | None = None,
        *,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        self._provider = llm_provider
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    def can_handle(self, report: CompatibilityReport) -> bool:
        return report.level == CompatLevel.SEMANTIC and self._provider is not None

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM, handling both protocol and callable providers."""
        from atlas.llm.provider import LLMProvider as LLMProviderProto

        if isinstance(self._provider, LLMProviderProto):
            response = await self._provider.complete(prompt)
            self._total_input_tokens += response.input_tokens
            self._total_output_tokens += response.output_tokens
            return response.text
        elif callable(self._provider):
            return await self._provider(prompt)
        else:
            raise RuntimeError("LLM provider not configured")

    async def transform(
        self,
        data: dict,
        report: CompatibilityReport,
        context: MediationContext,
    ) -> dict:
        if not self._provider:
            raise RuntimeError("LLM provider not configured")

        prompt = _build_bridge_prompt(data, context.source_schema, context.target_schema)
        last_error: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                if attempt == 0:
                    response = await self._call_llm(prompt)
                else:
                    retry_prompt = (
                        f"Your previous response was not valid JSON. "
                        f"Please try again.\n\n{prompt}"
                    )
                    response = await self._call_llm(retry_prompt)

                result = json.loads(_extract_json(response))

                if not isinstance(result, dict):
                    raise ValueError(f"Expected dict, got {type(result).__name__}")

                # Validate required fields
                target_required = set(context.target_schema.required)
                missing = target_required - set(result.keys())
                if missing:
                    raise ValueError(
                        f"Missing required fields: {', '.join(sorted(missing))}"
                    )

                return result

            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_base_delay * (2 ** attempt))

        raise RuntimeError(
            f"LLM bridge failed after {self._max_retries} attempts: {last_error}"
        )


# --- Helpers ---

_SENTINEL = object()


def _resolve_path(path: str, data: dict, chain_data: dict) -> Any:
    """Resolve a simple JSONPath-like expression.

    Supports:
    - $.field — top-level field from current data
    - $.steps[N].output.field — field from a previous chain step (by index)
    - $.steps.step_name.output.field — field from a named chain step
    - $.trigger.field — field from trigger input
    """
    if not path.startswith("$."):
        return _SENTINEL

    path = path[2:]  # Strip "$."

    # Determine root object
    if path.startswith("steps["):
        root = chain_data
    elif path.startswith("steps."):
        # Named step: $.steps.step_name.output.field
        parts = path.split(".", 2)  # ['steps', 'step_name', 'output.field']
        if len(parts) >= 2:
            step_name = parts[1]
            steps_by_name = chain_data.get("steps_by_name", {})
            step_ctx = steps_by_name.get(step_name)
            if step_ctx is None:
                return _SENTINEL
            remaining = parts[2] if len(parts) > 2 else ""
            if remaining:
                return _walk_path(step_ctx, remaining)
            return step_ctx
        return _SENTINEL
    elif path.startswith("trigger."):
        root = chain_data.get("trigger", {})
        path = path[len("trigger."):]
    else:
        root = data

    return _walk_path(root, path)


def _walk_path(obj: Any, path: str) -> Any:
    """Walk a dotted/bracketed path through an object."""
    parts = _split_path(path)
    current = obj

    for part in parts:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return _SENTINEL
        elif isinstance(current, (list, tuple)):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return _SENTINEL
        else:
            return _SENTINEL

    return current


def _split_path(path: str) -> list[str]:
    """Split 'steps[0].output.field' into ['steps', '0', 'output', 'field']."""
    parts = []
    for segment in path.split("."):
        # Handle array indexing: steps[0] → ['steps', '0']
        match = re.match(r"(\w+)\[(\d+)\]", segment)
        if match:
            parts.append(match.group(1))
            parts.append(match.group(2))
        else:
            parts.append(segment)
    return parts


def _cast_value(value: Any, to_type: str) -> Any:
    """Cast a value to a target JSON Schema type."""
    if to_type == "string":
        return str(value)
    if to_type == "integer":
        return int(value)
    if to_type == "number":
        return float(value)
    if to_type == "boolean":
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _build_bridge_prompt(
    data: dict,
    source_schema: SchemaSpec,
    target_schema: SchemaSpec,
) -> str:
    """Build an LLM prompt for schema transformation with few-shot examples."""
    target_props = target_schema.properties
    examples = _build_few_shot_examples(target_props)

    return f"""Transform this data from the source schema to match the target schema.

SOURCE SCHEMA:
{json.dumps(source_schema.to_json_schema(), indent=2)}

TARGET SCHEMA:
{json.dumps(target_schema.to_json_schema(), indent=2)}

DATA TO TRANSFORM:
{json.dumps(data, indent=2)}
{examples}
Return ONLY valid JSON matching the target schema. No explanation, no markdown fences."""


def _build_few_shot_examples(target_props: dict[str, Any]) -> str:
    """Build type-appropriate few-shot examples for the target schema."""
    if not target_props:
        return ""

    example = _build_example_value_for_props(target_props)

    return f"\nEXPECTED OUTPUT FORMAT:\n{json.dumps(example, indent=2)}\n"


def _build_example_value_for_props(props: dict[str, Any]) -> dict[str, Any]:
    """Recursively build example values for a set of properties."""
    example = {}
    for field_name, spec in props.items():
        example[field_name] = _build_example_value(field_name, spec)
    return example


def _build_example_value(field_name: str, spec: Any) -> Any:
    """Build a type-appropriate example value, recursing into objects/arrays."""
    if not isinstance(spec, dict):
        return f"<{field_name} value>"

    field_type = spec.get("type", "string")
    if field_type == "string":
        return f"<{field_name} value>"
    elif field_type == "integer":
        return 0
    elif field_type == "number":
        return 0.0
    elif field_type == "boolean":
        return True
    elif field_type == "array":
        items_spec = spec.get("items", {})
        if isinstance(items_spec, dict) and items_spec.get("type") == "object":
            inner_props = items_spec.get("properties", {})
            if inner_props:
                return [_build_example_value_for_props(inner_props)]
        return ["<item>"]
    elif field_type == "object":
        inner_props = spec.get("properties", {})
        if inner_props:
            return _build_example_value_for_props(inner_props)
        return {}
    else:
        return None


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown fences."""
    text = text.strip()

    # Try to extract from ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try raw JSON
    if text.startswith("{"):
        return text

    raise ValueError(f"Could not extract JSON from response: {text[:100]}")
