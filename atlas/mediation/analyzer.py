"""Schema compatibility analysis — the core of chain mediation.

Analyzes how well one agent's output schema feeds into another's input schema,
and determines what transformation strategy is needed to bridge the gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any

from atlas.contract.types import SchemaSpec


class CompatLevel(Enum):
    """How compatible two schemas are, from cheapest to most expensive to bridge."""

    IDENTICAL = "identical"       # Same schema — zero-cost passthrough
    SUPERSET = "superset"         # Output has all input fields + extras — just drop fields
    MAPPABLE = "mappable"         # Explicit input_map resolves everything
    COERCIBLE = "coercible"       # Deterministic transforms bridge the gap (rename, cast, flatten)
    SEMANTIC = "semantic"         # Needs LLM to transform (schemas semantically related)
    INCOMPATIBLE = "incompatible" # Can't bridge — schemas have nothing in common


@dataclass
class CoercionOp:
    """A single deterministic transformation operation."""

    op_type: str  # "rename", "cast", "flatten", "nest", "default"
    source_field: str
    target_field: str
    details: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if self.op_type == "rename":
            return f"rename '{self.source_field}' → '{self.target_field}'"
        if self.op_type == "cast":
            return f"cast '{self.source_field}' ({self.details.get('from_type')}) → '{self.target_field}' ({self.details.get('to_type')})"
        if self.op_type == "default":
            return f"default '{self.target_field}' = {self.details.get('value')!r}"
        return f"{self.op_type}: {self.source_field} → {self.target_field}"


@dataclass
class CompatibilityReport:
    """Full analysis of how to bridge two schemas."""

    level: CompatLevel
    direct_mappings: dict[str, str] = field(default_factory=dict)  # target_field → source_field
    coercions: list[CoercionOp] = field(default_factory=list)
    unmapped_required: list[str] = field(default_factory=list)  # Required input fields with no source
    unmapped_optional: list[str] = field(default_factory=list)  # Optional input fields with no source
    confidence: float = 1.0  # 0-1, how confident the analysis is
    notes: list[str] = field(default_factory=list)  # Human-readable explanation

    @property
    def can_bridge(self) -> bool:
        return self.level != CompatLevel.INCOMPATIBLE


def analyze_compatibility(
    output_schema: SchemaSpec,
    input_schema: SchemaSpec,
    input_map: dict[str, str] | None = None,
) -> CompatibilityReport:
    """Analyze how well output_schema feeds into input_schema.

    Strategy hierarchy (cheapest first):
    1. IDENTICAL — schemas are the same
    2. SUPERSET — output has all required input fields
    3. MAPPABLE — explicit input_map covers all required fields
    4. COERCIBLE — deterministic transforms (rename, cast) can bridge
    5. SEMANTIC — LLM needed
    6. INCOMPATIBLE — can't bridge
    """
    out_props = output_schema.properties
    in_props = input_schema.properties
    in_required = set(input_schema.required)
    in_optional = set(in_props.keys()) - in_required

    # Trivial case: empty input schema accepts anything
    if not in_props:
        return CompatibilityReport(level=CompatLevel.IDENTICAL, confidence=1.0)

    # Phase 1: Find exact field name matches
    direct_mappings: dict[str, str] = {}  # target → source
    for target_field in in_props:
        if target_field in out_props:
            direct_mappings[target_field] = target_field

    # Check for IDENTICAL
    if set(in_props.keys()) == set(out_props.keys()):
        if _schemas_type_compatible(out_props, in_props, in_props.keys()):
            return CompatibilityReport(
                level=CompatLevel.IDENTICAL,
                direct_mappings=direct_mappings,
                confidence=1.0,
            )

    # Check for SUPERSET
    if in_required.issubset(direct_mappings.keys()):
        if _schemas_type_compatible(out_props, in_props, direct_mappings.keys()):
            unmapped_optional = [f for f in in_optional if f not in direct_mappings]
            return CompatibilityReport(
                level=CompatLevel.SUPERSET,
                direct_mappings=direct_mappings,
                unmapped_optional=unmapped_optional,
                confidence=1.0,
            )

    # Phase 2: Check explicit input_map
    if input_map:
        mapped_report = _check_input_map(input_map, in_required, in_optional, out_props, in_props)
        if mapped_report:
            return mapped_report

    # Phase 3: Try coercion (fuzzy field matching + type casting)
    coercion_report = _try_coercion(out_props, in_props, in_required, in_optional, direct_mappings)
    if coercion_report:
        return coercion_report

    # Phase 4: Check if semantic bridging could work
    semantic_report = _check_semantic(out_props, in_props, in_required)
    if semantic_report:
        return semantic_report

    # Phase 5: Incompatible
    return CompatibilityReport(
        level=CompatLevel.INCOMPATIBLE,
        direct_mappings=direct_mappings,
        unmapped_required=list(in_required - set(direct_mappings.keys())),
        unmapped_optional=list(in_optional - set(direct_mappings.keys())),
        confidence=1.0,
        notes=["No viable transformation path found"],
    )


def _schemas_type_compatible(
    out_props: dict[str, Any],
    in_props: dict[str, Any],
    fields: set[str] | dict[str, str] | Any,
) -> bool:
    """Check if mapped fields have compatible types."""
    for field_name in fields:
        out_type = _get_type(out_props.get(field_name, {}))
        in_type = _get_type(in_props.get(field_name, {}))
        if out_type and in_type and out_type != in_type:
            if not _type_castable(out_type, in_type):
                return False
    return True


def _get_type(prop: dict[str, Any] | Any) -> str:
    """Extract type from a JSON Schema property."""
    if isinstance(prop, dict):
        return prop.get("type", "")
    return ""


def _type_castable(from_type: str, to_type: str) -> bool:
    """Check if a type can be losslessly cast to another."""
    # Lossless casts
    castable = {
        ("integer", "string"),
        ("integer", "number"),
        ("number", "string"),
        ("boolean", "string"),
    }
    return (from_type, to_type) in castable


def _check_input_map(
    input_map: dict[str, str],
    in_required: set[str],
    in_optional: set[str],
    out_props: dict[str, Any],
    in_props: dict[str, Any],
) -> CompatibilityReport | None:
    """Check if an explicit input_map covers all required fields."""
    mapped_fields = set(input_map.keys())
    unmapped_required = in_required - mapped_fields
    unmapped_optional = in_optional - mapped_fields

    if not unmapped_required:
        # All required fields covered by the map
        # Verify source fields exist (simple check — no JSONPath resolution here)
        notes = []
        for target, source in input_map.items():
            if source.startswith("$."):
                notes.append(f"JSONPath: {target} ← {source}")
            elif source not in out_props:
                notes.append(f"Static value: {target} = {source!r}")

        return CompatibilityReport(
            level=CompatLevel.MAPPABLE,
            direct_mappings=input_map,
            unmapped_optional=list(unmapped_optional),
            confidence=0.95,
            notes=notes,
        )
    return None


def _try_coercion(
    out_props: dict[str, Any],
    in_props: dict[str, Any],
    in_required: set[str],
    in_optional: set[str],
    direct_mappings: dict[str, str],
) -> CompatibilityReport | None:
    """Try to bridge via deterministic transformations (rename, cast)."""
    coercions: list[CoercionOp] = []
    all_mappings = dict(direct_mappings)  # target → source

    # Find unmapped required fields
    unmapped = in_required - set(direct_mappings.keys())
    available_sources = set(out_props.keys()) - set(direct_mappings.values())

    for target_field in list(unmapped):
        target_spec = in_props.get(target_field, {})
        best_match: str | None = None
        best_score: float = 0.0

        for source_field in available_sources:
            score = _field_similarity(source_field, target_field)
            if score > best_score:
                best_score = score
                best_match = source_field

        if best_match and best_score >= 0.5:
            source_spec = out_props.get(best_match, {})
            source_type = _get_type(source_spec)
            target_type = _get_type(target_spec)

            if source_type == target_type or not source_type or not target_type:
                # Same type or unknown — just rename
                coercions.append(CoercionOp(
                    op_type="rename",
                    source_field=best_match,
                    target_field=target_field,
                    details={"similarity": best_score},
                ))
            elif _type_castable(source_type, target_type):
                coercions.append(CoercionOp(
                    op_type="cast",
                    source_field=best_match,
                    target_field=target_field,
                    details={
                        "from_type": source_type,
                        "to_type": target_type,
                        "similarity": best_score,
                    },
                ))
            else:
                continue  # Can't coerce this pair

            all_mappings[target_field] = best_match
            available_sources.discard(best_match)
            unmapped.discard(target_field)

    # Check if defaults can fill remaining required fields
    for target_field in list(unmapped):
        target_spec = in_props.get(target_field, {})
        if isinstance(target_spec, dict) and "default" in target_spec:
            coercions.append(CoercionOp(
                op_type="default",
                source_field="",
                target_field=target_field,
                details={"value": target_spec["default"]},
            ))
            unmapped.discard(target_field)

    if not unmapped:
        # All required fields resolved
        confidence = min(
            (c.details.get("similarity", 1.0) for c in coercions if "similarity" in c.details),
            default=0.9,
        )
        return CompatibilityReport(
            level=CompatLevel.COERCIBLE,
            direct_mappings=all_mappings,
            coercions=coercions,
            unmapped_optional=list(in_optional - set(all_mappings.keys())),
            confidence=round(confidence, 2),
            notes=[c.describe() for c in coercions],
        )

    return None


def _field_similarity(a: str, b: str) -> float:
    """Score how similar two field names are (0-1).

    Handles common patterns:
    - Exact match: 1.0
    - Case-insensitive: 0.95
    - Normalized match (camelCase == snake_case): 0.95
    - Abbreviation expansion: 0.9
    - Token overlap: high score
    - Subsequence match: medium score
    """
    if a == b:
        return 1.0
    if a.lower() == b.lower():
        return 0.95

    # Tokenize and normalize
    a_tokens = _tokenize_field(a)
    b_tokens = _tokenize_field(b)

    # Expand abbreviations
    a_expanded = [_expand_abbreviation(t) for t in a_tokens]
    b_expanded = [_expand_abbreviation(t) for t in b_tokens]

    # Exact match after normalization (e.g. translatedText == translated_text)
    if a_expanded == b_expanded:
        return 0.95

    # Token overlap (Jaccard-like but weighted toward the target)
    a_set = set(a_expanded)
    b_set = set(b_expanded)
    if a_set and b_set:
        overlap = a_set & b_set
        if overlap:
            # Weight by how much of the target is covered
            coverage = len(overlap) / len(b_set)
            return min(0.9, 0.5 + coverage * 0.4)

    # Fall back to sequence matching
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Common abbreviation expansions
_ABBREVIATIONS: dict[str, str] = {
    "msg": "message",
    "desc": "description",
    "txt": "text",
    "src": "source",
    "dst": "destination",
    "val": "value",
    "num": "number",
    "cnt": "count",
    "idx": "index",
    "len": "length",
    "fmt": "format",
    "cfg": "config",
    "ctx": "context",
    "err": "error",
    "res": "result",
    "req": "request",
    "resp": "response",
}


def _expand_abbreviation(token: str) -> str:
    """Expand common abbreviations to full forms."""
    return _ABBREVIATIONS.get(token, token)


def _tokenize_field(name: str) -> list[str]:
    """Split a field name into semantic tokens.

    'translated_text' → ['translated', 'text']
    'translatedText' → ['translated', 'text']
    'max_length' → ['max', 'length']
    'XMLParser' → ['xml', 'parser']
    """
    # Split on underscores first
    parts = name.split("_")
    tokens = []
    for part in parts:
        # Split camelCase/PascalCase with proper acronym handling
        # XMLParser → XML, Parser → xml, parser
        # translatedText → translated, Text → translated, text
        subtokens = re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", part)
        tokens.extend(t.lower() for t in subtokens)
    return tokens


def _check_semantic(
    out_props: dict[str, Any],
    in_props: dict[str, Any],
    in_required: set[str],
) -> CompatibilityReport | None:
    """Check if an LLM could plausibly bridge these schemas.

    Heuristic: if both schemas have string-typed fields, there's a chance
    an LLM can reason about the transformation. If schemas are completely
    disjoint types (e.g., numbers in vs. arrays out), probably not.
    """
    bridgeable_types = ("string", "object", "array", "")
    out_has_bridgeable = any(
        _get_type(v) in bridgeable_types for v in out_props.values()
    )
    in_needs_bridgeable = any(
        _get_type(in_props.get(f, {})) in bridgeable_types for f in in_required
    )

    if out_has_bridgeable and in_needs_bridgeable:
        return CompatibilityReport(
            level=CompatLevel.SEMANTIC,
            unmapped_required=list(in_required),
            confidence=0.5,
            notes=[
                "Schemas appear semantically bridgeable",
                f"Output has {len(out_props)} fields, input needs {len(in_required)} required fields",
                "An LLM could attempt this transformation",
            ],
        )

    return None
