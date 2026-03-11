"""MediationEngine — pick the cheapest viable strategy to bridge agent I/O."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas.contract.types import SchemaSpec
from atlas.logging import get_logger
from atlas.mediation.analyzer import CompatibilityReport, CompatLevel, analyze_compatibility
from atlas.mediation.strategies import (
    CoerceStrategy,
    DirectStrategy,
    LLMBridgeStrategy,
    LLMCallable,
    MappedStrategy,
    MediationContext,
    MediationStrategy,
)

logger = get_logger(__name__)


@dataclass
class MediationResult:
    """Result of a mediation attempt."""

    success: bool
    data: dict[str, Any] | None = None
    strategy_used: str = ""
    report: CompatibilityReport | None = None
    error: str = ""
    cost: float = 0.0  # Relative cost of the strategy used


class MediationEngine:
    """Try strategies in cost order to bridge agent I/O schemas.

    The engine runs the compatibility analyzer, then iterates through
    strategies from cheapest (direct passthrough) to most expensive
    (LLM bridge), returning the first successful result.
    """

    def __init__(self, llm_provider: LLMCallable | None = None) -> None:
        self._llm_bridge: LLMBridgeStrategy | None = None
        self._strategies: list[MediationStrategy] = [
            DirectStrategy(),
            MappedStrategy(),
            CoerceStrategy(),
        ]
        if llm_provider:
            self._llm_bridge = LLMBridgeStrategy(llm_provider)
            self._strategies.append(self._llm_bridge)

    @property
    def total_llm_tokens(self) -> int:
        """Total tokens used by LLM bridge calls."""
        if self._llm_bridge:
            return self._llm_bridge.total_tokens
        return 0

    async def mediate(
        self,
        source_output: dict,
        source_schema: SchemaSpec,
        target_schema: SchemaSpec,
        *,
        input_map: dict[str, str] | None = None,
        chain_context: dict[str, Any] | None = None,
    ) -> MediationResult:
        """Try to transform source_output to match target_schema.

        Runs the analyzer first, then picks the cheapest strategy that
        can handle the compatibility level.
        """
        report = analyze_compatibility(source_schema, target_schema, input_map)
        logger.debug("Compatibility: %s (confidence=%.2f)", report.level.value, report.confidence)

        if not report.can_bridge:
            return MediationResult(
                success=False,
                report=report,
                error=f"Schemas are incompatible: {', '.join(report.notes)}",
            )

        context = MediationContext(
            source_schema=source_schema,
            target_schema=target_schema,
            chain_context=chain_context or {},
        )

        for strategy in self._strategies:
            if strategy.can_handle(report):
                try:
                    logger.debug("Trying strategy: %s", strategy.name)
                    result = await strategy.transform(source_output, report, context)
                    return MediationResult(
                        success=True,
                        data=result,
                        strategy_used=strategy.name,
                        report=report,
                        cost=strategy.cost,
                    )
                except Exception as e:
                    return MediationResult(
                        success=False,
                        report=report,
                        strategy_used=strategy.name,
                        error=f"Strategy '{strategy.name}' failed: {e}",
                    )

        # No strategy could handle it
        return MediationResult(
            success=False,
            report=report,
            error=f"No strategy available for compatibility level: {report.level.value}",
        )

    async def analyze(
        self,
        source_schema: SchemaSpec,
        target_schema: SchemaSpec,
        *,
        input_map: dict[str, str] | None = None,
    ) -> CompatibilityReport:
        """Run analysis without transformation — for introspection/debugging."""
        return analyze_compatibility(source_schema, target_schema, input_map)
