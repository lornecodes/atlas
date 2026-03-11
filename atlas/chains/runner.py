"""ChainRunner — execute chains with automatic mediation between steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atlas.chains.definition import ChainDefinition
from atlas.contract.registry import AgentRegistry
from atlas.logging import get_logger
from atlas.mediation.engine import MediationEngine, MediationResult
from atlas.runtime.context import AgentContext
from atlas.runtime.runner import AgentResult, RunError, run_agent

logger = get_logger(__name__)


@dataclass
class StepResult:
    """Result of a single chain step."""

    agent_name: str
    agent_result: AgentResult
    mediation: MediationResult | None = None  # How input was prepared


@dataclass
class ChainResult:
    """Result of executing a full chain."""

    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    steps: list[StepResult] = field(default_factory=list)
    failed_at: int = -1
    error: str = ""

    @property
    def partial_outputs(self) -> list[dict[str, Any]]:
        """Outputs from completed steps (useful on failure)."""
        return [
            s.agent_result.data
            for s in self.steps
            if s.agent_result.success
        ]

    @property
    def mediation_summary(self) -> list[dict[str, str]]:
        """Summary of mediation strategies used at each step."""
        return [
            {
                "step": i,
                "agent": s.agent_name,
                "strategy": s.mediation.strategy_used if s.mediation else "none",
                "cost": s.mediation.cost if s.mediation else 0.0,
            }
            for i, s in enumerate(self.steps)
        ]


class ChainRunner:
    """Execute a chain of agents, mediating I/O between steps."""

    def __init__(
        self,
        registry: AgentRegistry,
        mediation: MediationEngine,
    ) -> None:
        self._registry = registry
        self._mediation = mediation

    async def execute(
        self,
        chain: ChainDefinition,
        trigger_input: dict,
        *,
        providers: dict[str, dict[str, Any]] | None = None,
    ) -> ChainResult:
        """Execute a chain, mediating between each step.

        The trigger input feeds into step 0. Each subsequent step receives
        mediated output from the previous step.

        Args:
            providers: Optional per-agent dependency injection. Maps agent
                names to provider dicts that get set on ``AgentContext.providers``.
                Example: ``{"claude-writer": {"llm_provider": my_provider}}``
        """
        # Pre-validate: all agents must exist before we start executing
        missing = [
            s.agent_name for s in chain.steps
            if not self._registry.get(s.agent_name)
        ]
        if missing:
            return ChainResult(
                success=False,
                error=f"Chain '{chain.name}' references unknown agents: {', '.join(missing)}",
            )

        steps: list[StepResult] = []
        step_outputs: list[dict] = []

        logger.info("Starting chain '%s' (%d steps)", chain.name, len(chain.steps))

        for i, step in enumerate(chain.steps):
            step_name = chain.step_name(i)
            logger.debug("Chain '%s' step %d (%s): agent=%s", chain.name, i, step_name, step.agent_name)

            # Prepare input for this step
            if i == 0:
                # First step gets trigger input directly (or mediated if schemas differ)
                step_input = trigger_input
                mediation_result = None
            else:
                # Mediate between previous output and this step's input
                prev_output = step_outputs[-1]
                prev_agent = self._registry.get(chain.steps[i - 1].agent_name)
                curr_agent = self._registry.get(step.agent_name)

                if not prev_agent or not curr_agent:
                    return ChainResult(
                        success=False,
                        output={"partial_outputs": [s.agent_result.data for s in steps if s.agent_result.success], "failed_step": i},
                        steps=steps,
                        failed_at=i,
                        error=f"Agent not found for step {i}",
                    )

                # Build chain context with named step access
                steps_context = []
                for j, out in enumerate(step_outputs):
                    sn = chain.step_name(j)
                    steps_context.append({"output": out, "name": sn})

                mediation_result = await self._mediation.mediate(
                    source_output=prev_output,
                    source_schema=prev_agent.contract.output_schema,
                    target_schema=curr_agent.contract.input_schema,
                    input_map=step.input_map,
                    chain_context={
                        "trigger": trigger_input,
                        "steps": steps_context,
                        "steps_by_name": {
                            ctx["name"]: ctx for ctx in steps_context
                        },
                    },
                )

                if not mediation_result.success:
                    logger.warning("Mediation failed at step %d (%s): %s", i, step_name, mediation_result.error)
                    return ChainResult(
                        success=False,
                        output={"partial_outputs": [s.agent_result.data for s in steps if s.agent_result.success], "failed_step": i},
                        steps=steps,
                        failed_at=i,
                        error=f"Mediation failed at step {i} ({step.agent_name}): {mediation_result.error}",
                    )

                step_input = mediation_result.data

            # Build context with chain info + injected providers
            agent_providers = (providers or {}).get(step.agent_name, {})
            ctx = AgentContext(
                chain_name=chain.name,
                step_index=i,
                chain_data={
                    "trigger": trigger_input,
                    "step_outputs": list(step_outputs),
                },
                providers=agent_providers,
            )

            # Execute the agent
            try:
                agent_result = await run_agent(
                    self._registry,
                    step.agent_name,
                    step_input,
                    context=ctx,
                )
            except RunError as e:
                logger.error("Chain '%s' RunError at step %d: %s", chain.name, i, e)
                steps.append(StepResult(
                    agent_name=step.agent_name,
                    agent_result=AgentResult(error=str(e), agent_name=step.agent_name),
                    mediation=mediation_result,
                ))
                return ChainResult(
                    success=False,
                    output={"partial_outputs": [s.agent_result.data for s in steps if s.agent_result.success], "failed_step": i},
                    steps=steps,
                    failed_at=i,
                    error=str(e),
                )

            steps.append(StepResult(
                agent_name=step.agent_name,
                agent_result=agent_result,
                mediation=mediation_result,
            ))

            if not agent_result.success:
                logger.warning("Chain '%s' agent failed at step %d (%s): %s", chain.name, i, step_name, agent_result.error)
                return ChainResult(
                    success=False,
                    output={"partial_outputs": [s.agent_result.data for s in steps if s.agent_result.success], "failed_step": i},
                    steps=steps,
                    failed_at=i,
                    error=f"Agent failed at step {i} ({step.agent_name}): {agent_result.error}",
                )

            step_outputs.append(agent_result.data)

        logger.info("Chain '%s' completed successfully", chain.name)
        return ChainResult(
            success=True,
            output=step_outputs[-1] if step_outputs else {},
            steps=steps,
        )
