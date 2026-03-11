"""Agent runner — load, validate, execute, return."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from atlas.contract.registry import AgentRegistry
from atlas.contract.schema import validate_input, validate_output
from atlas.logging import get_logger
from atlas.runtime.context import AgentContext

logger = get_logger(__name__)


@dataclass
class AgentResult:
    """Standardized result from running an agent."""

    success: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    agent_name: str = ""
    validation_errors: list[str] = field(default_factory=list)


class RunError(Exception):
    """Fatal runtime error — agent not found, no implementation."""


async def run_agent(
    registry: AgentRegistry,
    agent_name: str,
    input_data: dict,
    *,
    context: AgentContext | None = None,
) -> AgentResult:
    """Load an agent from the registry, validate I/O, execute, return result.

    Raises RunError for fatal failures (agent not found, no implementation).
    Returns AgentResult(success=False) for validation/execution failures.
    """
    # Resolve agent
    entry = registry.get(agent_name)
    if not entry:
        raise RunError(f"Agent not found: {agent_name}")

    agent_class = entry.agent_class
    if not agent_class:
        raise RunError(f"No agent implementation found for: {agent_name}")

    # Validate input
    input_errors = validate_input(entry.contract, input_data)
    if input_errors:
        logger.warning("Input validation failed for %s: %s", agent_name, input_errors)
        return AgentResult(
            error="Input validation failed",
            agent_name=agent_name,
            validation_errors=input_errors,
        )

    # Instantiate and execute
    ctx = context or AgentContext()
    agent = agent_class(entry.contract, ctx)
    timeout = entry.contract.execution_timeout

    try:
        await agent.on_startup()
        logger.debug("Executing %s (timeout=%.1fs)", agent_name, timeout)
        output = await asyncio.wait_for(agent.execute(input_data), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("Agent %s timed out after %.1fs", agent_name, timeout)
        return AgentResult(
            error=f"Execution timed out after {timeout}s",
            agent_name=agent_name,
        )
    except Exception as e:
        logger.error("Agent %s raised: %s", agent_name, e)
        return AgentResult(error=str(e), agent_name=agent_name)
    finally:
        try:
            await agent.on_shutdown()
        except Exception:
            pass  # Don't mask execution errors with shutdown errors

    # Validate output
    output_errors = validate_output(entry.contract, output)
    if output_errors:
        logger.warning("Output validation failed for %s: %s", agent_name, output_errors)
        return AgentResult(
            error="Output validation failed",
            agent_name=agent_name,
            data=output,
            validation_errors=output_errors,
        )

    logger.debug("Agent %s completed successfully", agent_name)
    return AgentResult(success=True, data=output, agent_name=agent_name)
