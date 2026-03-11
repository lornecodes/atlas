"""Output formatting helpers for the Atlas CLI."""

from __future__ import annotations

import json
from typing import Any

from atlas.contract.types import AgentContract


def format_agent_list(agents: list[dict[str, Any]]) -> str:
    """Format a list of agents for human-readable output."""
    if not agents:
        return "No agents found."

    lines = []
    for agent in agents:
        caps = ", ".join(agent.get("capabilities", []))
        line = f"  {agent['name']}  v{agent['version']}  [{caps}]"
        lines.append(line)

    header = f"Found {len(agents)} agent(s):\n"
    return header + "\n".join(lines)


def format_contract(contract: AgentContract) -> str:
    """Format agent contract details for inspection."""
    lines = [
        f"Agent: {contract.name}",
        f"Version: {contract.version}",
        f"Description: {contract.description}",
        "",
        "Input Schema:",
        json.dumps(contract.input_schema.to_json_schema(), indent=2),
        "",
        "Output Schema:",
        json.dumps(contract.output_schema.to_json_schema(), indent=2),
    ]

    if contract.capabilities:
        lines.extend(["", f"Capabilities: {', '.join(contract.capabilities)}"])

    if contract.execution_timeout != 60.0:
        lines.extend(["", f"Execution Timeout: {contract.execution_timeout}s"])

    return "\n".join(lines)


def format_result(result: dict[str, Any], *, json_output: bool = False) -> str:
    """Format an agent execution result."""
    if json_output:
        return json.dumps(result, indent=2)
    lines = ["Result:"]
    for key, value in result.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def format_validation_errors(errors: list[str]) -> str:
    """Format validation errors."""
    if not errors:
        return "Valid."
    lines = [f"Found {len(errors)} error(s):"]
    for err in errors:
        lines.append(f"  - {err}")
    return "\n".join(lines)


def format_job(job: dict[str, Any], *, json_output: bool = False) -> str:
    """Format a single job for display."""
    if json_output:
        return json.dumps(job, indent=2)

    lines = [
        f"Job: {job['id']}",
        f"  Agent: {job['agent_name']}",
        f"  Status: {job['status']}",
    ]

    if job.get("error"):
        lines.append(f"  Error: {job['error']}")

    if job.get("output_data"):
        lines.append(f"  Output: {json.dumps(job['output_data'])}")

    # Timing
    timing_parts = []
    if job.get("warmup_ms"):
        timing_parts.append(f"warmup={job['warmup_ms']:.1f}ms")
    if job.get("execution_ms"):
        timing_parts.append(f"exec={job['execution_ms']:.1f}ms")
    if timing_parts:
        lines.append(f"  Timing: {', '.join(timing_parts)}")

    if job.get("priority"):
        lines.append(f"  Priority: {job['priority']}")

    return "\n".join(lines)


def format_job_list(jobs: list[dict[str, Any]], *, json_output: bool = False) -> str:
    """Format a list of jobs for display."""
    if json_output:
        return json.dumps(jobs, indent=2)

    if not jobs:
        return "No jobs found."

    lines = [f"Found {len(jobs)} job(s):\n"]
    for job in jobs:
        status = job["status"]
        agent = job["agent_name"]
        total_ms = job.get("warmup_ms", 0) + job.get("execution_ms", 0)
        timing = f"  ({total_ms:.0f}ms)" if total_ms else ""
        error = f"  [{job['error'][:50]}...]" if job.get("error") else ""
        lines.append(f"  {job['id']}  {agent}  {status}{timing}{error}")

    return "\n".join(lines)
