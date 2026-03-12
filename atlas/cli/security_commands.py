"""CLI commands for security policy management."""

from __future__ import annotations

import json
from pathlib import Path

import typer

security_app = typer.Typer(
    name="security",
    help="Manage security — validate policies, check agent permissions.",
    no_args_is_help=True,
)


@security_app.command("validate")
def validate_policy(
    policy_path: str = typer.Argument(help="Path to security policy YAML file"),
) -> None:
    """Validate a security policy YAML file."""
    target = Path(policy_path)
    if not target.exists():
        typer.echo(f"File not found: {target}", err=True)
        raise typer.Exit(1)

    from atlas.security.policy import SecurityPolicy

    try:
        policy = SecurityPolicy.from_yaml(target)
        typer.echo(f"Valid security policy:")
        typer.echo(f"  Container image: {policy.container_image}")
        typer.echo(f"  Container network: {policy.container_network}")
        typer.echo(f"  Secret provider: {policy.secret_provider}")
        typer.echo(f"  Allowed secrets: {sorted(policy.allowed_secrets) or '(none)'}")
        typer.echo(f"  Max memory: {policy.max_memory_mb}MB")
        typer.echo(f"  Max CPU: {policy.max_cpu_seconds}s")
    except Exception as e:
        typer.echo(f"Invalid policy: {e}", err=True)
        raise typer.Exit(1)


@security_app.command("check")
def check_agent(
    agent_path: str = typer.Argument(help="Path to agent directory or agent.yaml"),
    policy_path: str | None = typer.Option(None, "--policy", "-p", help="Security policy YAML"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Show resolved permissions for an agent."""
    target = Path(agent_path)
    if target.is_dir():
        target = target / "agent.yaml"
    if not target.exists():
        typer.echo(f"File not found: {target}", err=True)
        raise typer.Exit(1)

    from atlas.contract.schema import load_contract, ContractError
    from atlas.security.policy import SecurityPolicy

    try:
        contract = load_contract(target)
    except (ContractError, Exception) as e:
        typer.echo(f"Invalid agent contract: {e}", err=True)
        raise typer.Exit(1)

    policy = SecurityPolicy()
    if policy_path:
        policy = SecurityPolicy.from_yaml(policy_path)

    resolved = policy.resolve_permissions(contract.permissions)

    if json_output:
        typer.echo(json.dumps({
            "agent": contract.name,
            "version": contract.version,
            "permissions": resolved.to_dict(),
        }, indent=2))
    else:
        typer.echo(f"Agent: {contract.name} v{contract.version}")
        typer.echo(f"Resolved permissions:")
        typer.echo(f"  Filesystem: {resolved.filesystem}")
        typer.echo(f"  Network: {resolved.network}")
        typer.echo(f"  Spawn: {resolved.spawn}")
        typer.echo(f"  Isolation: {resolved.isolation}")
        typer.echo(f"  Max memory: {resolved.max_memory_mb}MB")
        typer.echo(f"  Max CPU: {resolved.max_cpu_seconds}s")
        typer.echo(f"  Secrets: {resolved.secrets or '(none)'}")
        typer.echo(f"  Container image: {resolved.container_image or '(default)'}")
