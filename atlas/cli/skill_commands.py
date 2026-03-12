"""CLI commands for skill management."""

from __future__ import annotations

import json
from pathlib import Path

import typer

skill_app = typer.Typer(
    name="skill",
    help="Manage skills — list, describe, and inspect available skills.",
    no_args_is_help=True,
)


@skill_app.command("list")
def list_skills(
    skills_path: str = typer.Option("./skills", "--skills-path", help="Path to skills directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """List all available skills."""
    from atlas.skills.registry import SkillRegistry

    registry = SkillRegistry(search_paths=[skills_path])
    count = registry.discover()

    skills = registry.list_all()
    if not skills:
        typer.echo("No skills found.")
        return

    if json_output:
        data = [
            {
                "name": s.spec.name,
                "version": s.spec.version,
                "description": s.spec.description,
            }
            for s in skills
        ]
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"Found {len(skills)} skill(s):\n")
        for s in skills:
            desc = f" — {s.spec.description}" if s.spec.description else ""
            typer.echo(f"  {s.spec.name} v{s.spec.version}{desc}")


@skill_app.command("describe")
def describe_skill(
    name: str = typer.Argument(help="Skill name to describe"),
    skills_path: str = typer.Option("./skills", "--skills-path", help="Path to skills directory"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Show detailed information about a skill."""
    from atlas.skills.registry import SkillRegistry

    registry = SkillRegistry(search_paths=[skills_path])
    registry.discover()

    entry = registry.get(name)
    if not entry:
        typer.echo(f"Skill not found: {name}", err=True)
        raise typer.Exit(1)

    spec = entry.spec

    if json_output:
        typer.echo(json.dumps({
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "has_implementation": entry.callable is not None,
            "input_schema": spec.input_schema.to_json_schema(),
            "output_schema": spec.output_schema.to_json_schema(),
        }, indent=2))
    else:
        typer.echo(f"Skill: {spec.name} v{spec.version}")
        if spec.description:
            typer.echo(f"  Description: {spec.description}")
        typer.echo(f"  Implementation: {'yes' if entry.callable else 'no'}")
        if spec.input_schema.properties:
            typer.echo(f"  Input: {list(spec.input_schema.properties.keys())}")
        if spec.output_schema.properties:
            typer.echo(f"  Output: {list(spec.output_schema.properties.keys())}")
