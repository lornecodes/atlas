"""MCP server factory — wraps SkillRegistry as an MCP tool server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from atlas.logging import get_logger
from atlas.skills.types import SkillSpec

if TYPE_CHECKING:
    from atlas.skills.registry import SkillRegistry

logger = get_logger(__name__)


def create_mcp_server(skill_registry: "SkillRegistry") -> Server:
    """Create an MCP Server backed by the SkillRegistry.

    Every registered skill with a callable becomes an MCP tool.
    """
    app = Server("atlas")

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        tools = []
        for rs in skill_registry.list_all():
            if rs.callable is not None:
                tools.append(_spec_to_tool(rs.spec))
        return tools

    @app.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any] | None = None,
    ) -> list[TextContent]:
        entry = skill_registry.get(name)
        if not entry or not entry.callable:
            raise ValueError(f"Unknown tool: {name}")
        logger.debug("Calling tool '%s'", name)
        result = await entry.callable(arguments or {})
        return [TextContent(type="text", text=json.dumps(result))]

    return app


def _spec_to_tool(spec: SkillSpec) -> Tool:
    """Convert a SkillSpec to an MCP Tool definition."""
    return Tool(
        name=spec.name,
        description=spec.description or f"Skill: {spec.name}",
        inputSchema=spec.input_schema.to_json_schema(),
    )
