"""Stdio transport for the Atlas MCP server."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atlas.skills.registry import SkillRegistry


async def run_stdio(skill_registry: "SkillRegistry") -> None:
    """Run the Atlas MCP server over stdio.

    This is the entry point for ``atlas mcp`` — it reads JSON-RPC
    from stdin and writes responses to stdout, following the MCP
    stdio transport specification.
    """
    from mcp.server.stdio import stdio_server

    from atlas.mcp.server import create_mcp_server

    app = create_mcp_server(skill_registry)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )
