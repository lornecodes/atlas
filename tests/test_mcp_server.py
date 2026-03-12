"""Tests for the Atlas MCP server (Phase 9C)."""

from __future__ import annotations

import json

import pytest
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ListToolsRequest,
)

from atlas.contract.types import SchemaSpec
from atlas.mcp.server import _spec_to_tool, create_mcp_server
from atlas.skills.registry import SkillRegistry
from atlas.skills.types import SkillSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(name: str, description: str = "", version: str = "1.0.0") -> SkillSpec:
    return SkillSpec(
        name=name,
        version=version,
        description=description,
        input_schema=SchemaSpec({
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }),
    )


async def _echo(input_data: dict) -> dict:
    return {"echo": input_data.get("text", "")}


def _registry_with(*skills: SkillSpec) -> SkillRegistry:
    """Build a SkillRegistry with programmatic callables."""
    reg = SkillRegistry()
    for spec in skills:
        reg.register_callable(spec, _echo)
    return reg


async def _list_tools(server):
    """Invoke the list_tools handler on an MCP Server."""
    req = ListToolsRequest(method="tools/list")
    result = await server.request_handlers[ListToolsRequest](req)
    return result.root.tools


async def _call_tool(server, name: str, arguments: dict | None = None):
    """Invoke the call_tool handler on an MCP Server.

    Returns the full CallToolResult (has .content and .isError).
    """
    req = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=name, arguments=arguments or {}),
    )
    result = await server.request_handlers[CallToolRequest](req)
    return result.root


# ---------------------------------------------------------------------------
# TestCreateMcpServer
# ---------------------------------------------------------------------------

class TestCreateMcpServer:
    def test_creates_server_instance(self):
        reg = SkillRegistry()
        server = create_mcp_server(reg)
        assert server.name == "atlas"

    def test_registers_list_and_call_handlers(self):
        reg = SkillRegistry()
        server = create_mcp_server(reg)
        assert ListToolsRequest in server.request_handlers
        assert CallToolRequest in server.request_handlers


# ---------------------------------------------------------------------------
# TestListTools
# ---------------------------------------------------------------------------

class TestListTools:
    async def test_empty_registry_returns_no_tools(self):
        server = create_mcp_server(SkillRegistry())
        tools = await _list_tools(server)
        assert tools == []

    async def test_lists_all_registered_skills(self):
        spec_a = _make_spec("alpha", "First skill")
        spec_b = _make_spec("beta", "Second skill")
        server = create_mcp_server(_registry_with(spec_a, spec_b))
        tools = await _list_tools(server)
        names = {t.name for t in tools}
        assert names == {"alpha", "beta"}

    async def test_tool_has_name_and_description(self):
        spec = _make_spec("foo", "Does foo things")
        server = create_mcp_server(_registry_with(spec))
        tools = await _list_tools(server)
        assert tools[0].name == "foo"
        assert tools[0].description == "Does foo things"

    async def test_tool_has_input_schema(self):
        spec = _make_spec("bar")
        server = create_mcp_server(_registry_with(spec))
        tools = await _list_tools(server)
        schema = tools[0].inputSchema
        assert schema["type"] == "object"
        assert "text" in schema["properties"]

    async def test_skills_without_callable_excluded(self):
        reg = SkillRegistry()
        # Register with callable
        reg.register_callable(_make_spec("has-fn"), _echo)
        # Register without callable (simulate spec-only)
        from atlas.skills.registry import RegisteredSkill
        reg._skills["no-fn"] = RegisteredSkill(spec=_make_spec("no-fn"))

        server = create_mcp_server(reg)
        tools = await _list_tools(server)
        names = {t.name for t in tools}
        assert "has-fn" in names
        assert "no-fn" not in names

    async def test_platform_tools_included(self):
        """Platform tools (atlas.*) appear in the MCP tool list."""
        spec = _make_spec("atlas.registry.list", "List agents")
        server = create_mcp_server(_registry_with(spec))
        tools = await _list_tools(server)
        assert tools[0].name == "atlas.registry.list"


# ---------------------------------------------------------------------------
# TestCallTool
# ---------------------------------------------------------------------------

class TestCallTool:
    async def test_call_existing_skill(self):
        spec = _make_spec("echo-skill", "Echoes text")
        server = create_mcp_server(_registry_with(spec))
        result = await _call_tool(server, "echo-skill", {"text": "hello"})
        data = json.loads(result.content[0].text)
        assert data == {"echo": "hello"}

    async def test_call_returns_text_content(self):
        spec = _make_spec("tc")
        server = create_mcp_server(_registry_with(spec))
        result = await _call_tool(server, "tc", {"text": "x"})
        assert len(result.content) == 1
        assert result.content[0].type == "text"

    async def test_call_unknown_tool_returns_error(self):
        """MCP SDK catches ValueError and sets isError=True."""
        server = create_mcp_server(SkillRegistry())
        result = await _call_tool(server, "nonexistent", {})
        assert result.isError is True

    async def test_call_with_optional_schema(self):
        """Skills with no required fields accept empty arguments."""
        spec = SkillSpec(
            name="optional-args",
            version="1.0.0",
            description="No required fields",
            input_schema=SchemaSpec({"type": "object", "properties": {"text": {"type": "string"}}}),
        )
        reg = SkillRegistry()
        reg.register_callable(spec, _echo)
        server = create_mcp_server(reg)
        result = await _call_tool(server, "optional-args", {})
        data = json.loads(result.content[0].text)
        assert data == {"echo": ""}

    async def test_call_platform_tool(self):
        spec = _make_spec("atlas.monitor.health", "Health check")
        server = create_mcp_server(_registry_with(spec))
        result = await _call_tool(server, "atlas.monitor.health", {"text": "ok"})
        data = json.loads(result.content[0].text)
        assert data == {"echo": "ok"}


# ---------------------------------------------------------------------------
# TestSpecToTool
# ---------------------------------------------------------------------------

class TestSpecToTool:
    def test_converts_name(self):
        spec = _make_spec("my-tool")
        tool = _spec_to_tool(spec)
        assert tool.name == "my-tool"

    def test_converts_description(self):
        spec = _make_spec("d", "A description")
        tool = _spec_to_tool(spec)
        assert tool.description == "A description"

    def test_description_fallback(self):
        spec = _make_spec("no-desc", "")
        tool = _spec_to_tool(spec)
        assert tool.description == "Skill: no-desc"

    def test_converts_input_schema(self):
        spec = _make_spec("s")
        tool = _spec_to_tool(spec)
        assert tool.inputSchema["type"] == "object"
        assert "text" in tool.inputSchema["properties"]


# ---------------------------------------------------------------------------
# TestStdioEntry
# ---------------------------------------------------------------------------

class TestStdioEntry:
    def test_run_stdio_is_async(self):
        import asyncio
        from atlas.mcp.stdio import run_stdio
        assert asyncio.iscoroutinefunction(run_stdio)


# ---------------------------------------------------------------------------
# TestMcpCliCommand
# ---------------------------------------------------------------------------

class TestMcpCliCommand:
    def test_mcp_command_registered(self):
        """The 'mcp' command is registered on the CLI app."""
        from atlas.cli.app import app
        command_names = [cmd.name for cmd in app.registered_commands]
        assert "mcp" in command_names
