"""Tests for MCP client — remote tool federation (Phase 10B)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from atlas.contract.types import SchemaSpec
from atlas.mcp.client import RemoteServer, RemoteToolProvider, parse_remote_spec
from atlas.skills.registry import SkillRegistry
from atlas.skills.types import SkillSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_tool(name: str, description: str = "", input_schema: dict | None = None):
    """Create a mock MCP Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description or f"Tool {name}"
    tool.inputSchema = input_schema or {"type": "object", "properties": {}}
    return tool


def _make_mock_session(tools: list | None = None):
    """Create a mock ClientSession with list_tools and call_tool."""
    session = AsyncMock()

    # list_tools returns a result with .tools
    tools_result = MagicMock()
    tools_result.tools = tools or []
    session.list_tools = AsyncMock(return_value=tools_result)
    session.initialize = AsyncMock()

    return session


def _make_call_result(data: dict, is_error: bool = False):
    """Create a mock CallToolResult."""
    result = MagicMock()
    result.isError = is_error
    content = MagicMock()
    content.text = json.dumps(data)
    result.content = [content]
    return result


# ---------------------------------------------------------------------------
# TestRemoteServer
# ---------------------------------------------------------------------------

class TestRemoteServer:
    def test_defaults(self):
        server = RemoteServer(name="lab", url="http://localhost:8400/mcp")
        assert server.name == "lab"
        assert server.url == "http://localhost:8400/mcp"
        assert server.transport == "streamable"
        assert server.auth_token is None
        assert server.timeout == 30.0

    def test_from_dict(self):
        server = RemoteServer.from_dict({
            "name": "prod",
            "url": "http://host:8400/mcp",
            "transport": "sse",
            "auth_token": "secret",
            "timeout": 60.0,
        })
        assert server.name == "prod"
        assert server.transport == "sse"
        assert server.auth_token == "secret"
        assert server.timeout == 60.0

    def test_from_dict_defaults(self):
        server = RemoteServer.from_dict({"name": "x", "url": "http://x"})
        assert server.transport == "streamable"
        assert server.auth_token is None


# ---------------------------------------------------------------------------
# TestParseRemoteSpec
# ---------------------------------------------------------------------------

class TestParseRemoteSpec:
    def test_name_url(self):
        server = parse_remote_spec("lab=http://localhost:8401/mcp")
        assert server.name == "lab"
        assert server.url == "http://localhost:8401/mcp"
        assert server.auth_token is None

    def test_name_url_token(self):
        server = parse_remote_spec("lab=http://localhost:8401/mcp@my-secret")
        assert server.name == "lab"
        assert server.url == "http://localhost:8401/mcp"
        assert server.auth_token == "my-secret"

    def test_url_with_port_and_token(self):
        server = parse_remote_spec("prod=http://host:8400/mcp@tok123")
        assert server.url == "http://host:8400/mcp"
        assert server.auth_token == "tok123"

    def test_invalid_no_equals(self):
        with pytest.raises(ValueError, match="expected"):
            parse_remote_spec("justanurl")

    def test_invalid_empty_name(self):
        with pytest.raises(ValueError, match="empty name"):
            parse_remote_spec("=http://host/mcp")

    def test_invalid_empty_url(self):
        with pytest.raises(ValueError, match="empty URL"):
            parse_remote_spec("lab=")


# ---------------------------------------------------------------------------
# TestRemoteToolProvider
# ---------------------------------------------------------------------------

class TestRemoteToolProvider:
    def test_starts_empty(self):
        provider = RemoteToolProvider()
        assert provider.connected_servers == []

    async def test_connect_registers_namespaced_tools(self):
        """Tools from remote server are registered with namespace prefix."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        tools = [
            _make_mock_tool("echo", "Echo tool"),
            _make_mock_tool("reverse", "Reverse tool"),
        ]
        session = _make_mock_session(tools)

        # Monkey-patch _open_session to return our mock
        provider._open_session = AsyncMock(return_value=session)

        server = RemoteServer(name="lab", url="http://localhost:8400/mcp")
        count = await provider.connect(server, registry)

        assert count == 2
        assert "lab.echo" in registry
        assert "lab.reverse" in registry
        assert "lab" in provider.connected_servers

    async def test_connect_tool_has_correct_spec(self):
        """Registered skill spec has correct name and description."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        tools = [_make_mock_tool("greet", "Say hello", {"type": "object", "properties": {"name": {"type": "string"}}})]
        session = _make_mock_session(tools)
        provider._open_session = AsyncMock(return_value=session)

        server = RemoteServer(name="remote", url="http://x/mcp")
        await provider.connect(server, registry)

        entry = registry.get("remote.greet")
        assert entry is not None
        assert entry.spec.description == "Say hello"
        assert entry.spec.name == "remote.greet"

    async def test_remote_callable_proxies_call(self):
        """Calling a registered remote skill proxies to session.call_tool."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        tools = [_make_mock_tool("echo")]
        session = _make_mock_session(tools)
        session.call_tool = AsyncMock(
            return_value=_make_call_result({"echoed": "hello"})
        )
        provider._open_session = AsyncMock(return_value=session)

        server = RemoteServer(name="r", url="http://x/mcp")
        await provider.connect(server, registry)

        entry = registry.get("r.echo")
        result = await entry.callable({"text": "hello"})
        assert result == {"echoed": "hello"}
        session.call_tool.assert_called_once_with("echo", {"text": "hello"})

    async def test_remote_callable_handles_error(self):
        """Remote tool errors are returned as error dicts."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        tools = [_make_mock_tool("fail")]
        session = _make_mock_session(tools)
        session.call_tool = AsyncMock(
            return_value=_make_call_result({"error": "boom"}, is_error=True)
        )
        provider._open_session = AsyncMock(return_value=session)

        server = RemoteServer(name="r", url="http://x/mcp")
        await provider.connect(server, registry)

        result = await registry.get("r.fail").callable({})
        assert "error" in result

    async def test_remote_callable_handles_transport_exception(self):
        """Transport exceptions (network drop) return error dicts, not raw exceptions."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        tools = [_make_mock_tool("flaky")]
        session = _make_mock_session(tools)
        session.call_tool = AsyncMock(side_effect=ConnectionError("server disconnected"))
        provider._open_session = AsyncMock(return_value=session)

        server = RemoteServer(name="r", url="http://x/mcp")
        await provider.connect(server, registry)

        result = await registry.get("r.flaky").callable({})
        assert "error" in result
        assert "server disconnected" in result["error"]

    async def test_disconnect_removes_skills(self):
        """Disconnecting unregisters the remote skills."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        tools = [_make_mock_tool("tool-a")]
        session = _make_mock_session(tools)
        provider._open_session = AsyncMock(return_value=session)

        server = RemoteServer(name="lab", url="http://x/mcp")
        await provider.connect(server, registry)
        assert "lab.tool-a" in registry

        await provider.disconnect("lab", registry)
        assert "lab.tool-a" not in registry
        assert "lab" not in provider.connected_servers

    async def test_disconnect_all(self):
        """disconnect_all cleans up all connections."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        for name in ["a", "b"]:
            session = _make_mock_session([_make_mock_tool(f"tool-{name}")])
            provider._open_session = AsyncMock(return_value=session)
            await provider.connect(
                RemoteServer(name=name, url=f"http://x-{name}/mcp"),
                registry,
            )

        assert len(provider.connected_servers) == 2
        await provider.disconnect_all(registry)
        assert provider.connected_servers == []
        assert len(registry) == 0

    async def test_duplicate_server_name_errors(self):
        """Connecting with a duplicate name raises ValueError."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        session = _make_mock_session([])
        provider._open_session = AsyncMock(return_value=session)

        server = RemoteServer(name="lab", url="http://x/mcp")
        await provider.connect(server, registry)

        with pytest.raises(ValueError, match="Already connected"):
            await provider.connect(server, registry)

    async def test_connect_multiple_servers(self):
        """Can connect to multiple servers with different namespaces."""
        provider = RemoteToolProvider()
        registry = SkillRegistry()

        for name, tool_name in [("lab", "echo"), ("prod", "classify")]:
            session = _make_mock_session([_make_mock_tool(tool_name)])
            provider._open_session = AsyncMock(return_value=session)
            await provider.connect(
                RemoteServer(name=name, url=f"http://{name}/mcp"),
                registry,
            )

        assert "lab.echo" in registry
        assert "prod.classify" in registry
        assert set(provider.connected_servers) == {"lab", "prod"}


# ---------------------------------------------------------------------------
# TestMakeRemoteCallable
# ---------------------------------------------------------------------------

class TestMakeRemoteCallable:
    async def test_returns_parsed_json(self):
        """Callable parses JSON from text content."""
        session = AsyncMock()
        session.call_tool = AsyncMock(
            return_value=_make_call_result({"key": "value"})
        )
        provider = RemoteToolProvider()
        fn = provider._make_remote_callable(session, "test-tool")
        result = await fn({"input": "data"})
        assert result == {"key": "value"}

    async def test_non_json_text_returned_as_result(self):
        """Non-JSON text content is wrapped in a result dict."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.isError = False
        content = MagicMock()
        content.text = "plain text output"
        result_mock.content = [content]
        session.call_tool = AsyncMock(return_value=result_mock)

        provider = RemoteToolProvider()
        fn = provider._make_remote_callable(session, "tool")
        result = await fn({})
        assert result == {"result": "plain text output"}

    async def test_empty_content_returns_none_result(self):
        """Empty content list returns {result: None}."""
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.isError = False
        result_mock.content = []
        session.call_tool = AsyncMock(return_value=result_mock)

        provider = RemoteToolProvider()
        fn = provider._make_remote_callable(session, "tool")
        result = await fn({})
        assert result == {"result": None}


# ---------------------------------------------------------------------------
# TestCliIntegration
# ---------------------------------------------------------------------------

class TestCliIntegration:
    def test_serve_has_remote_option(self):
        from atlas.cli.app import serve
        import inspect
        params = list(inspect.signature(serve).parameters)
        assert "remote" in params

    def test_mcp_has_remote_option(self):
        from atlas.cli.app import mcp_server
        import inspect
        params = list(inspect.signature(mcp_server).parameters)
        assert "remote" in params
