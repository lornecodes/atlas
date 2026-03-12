"""MCP client — connect to remote MCP servers and register their tools as local skills.

This is the federation primitive: one Atlas instance discovers tools on another
and registers them as local skills. Agents call remote tools transparently via
the normal skill resolution path.

Follows the PlatformToolProvider closure pattern:
  discover remote tools → create closures → register_callable()
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from atlas.logging import get_logger
from atlas.skills.types import SkillCallable, SkillSpec

if TYPE_CHECKING:
    from mcp.client.session import ClientSession

    from atlas.skills.registry import SkillRegistry

logger = get_logger(__name__)


@dataclass
class RemoteServer:
    """Configuration for a remote MCP server connection."""

    name: str           # Namespace prefix (e.g., "lab" → "lab.tool-name")
    url: str            # MCP endpoint (e.g., "http://host:8400/mcp")
    transport: str = "streamable"  # "streamable" or "sse"
    auth_token: str | None = None
    timeout: float = 30.0

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RemoteServer:
        """Parse a dict into a RemoteServer."""
        return RemoteServer(
            name=d["name"],
            url=d["url"],
            transport=d.get("transport", "streamable"),
            auth_token=d.get("auth_token"),
            timeout=d.get("timeout", 30.0),
        )


@dataclass
class _Connection:
    """Internal: tracks a live connection to a remote server."""

    server: RemoteServer
    session: ClientSession
    stack: AsyncExitStack
    tool_names: list[str] = field(default_factory=list)


class RemoteToolProvider:
    """Connects to remote MCP servers and registers their tools as local skills.

    Each remote tool becomes a closure that proxies calls over the MCP protocol.
    Tools are namespaced with the server name to avoid collisions:
    ``{server.name}.{tool.name}``
    """

    def __init__(self) -> None:
        self._connections: dict[str, _Connection] = {}

    async def connect(
        self,
        server: RemoteServer,
        skill_registry: SkillRegistry,
    ) -> int:
        """Connect to a remote MCP server, discover tools, register as skills.

        Returns the number of tools registered.
        """
        if server.name in self._connections:
            raise ValueError(f"Already connected to '{server.name}'")

        stack = AsyncExitStack()
        try:
            session = await self._open_session(server, stack)

            # Initialize the MCP session
            await session.initialize()

            # Discover remote tools
            result = await session.list_tools()
            tools = result.tools

            conn = _Connection(server=server, session=session, stack=stack)

            for tool in tools:
                local_name = f"{server.name}.{tool.name}"
                spec = SkillSpec(
                    name=local_name,
                    version="1.0.0",
                    description=tool.description or f"Remote tool: {tool.name}",
                    input_schema=_schema_from_dict(tool.inputSchema),
                )
                fn = self._make_remote_callable(session, tool.name)
                skill_registry.register_callable(spec, fn)
                conn.tool_names.append(local_name)

            self._connections[server.name] = conn
            logger.info(
                "Connected to '%s' at %s: %d tools",
                server.name, server.url, len(tools),
            )
            return len(tools)

        except Exception:
            await stack.aclose()
            raise

    async def disconnect(self, name: str, skill_registry: SkillRegistry | None = None) -> None:
        """Disconnect from a remote server and optionally unregister its skills."""
        conn = self._connections.pop(name, None)
        if not conn:
            return
        if skill_registry:
            for tool_name in conn.tool_names:
                skill_registry._skills.pop(tool_name, None)
        await conn.stack.aclose()
        logger.info("Disconnected from '%s'", name)

    async def disconnect_all(self, skill_registry: SkillRegistry | None = None) -> None:
        """Clean shutdown of all connections."""
        names = list(self._connections.keys())
        for name in names:
            await self.disconnect(name, skill_registry)

    @property
    def connected_servers(self) -> list[str]:
        """Names of currently connected servers."""
        return list(self._connections.keys())

    def _make_remote_callable(self, session: "ClientSession", tool_name: str) -> SkillCallable:
        """Factory: returns a closure that calls a remote tool via MCP."""

        async def _fn(input_data: dict[str, Any]) -> dict[str, Any]:
            try:
                result = await session.call_tool(tool_name, input_data)
            except Exception as e:
                return {"error": f"Remote call to '{tool_name}' failed: {e}"}
            if result.isError:
                # Extract error text from content
                error_text = ""
                for content in result.content:
                    if hasattr(content, "text"):
                        error_text += content.text
                return {"error": error_text or "Remote tool error"}
            # Parse the first text content block as JSON
            for content in result.content:
                if hasattr(content, "text"):
                    try:
                        return json.loads(content.text)
                    except json.JSONDecodeError:
                        return {"result": content.text}
            return {"result": None}

        return _fn

    async def _open_session(self, server: RemoteServer, stack: AsyncExitStack) -> "ClientSession":
        """Open an MCP client session using the configured transport."""
        from mcp.client.session import ClientSession

        if server.transport == "sse":
            from mcp.client.sse import sse_client

            headers = {}
            if server.auth_token:
                headers["Authorization"] = f"Bearer {server.auth_token}"

            read_stream, write_stream = await stack.enter_async_context(
                sse_client(server.url, headers=headers, timeout=server.timeout)
            )
        else:
            from mcp.client.streamable_http import streamablehttp_client

            headers = {}
            if server.auth_token:
                headers["Authorization"] = f"Bearer {server.auth_token}"

            read_stream, write_stream, _ = await stack.enter_async_context(
                streamablehttp_client(
                    server.url, headers=headers, timeout=server.timeout,
                )
            )

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        return session


def _schema_from_dict(schema: dict[str, Any] | None) -> Any:
    """Convert a JSON Schema dict to a SchemaSpec."""
    from atlas.contract.types import SchemaSpec
    return SchemaSpec(schema or {"type": "object", "properties": {}})


def parse_remote_spec(spec: str) -> RemoteServer:
    """Parse a CLI remote spec string into a RemoteServer.

    Format: ``name=url`` or ``name=url@token``

    Examples:
        ``lab=http://localhost:8401/mcp``
        ``lab=http://localhost:8401/mcp@my-secret-token``
    """
    if "=" not in spec:
        raise ValueError(f"Invalid remote spec '{spec}': expected 'name=url' or 'name=url@token'")

    name, rest = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid remote spec '{spec}': empty name")

    auth_token = None
    if "@" in rest:
        # Split on last @ to handle URLs that might contain @
        url, auth_token = rest.rsplit("@", 1)
    else:
        url = rest

    url = url.strip()
    if not url:
        raise ValueError(f"Invalid remote spec '{spec}': empty URL")

    return RemoteServer(name=name, url=url, auth_token=auth_token or None)
