"""Atlas MCP — Model Context Protocol server and client for Atlas skills."""

from atlas.mcp.client import RemoteServer, RemoteToolProvider
from atlas.mcp.remote_agents import RemoteAgentProvider
from atlas.mcp.server import create_mcp_server
from atlas.mcp.transport import make_mcp_app, run_mcp_http

__all__ = [
    "RemoteAgentProvider",
    "RemoteServer",
    "RemoteToolProvider",
    "create_mcp_server",
    "make_mcp_app",
    "run_mcp_http",
]
