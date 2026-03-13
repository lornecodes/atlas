"""MCP HTTP transport — Streamable HTTP + legacy SSE over uvicorn.

Builds an ASGI app that serves the Atlas MCP server over:
  POST /mcp          — Streamable HTTP (session-persistent, modern clients)
  GET  /sse          — Legacy SSE (backwards compat)
  POST /messages/    — Legacy SSE message endpoint
  GET  /health       — Always open, no auth required
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

from atlas.logging import get_logger
from atlas.mcp.auth import BearerAuthMiddleware
from atlas.mcp.server import create_mcp_server

if TYPE_CHECKING:
    from atlas.skills.registry import SkillRegistry

logger = get_logger(__name__)


def make_mcp_app(
    skill_registry: "SkillRegistry",
    *,
    auth_token: str | None = None,
) -> Callable:
    """Build an ASGI app serving Atlas MCP over Streamable HTTP + SSE.

    Returns a raw ASGI callable suitable for uvicorn.
    """
    from mcp.server.sse import SseServerTransport
    from mcp.server.streamable_http import StreamableHTTPServerTransport

    mcp_server = create_mcp_server(skill_registry)
    sse_transport = SseServerTransport("/messages/")

    # Session registry — maps session_id → (transport, created_at)
    _sessions: dict[str, tuple[StreamableHTTPServerTransport, float]] = {}
    _SESSION_TTL = 300.0  # 5 minutes

    def _reap_stale_sessions():
        now = time.monotonic()
        stale = [sid for sid, (_, created) in _sessions.items()
                 if now - created > _SESSION_TTL]
        for sid in stale:
            _sessions.pop(sid, None)
            logger.debug("Reaped stale MCP session: %s", sid)

    # ── Streamable HTTP handler ──────────────────────────────────────
    async def handle_streamable(scope, receive, send):
        """Streamable HTTP transport with session persistence.

        On initialize (no Mcp-Session-Id header), creates a new session
        and spawns a background task for the MCP protocol loop.
        Subsequent requests with matching Mcp-Session-Id reuse the transport.
        """
        headers = dict(scope.get("headers", []))
        session_id = headers.get(b"mcp-session-id", b"").decode() or None

        _reap_stale_sessions()

        if session_id and session_id in _sessions:
            transport, _ = _sessions[session_id]
            await transport.handle_request(scope, receive, send)
            return

        # New session
        session_id = str(uuid.uuid4())
        transport = StreamableHTTPServerTransport(
            mcp_session_id=session_id,
            is_json_response_enabled=True,
        )
        _sessions[session_id] = (transport, time.monotonic())

        async def _run_session():
            logger.debug("Streamable HTTP session started: %s", session_id)
            try:
                async with transport.connect() as (read_stream, write_stream):
                    await mcp_server.run(
                        read_stream, write_stream,
                        mcp_server.create_initialization_options(),
                    )
            finally:
                _sessions.pop(session_id, None)
                logger.debug("Streamable HTTP session closed: %s", session_id)

        asyncio.create_task(_run_session())
        await asyncio.sleep(0.05)  # Let connect() set up internal state
        await transport.handle_request(scope, receive, send)

    # ── Legacy SSE handlers ──────────────────────────────────────────
    async def handle_sse(scope, receive, send):
        """Legacy SSE connection — raw ASGI."""
        from starlette.requests import Request

        request = Request(scope, receive, send)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream, write_stream,
                mcp_server.create_initialization_options(),
            )

    # ── Health ───────────────────────────────────────────────────────
    async def handle_health(scope, receive, send):
        tool_count = sum(1 for rs in skill_registry.list_all() if rs.callable is not None)
        body = json.dumps({
            "status": "ok",
            "tools": tool_count,
        }).encode()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({"type": "http.response.body", "body": body})

    # ── ASGI dispatcher ─────────────────────────────────────────────
    async def asgi_app(scope, receive, send):
        if scope["type"] != "http":
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET")

        if path == "/mcp" and method in ("POST", "GET", "DELETE"):
            await handle_streamable(scope, receive, send)
        elif path == "/sse" and method == "GET":
            await handle_sse(scope, receive, send)
        elif path.startswith("/messages"):
            await sse_transport.handle_post_message(scope, receive, send)
        elif path == "/health":
            await handle_health(scope, receive, send)
        else:
            body = b'{"error": "not found"}'
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"content-length", str(len(body)).encode()],
                ],
            })
            await send({"type": "http.response.body", "body": body})

    if auth_token:
        return BearerAuthMiddleware(asgi_app, auth_token=auth_token)
    return asgi_app


async def run_mcp_http(
    skill_registry: "SkillRegistry",
    *,
    host: str = "127.0.0.1",
    port: int = 8400,
    auth_token: str | None = None,
) -> None:
    """Run the MCP HTTP server via uvicorn."""
    import uvicorn

    app = make_mcp_app(skill_registry, auth_token=auth_token)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
