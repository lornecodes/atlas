"""Bearer token auth middleware for MCP HTTP servers.

Validates Authorization: Bearer <token> against ATLAS_AUTH_TOKEN env var.
/health is always exempt. Empty token = dev mode (all requests pass through).

Uses raw ASGI middleware (not BaseHTTPMiddleware) because SSE endpoints
write directly to the ASGI send callable and don't return Response objects.
"""

from __future__ import annotations

import json
import os
import secrets


class BearerAuthMiddleware:
    """Raw ASGI middleware — validates Bearer token, SSE-compatible."""

    def __init__(self, app, auth_token: str | None = None):
        self.app = app
        # None = dev mode (no auth). Empty string from env var is treated as unset.
        self._token = auth_token or os.getenv("ATLAS_AUTH_TOKEN") or None

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        # Health always open
        if path == "/health":
            await self.app(scope, receive, send)
            return

        # Dev mode — no token configured, pass through
        if self._token is None:
            await self.app(scope, receive, send)
            return

        # Extract Authorization header
        headers = dict(scope.get("headers", []))
        auth_value = headers.get(b"authorization", b"").decode()

        if not auth_value.startswith("Bearer "):
            await self._send_json(send, 401, {"error": "Missing or invalid Authorization header"})
            return

        token = auth_value.removeprefix("Bearer ").strip()
        if not secrets.compare_digest(token, self._token):
            await self._send_json(send, 403, {"error": "Invalid token"})
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _send_json(send, status: int, body: dict):
        """Send a JSON error response and close the connection."""
        payload = json.dumps(body).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(payload)).encode()],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": payload,
        })
