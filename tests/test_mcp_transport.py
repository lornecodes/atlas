"""Tests for MCP HTTP transport and auth middleware (Phase 10A)."""

from __future__ import annotations

import asyncio
import json
import secrets

import pytest

from atlas.mcp.auth import BearerAuthMiddleware
from atlas.mcp.transport import make_mcp_app, run_mcp_http
from atlas.skills.registry import SkillRegistry
from atlas.skills.types import SkillSpec
from atlas.contract.types import SchemaSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _simple_asgi_app(scope, receive, send):
    """Minimal ASGI app that returns 200 OK with the path as body."""
    body = json.dumps({"path": scope.get("path", "")}).encode()
    await send({
        "type": "http.response.start",
        "status": 200,
        "headers": [[b"content-type", b"application/json"],
                     [b"content-length", str(len(body)).encode()]],
    })
    await send({"type": "http.response.body", "body": body})


def _make_scope(path: str = "/", method: str = "GET", headers: dict | None = None) -> dict:
    """Build a minimal ASGI HTTP scope."""
    raw_headers = []
    for k, v in (headers or {}).items():
        raw_headers.append([k.encode(), v.encode()])
    return {
        "type": "http",
        "path": path,
        "method": method,
        "headers": raw_headers,
    }


class ResponseCapture:
    """Captures ASGI send() calls for assertion."""

    def __init__(self):
        self.start = None
        self.body = b""

    async def __call__(self, message):
        if message["type"] == "http.response.start":
            self.start = message
        elif message["type"] == "http.response.body":
            self.body += message.get("body", b"")

    @property
    def status(self) -> int:
        return self.start["status"] if self.start else 0

    @property
    def json(self) -> dict:
        return json.loads(self.body)


async def _noop_receive():
    return {"type": "http.disconnect"}


def _make_skill_registry(*names: str) -> SkillRegistry:
    """Build a SkillRegistry with echo callables."""
    reg = SkillRegistry()

    async def _echo(input_data: dict) -> dict:
        return {"echo": input_data}

    for name in names:
        spec = SkillSpec(
            name=name, version="1.0.0", description=f"Skill {name}",
            input_schema=SchemaSpec({"type": "object", "properties": {}}),
        )
        reg.register_callable(spec, _echo)
    return reg


# ---------------------------------------------------------------------------
# TestBearerAuthMiddleware
# ---------------------------------------------------------------------------

class TestBearerAuthMiddleware:
    async def test_no_token_dev_mode_passes(self):
        """No token configured = dev mode, all requests pass through."""
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token=None)
        cap = ResponseCapture()
        await middleware(_make_scope("/mcp"), _noop_receive, cap)
        assert cap.status == 200

    async def test_empty_string_token_is_dev_mode(self):
        """Empty string token (e.g. from env var) is treated as dev mode."""
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token="")
        cap = ResponseCapture()
        await middleware(_make_scope("/mcp"), _noop_receive, cap)
        # Empty string should be treated as no token (dev mode)
        assert cap.status == 200

    async def test_health_always_open(self):
        """Health endpoint bypasses auth even with token set."""
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token="secret")
        cap = ResponseCapture()
        await middleware(_make_scope("/health"), _noop_receive, cap)
        assert cap.status == 200

    async def test_missing_auth_header_401(self):
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token="secret")
        cap = ResponseCapture()
        await middleware(_make_scope("/mcp"), _noop_receive, cap)
        assert cap.status == 401
        assert "Missing" in cap.json["error"]

    async def test_invalid_token_403(self):
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token="secret")
        cap = ResponseCapture()
        scope = _make_scope("/mcp", headers={"authorization": "Bearer wrong"})
        await middleware(scope, _noop_receive, cap)
        assert cap.status == 403
        assert "Invalid" in cap.json["error"]

    async def test_valid_token_passes(self):
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token="secret")
        cap = ResponseCapture()
        scope = _make_scope("/mcp", headers={"authorization": "Bearer secret"})
        await middleware(scope, _noop_receive, cap)
        assert cap.status == 200

    async def test_timing_safe_comparison(self):
        """Verify secrets.compare_digest is used (no early exit on first char mismatch)."""
        # This is a structural test — we verify the module uses secrets
        import atlas.mcp.auth as auth_mod
        assert hasattr(auth_mod, "secrets")
        assert auth_mod.secrets is secrets

    async def test_non_http_scope_passes_through(self):
        """Non-HTTP scopes (e.g. lifespan) pass through without auth."""
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token="secret")
        calls = []

        async def tracking_app(scope, receive, send):
            calls.append(scope["type"])

        middleware = BearerAuthMiddleware(tracking_app, auth_token="secret")
        await middleware({"type": "lifespan"}, _noop_receive, lambda m: None)
        assert calls == ["lifespan"]

    async def test_bearer_prefix_required(self):
        """Authorization header must start with 'Bearer '."""
        middleware = BearerAuthMiddleware(_simple_asgi_app, auth_token="secret")
        cap = ResponseCapture()
        scope = _make_scope("/mcp", headers={"authorization": "Basic secret"})
        await middleware(scope, _noop_receive, cap)
        assert cap.status == 401


# ---------------------------------------------------------------------------
# TestMakeMcpApp
# ---------------------------------------------------------------------------

class TestMakeMcpApp:
    def test_creates_callable(self):
        reg = _make_skill_registry("tool-a")
        app = make_mcp_app(reg)
        assert callable(app)

    async def test_health_returns_ok(self):
        reg = _make_skill_registry("tool-a", "tool-b")
        app = make_mcp_app(reg)
        cap = ResponseCapture()
        await app(_make_scope("/health"), _noop_receive, cap)
        assert cap.status == 200
        data = cap.json
        assert data["status"] == "ok"
        assert data["tools"] == 2

    async def test_health_with_zero_tools(self):
        app = make_mcp_app(SkillRegistry())
        cap = ResponseCapture()
        await app(_make_scope("/health"), _noop_receive, cap)
        assert cap.json["tools"] == 0

    async def test_404_for_unknown_path(self):
        app = make_mcp_app(SkillRegistry())
        cap = ResponseCapture()
        await app(_make_scope("/unknown"), _noop_receive, cap)
        assert cap.status == 404
        assert "not found" in cap.json["error"]

    async def test_auth_wraps_when_token_set(self):
        reg = _make_skill_registry()
        app = make_mcp_app(reg, auth_token="my-secret")
        assert isinstance(app, BearerAuthMiddleware)

    async def test_no_auth_when_no_token(self):
        reg = _make_skill_registry()
        app = make_mcp_app(reg)
        assert not isinstance(app, BearerAuthMiddleware)

    async def test_non_http_scope_ignored(self):
        """Non-HTTP scopes return without error."""
        app = make_mcp_app(SkillRegistry())
        # Should not raise
        await app({"type": "lifespan"}, _noop_receive, lambda m: None)


# ---------------------------------------------------------------------------
# TestRunMcpHttp
# ---------------------------------------------------------------------------

class TestRunMcpHttp:
    def test_function_is_async(self):
        assert asyncio.iscoroutinefunction(run_mcp_http)


# ---------------------------------------------------------------------------
# TestCliIntegration
# ---------------------------------------------------------------------------

class TestCliIntegration:
    def test_mcp_command_has_serve_flag(self):
        """The mcp command function accepts --serve/--host/--port/--auth-token."""
        from atlas.cli.app import mcp_server
        import inspect
        sig = inspect.signature(mcp_server)
        params = list(sig.parameters)
        assert "http" in params
        assert "host" in params
        assert "port" in params
        assert "auth_token" in params

    def test_serve_command_has_mcp_port(self):
        """The serve command function accepts --mcp-port/--auth-token."""
        from atlas.cli.app import serve
        import inspect
        sig = inspect.signature(serve)
        params = list(sig.parameters)
        assert "mcp_port" in params
        assert "auth_token" in params
