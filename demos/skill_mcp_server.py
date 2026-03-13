#!/usr/bin/env python
"""Demo: Skills exposed as MCP tools over HTTP.

Registers custom skills, creates an MCP server, and starts an HTTP endpoint.
Shows the skill → MCP → HTTP pipeline.

Usage:
    python demos/skill_mcp_server.py

Then test with:
    curl http://localhost:8400/health
    # MCP clients can connect to http://localhost:8400/mcp
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas.mcp.server import create_mcp_server
from atlas.mcp.transport import make_mcp_app
from atlas.skills.registry import SkillRegistry
from atlas.skills.types import SkillSpec


# --- Define custom skills ---


async def word_count(input_data: dict[str, Any]) -> dict[str, Any]:
    """Count words in text."""
    text = input_data.get("text", "")
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "unique_words": len(set(w.lower() for w in words)),
    }


async def reverse_text(input_data: dict[str, Any]) -> dict[str, Any]:
    """Reverse the input text."""
    text = input_data.get("text", "")
    return {"reversed": text[::-1]}


async def calculate(input_data: dict[str, Any]) -> dict[str, Any]:
    """Simple calculator: add, subtract, multiply, divide."""
    a = input_data.get("a", 0)
    b = input_data.get("b", 0)
    op = input_data.get("op", "add")
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else None,
    }
    result = ops.get(op)
    return {"result": result, "expression": f"{a} {op} {b}"}


async def main():
    # --- Register skills ---
    registry = SkillRegistry()

    registry.register_callable(
        SkillSpec(
            name="word-count",
            version="1.0.0",
            description="Count words, characters, and unique words in text",
        ),
        word_count,
    )
    registry.register_callable(
        SkillSpec(
            name="reverse-text",
            version="1.0.0",
            description="Reverse input text",
        ),
        reverse_text,
    )
    registry.register_callable(
        SkillSpec(
            name="calculate",
            version="1.0.0",
            description="Simple calculator: add, subtract, multiply, divide",
        ),
        calculate,
    )

    print(f"Registered {len(registry)} skills:")
    for skill in registry.list_all():
        callable_status = "callable" if skill.callable else "no callable"
        print(f"  - {skill.spec.name} v{skill.spec.version} ({callable_status})")

    # --- Create MCP server and inspect tools ---
    mcp_server = create_mcp_server(registry)
    print(f"\nMCP server created: {mcp_server.name}")

    # --- Build ASGI app ---
    app = make_mcp_app(registry, auth_token=None)  # No auth for demo
    print("\nASGI app ready. Endpoints:")
    print("  POST /mcp          — Streamable HTTP (MCP clients)")
    print("  GET  /sse          — Legacy SSE")
    print("  POST /messages/    — SSE message endpoint")
    print("  GET  /health       — Health check")

    # --- Quick self-test ---
    print("\n--- Self-test ---")

    # Test word-count skill directly
    result = await word_count({"text": "The quick brown fox jumps over the lazy dog"})
    print(f"  word-count: {result}")

    # Test calculate
    result = await calculate({"a": 42, "b": 8, "op": "multiply"})
    print(f"  calculate:  {result}")

    # Test reverse
    result = await reverse_text({"text": "Atlas"})
    print(f"  reverse:    {result}")

    print("\n--- Starting HTTP server ---")
    print("Listening on http://127.0.0.1:8400")
    print("Press Ctrl+C to stop\n")

    # --- Start HTTP server ---
    from atlas.mcp.transport import run_mcp_http
    await run_mcp_http(registry, host="127.0.0.1", port=8400)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
