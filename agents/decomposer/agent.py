"""Decomposer agent — spawns echo for each message in the input list."""

from __future__ import annotations

from typing import Any

from atlas.runtime.base import AgentBase


class DecomposerAgent(AgentBase):
    """Decomposes a list of messages by spawning echo for each one."""

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        messages = input_data["messages"]
        results = []

        for msg in messages:
            result = await self.context.spawn(
                "echo",
                {"message": msg},
            )
            results.append({
                "success": result.success,
                "data": result.data,
                "error": result.error,
            })

        return {
            "results": results,
            "count": len(results),
        }
