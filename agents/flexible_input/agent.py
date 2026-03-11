"""Flexible input agent — demonstrates optional complex defaults."""

from atlas.runtime.base import AgentBase


class FlexibleInputAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        text = input_data["text"]
        options = input_data.get("options", {})
        fmt = options.get("format", "plain")
        max_length = options.get("max_length", 500)

        result = text[:max_length]
        if fmt == "uppercase":
            result = result.upper()

        return {
            "result": result,
            "format_used": fmt,
        }
