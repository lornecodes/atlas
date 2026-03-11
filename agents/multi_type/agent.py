"""Multi-type agent — handles polymorphic input."""

from atlas.runtime.base import AgentBase


class MultiTypeAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        data = input_data["data"]
        mode = input_data["mode"]

        if isinstance(data, str):
            processed = data
        elif isinstance(data, dict):
            processed = data.get("value", str(data))
        else:
            processed = str(data)

        return {
            "processed": processed,
            "mode_used": mode,
        }
