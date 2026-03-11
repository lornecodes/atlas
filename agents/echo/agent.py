"""Echo agent — returns input unchanged."""

from atlas.runtime.base import AgentBase


class EchoAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        return {"message": input_data["message"]}
