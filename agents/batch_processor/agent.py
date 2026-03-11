"""Batch processor agent — handles arrays of items."""

from atlas.runtime.base import AgentBase


class BatchProcessorAgent(AgentBase):
    async def execute(self, input_data: dict) -> dict:
        items = input_data["items"]
        currency = input_data.get("currency", "USD")

        total = sum(item["price"] for item in items)
        return {
            "total": total,
            "count": len(items),
            "currency": currency,
        }
