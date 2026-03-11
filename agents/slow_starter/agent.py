"""Slow-starter agent — simulates expensive model loading on startup."""

import asyncio

from atlas.runtime.base import AgentBase

# Module-level counter to track how many times startup is called
_startup_count = 0


class SlowStarterAgent(AgentBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._started = False

    async def on_startup(self) -> None:
        global _startup_count
        await asyncio.sleep(0.5)  # Simulate model loading (kept short for tests)
        _startup_count += 1
        self._started = True

    async def execute(self, input_data: dict) -> dict:
        return {
            "result": f"processed: {input_data['text']}",
            "startup_count": _startup_count,
        }

    async def on_shutdown(self) -> None:
        self._started = False
