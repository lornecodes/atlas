"""EventBus — multi-subscriber event system for job lifecycle transitions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from atlas.logging import get_logger

if TYPE_CHECKING:
    from atlas.pool.job import JobData

logger = get_logger(__name__)

# Callback signature: (job, old_status, new_status) -> None
EventCallback = Callable[["JobData", str, str], Awaitable[None]]


class EventBus:
    """Multi-subscriber event bus for job lifecycle events.

    Subscribers receive (job, old_status, new_status) on every status
    transition. Exceptions in one subscriber never crash the bus or
    affect other subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: list[EventCallback] = []

    def subscribe(self, callback: EventCallback) -> None:
        """Add a subscriber. No-op if already subscribed."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """Remove a subscriber. No-op if not subscribed."""
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    async def emit(self, job: JobData, old_status: str, new_status: str) -> None:
        """Fire all subscribers for a status transition.

        Each subscriber is called independently — a failure in one
        does not prevent others from running.
        """
        for callback in list(self._subscribers):  # Copy to allow unsub during emit
            try:
                await callback(job, old_status, new_status)
            except Exception as e:
                logger.error(
                    "Event subscriber %s failed for job %s (%s→%s): %s",
                    getattr(callback, "__name__", callback),
                    job.id,
                    old_status,
                    new_status,
                    e,
                )

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
