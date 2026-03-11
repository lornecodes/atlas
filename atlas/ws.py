"""WebSocket streaming — real-time job lifecycle events."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

from aiohttp import web, WSMsgType

from atlas.app_keys import EVENT_BUS as _EVENT_BUS_KEY, JOB_TO_DICT as _JOB_TO_DICT_KEY
from atlas.events import EventBus
from atlas.constants import TERMINAL_STATUSES
from atlas.logging import get_logger
from atlas.pool.job import JobData

logger = get_logger(__name__)


def _build_event_frame(
    job: JobData,
    old_status: str,
    new_status: str,
    job_to_dict: Callable[[JobData], dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSON-serializable event frame."""
    return {
        "job_id": job.id,
        "agent_name": job.agent_name,
        "old_status": old_status,
        "new_status": new_status,
        "timestamp": time.time(),
        "job": job_to_dict(job),
    }


async def handle_events_ws(request: web.Request) -> web.WebSocketResponse:
    """GET /api/events — stream ALL job lifecycle events."""
    bus: EventBus | None = request.app.get(_EVENT_BUS_KEY)
    if not bus:
        return web.json_response({"error": "Events not available"}, status=503)

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    job_to_dict = request.app[_JOB_TO_DICT_KEY]
    queue: asyncio.Queue = asyncio.Queue()

    async def on_event(job: JobData, old_status: str, new_status: str) -> None:
        frame = _build_event_frame(job, old_status, new_status, job_to_dict)
        await queue.put(frame)

    bus.subscribe(on_event)

    try:
        send_task = asyncio.create_task(_drain_queue_to_ws(ws, queue))
        async for msg in ws:
            if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
        send_task.cancel()
    finally:
        bus.unsubscribe(on_event)

    return ws


async def handle_job_events_ws(request: web.Request) -> web.WebSocketResponse:
    """GET /api/jobs/{id}/events — stream events for a single job until terminal."""
    bus: EventBus | None = request.app.get(_EVENT_BUS_KEY)
    if not bus:
        return web.json_response({"error": "Events not available"}, status=503)

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    job_id = request.match_info["id"]
    job_to_dict = request.app[_JOB_TO_DICT_KEY]
    queue: asyncio.Queue = asyncio.Queue()

    async def on_event(job: JobData, old_status: str, new_status: str) -> None:
        if job.id != job_id:
            return
        frame = _build_event_frame(job, old_status, new_status, job_to_dict)
        await queue.put(frame)
        if new_status in TERMINAL_STATUSES:
            await queue.put(None)  # Sentinel to close

    bus.subscribe(on_event)

    try:
        send_task = asyncio.create_task(
            _drain_queue_to_ws(ws, queue, close_on_sentinel=True)
        )
        async for msg in ws:
            if msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                break
        send_task.cancel()
    finally:
        bus.unsubscribe(on_event)

    return ws


async def _drain_queue_to_ws(
    ws: web.WebSocketResponse,
    queue: asyncio.Queue,
    close_on_sentinel: bool = False,
) -> None:
    """Pull frames from the queue and send over WebSocket."""
    try:
        while not ws.closed:
            frame = await queue.get()
            if frame is None and close_on_sentinel:
                await ws.close()
                return
            if frame is not None:
                await ws.send_json(frame)
    except asyncio.CancelledError:
        pass
    except ConnectionResetError:
        pass
