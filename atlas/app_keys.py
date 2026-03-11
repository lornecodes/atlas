"""Typed aiohttp AppKey definitions — shared across serve.py and ws.py."""

from __future__ import annotations

from typing import Any

from aiohttp import web

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.pool.executor import ExecutionPool
from atlas.pool.queue import JobQueue
from atlas.store.job_store import JobStore

REGISTRY = web.AppKey("registry", AgentRegistry)
QUEUE = web.AppKey("queue", JobQueue)
POOL = web.AppKey("pool", ExecutionPool)
STORE = web.AppKey("store", JobStore)
EVENT_BUS = web.AppKey("event_bus", EventBus)
JOB_TO_DICT: web.AppKey[Any] = web.AppKey("job_to_dict")
METRICS: web.AppKey[Any] = web.AppKey("metrics")
CHAIN_EXECUTOR: web.AppKey[Any] = web.AppKey("chain_executor")
TRACE_COLLECTOR: web.AppKey[Any] = web.AppKey("trace_collector")
