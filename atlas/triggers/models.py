"""Trigger data models — defines trigger types and their configuration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from atlas.triggers.cron import CronExpr

VALID_TRIGGER_TYPES = frozenset({"cron", "interval", "one_shot", "webhook"})


@dataclass
class TriggerDefinition:
    """A trigger that submits jobs to the pool on a schedule or event.

    Trigger types:
        cron     — fires on a cron schedule (e.g., "*/5 * * * *")
        interval — fires every N seconds
        one_shot — fires once at a specific time (or immediately if in the past)
        webhook  — fires when an HTTP request hits the webhook endpoint
    """

    id: str = ""
    name: str = ""
    trigger_type: str = ""  # cron | interval | one_shot | webhook
    enabled: bool = True

    # Target
    agent_name: str = ""       # agent to run (mutually exclusive with chain_name)
    chain_name: str = ""       # chain to run
    input_data: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Schedule config (type-specific)
    cron_expr: str = ""        # cron expression, e.g. "0 9 * * *"
    interval_seconds: float = 0.0
    fire_at: float = 0.0      # unix timestamp for one_shot

    # Webhook config
    webhook_secret: str = ""   # HMAC secret for validating payloads

    # State
    last_fired: float = 0.0
    next_fire: float = 0.0
    fire_count: int = 0
    last_job_id: str = ""
    created_at: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = f"trigger-{uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    def validate(self) -> None:
        """Validate trigger configuration. Raises ValueError on problems."""
        if self.trigger_type not in VALID_TRIGGER_TYPES:
            raise ValueError(
                f"Invalid trigger_type {self.trigger_type!r}, "
                f"must be one of {sorted(VALID_TRIGGER_TYPES)}"
            )
        if not self.agent_name and not self.chain_name:
            raise ValueError("Trigger must specify agent_name or chain_name")
        if self.agent_name and self.chain_name:
            raise ValueError("Trigger cannot specify both agent_name and chain_name")

        if self.trigger_type == "cron":
            if not self.cron_expr:
                raise ValueError("Cron trigger requires cron_expr")
            CronExpr.parse(self.cron_expr)  # validates syntax
        elif self.trigger_type == "interval":
            if self.interval_seconds <= 0:
                raise ValueError("Interval trigger requires interval_seconds > 0")
        elif self.trigger_type == "one_shot":
            if self.fire_at <= 0:
                raise ValueError("One-shot trigger requires fire_at > 0")

    def compute_next_fire(self, now: float | None = None) -> float:
        """Compute the next fire time based on trigger type and state.

        Returns 0.0 for webhook triggers (event-driven, no schedule).
        """
        if now is None:
            now = time.time()

        if self.trigger_type == "cron":
            after = self.last_fired if self.last_fired > 0 else now
            return CronExpr.parse(self.cron_expr).next_fire(after)
        elif self.trigger_type == "interval":
            if self.last_fired > 0:
                return self.last_fired + self.interval_seconds
            return now + self.interval_seconds
        elif self.trigger_type == "one_shot":
            return self.fire_at
        else:  # webhook
            return 0.0

    @property
    def target(self) -> str:
        """Human-readable target identifier."""
        return self.chain_name if self.chain_name else self.agent_name

    @property
    def is_recurring(self) -> bool:
        return self.trigger_type in ("cron", "interval")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trigger_type": self.trigger_type,
            "enabled": self.enabled,
            "agent_name": self.agent_name,
            "chain_name": self.chain_name,
            "input_data": self.input_data,
            "priority": self.priority,
            "metadata": self.metadata,
            "cron_expr": self.cron_expr,
            "interval_seconds": self.interval_seconds,
            "fire_at": self.fire_at,
            "webhook_secret": self.webhook_secret,
            "last_fired": self.last_fired,
            "next_fire": self.next_fire,
            "fire_count": self.fire_count,
            "last_job_id": self.last_job_id,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TriggerDefinition:
        """Create a TriggerDefinition from a dict (API body or YAML content).

        Accepts both flat dicts and dicts with a top-level 'trigger' key.
        """
        if "trigger" in data and isinstance(data["trigger"], dict):
            data = data["trigger"]

        trigger = TriggerDefinition(
            id=data.get("id", ""),
            name=data.get("name", ""),
            trigger_type=data.get("trigger_type", ""),
            enabled=data.get("enabled", True),
            agent_name=data.get("agent_name", ""),
            chain_name=data.get("chain_name", ""),
            input_data=data.get("input_data", {}),
            priority=int(data.get("priority", 0)),
            metadata=data.get("metadata", {}),
            cron_expr=data.get("cron_expr", ""),
            interval_seconds=float(data.get("interval_seconds", 0.0)),
            fire_at=float(data.get("fire_at", 0.0)),
            webhook_secret=data.get("webhook_secret", ""),
            last_fired=float(data.get("last_fired", 0.0)),
            next_fire=float(data.get("next_fire", 0.0)),
            fire_count=int(data.get("fire_count", 0)),
            last_job_id=data.get("last_job_id", ""),
            created_at=float(data.get("created_at", 0.0)),
        )
        return trigger

    @staticmethod
    def from_yaml(path: str | Path) -> TriggerDefinition:
        """Load a TriggerDefinition from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return TriggerDefinition.from_dict(data)
