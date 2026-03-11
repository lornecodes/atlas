"""Job data model — lightweight, in-memory, no Redis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from atlas.constants import TERMINAL_STATUSES


@dataclass
class JobData:
    """A unit of work in the execution pool."""

    id: str = ""
    agent_name: str = ""
    status: str = "pending"  # pending | running | completed | failed | cancelled
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] | None = None
    error: str = ""
    priority: int = 0  # Higher = more important

    # Timing
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    # Lifecycle metrics
    warmup_ms: float = 0.0     # How long on_startup took (0 if warm slot reused)
    execution_ms: float = 0.0  # How long execute() took

    # Retry tracking
    retry_count: int = 0
    original_job_id: str = ""  # Links retries back to the original job

    # Arbitrary metadata (e.g., _spawn_depth for child jobs)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"job-{uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = time.time()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def total_ms(self) -> float:
        return self.warmup_ms + self.execution_ms
