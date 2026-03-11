"""Eval hooks — declarative YAML-based output validation for agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from atlas.events import EventBus, EventCallback
from atlas.logging import get_logger
from atlas.trace import EvalResult

if TYPE_CHECKING:
    from atlas.contract.registry import AgentRegistry
    from atlas.pool.job import JobData
    from atlas.trace import TraceCollector

logger = get_logger(__name__)


@dataclass
class EvalCheck:
    """A single eval check definition."""

    name: str
    type: str  # "equals", "contains", "range", "regex", "exists"
    field: str  # key in output_data
    expected: Any = None
    min_val: float | None = None
    max_val: float | None = None
    pattern: str = ""

    def run(self, output: dict[str, Any]) -> EvalResult:
        """Execute this check against an output dict."""
        value = output.get(self.field)

        if self.type == "exists":
            passed = value is not None and value != ""
            return EvalResult(
                name=self.name,
                passed=passed,
                expected="non-empty value",
                actual=value,
                message="" if passed else f"Field '{self.field}' is missing or empty",
            )

        if self.type == "equals":
            passed = value == self.expected
            return EvalResult(
                name=self.name,
                passed=passed,
                expected=self.expected,
                actual=value,
                message="" if passed else f"Expected {self.expected!r}, got {value!r}",
            )

        if self.type == "contains":
            if isinstance(value, str):
                passed = str(self.expected) in value
            elif isinstance(value, (list, dict)):
                passed = self.expected in value
            else:
                passed = False
            return EvalResult(
                name=self.name,
                passed=passed,
                expected=f"contains {self.expected!r}",
                actual=value,
                message="" if passed else f"'{self.field}' does not contain {self.expected!r}",
            )

        if self.type == "range":
            if not isinstance(value, (int, float)):
                return EvalResult(
                    name=self.name,
                    passed=False,
                    expected=f"[{self.min_val}, {self.max_val}]",
                    actual=value,
                    message=f"'{self.field}' is not numeric: {value!r}",
                )
            in_range = True
            if self.min_val is not None and value < self.min_val:
                in_range = False
            if self.max_val is not None and value > self.max_val:
                in_range = False
            return EvalResult(
                name=self.name,
                passed=in_range,
                expected=f"[{self.min_val}, {self.max_val}]",
                actual=value,
                message="" if in_range else f"'{self.field}' = {value} outside [{self.min_val}, {self.max_val}]",
            )

        if self.type == "regex":
            if not isinstance(value, str):
                return EvalResult(
                    name=self.name,
                    passed=False,
                    expected=f"matches /{self.pattern}/",
                    actual=value,
                    message=f"'{self.field}' is not a string",
                )
            passed = bool(re.search(self.pattern, value))
            return EvalResult(
                name=self.name,
                passed=passed,
                expected=f"matches /{self.pattern}/",
                actual=value,
                message="" if passed else f"'{self.field}' does not match /{self.pattern}/",
            )

        return EvalResult(
            name=self.name,
            passed=False,
            message=f"Unknown check type: {self.type}",
        )


@dataclass
class EvalDefinition:
    """Eval definition for an agent — loaded from eval.yaml."""

    agent_name: str
    checks: list[EvalCheck] = field(default_factory=list)

    @staticmethod
    def from_yaml(path: Path) -> EvalDefinition:
        """Load an eval definition from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        eval_block = raw.get("eval", raw)
        agent_name = eval_block.get("agent_name", path.parent.name)
        checks = []
        for c in eval_block.get("checks", []):
            checks.append(EvalCheck(
                name=c["name"],
                type=c["type"],
                field=c.get("field", ""),
                expected=c.get("expected"),
                min_val=c.get("min_val"),
                max_val=c.get("max_val"),
                pattern=c.get("pattern", ""),
            ))
        return EvalDefinition(agent_name=agent_name, checks=checks)


class EvalRunner:
    """Runs eval checks against an output dict."""

    def run(
        self, definition: EvalDefinition, output_data: dict[str, Any]
    ) -> list[EvalResult]:
        """Execute all checks in the definition. Returns results."""
        return [check.run(output_data) for check in definition.checks]


class EvalSubscriber:
    """EventBus subscriber — runs evals async on job completion.

    Discovers eval.yaml from the agent's source path in the registry.
    Results are attached to the trace via TraceCollector.
    """

    def __init__(
        self,
        bus: EventBus,
        registry: AgentRegistry,
        trace_collector: TraceCollector,
    ) -> None:
        self._registry = registry
        self._traces = trace_collector
        self._runner = EvalRunner()
        self._bus = bus
        self._cache: dict[str, EvalDefinition | None] = {}
        self._callback: EventCallback = self._on_event
        bus.subscribe(self._callback)

    def _load_eval(self, agent_name: str) -> EvalDefinition | None:
        """Load and cache eval.yaml for an agent."""
        if agent_name in self._cache:
            return self._cache[agent_name]

        entry = self._registry.get(agent_name)
        if not entry or not entry.source_path:
            self._cache[agent_name] = None
            return None

        eval_path = Path(entry.source_path).parent / "eval.yaml"
        if not eval_path.exists():
            self._cache[agent_name] = None
            return None

        try:
            definition = EvalDefinition.from_yaml(eval_path)
            self._cache[agent_name] = definition
            return definition
        except Exception as e:
            logger.warning("Failed to load eval.yaml for %s: %s", agent_name, e)
            self._cache[agent_name] = None
            return None

    async def _on_event(
        self, job: JobData, old_status: str, new_status: str
    ) -> None:
        if new_status != "completed":
            return
        if not job.output_data:
            return

        definition = self._load_eval(job.agent_name)
        if not definition:
            return

        results = self._runner.run(definition, job.output_data)
        self._traces.attach_eval_results(job.id, results)

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        if passed < total:
            logger.warning(
                "Eval for job %s (%s): %d/%d checks passed",
                job.id, job.agent_name, passed, total,
            )

    def close(self) -> None:
        self._bus.unsubscribe(self._callback)
