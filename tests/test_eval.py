"""Tests for the eval hooks system."""

from __future__ import annotations

import asyncio
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from atlas.contract.registry import AgentRegistry
from atlas.events import EventBus
from atlas.eval import EvalCheck, EvalDefinition, EvalRunner, EvalSubscriber
from atlas.pool.job import JobData
from atlas.trace import EvalResult, TraceCollector

AGENTS_DIR = Path(__file__).parent.parent / "agents"


# ---------------------------------------------------------------------------
# EvalCheck — individual check types
# ---------------------------------------------------------------------------

class TestEvalCheckExists:
    def test_exists_passes(self):
        r = EvalCheck(name="t", type="exists", field="key").run({"key": "val"})
        assert r.passed is True

    def test_exists_fails_missing(self):
        r = EvalCheck(name="t", type="exists", field="key").run({})
        assert r.passed is False

    def test_exists_fails_empty(self):
        r = EvalCheck(name="t", type="exists", field="key").run({"key": ""})
        assert r.passed is False

    def test_exists_zero_is_present(self):
        r = EvalCheck(name="t", type="exists", field="key").run({"key": 0})
        assert r.passed is True


class TestEvalCheckEquals:
    def test_equals_pass(self):
        r = EvalCheck(name="t", type="equals", field="x", expected=42).run({"x": 42})
        assert r.passed is True

    def test_equals_fail(self):
        r = EvalCheck(name="t", type="equals", field="x", expected=42).run({"x": 99})
        assert r.passed is False
        assert "99" in r.message

    def test_equals_string(self):
        r = EvalCheck(name="t", type="equals", field="s", expected="hello").run({"s": "hello"})
        assert r.passed is True


class TestEvalCheckContains:
    def test_contains_string(self):
        r = EvalCheck(name="t", type="contains", field="s", expected="world").run(
            {"s": "hello world"}
        )
        assert r.passed is True

    def test_contains_string_fail(self):
        r = EvalCheck(name="t", type="contains", field="s", expected="xyz").run(
            {"s": "hello"}
        )
        assert r.passed is False

    def test_contains_list(self):
        r = EvalCheck(name="t", type="contains", field="lst", expected=3).run(
            {"lst": [1, 2, 3]}
        )
        assert r.passed is True


class TestEvalCheckRange:
    def test_in_range(self):
        r = EvalCheck(name="t", type="range", field="x", min_val=0, max_val=100).run(
            {"x": 50}
        )
        assert r.passed is True

    def test_below_min(self):
        r = EvalCheck(name="t", type="range", field="x", min_val=10, max_val=100).run(
            {"x": 5}
        )
        assert r.passed is False

    def test_above_max(self):
        r = EvalCheck(name="t", type="range", field="x", min_val=0, max_val=10).run(
            {"x": 50}
        )
        assert r.passed is False

    def test_non_numeric(self):
        r = EvalCheck(name="t", type="range", field="x", min_val=0, max_val=10).run(
            {"x": "hello"}
        )
        assert r.passed is False
        assert "not numeric" in r.message

    def test_min_only(self):
        r = EvalCheck(name="t", type="range", field="x", min_val=5).run({"x": 10})
        assert r.passed is True

    def test_max_only(self):
        r = EvalCheck(name="t", type="range", field="x", max_val=100).run({"x": 50})
        assert r.passed is True


class TestEvalCheckRegex:
    def test_regex_match(self):
        r = EvalCheck(name="t", type="regex", field="s", pattern=r"^\d+$").run(
            {"s": "12345"}
        )
        assert r.passed is True

    def test_regex_no_match(self):
        r = EvalCheck(name="t", type="regex", field="s", pattern=r"^\d+$").run(
            {"s": "abc"}
        )
        assert r.passed is False

    def test_regex_non_string(self):
        r = EvalCheck(name="t", type="regex", field="s", pattern=r".*").run(
            {"s": 42}
        )
        assert r.passed is False


class TestEvalCheckUnknown:
    def test_unknown_type(self):
        r = EvalCheck(name="t", type="nope", field="x").run({"x": 1})
        assert r.passed is False
        assert "Unknown check type" in r.message


# ---------------------------------------------------------------------------
# EvalDefinition — YAML loading
# ---------------------------------------------------------------------------

class TestEvalDefinition:
    def test_from_yaml(self, tmp_path):
        eval_file = tmp_path / "eval.yaml"
        eval_file.write_text(textwrap.dedent("""\
            eval:
              checks:
                - name: has_output
                  type: exists
                  field: result
                - name: score_valid
                  type: range
                  field: score
                  min_val: 0
                  max_val: 1.0
        """))

        defn = EvalDefinition.from_yaml(eval_file)
        assert defn.agent_name == tmp_path.name
        assert len(defn.checks) == 2
        assert defn.checks[0].name == "has_output"
        assert defn.checks[0].type == "exists"
        assert defn.checks[1].min_val == 0
        assert defn.checks[1].max_val == 1.0

    def test_from_yaml_with_agent_name(self, tmp_path):
        eval_file = tmp_path / "eval.yaml"
        eval_file.write_text(textwrap.dedent("""\
            eval:
              agent_name: custom-agent
              checks:
                - name: check1
                  type: equals
                  field: x
                  expected: 42
        """))

        defn = EvalDefinition.from_yaml(eval_file)
        assert defn.agent_name == "custom-agent"


# ---------------------------------------------------------------------------
# EvalRunner
# ---------------------------------------------------------------------------

class TestEvalRunner:
    def test_run_all_pass(self):
        defn = EvalDefinition(
            agent_name="test",
            checks=[
                EvalCheck(name="a", type="exists", field="x"),
                EvalCheck(name="b", type="equals", field="y", expected=10),
            ],
        )
        results = EvalRunner().run(defn, {"x": "hello", "y": 10})
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_run_partial_fail(self):
        defn = EvalDefinition(
            agent_name="test",
            checks=[
                EvalCheck(name="a", type="exists", field="x"),
                EvalCheck(name="b", type="equals", field="y", expected=10),
            ],
        )
        results = EvalRunner().run(defn, {"x": "hello", "y": 99})
        assert results[0].passed is True
        assert results[1].passed is False


# ---------------------------------------------------------------------------
# EvalSubscriber — integration
# ---------------------------------------------------------------------------

class TestEvalSubscriber:
    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def registry(self):
        reg = AgentRegistry(search_paths=[AGENTS_DIR])
        reg.discover()
        return reg

    @pytest.fixture
    def trace_collector(self, bus):
        tc = TraceCollector(bus, max_traces=100)
        yield tc
        tc.close()

    @pytest.fixture
    def subscriber(self, bus, registry, trace_collector):
        sub = EvalSubscriber(bus, registry, trace_collector)
        yield sub
        sub.close()

    async def test_evals_attached_on_completion(self, bus, subscriber, trace_collector):
        """Echo agent has eval.yaml — evals should run and attach to trace."""
        job = JobData(
            agent_name="echo",
            status="completed",
            output_data={"message": "hello"},
            started_at=time.time() - 1,
            completed_at=time.time(),
        )
        await bus.emit(job, "running", "completed")

        trace = trace_collector.get(job.id)
        assert trace is not None
        assert len(trace.eval_results) == 2
        assert all(r.passed for r in trace.eval_results)

    async def test_no_eval_for_agent_without_yaml(self, bus, subscriber, trace_collector):
        """Agents without eval.yaml should not have eval results."""
        job = JobData(
            agent_name="formatter",
            status="completed",
            output_data={"formatted": "test"},
            started_at=time.time() - 1,
            completed_at=time.time(),
        )
        await bus.emit(job, "running", "completed")

        trace = trace_collector.get(job.id)
        assert trace is not None
        assert trace.eval_results == []

    async def test_eval_failure_logged(self, bus, subscriber, trace_collector):
        """Eval checks that fail should be attached as failed results."""
        job = JobData(
            agent_name="echo",
            status="completed",
            output_data={"message": ""},  # empty string fails "exists" check
            started_at=time.time() - 1,
            completed_at=time.time(),
        )
        await bus.emit(job, "running", "completed")

        trace = trace_collector.get(job.id)
        assert trace is not None
        # has_message should fail (empty), message_is_string should also fail
        failed = [r for r in trace.eval_results if not r.passed]
        assert len(failed) >= 1

    async def test_skips_failed_jobs(self, bus, subscriber, trace_collector):
        """Evals should not run for failed jobs."""
        job = JobData(
            agent_name="echo",
            status="failed",
            output_data=None,
            started_at=time.time() - 1,
            completed_at=time.time(),
        )
        await bus.emit(job, "running", "failed")

        trace = trace_collector.get(job.id)
        assert trace is not None
        assert trace.eval_results == []
