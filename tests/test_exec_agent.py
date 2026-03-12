"""Tests for ExecAgent — subprocess-based agent execution."""

from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from atlas.contract.types import AgentContract, ProviderSpec
from atlas.runtime.context import AgentContext
from atlas.runtime.exec_agent import ExecAgent


def _make_contract(command: list[str], timeout: float = 10.0) -> AgentContract:
    return AgentContract(
        name="test-exec",
        version="1.0.0",
        provider=ProviderSpec(type="exec", command=command),
        execution_timeout=timeout,
    )


def _make_context(**kwargs) -> AgentContext:
    return AgentContext(job_id="job-test", **kwargs)


def _write_script(tmp_path: Path, filename: str, code: str) -> Path:
    """Write a Python script and return the path."""
    script = tmp_path / filename
    script.write_text(textwrap.dedent(code))
    return script


class TestExecAgentBasic:
    """Basic exec agent functionality."""

    @pytest.mark.asyncio
    async def test_echo_agent(self, tmp_path):
        script = _write_script(tmp_path, "echo.py", """\
            import json, sys
            envelope = json.loads(sys.stdin.read())
            print(json.dumps(envelope["input"]))
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({"message": "hello"})
        assert result == {"message": "hello"}

    @pytest.mark.asyncio
    async def test_envelope_contains_context(self, tmp_path):
        script = _write_script(tmp_path, "dump.py", """\
            import json, sys
            envelope = json.loads(sys.stdin.read())
            print(json.dumps({"ctx": envelope["context"]}))
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(
            metadata={"_agent_dir": str(tmp_path)},
            chain_name="my-chain",
            step_index=2,
        )
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({"x": 1})
        assert result["ctx"]["job_id"] == "job-test"
        assert result["ctx"]["chain_name"] == "my-chain"
        assert result["ctx"]["step_index"] == 2

    @pytest.mark.asyncio
    async def test_envelope_contains_memory(self, tmp_path):
        script = _write_script(tmp_path, "mem.py", """\
            import json, sys
            envelope = json.loads(sys.stdin.read())
            print(json.dumps({"mem": envelope.get("memory", "")}))
        """)
        contract = _make_contract(["python", str(script)])
        mock_mem = AsyncMock()
        mock_mem.read = AsyncMock(return_value="previous learning")
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        ctx._memory_provider = mock_mem
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({"x": 1})
        assert result["mem"] == "previous learning"

    @pytest.mark.asyncio
    async def test_transform_input(self, tmp_path):
        script = _write_script(tmp_path, "upper.py", """\
            import json, sys
            envelope = json.loads(sys.stdin.read())
            msg = envelope["input"]["message"]
            print(json.dumps({"message": msg.upper()}))
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({"message": "hello"})
        assert result == {"message": "HELLO"}


class TestExecAgentErrors:
    """Error handling for exec agents."""

    @pytest.mark.asyncio
    async def test_no_command_raises(self):
        contract = AgentContract(
            name="test", version="1.0.0",
            provider=ProviderSpec(type="exec", command=[]),
        )
        agent = ExecAgent(contract, _make_context())
        with pytest.raises(RuntimeError, match="requires 'command'"):
            await agent.execute({})

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, tmp_path):
        script = _write_script(tmp_path, "fail.py", """\
            import sys
            sys.stderr.write("something went wrong")
            sys.exit(1)
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        with pytest.raises(RuntimeError, match="exited with code 1"):
            await agent.execute({})

    @pytest.mark.asyncio
    async def test_invalid_json_output(self, tmp_path):
        script = _write_script(tmp_path, "bad.py", """\
            print("not json")
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        with pytest.raises(RuntimeError, match="Invalid JSON output"):
            await agent.execute({})

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        script = _write_script(tmp_path, "slow.py", """\
            import time
            time.sleep(10)
        """)
        contract = _make_contract(["python", str(script)], timeout=0.5)
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        with pytest.raises(RuntimeError, match="timed out"):
            await agent.execute({})

    @pytest.mark.asyncio
    async def test_stderr_in_error_message(self, tmp_path):
        script = _write_script(tmp_path, "err.py", """\
            import sys
            sys.stderr.write("detailed error info")
            sys.exit(42)
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        with pytest.raises(RuntimeError, match="detailed error info"):
            await agent.execute({})


class TestExecAgentMemoryAppend:
    """Memory append from exec agent output."""

    @pytest.mark.asyncio
    async def test_memory_append_stripped_and_written(self, tmp_path):
        script = _write_script(tmp_path, "learn.py", """\
            import json, sys
            print(json.dumps({
                "result": "done",
                "_memory_append": "Learned: API limit is 100/min"
            }))
        """)
        contract = _make_contract(["python", str(script)])
        mock_mem = AsyncMock()
        mock_mem.read = AsyncMock(return_value="")
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        ctx._memory_provider = mock_mem
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({})
        assert result == {"result": "done"}
        assert "_memory_append" not in result
        mock_mem.append.assert_called_once_with("Learned: API limit is 100/min")

    @pytest.mark.asyncio
    async def test_memory_append_stripped_without_provider(self, tmp_path):
        script = _write_script(tmp_path, "learn2.py", """\
            import json, sys
            print(json.dumps({"result": "ok", "_memory_append": "something"}))
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({})
        assert result == {"result": "ok"}
        assert "_memory_append" not in result


class TestExecAgentWorkingDirectory:
    """Working directory resolution."""

    @pytest.mark.asyncio
    async def test_cwd_is_agent_dir(self, tmp_path):
        script = _write_script(tmp_path, "cwd.py", """\
            import json, os, sys
            print(json.dumps({"cwd": os.getcwd()}))
        """)
        contract = _make_contract(["python", str(script)])
        ctx = _make_context(metadata={"_agent_dir": str(tmp_path)})
        agent = ExecAgent(contract, ctx)
        result = await agent.execute({})
        # Normalize paths for comparison
        assert Path(result["cwd"]).resolve() == tmp_path.resolve()
