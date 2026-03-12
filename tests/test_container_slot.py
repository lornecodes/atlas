"""Tests for ContainerSlot — Docker-based agent execution."""

import asyncio
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.contract.permissions import PermissionsSpec
from atlas.security.container import ContainerError, ContainerSlot


class TestContainerSlotStartup:
    @pytest.mark.asyncio
    async def test_startup_finds_docker(self):
        with patch("shutil.which", return_value="/usr/bin/docker"):
            slot = ContainerSlot("python:3.12-slim")
            await slot.on_startup()
            assert slot._docker_cmd == "/usr/bin/docker"

    @pytest.mark.asyncio
    async def test_startup_docker_not_found(self):
        with patch("shutil.which", return_value=None):
            slot = ContainerSlot("python:3.12-slim")
            with pytest.raises(ContainerError, match="Docker not found"):
                await slot.on_startup()

    @pytest.mark.asyncio
    async def test_execute_before_startup(self):
        slot = ContainerSlot("python:3.12-slim")
        with pytest.raises(ContainerError, match="on_startup"):
            await slot.execute({"key": "val"})


class TestContainerSlotBuildCommand:
    def _make_slot(self, **kwargs):
        slot = ContainerSlot("test-image:v1", **kwargs)
        slot._docker_cmd = "/usr/bin/docker"
        return slot

    def test_basic_command(self):
        slot = self._make_slot()
        cmd = slot._build_command()
        assert cmd[0] == "/usr/bin/docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "-i" in cmd
        assert "test-image:v1" == cmd[-1]

    def test_network_none(self):
        slot = self._make_slot(network="none")
        cmd = slot._build_command()
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

    def test_network_bridge(self):
        slot = self._make_slot(network="bridge")
        cmd = slot._build_command()
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "bridge"

    def test_memory_limit(self):
        perms = PermissionsSpec(max_memory_mb=256)
        slot = self._make_slot(permissions=perms)
        cmd = slot._build_command()
        idx = cmd.index("--memory")
        assert cmd[idx + 1] == "256m"

    def test_cpu_limit(self):
        perms = PermissionsSpec(max_cpu_seconds=30)
        slot = self._make_slot(permissions=perms)
        cmd = slot._build_command()
        assert "--cpus" in cmd

    def test_secrets_as_env(self):
        slot = self._make_slot(secrets={"API_KEY": "abc123", "DB_URL": "pg://x"})
        cmd = slot._build_command()
        assert "-e" in cmd
        assert "API_KEY=abc123" in cmd
        assert "DB_URL=pg://x" in cmd

    def test_working_dir_read_only(self):
        perms = PermissionsSpec(filesystem=["read"])
        slot = self._make_slot(permissions=perms, working_dir="/tmp/agent")
        cmd = slot._build_command()
        assert "/tmp/agent:/workspace:ro" in cmd

    def test_working_dir_read_write(self):
        perms = PermissionsSpec(filesystem=["read", "write"])
        slot = self._make_slot(permissions=perms, working_dir="/tmp/agent")
        cmd = slot._build_command()
        assert "/tmp/agent:/workspace" in cmd
        # Should NOT have :ro
        for arg in cmd:
            if "/workspace" in arg:
                assert not arg.endswith(":ro")

    def test_no_working_dir(self):
        slot = self._make_slot()
        cmd = slot._build_command()
        assert "-v" not in cmd
        assert "-w" not in cmd

    def test_no_filesystem_no_mount(self):
        perms = PermissionsSpec(filesystem=[])
        slot = self._make_slot(permissions=perms, working_dir="/tmp/agent")
        cmd = slot._build_command()
        assert "-v" not in cmd


class TestContainerSlotExecute:
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        slot = ContainerSlot("test:v1")
        slot._docker_cmd = "/usr/bin/docker"

        output_json = json.dumps({"output": {"result": "hello"}})

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (output_json.encode(), b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await slot.execute({"input_key": "val"})

        assert result == {"result": "hello"}

    @pytest.mark.asyncio
    async def test_container_error_response(self):
        slot = ContainerSlot("test:v1")
        slot._docker_cmd = "/usr/bin/docker"

        output_json = json.dumps({"error": "agent crashed"})

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (output_json.encode(), b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ContainerError, match="agent crashed"):
                await slot.execute({"key": "val"})

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self):
        slot = ContainerSlot("test:v1")
        slot._docker_cmd = "/usr/bin/docker"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"OOM killed")
        mock_proc.returncode = 137

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ContainerError, match="exited with code 137"):
                await slot.execute({})

    @pytest.mark.asyncio
    async def test_empty_stdout(self):
        slot = ContainerSlot("test:v1")
        slot._docker_cmd = "/usr/bin/docker"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ContainerError, match="no output"):
                await slot.execute({})

    @pytest.mark.asyncio
    async def test_invalid_json_output(self):
        slot = ContainerSlot("test:v1")
        slot._docker_cmd = "/usr/bin/docker"

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"not json at all", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ContainerError, match="Invalid container response"):
                await slot.execute({})

    @pytest.mark.asyncio
    async def test_timeout(self):
        slot = ContainerSlot("test:v1", timeout=0.01)
        slot._docker_cmd = "/usr/bin/docker"

        mock_proc = AsyncMock()
        mock_proc.communicate.side_effect = asyncio.TimeoutError()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(ContainerError, match="timed out"):
                await slot.execute({})

    @pytest.mark.asyncio
    async def test_multiline_stdout_uses_last_line(self):
        """Agent may print logs before the JSON response — use last line."""
        slot = ContainerSlot("test:v1")
        slot._docker_cmd = "/usr/bin/docker"

        output = "Loading model...\nReady\n" + json.dumps({"output": {"ok": True}})

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (output.encode(), b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await slot.execute({})

        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_sends_input_and_context(self):
        slot = ContainerSlot("test:v1")
        slot._docker_cmd = "/usr/bin/docker"

        output_json = json.dumps({"output": {}})
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (output_json.encode(), b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await slot.execute({"key": "val"}, context={"job_id": "j-1"})

        # Verify stdin data
        call_args = mock_proc.communicate.call_args
        stdin_data = call_args.kwargs.get("input") or call_args[1].get("input")
        parsed = json.loads(stdin_data)
        assert parsed["input"] == {"key": "val"}
        assert parsed["context"] == {"job_id": "j-1"}


class TestContainerSlotShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_is_noop(self):
        slot = ContainerSlot("test:v1")
        await slot.on_shutdown()  # should not raise
