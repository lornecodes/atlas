"""ContainerSlot — Docker-based agent execution with stdin/stdout JSON protocol."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from atlas.contract.permissions import PermissionsSpec
from atlas.logging import get_logger
from atlas.security.protocol import ContainerMessage, ContainerResponse

logger = get_logger(__name__)


class ContainerError(Exception):
    """Raised when container execution fails."""


class ContainerSlot:
    """Executes an agent in a Docker container via stdin/stdout JSON protocol.

    Lifecycle:
        on_startup()  — validates Docker is available
        execute()     — runs ``docker run`` with resource limits, sends input via stdin,
                        reads output from stdout
        on_shutdown()  — no-op (container is removed after each execute)

    The container receives a JSON message on stdin::

        {"input": {...}, "context": {...}}

    And must write a JSON response on stdout::

        {"output": {...}}          # success
        {"error": "message"}       # failure
    """

    def __init__(
        self,
        image: str,
        *,
        permissions: PermissionsSpec | None = None,
        secrets: dict[str, str] | None = None,
        network: str = "none",
        timeout: float = 60.0,
        working_dir: str = "",
    ) -> None:
        self._image = image
        self._permissions = permissions or PermissionsSpec()
        self._secrets = secrets or {}
        self._network = network
        self._timeout = timeout
        self._working_dir = working_dir
        self._docker_cmd: str | None = None

    async def on_startup(self) -> None:
        """Validate that Docker is available on the system."""
        self._docker_cmd = shutil.which("docker")
        if not self._docker_cmd:
            raise ContainerError(
                "Docker not found on PATH. Install Docker to use container isolation."
            )

    async def execute(
        self,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the container, send input via stdin, read output from stdout."""
        if not self._docker_cmd:
            raise ContainerError("on_startup() must be called before execute()")

        cmd = self._build_command()
        message = ContainerMessage(input=input_data, context=context or {})

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdin_data = (message.to_json() + "\n").encode()

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            # Kill the container process
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            raise ContainerError(
                f"Container execution timed out after {self._timeout}s"
            )
        except FileNotFoundError:
            raise ContainerError(f"Docker command not found: {self._docker_cmd}")

        if proc.returncode != 0:
            stderr_str = stderr_bytes.decode(errors="replace").strip()
            raise ContainerError(
                f"Container exited with code {proc.returncode}: {stderr_str}"
            )

        stdout_str = stdout_bytes.decode(errors="replace").strip()
        if not stdout_str:
            raise ContainerError("Container produced no output on stdout")

        # Parse the last line of stdout (agent may print logs before the JSON)
        last_line = stdout_str.split("\n")[-1]
        try:
            response = ContainerResponse.from_json(last_line)
        except (json.JSONDecodeError, KeyError) as e:
            raise ContainerError(f"Invalid container response: {e}")

        if not response.success:
            raise ContainerError(f"Container agent error: {response.error}")

        return response.output

    async def on_shutdown(self) -> None:
        """No-op — containers are removed after each execute via --rm."""
        pass

    def _build_command(self) -> list[str]:
        """Build the ``docker run`` command with resource limits and env vars."""
        cmd = [
            self._docker_cmd,
            "run",
            "--rm",           # remove container after exit
            "-i",             # attach stdin
            "--network", self._network,
        ]

        # Resource limits
        if self._permissions.max_memory_mb > 0:
            cmd.extend(["--memory", f"{self._permissions.max_memory_mb}m"])

        if self._permissions.max_cpu_seconds > 0:
            # Use --cpus to limit CPU shares (map seconds to a rough CPU limit)
            # max_cpu_seconds is a timeout concept; for container caps we limit
            # to 1 CPU by default, with timeout handled by asyncio.wait_for
            cmd.extend(["--cpus", "1"])

        # Secrets as environment variables
        for name, value in self._secrets.items():
            cmd.extend(["-e", f"{name}={value}"])

        # Working directory mount (read-only unless write permission)
        if self._working_dir:
            if ".." in Path(self._working_dir).parts:
                raise ContainerError(
                    f"working_dir must not contain '..': {self._working_dir}"
                )
            wd = self._working_dir
            if "write" in self._permissions.filesystem:
                cmd.extend(["-v", f"{wd}:/workspace"])
            elif "read" in self._permissions.filesystem:
                cmd.extend(["-v", f"{wd}:/workspace:ro"])
            cmd.extend(["-w", "/workspace"])

        cmd.append(self._image)
        return cmd
