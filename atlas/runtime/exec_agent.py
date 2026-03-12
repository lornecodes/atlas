"""ExecAgent — agent backed by an external executable (any language).

JSON envelope on stdin, JSON output on stdout. Cold-start per execution.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from atlas.runtime.base import AgentBase


class ExecAgent(AgentBase):
    """Agent backed by an external process.

    The contract declares ``provider: {type: exec, command: ["./my-agent"]}``.
    Atlas pipes a JSON envelope on stdin and reads JSON output from stdout.

    Stdin envelope::

        {
            "input": {<agent input data>},
            "context": {"job_id": "...", "chain_name": "...", "step_index": 0},
            "memory": "shared memory contents (if enabled)"
        }

    Stdout: JSON object matching the contract's output schema.
    If the output contains a ``_memory_append`` key, that value is appended
    to shared memory and the key is stripped before schema validation.
    """

    async def execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        command = self.contract.provider.command
        if not command:
            raise RuntimeError("exec provider requires 'command' in contract")

        agent_dir = self.context.metadata.get("_agent_dir", ".")

        # Build envelope
        envelope: dict[str, Any] = {
            "input": input_data,
            "context": {
                "job_id": self.context.job_id,
                "chain_name": self.context.chain_name,
                "step_index": self.context.step_index,
            },
        }

        # Include memory if available
        if self.context._memory_provider:
            envelope["memory"] = await self.context.memory_read()

        # Include relevant knowledge if available
        if self.context._knowledge_provider:
            from atlas.runtime.dynamic_llm_agent import _extract_search_text
            search_query = _extract_search_text(input_data)
            results = await self.context.knowledge_search(search_query, limit=5) if search_query else []
            envelope["knowledge"] = [
                {"id": e.id, "domain": e.domain, "content": e.content, "tags": e.tags}
                for e in results
            ]

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=agent_dir,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(json.dumps(envelope).encode()),
                timeout=self.contract.execution_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(
                f"Process timed out after {self.contract.execution_timeout}s"
            )

        if proc.returncode != 0:
            err_msg = stderr.decode().strip() if stderr else "unknown error"
            raise RuntimeError(
                f"Process exited with code {proc.returncode}: {err_msg}"
            )

        try:
            output = json.loads(stdout.decode())
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON output from process: {e}\n"
                f"stdout: {stdout.decode()[:500]}"
            )

        # Handle memory append from exec agent
        if "_memory_append" in output and self.context._memory_provider:
            await self.context.memory_append(str(output.pop("_memory_append")))
        elif "_memory_append" in output:
            output.pop("_memory_append")  # Strip even if no memory provider

        # Handle knowledge store from exec agent
        if "_knowledge_store" in output and self.context._knowledge_provider:
            store_data = output.pop("_knowledge_store")
            items = store_data if isinstance(store_data, list) else [store_data]
            for item in items:
                if isinstance(item, dict):
                    await self.context.knowledge_store(
                        content=item.get("content", ""),
                        domain=item.get("domain", "general"),
                        tags=item.get("tags", []),
                    )
        elif "_knowledge_store" in output:
            output.pop("_knowledge_store")

        return output
