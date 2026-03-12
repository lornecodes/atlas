"""Remote agent provider — discover remote agents and register as virtual local agents.

Connects to a remote Atlas instance's MCP server, discovers its agents via
the ``atlas.registry.list`` and ``atlas.registry.describe`` platform tools,
and registers virtual agents in the local AgentRegistry. Each virtual agent's
execute() calls ``atlas.exec.run`` on the remote instance.

This is the federation primitive for chains: a chain on Instance A can
reference ``lab.translator`` as a step, and it executes transparently on
Instance B.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from atlas.logging import get_logger
from atlas.runtime.base import AgentBase

if TYPE_CHECKING:
    from atlas.contract.registry import AgentRegistry
    from atlas.mcp.client import RemoteServer
    from atlas.skills.registry import SkillRegistry
    from atlas.skills.types import SkillCallable

logger = get_logger(__name__)


class RemoteAgent(AgentBase):
    """Virtual agent that proxies execution to a remote Atlas instance.

    Its execute() calls the remote ``atlas.exec.run`` tool, which runs
    the agent synchronously on the remote instance and returns the result.
    """

    # Set by RemoteAgentProvider before registration.
    _exec_run_skill: SkillCallable | None = None
    _remote_agent_name: str = ""

    async def execute(self, input_data: dict) -> dict:
        if self._exec_run_skill is None:
            raise RuntimeError("RemoteAgent not properly initialized: no exec_run skill")
        result = await self._exec_run_skill(
            {"agent": self._remote_agent_name, "input": input_data}
        )
        if not result.get("success"):
            raise RuntimeError(
                f"Remote agent '{self._remote_agent_name}' failed: "
                f"{result.get('error', 'unknown error')}"
            )
        return result.get("data", {})


def _make_remote_agent_class(
    exec_run_skill: SkillCallable,
    remote_agent_name: str,
) -> type:
    """Create a RemoteAgent subclass bound to a specific remote agent.

    Each remote agent gets its own class so the class attributes
    (exec_run_skill, remote_agent_name) don't collide.
    """
    cls = type(
        f"RemoteAgent_{remote_agent_name}",
        (RemoteAgent,),
        {
            "_exec_run_skill": staticmethod(exec_run_skill),
            "_remote_agent_name": remote_agent_name,
        },
    )
    return cls


class RemoteAgentProvider:
    """Discovers remote agents and registers them as virtual local agents.

    Uses the platform tools already registered by RemoteToolProvider (10B):
    - ``{name}.atlas.registry.list`` to discover agents
    - ``{name}.atlas.registry.describe`` to get contracts
    - ``{name}.atlas.exec.run`` to execute agents remotely

    Agents are namespaced: ``{server.name}.{agent.name}``.
    """

    def __init__(self) -> None:
        self._registered: dict[str, list[str]] = {}  # server_name -> [agent_names]

    async def connect(
        self,
        server: "RemoteServer",
        agent_registry: "AgentRegistry",
        skill_registry: "SkillRegistry",
    ) -> int:
        """Discover remote agents and register as virtual agents.

        Requires that RemoteToolProvider.connect() has already been called
        for this server (so the platform tool skills are available).

        Returns the number of agents registered.
        """
        if server.name in self._registered:
            raise ValueError(f"Already connected to '{server.name}'")

        prefix = server.name

        # Look up the remote platform tools in skill_registry
        list_skill = skill_registry.get(f"{prefix}.atlas.registry.list")
        describe_skill = skill_registry.get(f"{prefix}.atlas.registry.describe")
        exec_run_skill = skill_registry.get(f"{prefix}.atlas.exec.run")

        if not list_skill or not list_skill.callable:
            raise ValueError(
                f"Remote '{prefix}' does not expose atlas.registry.list — "
                f"ensure RemoteToolProvider.connect() was called first"
            )
        if not exec_run_skill or not exec_run_skill.callable:
            raise ValueError(
                f"Remote '{prefix}' does not expose atlas.exec.run — "
                f"remote instance may need platform_tools enabled"
            )

        # Discover remote agents
        list_result = await list_skill.callable({})
        agents_list = list_result.get("agents", [])

        registered_names: list[str] = []

        for agent_info in agents_list:
            remote_name = agent_info.get("name", "")
            if not remote_name:
                continue

            # Get full contract details if describe is available
            if describe_skill and describe_skill.callable:
                details = await describe_skill.callable({"name": remote_name})
            else:
                details = agent_info

            # Build a synthetic AgentContract
            contract = _build_contract(
                local_name=f"{prefix}.{remote_name}",
                details=details,
            )

            # Create a bound RemoteAgent class for this agent
            agent_class = _make_remote_agent_class(
                exec_run_skill=exec_run_skill.callable,
                remote_agent_name=remote_name,
            )

            agent_registry.register_virtual(contract, agent_class)
            registered_names.append(f"{prefix}.{remote_name}")

        self._registered[server.name] = registered_names
        logger.info(
            "Registered %d remote agents from '%s'",
            len(registered_names), server.name,
        )
        return len(registered_names)

    def disconnect(
        self,
        name: str,
        agent_registry: "AgentRegistry | None" = None,
    ) -> None:
        """Unregister remote agents for a server."""
        agent_names = self._registered.pop(name, [])
        if agent_registry:
            for agent_name in agent_names:
                agent_registry.unregister(agent_name)
        logger.info("Unregistered %d remote agents from '%s'", len(agent_names), name)

    def disconnect_all(
        self,
        agent_registry: "AgentRegistry | None" = None,
    ) -> None:
        """Unregister all remote agents."""
        names = list(self._registered.keys())
        for name in names:
            self.disconnect(name, agent_registry)

    @property
    def connected_servers(self) -> list[str]:
        """Names of servers with registered agents."""
        return list(self._registered.keys())


def _build_contract(
    local_name: str,
    details: dict[str, Any],
) -> Any:
    """Build an AgentContract from remote describe response."""
    from atlas.contract.types import AgentContract, SchemaSpec

    return AgentContract(
        name=local_name,
        version=details.get("version", "1.0.0"),
        type=details.get("type", "agent"),
        description=details.get("description", f"Remote agent: {local_name}"),
        input_schema=SchemaSpec.from_dict(details.get("input_schema")),
        output_schema=SchemaSpec.from_dict(details.get("output_schema")),
        capabilities=details.get("capabilities", []),
        execution_timeout=float(details.get("execution_timeout", 120.0)),
    )
