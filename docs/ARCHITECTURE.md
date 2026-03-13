# Architecture

Deep dive into how Atlas works internally. For the high-level overview and motivation, see the [README](../README.md).

---

## System Overview

```mermaid
graph TB
    subgraph Submission
        CLI[CLI / API]
        Chain[Chain Executor]
    end

    subgraph Core
        Q[JobQueue<br/><i>priority heap + backpressure</i>]
        Pool[ExecutionPool<br/><i>semaphore-bounded concurrency</i>]
        SM[SlotManager<br/><i>warm slot reuse + lifecycle</i>]
        Orch[Orchestrator<br/><i>route / reject / redirect</i>]
    end

    subgraph Agents
        Reg[AgentRegistry<br/><i>discovery + semver resolution</i>]
        A1[echo]
        A2[formatter]
        A3[llm-summarizer]
        AN[your-agent]
    end

    subgraph Observers
        EB[EventBus]
        Metrics[MetricsCollector]
        Traces[TraceCollector]
        Store[JobStore<br/><i>SQLite persistence</i>]
        Eval[EvalSubscriber]
        Retry[RetrySubscriber]
    end

    CLI --> Q
    Chain --> Q
    Q --> Pool
    Pool --> Orch
    Orch --> SM
    SM --> Reg
    Reg --> A1 & A2 & A3 & AN
    Pool -- status change --> EB
    EB --> Metrics & Traces & Store & Eval & Retry
    Retry -- resubmit --> Q
```

---

## Job Lifecycle

Every job follows a deterministic state machine. Status transitions emit events to all EventBus subscribers **before** waiters are notified — guaranteeing that traces, metrics, and store are consistent when `wait_for_terminal` returns.

```mermaid
stateDiagram-v2
    [*] --> pending: submit
    pending --> running: pool dequeues
    running --> completed: execute() returns
    running --> failed: execute() raises
    pending --> cancelled: cancel()
    failed --> pending: retry (new job)
```

### Ordering Guarantee

When a job reaches a terminal state, the queue processes side effects in strict order:

1. **Store save** — persist updated job to SQLite
2. **EventBus emit** — all subscribers process the event (metrics, traces, eval, retry)
3. **Waiter signal** — `wait_for_terminal()` callers are unblocked

This means any code that awaits `wait_for_terminal` is guaranteed that the store, metrics, and traces are already populated.

---

## Execution Pool

### Concurrency Model

The pool uses an `asyncio.Semaphore` to bound concurrent executions. When a job is dequeued:

1. Acquire semaphore (blocks if at `max_concurrent`)
2. Route through orchestrator (allow / reject / redirect)
3. Acquire or create a warm slot via SlotManager
4. Validate input against contract schema
5. Execute agent's `execute()` method
6. Validate output against contract schema
7. Update job status → triggers EventBus cascade
8. Release semaphore

### Warm Slot Reuse

```mermaid
flowchart TD
    Job[Incoming Job] --> Check{Warm slot<br/>available?}
    Check -->|Yes| Reuse[Reuse existing slot<br/><i>warmup_ms = 0</i>]
    Check -->|No| Cold[Create new slot<br/><i>call on_startup</i>]
    Reuse --> Exec[execute]
    Cold --> Exec
    Exec --> Return[Return to warm pool]
    Return --> Idle{Idle timeout<br/>exceeded?}
    Idle -->|Yes| Evict[Call on_shutdown<br/>Destroy slot]
    Idle -->|No| Wait[Wait for next job]
```

Agents call `on_startup()` once when a slot is created (load models, open connections). The slot persists across multiple jobs. After `idle_timeout`, the slot is evicted and `on_shutdown()` is called.

The `warm_pool_size` parameter controls how many slots are kept alive. First job to an agent is a cold start; subsequent jobs reuse the warm slot.

---

## Chain Mediation

When chaining agents with different I/O schemas, the MediationEngine bridges them automatically by trying strategies in order of simplicity:

```mermaid
flowchart TD
    Output[Step N output] --> Direct{Direct?}
    Direct -->|schemas match| Pass[Pass through]
    Direct -->|no| Mapped{Mapped?}
    Mapped -->|input_map defined| Apply[Apply field mapping]
    Mapped -->|no| Coerce{Coercible?}
    Coerce -->|type conversion possible| Convert[Coerce types]
    Coerce -->|no| LLM{LLM provider<br/>available?}
    LLM -->|yes| Bridge[LLM semantic bridge]
    LLM -->|no| Fail[MediationFailed]

    Pass --> Next[Step N+1 input]
    Apply --> Next
    Convert --> Next
    Bridge --> Next
```

### Strategy Details

| Strategy | When Used | Example |
|---|---|---|
| **Direct** | Output schema is a superset of input schema | `{text, score}` → `{text}` |
| **Mapped** | Chain definition includes `input_map` | `content: summary` maps field names |
| **Coerce** | Types differ but are convertible | `"42"` → `42`, scalar → `{value: scalar}` |
| **LLM Bridge** | Schemas are semantically related but structurally incompatible | Free-text summary → structured JSON |

### Compatibility Analysis

Before running a chain, you can analyze compatibility between steps:

```python
from atlas.mediation.analyzer import analyze_compatibility

compat = analyze_compatibility(agent_a.output_schema, agent_b.input_schema)
# Returns: compatible (bool), strategy (str), field_mapping (dict), warnings (list)
```

---

## Orchestrator Pipeline

Orchestrators sit between the queue and execution. They implement a simple protocol:

```python
class Orchestrator(Protocol):
    async def route(self, job: JobData, registry: AgentRegistry) -> RoutingDecision: ...
    async def on_job_complete(self, job: JobData) -> None: ...
    async def on_job_failed(self, job: JobData) -> None: ...
```

### Routing Decisions

| Action | Effect |
|---|---|
| `allow` | Execute as-is |
| `reject` | Fail the job immediately with a reason |
| `redirect` | Change the target agent (e.g., echo → formatter) |
| `allow` + `priority` | Override the job's priority |

### Hot-Swap

Orchestrators can be swapped at runtime via `pool.set_orchestrator()` or the `POST /api/orchestrator` endpoint. The new orchestrator takes effect on the next job dequeued — no restart, no downtime.

---

## EventBus

The EventBus is a simple pub/sub system for job status transitions. Subscribers are async callables:

```python
async def callback(job: JobData, old_status: str, new_status: str) -> None: ...
```

### Subscriber Isolation

- Each subscriber runs independently — a failure in one does not affect others
- Exceptions are logged with full tracebacks but swallowed
- Subscribers are called sequentially per event (not parallel)

### Built-in Subscribers

| Subscriber | Purpose |
|---|---|
| `MetricsCollector` | Latency percentiles, warm hit rate, status counts per agent |
| `TraceCollector` | Per-job execution traces with token counts and cost estimates |
| `JobStore` | Persistence to SQLite (via queue) |
| `EvalSubscriber` | Runs YAML eval checks on completed jobs, attaches results to traces |
| `RetrySubscriber` | Resubmits failed jobs based on agent retry config |

---

## Agent Registry

### Discovery

The registry scans directories for `agent.yaml` files, validates contracts, and loads agent implementations:

```
agents/
├── echo/
│   ├── agent.yaml    # contract (name, schemas, capabilities)
│   └── agent.py      # implementation (class Agent(AgentBase))
├── formatter/
│   ├── agent.yaml
│   └── agent.py
└── summarizer/
    ├── agent.yaml
    ├── agent.py
    └── eval.yaml     # optional eval checks
```

### Semver Resolution

Agents are versioned. The registry supports semver range queries:

```python
registry.get("summarizer", "^1.0.0")  # latest 1.x.x
registry.get("summarizer", "~1.2.0")  # latest 1.2.x
registry.get("summarizer")            # latest version
```

### Capability Search

Agents declare capabilities in their contract. The registry supports capability-based lookup:

```python
agents = registry.search("text-processing")  # all agents with this capability
```

---

## LLM Provider Abstraction

Atlas abstracts LLM providers behind a common interface:

```python
class LLMProvider(Protocol):
    async def complete(self, prompt: str, **kwargs) -> LLMResponse: ...
```

Built-in providers:
- `AnthropicProvider` — Claude models via the Anthropic SDK
- `OpenAIProvider` — GPT models via the OpenAI SDK
- `LangChainProvider` — any LangChain-compatible model

Token counts and model name flow back through `LLMResponse` → `AgentContext.execution_metadata` → job metadata → `ExecutionTrace`.

---

## Triggers & Scheduling

The trigger system submits jobs to the pool on a schedule or in response to events.

### Trigger Types

| Type | Fires When | Schedule Field |
|---|---|---|
| `cron` | Cron expression matches | `cron_expr` (5-field) |
| `interval` | Every N seconds | `interval_seconds` |
| `one_shot` | Once at a specific time, then disables | `fire_at` (unix timestamp) |
| `webhook` | HTTP request hits `/api/hooks/{id}` | N/A (event-driven) |

### Scheduler

The `TriggerScheduler` runs as an async background task, polling the `TriggerStore` every `poll_interval` seconds for due triggers. When a trigger fires:

1. Create a `JobData` from the trigger's `agent_name`, `input_data`, and `priority`
2. Submit to the pool via `pool.submit()`
3. Update trigger state: `last_fired`, `fire_count`, `last_job_id`
4. Compute `next_fire` for recurring triggers; disable one-shot triggers
5. Save updated trigger to store

Webhook triggers bypass the polling loop — they fire immediately via `fire_webhook()` when an HTTP request arrives.

### Webhook Security

Webhook triggers support optional HMAC-SHA256 signature validation. If `webhook_secret` is set, the endpoint validates the `X-Atlas-Signature` header against the request body before firing.

---

## MCP Federation

Atlas instances communicate via the [Model Context Protocol](https://modelcontextprotocol.io). Federation has three layers:

### Layer 1: MCP Server (Phase 10A)

Every Atlas instance can expose its skills as MCP tools over HTTP:

```
atlas serve --mcp-port 8400 --auth-token secret
```

The MCP server uses Streamable HTTP transport with optional bearer token auth. The `/health` endpoint is always open. SSE transport is supported for legacy clients.

### Layer 2: Remote Tool Federation (Phase 10B)

`RemoteToolProvider` connects to a remote MCP server, discovers its tools, and registers them as local skills with a namespace prefix:

```
atlas serve --remote "lab=http://host:8400/mcp@secret"
```

Remote tools appear as `lab.tool-name` in the local skill registry. Agents declare them as dependencies via `requires.skills: ["lab.tool-name"]` and call them via `context.skill()`.

### Layer 3: Federated Chains (Phase 10C)

`RemoteAgentProvider` discovers remote agents (via `atlas.registry.list` / `atlas.registry.describe`) and registers them as virtual agents in the local `AgentRegistry`. Each virtual agent's `execute()` calls `atlas.exec.run` on the remote instance.

Chains reference remote agents directly — no wrapper code needed:

```yaml
chain:
  name: cross-instance
  steps:
    - agent: lab.translator    # executes on remote instance
    - agent: local-formatter   # executes locally
```

The `ChainRunner` resolves and injects skills for each step via an optional `SkillResolver`, matching the same injection path used by the `ExecutionPool`.

---

## Skills & Platform Tools

### Skill System

Skills are named async callables with typed I/O schemas. Agents declare dependencies via `requires.skills` in their contract, and the runtime injects them at execution time.

```
Agent contract (requires.skills: ["embedder"]) → SkillResolver → SkillRegistry → callable injected into AgentContext._skills
```

### Platform Tools (12 tools)

| Tool | Description |
|---|---|
| `atlas.registry.list` | List registered agents |
| `atlas.registry.describe` | Describe an agent's contract |
| `atlas.registry.search` | Search agents by capability |
| `atlas.exec.run` | Execute an agent synchronously (federation primitive) |
| `atlas.exec.spawn` | Submit a job to the pool |
| `atlas.exec.status` | Get a job's status |
| `atlas.exec.cancel` | Cancel a pending job |
| `atlas.queue.inspect` | Inspect the job queue |
| `atlas.monitor.health` | Pool health stats |
| `atlas.monitor.metrics` | Per-agent metrics |
| `atlas.monitor.trace` | Get a single trace |
| `atlas.monitor.traces` | List execution traces |

Agents opt in via `requires.platform_tools: true`.

---

## Agent Spawning & Fan-Out

Agents can decompose work by spawning child agents during execution. The decomposer pattern fans out input across multiple child jobs, then collects results.

### Spawn Flow

```mermaid
sequenceDiagram
    participant Parent as Parent Agent
    participant Ctx as AgentContext
    participant Q as JobQueue
    participant Pool as ExecutionPool
    participant Child as Child Agent

    Parent->>Ctx: spawn("echo", {msg})
    Ctx->>Ctx: Check spawn_allowed + depth < max_depth
    Ctx->>Q: submit(child JobData, depth+1)
    Q->>Pool: dequeue child
    Pool->>Child: execute()
    Child-->>Pool: result
    Pool-->>Q: status = completed
    Q-->>Ctx: wait_for_terminal returns
    Ctx-->>Parent: SpawnResult(success, data)
```

### Guards

- **Permission** — only agents with `requires.spawn_agents: true` in their contract can call `context.spawn()`. Enforced in `AgentContext.spawn()`.
- **Depth limit** — default max depth of 3, configurable via `AgentContext.max_depth`. Each child increments `_spawn_depth` in metadata.
- **Queue coordination** — children flow through the same `JobQueue` and `ExecutionPool` as top-level jobs, bounded by the same `max_concurrent` semaphore.

### Spawn Callback Injection

The `ExecutionPool` injects a spawn callback into the `AgentContext` before execution. This callback:

1. Creates a child `JobData` with incremented depth and parent trace ID in metadata
2. Submits it to the queue
3. Blocks on `queue.wait_for_terminal()` until the child reaches a terminal state
4. Returns `SpawnResult(success=True/False, data=..., error=...)`

Children execute in parallel (bounded by pool concurrency), but each parent waits for its spawned children sequentially.

---

## Dynamic Agent Providers

Atlas supports three provider types — all discovered, registered, and executed identically through the same pool and registry.

### Provider Types

| Provider | Implementation | Use Case |
|---|---|---|
| `python` (default) | `AgentBase` subclass in `agent.py` | Full Python control |
| `exec` | External process, JSON on stdin/stdout | Any language (Rust, Go, Node, shell) |
| `llm` | Pure YAML, no code | LLM agents with system prompt + tools |

### Exec Provider (`runtime/exec_agent.py`)

Runs any executable as an agent. The runtime sends a JSON envelope on stdin:

```json
{"input": {...}, "context": {...}, "memory": "..."}
```

The agent process writes JSON to stdout. Memory writes return via `_memory_append` key. Knowledge writes via `_knowledge_store` key.

### LLM Provider (`runtime/dynamic_llm_agent.py`)

Defines LLM agents in pure YAML — system prompt, model preference, output format. Skills declared in `requires.skills` are automatically exposed as tools. The runtime handles the tool-use loop internally.

---

## Hardware Scheduling

Agents declare hardware requirements in their contracts. The pool tracks a hardware inventory and gates job execution on resource availability.

### Allocation Flow

```mermaid
flowchart TD
    Job[Job submitted] --> Check{can_satisfy<br/>HardwareSpec?}
    Check -->|Yes| Alloc[allocate slot_id, spec<br/>GPUs + memory + CPU reserved]
    Check -->|No| Fail[ResourceUnavailable]
    Alloc --> Exec[Agent executes]
    Exec --> Release[release slot_id<br/>resources returned to pool]
```

### HardwareInventory

Tracks total and free resources: GPUs (with per-GPU VRAM), system memory, CPU cores, architecture, and available devices. Allocation is per-slot — resources are reserved when a slot is acquired and released when the slot is returned or destroyed.

### Constraint Checks

| Constraint | Check |
|---|---|
| GPU count | `free_gpus >= 1` when `gpu: true` |
| VRAM | At least one free GPU with `vram_gb >= gpu_vram_gb` |
| Memory | `free_memory_gb >= min_memory_gb` |
| CPU cores | `free_cpu_cores >= min_cpu_cores` |
| Architecture | `architecture` matches or is `"any"` |
| Device access | All `device_access` entries in `available_devices` |

---

## Knowledge & Memory

Two orthogonal systems for agent learning:

### Shared Memory

"What happened this session" — agents opt in with `requires.memory: true`. All participating agents share a memory pool that persists across executions.

- **FileMemoryProvider** — local markdown file (`memory.md`)
- **HttpMemoryProvider** — REST hook for external systems (Redis, vector DB)

For `exec` agents, memory arrives in the stdin envelope. For `llm` agents, memory is injected into the system prompt.

### Knowledge Base

"What do we know about X" — structured knowledge scoped by domain with per-agent ACLs.

- **FileKnowledgeProvider** — markdown files with YAML frontmatter, organized by domain subdirectories
- **HttpKnowledgeProvider** — REST hook for external knowledge systems
- **MCPKnowledgeProvider** — delegates to an MCP server (e.g., Kronos vault)

### Access Control

Agents declare `read_domains` and `write_domains` in their contract. Protected domains block wildcard writes — an agent with `write_domains: ["*"]` can't write to a protected domain unless explicitly listed.

---

## Agent Marketplace

Package, publish, and pull agents across registries. Two pluggable providers:

- **FileRegistryProvider** — directory-based (manifest.json + package.tar.gz per version)
- **HttpRegistryProvider** — REST client for remote Atlas instances

Agents declare dependencies on other agents via `requires.agents` with optional semver ranges. Dependencies are checked at job submission — missing agents produce clear errors with install hints.

---

## Module Map

```
atlas/
├── contract/
│   ├── registry.py        # AgentRegistry — discovery, semver, virtual agents
│   ├── schema.py          # JSON Schema validation (validate_input, validate_output)
│   ├── types.py           # AgentContract, SchemaSpec, HardwareSpec, RequiresSpec
│   └── permissions.py     # PermissionsSpec — file, network, subprocess, env scopes
├── pool/
│   ├── executor.py        # ExecutionPool — concurrency, warm slots, spawn, skill injection
│   ├── job.py             # JobData — job record with status, timing, metadata
│   ├── queue.py           # JobQueue — priority heap, backpressure, persistence
│   ├── slot_manager.py    # SlotManager — warm slot lifecycle (create/reuse/evict)
│   └── hardware.py        # HardwareInventory — GPU/memory/CPU tracking + allocation
├── chains/
│   ├── definition.py      # ChainDefinition, ChainStep — YAML chain specs
│   ├── runner.py          # ChainRunner — mediation + optional skill injection
│   └── executor.py        # ChainExecutor — async chain execution with status tracking
├── orchestrator/
│   ├── protocol.py        # Orchestrator protocol + RoutingDecision
│   └── default.py         # DefaultOrchestrator (allow-all)
├── mediation/
│   ├── engine.py          # MediationEngine — strategy cascade
│   ├── strategies.py      # Direct, Mapped, Coerce, LLMBridge strategies
│   └── analyzer.py        # Compatibility analysis between schemas
├── runtime/
│   ├── base.py            # AgentBase — abstract base class for all agents
│   ├── context.py         # AgentContext — spawn, skills, memory, knowledge, chain data
│   ├── runner.py          # run_agent() — standalone agent execution
│   ├── llm_agent.py       # LLMAgent — base class for LLM-powered agents
│   ├── dynamic_llm_agent.py # DynamicLLMAgent — YAML-only LLM agents (provider: llm)
│   └── exec_agent.py      # ExecAgent — external process agents (provider: exec)
├── skills/
│   ├── registry.py        # SkillRegistry — discovery + RegisteredSkill entries
│   ├── resolver.py        # SkillResolver — resolve skill names to callables
│   ├── platform.py        # PlatformToolProvider — 12 atlas.* platform tools
│   ├── schema.py          # YAML loading + validation for skill.yaml
│   └── types.py           # SkillSpec, SkillCallable, SkillError
├── mcp/
│   ├── server.py          # create_mcp_server() — wraps SkillRegistry as MCP tools
│   ├── transport.py       # ASGI app — Streamable HTTP + SSE + health endpoint
│   ├── auth.py            # BearerAuthMiddleware — timing-safe token validation
│   ├── client.py          # RemoteToolProvider — connect to remote MCP, register skills
│   ├── remote_agents.py   # RemoteAgentProvider — virtual agents for federation
│   └── stdio.py           # stdio transport for MCP
├── security/
│   ├── policy.py          # SecurityPolicy — YAML-defined permission + secret rules
│   ├── protocol.py        # SecurityProvider protocol
│   ├── container.py       # ContainerSlot — Docker container isolation for agents
│   └── secrets.py         # SecretResolver, EnvSecretProvider, FileSecretProvider
├── knowledge/
│   ├── provider.py        # KnowledgeProvider protocol + KnowledgeEntry dataclass
│   ├── acl.py             # KnowledgeACL — domain-scoped read/write access control
│   ├── file_provider.py   # FileKnowledgeProvider — markdown + YAML frontmatter
│   ├── http_provider.py   # HttpKnowledgeProvider — REST hook
│   └── mcp_provider.py    # MCPKnowledgeProvider — delegates to MCP server
├── memory/
│   ├── provider.py        # MemoryProvider protocol
│   ├── file_provider.py   # FileMemoryProvider — local markdown file
│   └── http_provider.py   # HttpMemoryProvider — REST hook for external systems
├── registry/
│   ├── provider.py        # RegistryProvider protocol
│   ├── config.py          # Registry configuration and CLI integration
│   ├── resolver.py        # Dependency resolution for agent requirements
│   ├── package.py         # Agent packaging (tar.gz + manifest)
│   ├── file_provider.py   # FileRegistryProvider — directory-based marketplace
│   └── http_provider.py   # HttpRegistryProvider — REST client for remote registries
├── llm/
│   ├── provider.py        # LLMProvider protocol + LLMResponse
│   ├── anthropic.py       # AnthropicProvider
│   └── openai.py          # OpenAIProvider
├── triggers/
│   ├── models.py          # TriggerDefinition — cron, interval, one_shot, webhook
│   ├── cron.py            # CronExpr — lightweight 5-field cron parser
│   ├── scheduler.py       # TriggerScheduler — async tick loop, fires due triggers
│   └── routes.py          # HTTP routes for trigger CRUD + webhook endpoint
├── store/
│   ├── job_store.py       # JobStore — SQLite persistence via aiosqlite
│   └── trigger_store.py   # TriggerStore — SQLite persistence for triggers
├── cli/
│   ├── app.py             # Typer CLI — run, serve, mcp, list, inspect, validate
│   ├── pool_commands.py   # discover, run, serve commands
│   ├── registry_commands.py # registry add/publish/pull/search
│   ├── orchestrator_commands.py # orchestrator list/set/reset
│   ├── trigger_commands.py # trigger create/list/get/delete/enable/disable
│   ├── security_commands.py # security policy validation
│   ├── skill_commands.py  # skill list/inspect
│   └── formatting.py     # Table and output formatting
├── app_keys.py            # aiohttp AppKey definitions for typed app state
├── constants.py           # Shared constants
├── logging.py             # Structured logging configuration
├── events.py              # EventBus — subscriber-isolated pub/sub
├── metrics.py             # MetricsCollector — latency, throughput, warm hits
├── trace.py               # TraceCollector + ExecutionTrace + cost estimation
├── eval.py                # EvalRunner + EvalSubscriber + YAML eval definitions
├── retry.py               # RetrySubscriber — backoff + resubmit
├── serve.py               # aiohttp HTTP server
└── ws.py                  # WebSocket event streaming
```
