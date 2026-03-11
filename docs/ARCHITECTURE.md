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

## Module Map

```
atlas/
├── contract/
│   ├── registry.py      # AgentRegistry — discovery, semver, capability search
│   ├── schema.py        # JSON Schema validation (validate_input, validate_output)
│   └── types.py         # AgentContract, SchemaSpec, ModelSpec, HardwareSpec
├── pool/
│   ├── executor.py      # ExecutionPool — concurrency, warm slots, orchestration
│   ├── job.py           # JobData — job record with status, timing, metadata
│   ├── queue.py         # JobQueue — priority heap, backpressure, persistence
│   └── slot_manager.py  # SlotManager — warm slot lifecycle (create/reuse/evict)
├── chains/
│   ├── definition.py    # ChainDefinition, ChainStep — YAML chain specs
│   ├── runner.py        # ChainRunner — execute chains with mediation
│   └── executor.py      # ChainExecutor — async chain execution with status tracking
├── orchestrator/
│   ├── protocol.py      # Orchestrator protocol + RoutingDecision
│   └── default.py       # DefaultOrchestrator (allow-all)
├── mediation/
│   ├── engine.py        # MediationEngine — strategy cascade
│   ├── strategies.py    # Direct, Mapped, Coerce, LLMBridge strategies
│   └── analyzer.py      # Compatibility analysis between schemas
├── runtime/
│   ├── base.py          # AgentBase — abstract base class for all agents
│   ├── context.py       # AgentContext — runtime context, spawn support
│   ├── runner.py        # run_agent() — standalone agent execution
│   └── llm_agent.py     # LLMAgent — base class for LLM-powered agents
├── llm/
│   ├── provider.py      # LLMProvider protocol + LLMResponse
│   ├── anthropic.py     # AnthropicProvider
│   └── openai.py        # OpenAIProvider
├── store/
│   └── job_store.py     # JobStore — SQLite persistence via aiosqlite
├── cli/
│   ├── app.py           # Typer CLI entry point
│   ├── pool_commands.py # discover, run, serve commands
│   ├── orchestrator_commands.py # orchestrator list/set/reset
│   └── formatting.py    # Table and output formatting
├── events.py            # EventBus — subscriber-isolated pub/sub
├── metrics.py           # MetricsCollector — latency, throughput, warm hits
├── trace.py             # TraceCollector + ExecutionTrace + cost estimation
├── eval.py              # EvalRunner + EvalSubscriber + YAML eval definitions
├── retry.py             # RetrySubscriber — backoff + resubmit
├── serve.py             # aiohttp HTTP server
└── ws.py                # WebSocket event streaming
```
