# Atlas — Agent Runtime, Registry # Atlas — Agent Daemon & Registry Composition Engine

### An open-source runtime, registry, and composition engine for AI agents

> **Status:** Draft v0.1 — Foundational Specification  
> **Author:** Peter  
> **License:** TBD (open source)

---

## 1. What Is Atlas?

Atlas is a self-hostable daemon that acts as both a **runtime environment** and a **registry** for AI agents. It provides pooled execution, automatic chaining, intelligent orchestration, and a standardized agent contract — so that agents can be built once, shared freely, composed together, and consumed by anyone.

Think of it as what Docker did for applications, but for AI agents: a portable, composable, discoverable unit of **capability** rather than a unit of deployment.

### Core Principles

- **You own the runtime.** Self-host anywhere — your laptop, a homelab, a cloud VM, an enterprise cluster. Atlas doesn't phone home.
- **Agents are the unit.** Every agent conforms to one contract. If it conforms, it runs. No proprietary SDK, no vendor lock-in.
- **Composition is native.** Chaining agents, spawning sub-agents, and orchestrating workflows are first-class primitives — not bolted-on afterthoughts.
- **Entry points are decoupled.** A chat message, a cron job, a webhook, a monitoring alert, an API call — they're all just triggers. The runtime doesn't care where the signal came from.
- **Security is layered outside.** Auth, networking, and access control wrap the runtime. Agent developers don't think about infra. Infra operators don't think about agent internals.
- **Everything is MCP-native.** The platform exposes its own capabilities as MCP tools internally. Agents talk to the platform the same way external consumers talk to agents. One protocol, all the way down.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Atlas Control Plane                      │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌────────┐  │
│  │ Registry │  │ Orchestrator │  │ Execution │  │  Node  │   │
│  │          │  │ (pluggable)  │  │   Pool    │  │Sched.  │   │
│  │• Disc.   │  │              │  │           │  │        │   │
│  │• Vers.   │  │• Routing     │  │• Job Queue│  │• Place- │  │
│  │• Deps    │  │• Model Sel.  │  │• Lifecycle│  │  ment  │   │
│  │• Contr.  │  │• Mediation   │  │• Spawning │  │• Affin.│   │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  └───┬────┘  │
│       │               │                │             │       │
│  ┌────┴───────────────┴────────────────┴─────────────┴────┐  │
│  │                 Internal MCP Surface                    │  │
│  │    (all platform capabilities exposed as MCP tools)     │  │
│  └──────────────────────┬─────────────────────────────────┘  │
│                          │                                    │
│  ┌──────────────────────┴─────────────────────────────────┐  │
│  │            Monitoring & Evaluation Engine                │  │
│  │   • Execution traces • Cost tracking • Eval hooks       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                      Security Layer                          │
│            (auth, networking, access control)                 │
├──────────────────────────────────────────────────────────────┤
│                    External Interface                         │
│               MCP / HTTP / WebSocket / gRPC                   │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
  ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
  │  GPU Node │  │ CPU Pool  │  │ Edge Node │      ◄── Compute Nodes
  │  (RTX x2) │  │ (32 core) │  │ (Jetson)  │
  └───────────┘  └───────────┘  └───────────┘

        ▲           ▲            ▲           ▲
        │           │            │           │         ◄── Triggers
     Chat App    Cron Job     Webhook    API Client
```

---

## 3. Agent Contract

The agent contract is the foundational primitive. Every agent in the system — whether it's a simple tool wrapper or a complex multi-step reasoning chain — conforms to this interface.

### 3.1 Registration Schema

```yaml
# agent.yaml — the minimum viable agent definition
agent:
  name: "summarizer"
  version: "1.2.0"
  description: "Summarizes text input to a target length"

  # What this agent accepts
  input:
    schema:
      type: object
      properties:
        text:
          type: string
          description: "The text to summarize"
        max_length:
          type: integer
          default: 200
      required: [text]

  # What this agent returns
  output:
    schema:
      type: object
      properties:
        summary:
          type: string
        token_count:
          type: integer

  # Capabilities this agent declares
  capabilities:
    - text-processing
    - summarization

  # What this agent needs from the platform
  requires:
    platform_tools: false    # Does NOT need access to internal MCP surface
    spawn_agents: false      # Does NOT need to spawn sub-agents
    skills: []               # No platform skills required

  # Model preferences (overridable by orchestrator)
  model:
    preference: "fast"       # fast | balanced | powerful
    override_allowed: true   # Orchestrator can swap models

  # Hardware requirements (used by scheduler for node placement)
  hardware:
    gpu: false               # Requires GPU
    gpu_vram_gb: 0           # Minimum VRAM if GPU required
    min_memory_gb: 1         # Minimum system RAM
    min_cpu_cores: 1         # Minimum CPU cores
    architecture: "any"      # any | x86_64 | arm64
    node_affinity: ""        # Pin to a specific named node (optional)
    device_access: []        # Specific device paths (e.g., ["/dev/dri"])

  # Dependencies (external frameworks, packages)
  dependencies:
    python: ">=3.11"
    packages:
      - "langchain>=0.2.0"

  # Lifecycle hooks
  lifecycle:
    health: "/health"        # Health check endpoint
    startup: "on_startup"    # Function called on agent initialization
    shutdown: "on_shutdown"  # Function called on teardown
```

### 3.2 The Agent Wrapper

No SDK required. You wrap your agent in a class that implements the contract interface. If your agent uses a custom framework, that framework is declared as a dependency and pulled in during the build process.

```python
# Example: Minimal agent implementation
from atlas import AgentBase  # Thin base class, not a framework

class SummarizerAgent(AgentBase):
    """
    AgentBase provides:
    - Input/output validation against your declared schema
    - Lifecycle hook wiring
    - Optional access to platform MCP tools (if requires.platform_tools = true)
    
    You implement: execute()
    That's it.
    """

    async def execute(self, input: dict, context: AgentContext) -> dict:
        text = input["text"]
        max_length = input.get("max_length", 200)

        # Your logic — use any framework, any library, anything
        summary = await self.summarize(text, max_length)

        return {
            "summary": summary,
            "token_count": len(summary.split())
        }

    async def on_startup(self):
        # Load models, warm caches, etc.
        pass

    async def on_shutdown(self):
        # Cleanup
        pass
```

### 3.3 Remote Registration

Agents don't have to be co-located with the daemon. You can remotely register an agent running anywhere, as long as it exposes the contract interface over HTTP/MCP:

```bash
# Register a remote agent
atlas register --remote https://my-server.com/agents/summarizer --name summarizer

# Register a local agent from source
atlas register --path ./agents/summarizer/ --name summarizer

# Register from a public registry
atlas pull community/summarizer:1.2.0
```

---

## 4. The Orchestrator

The orchestrator is the daemon's brain — but it's **not** a fixed component. It's a pluggable agent that occupies a special slot in the runtime.

### 4.1 Default Orchestrator

Atlas ships with a default orchestrator that handles:

- **Routing:** Inspects the trigger input and determines which agent (or chain) should handle it.
- **Model Selection:** Profiles the task and routes to the appropriate model tier based on complexity, cost constraints, and latency requirements.
- **Chain Mediation:** When agent A's output doesn't perfectly match agent B's expected input, the orchestrator mediates the transformation.
- **Error Recovery:** Handles agent failures, retries, and fallback paths.

### 4.2 Pluggable Override

The orchestrator conforms to the same agent contract as everything else — it just implements the `OrchestratorInterface`:

```yaml
# custom-orchestrator.yaml
agent:
  name: "research-orchestrator"
  version: "1.0.0"
  type: orchestrator  # Special type flag

  # Orchestrators declare what strategies they implement
  orchestration:
    strategies:
      - chain_execution
      - parallel_fanout
      - conditional_routing
      - recursive_decomposition

  # Model selection configuration
  model_selection:
    enabled: true
    tiers:
      fast: ["claude-haiku", "gpt-4o-mini"]
      balanced: ["claude-sonnet"]
      powerful: ["claude-opus", "gpt-4o"]
    routing_strategy: "task_complexity"  # or "cost_optimized", "latency_optimized"
```

Swap it in:

```bash
# Replace the default orchestrator
atlas orchestrator set research-orchestrator

# Or scope it to specific chains
atlas orchestrator set research-orchestrator --chain "research-pipeline"

# Reset to default
atlas orchestrator reset
```

Because orchestrators are agents, they're **shareable through the registry.** Someone builds a killer orchestrator for financial analysis pipelines? Publish it. Others pull it in.

---

## 5. Execution Model

Atlas supports three execution modes natively. The runtime manages the pool, the queue, and the lifecycle for all of them.

### 5.1 Agent Chains (Declarative Pipelines)

Define a sequence of agents with data flow between them. The runtime (via the orchestrator) handles execution, mediation, and error recovery.

```yaml
# chains/research-pipeline.yaml
chain:
  name: "research-pipeline"
  description: "End-to-end research workflow"

  steps:
    - agent: "web-searcher"
      input_map:
        query: "$.trigger.query"

    - agent: "summarizer"
      input_map:
        text: "$.steps[0].output.results"
        max_length: 500

    - agent: "report-writer"
      input_map:
        summary: "$.steps[1].output.summary"
        format: "$.trigger.format"

  # Error handling
  on_failure:
    strategy: "retry_then_skip"  # retry_then_skip | halt | fallback
    max_retries: 2
```

### 5.2 On-Demand Agent Calls

Fire a single agent independently — no chain, no pipeline. Useful for simple tasks or when an external system needs a one-shot capability.

```bash
# CLI
atlas run summarizer --input '{"text": "...", "max_length": 100}'

# HTTP
POST /api/v1/agents/summarizer/run
{
  "text": "...",
  "max_length": 100
}
```

### 5.3 Recursive Spawning

Agents can spawn other agents during execution. This enables dynamic, adaptive workflows where the execution graph isn't known ahead of time.

```python
class ResearchAgent(AgentBase):
    async def execute(self, input: dict, context: AgentContext) -> dict:
        # Spawn sub-agents dynamically
        search_result = await context.spawn("web-searcher", {
            "query": input["topic"]
        })

        # Conditionally spawn more based on results
        if search_result["needs_deeper_analysis"]:
            analysis = await context.spawn("deep-analyzer", {
                "data": search_result["raw_data"]
            })
            return analysis

        return search_result
```

### 5.4 Guardrails

Recursive spawning requires safety mechanisms built into the runtime from day one:

| Guardrail | Default | Configurable |
|-----------|---------|-------------|
| Max spawn depth | 5 | Yes |
| Max concurrent agents | 20 | Yes |
| Per-chain timeout | 300s | Yes |
| Per-agent timeout | 60s | Yes |
| Resource budget (tokens) | 100k | Yes |
| Circuit breaker threshold | 3 failures | Yes |

```yaml
# atlas.config.yaml — runtime guardrails
execution:
  max_spawn_depth: 5
  max_concurrent_agents: 20
  timeouts:
    chain: 300
    agent: 60
  resource_budget:
    max_tokens_per_chain: 100000
  circuit_breaker:
    failure_threshold: 3
    recovery_timeout: 30
```

---

## 6. Hardware-Aware Scheduling

Atlas treats the daemon as a **control plane**, not a single process on a single machine. Agents are workloads, and workloads get scheduled onto **nodes** based on their declared hardware requirements. The agent developer declares what they need. The platform figures out where to run it.

### 6.1 Node Registration

Any machine can join the Atlas cluster as a compute node. Each node advertises its capabilities:

```bash
# Register a GPU node
atlas node join --name "gpu-box" --advertise

# Register a lightweight CPU node
atlas node join --name "cpu-worker-1" --advertise
```

Nodes are auto-profiled on join:

```yaml
# Auto-generated node profile
node:
  name: "gpu-box"
  address: "10.0.1.50:9090"
  status: "ready"
  resources:
    cpu_cores: 16
    memory_gb: 64
    gpus:
      - device: "nvidia-rtx-4090"
        vram_gb: 24
        index: 0
      - device: "nvidia-rtx-4090"
        vram_gb: 24
        index: 1
    architecture: "x86_64"
    devices: ["/dev/dri", "/dev/nvidia0", "/dev/nvidia1"]
  labels:
    zone: "homelab"
    tier: "high-compute"
```

### 6.2 Scheduling Rules

The execution pool uses hardware declarations from the agent contract to match agents to nodes:

```
Agent: "3d-modeler"                  Agent: "summarizer"
  hardware:                            hardware:
    gpu: true           ──────►          gpu: false         ──────►
    gpu_vram_gb: 16     Scheduled        min_memory_gb: 1   Scheduled
    min_memory_gb: 32   to gpu-box       min_cpu_cores: 1   to cpu-worker-1
```

Scheduling follows a priority order: explicit node affinity first (if the agent is pinned to a named node), then hardware constraints (GPU, VRAM, memory, CPU, architecture), then resource availability (which eligible node has the most headroom), and finally locality preference (prefer co-locating agents in the same chain to reduce network hops).

### 6.3 Node Affinity & Pinning

For specialized hardware — a machine with a specific GPU, a device with sensor access, an edge node with low-latency requirements — agents can be pinned:

```yaml
# This agent MUST run on the GPU box
hardware:
  node_affinity: "gpu-box"

# This agent can run on any node labeled "high-compute"
hardware:
  node_labels:
    tier: "high-compute"

# This agent needs direct device access
hardware:
  device_access: ["/dev/video0"]  # Camera input
  node_affinity: "edge-node-1"
```

### 6.4 Distributed Topology

A single Atlas control plane can manage a heterogeneous fleet:

```
                    ┌──────────────────────┐
                    │   Atlas Control     │
                    │      Plane           │
                    │  (orchestrator,      │
                    │   registry, queue)   │
                    └──────┬───────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
   ┌────────▼──────┐ ┌────▼───────┐ ┌────▼──────────┐
   │   gpu-box     │ │ cpu-pool   │ │  edge-node    │
   │               │ │            │ │               │
   │ • 3d-modeler  │ │ • summar.  │ │ • sensor-     │
   │ • vision      │ │ • writer   │ │   reader      │
   │ • training    │ │ • search   │ │ • local-llm   │
   │               │ │ • eval     │ │               │
   │ RTX 4090 x2   │ │ 32 cores   │ │ Jetson Orin   │
   └───────────────┘ └────────────┘ └───────────────┘
```

All nodes participate in the same pool, draw from the same registry, and are orchestrated as one system. An agent chain can span nodes transparently — step 1 runs on the GPU box, step 2 runs on a CPU node, step 3 runs on the edge device. The control plane handles data transfer between nodes.

### 6.5 Resource Budgets & Scheduling Config

```yaml
# atlas.config.yaml — scheduling configuration
scheduling:
  strategy: "bin_pack"           # bin_pack | spread | affinity_first
  preemption_enabled: false      # Can high-priority jobs evict running jobs?
  resource_reservation:
    gpu_headroom_pct: 10         # Keep 10% GPU capacity free
    memory_headroom_pct: 15      # Keep 15% memory free

  node_failure:
    detection_timeout: 30        # Seconds before marking node as unhealthy
    reschedule_strategy: "retry" # retry | fail | queue
```

---

## 7. Internal MCP Surface

The daemon exposes all of its own capabilities as MCP tools. Any agent with `requires.platform_tools: true` can access them.

### 7.1 Available Internal Tools

```
# Registry tools
atlas.registry.list          — List available agents
atlas.registry.describe      — Get agent contract details
atlas.registry.search        — Search agents by capability

# Execution tools
atlas.exec.spawn             — Spawn an agent
atlas.exec.spawn_chain       — Execute a defined chain
atlas.exec.status            — Check job status
atlas.exec.cancel            — Cancel a running job

# Queue tools
atlas.queue.inspect          — View current queue state
atlas.queue.priority         — Adjust job priority
atlas.queue.drain            — Drain the queue gracefully

# Node & scheduling tools
atlas.nodes.list             — List registered compute nodes
atlas.nodes.status           — Get node resource utilization
atlas.nodes.drain            — Drain a node (reschedule its agents)
atlas.schedule.explain       — Explain why an agent was placed on a node
atlas.schedule.migrate       — Move a running agent to a different node

# Monitoring tools
atlas.monitor.metrics        — Get execution metrics
atlas.monitor.trace          — Get execution trace for a job
atlas.monitor.health         — Platform health check

# Skill tools
atlas.skills.list            — List available skills
atlas.skills.invoke          — Invoke a platform skill
```

### 7.2 Skills

Skills are reusable capability packages (think: file operations, web search, code execution, database access) that the platform routes to agents natively. If an agent declares a skill requirement, the platform injects it automatically.

```yaml
# In agent.yaml
requires:
  skills:
    - web-search
    - file-ops
    - code-execution
```

The agent doesn't import these, configure them, or manage connections. The platform handles it. The agent just calls `context.skill("web-search", {...})`.

---

## 8. Monitoring & Evaluation

Native to the runtime, not bolted on. Every execution is observable.

### 8.1 Execution Traces

Every agent call, chain execution, and spawn event produces a structured trace:

```json
{
  "trace_id": "tr_abc123",
  "chain": "research-pipeline",
  "started_at": "2025-03-10T14:00:00Z",
  "completed_at": "2025-03-10T14:00:12Z",
  "status": "completed",
  "steps": [
    {
      "agent": "web-searcher",
      "model_used": "claude-haiku",
      "tokens_in": 150,
      "tokens_out": 800,
      "latency_ms": 2300,
      "status": "completed"
    },
    {
      "agent": "summarizer",
      "model_used": "claude-sonnet",
      "tokens_in": 900,
      "tokens_out": 200,
      "latency_ms": 1800,
      "status": "completed"
    }
  ],
  "total_tokens": 2050,
  "total_cost_usd": 0.0043
}
```

### 8.2 Evaluation Hooks

Plug in evaluation functions that run automatically against agent outputs. This is the Rubric pattern generalized — eval as a platform primitive.

```yaml
# evals/quality-check.yaml
eval:
  name: "summary-quality"
  target_agent: "summarizer"
  trigger: "every_execution"  # or "sampled", "manual"

  checks:
    - name: "length_compliance"
      type: "assertion"
      condition: "output.token_count <= input.max_length"

    - name: "relevance_score"
      type: "llm_judge"
      model: "claude-haiku"
      prompt: "Rate 1-5 how well this summary captures the key points..."
      threshold: 3.5
```

---

## 9. Entry Points & Triggers

Atlas doesn't care what initiates an execution. The trigger interface is standardized:

```yaml
# triggers/cron-nightly.yaml
trigger:
  type: cron
  schedule: "0 2 * * *"
  target:
    chain: "nightly-analysis"
  input:
    scope: "last_24h"
```

```yaml
# triggers/webhook.yaml
trigger:
  type: webhook
  path: "/hooks/deploy-alert"
  target:
    agent: "deploy-responder"
  input_map:
    event: "$.body"
```

```yaml
# triggers/chat.yaml
trigger:
  type: conversational
  interface: "http"  # or "discord", "slack", "websocket"
  target:
    agent: "front-door"
  input_map:
    message: "$.body.message"
    user_id: "$.body.user_id"
```

All triggers produce the same internal event structure. The runtime routes it to the target agent or chain. The agent never knows (or cares) how it was invoked.

---

## 10. Registry & Distribution

### 10.1 Self-Hosted Registry

Every Atlas instance is its own registry. Agents registered locally are immediately available to all chains and consumers on that instance.

### 10.2 Public Registry

Atlas supports pushing and pulling agents from public registries (think Docker Hub, but for agents):

```bash
# Publish an agent
atlas push my-org/summarizer:1.2.0

# Pull an agent
atlas pull community/web-searcher:latest

# Search the public registry
atlas search "memory agent"
```

### 10.3 Trust & Verification

Agents in the registry carry trust metadata:

```yaml
trust:
  publisher: "my-org"
  signed: true
  signature: "sha256:abc..."
  verified: true
  scan_status: "clean"         # Automated security scan
  review_status: "community"   # unreviewed | community | verified
  downloads: 12400
  rating: 4.7
```

Enterprise deployments can enforce policies:

```yaml
# atlas.config.yaml
registry:
  policy:
    allow_unsigned: false
    minimum_review_status: "verified"
    allowed_publishers: ["my-org", "trusted-partner"]
```

---

## 11. Deployment Models

Atlas is designed to serve multiple deployment scenarios from the same codebase:

| Model | Description |
|-------|-------------|
| **Personal** | Single instance on a homelab or laptop. Your own agent pool for personal automation. |
| **Team / Internal** | Shared instance within an org. Curated registry of blessed, supported agents. Teams build apps by composing from the internal pool. |
| **Platform / SaaS** | Exposed externally. Agents served as a service. Metering, billing, and access control via the security layer. |
| **Open Source Hub** | Public registry node. Community publishes and consumes agents freely. Trust and verification layers handle quality control. |
| **Hybrid** | Internal registry supplemented by pulls from public registries. Enterprise policy controls what's allowed in. |

---

## 12. The "Front Door" Pattern

For conversational applications, the only custom component needed is a **front door agent** — the one that greets the user, understands intent, and orchestrates everything else from the registry.

```yaml
agent:
  name: "front-door"
  version: "1.0.0"
  type: conversational

  requires:
    platform_tools: true   # Needs registry access to discover agents
    spawn_agents: true     # Needs to spawn agents on behalf of the user
    skills:
      - memory             # Conversation memory — provided by the platform
```

Everything behind the front door — memory, tool agents, domain workers, evaluation — is pulled from the registry. The front door is the thin custom layer. Everything else is reusable.

---

## 13. What's Next

This spec defines the architectural foundation. The build sequence is:

1. **Agent Contract & Registration** — Define the contract interface (including hardware declarations), build the registration mechanism, validate with a single working agent.
2. **Execution Pool** — Job queue, lifecycle management, basic pooling. One agent in, one agent out.
3. **Chaining** — Declarative chain definitions, input/output mapping, orchestrator mediation.
4. **Internal MCP Surface** — Expose platform capabilities as tools. Enable agent-to-platform and agent-to-agent communication.
5. **Monitoring & Eval** — Execution traces, cost tracking, evaluation hooks.
6. **Orchestrator Override** — Pluggable orchestrator interface, model selection, custom routing.
7. **Hardware-Aware Scheduling** — Node registration, resource profiling, placement engine, affinity rules, cross-node chain execution.
8. **Registry & Distribution** — Push/pull, versioning, trust metadata, public registry protocol.
9. **Triggers** — Cron, webhook, conversational, and custom trigger types.
10. **Security Layer** — Auth, access control, network policies, agent sandboxing.
11. **Skills System** — Platform-provided capabilities, skill routing, skill marketplace.

---

## Contributing

This project is open source. If you're interested in building the future of agent infrastructure, start here:

- Read this spec
- Pick a component from the build sequence
- Open an issue to discuss your approach
- Submit a PR

The goal is to build the standard, and let the community build the ecosystem on top of it.

---

*Atlas — because agents should be as easy to ship as containers.*