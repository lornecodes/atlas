# Contributing to Atlas

Thanks for your interest in contributing to Atlas! This document covers the basics of getting set up and submitting changes.

## Development Setup

```bash
# Clone and install in editable mode with dev dependencies
git clone https://github.com/your-org/atlas.git
cd atlas
pip install -e ".[dev,all]"

# Verify everything works
python -m pytest tests/ -x -q
```

**Requirements**: Python 3.11+

## Running Tests

```bash
# Full suite
python -m pytest tests/ -x -q

# Specific module
python -m pytest tests/test_pool.py -v

# E2E integration tests
python -m pytest tests/test_e2e_pool.py tests/test_e2e_orchestrator.py -v
```

All tests use `pytest-asyncio` with `asyncio_mode = "auto"` — async test functions are detected automatically.

## Project Layout

| Directory | What lives here |
|---|---|
| `atlas/` | Core library source |
| `agents/` | Example agent implementations (used in tests) |
| `chains/` | Example chain definitions (YAML) |
| `tests/` | All tests (unit, integration, e2e) |

## Adding an Agent

1. Create `agents/your-agent/agent.yaml` with a contract (see existing agents for examples)
2. Create `agents/your-agent/agent.py` with a class named `Agent` that extends `AgentBase`
3. Implement `async def execute(self, input_data: dict) -> dict`
4. Optionally add `eval.yaml` for automated output validation
5. Add tests in `tests/`

## Code Style

- **Type hints** on public APIs (function signatures, dataclass fields)
- **Docstrings** on public classes and non-obvious methods
- **No mocking of internal components** in e2e tests — they exercise the full stack
- **async-first** — all I/O-bound operations should be async
- Follow existing patterns — look at `echo` and `formatter` agents for reference

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Ensure `python -m pytest tests/ -x -q` passes (all 582+ tests)
4. Open a PR with a clear description of what and why

### PR Checklist

- [ ] Tests pass locally
- [ ] New functionality has tests
- [ ] Agent contracts have valid JSON Schema for inputs and outputs
- [ ] No breaking changes to existing contracts without version bump

## Architecture Principles

- **Contracts are the API** — agents communicate through typed schemas, not ad-hoc dicts
- **EventBus for observation, not control** — subscribers observe status changes but don't modify job flow
- **Strategies cascade** — mediation tries simple approaches before complex ones (direct → mapped → coerce → LLM)
- **Orchestrators are pluggable** — routing logic is separate from execution logic
- **No global state** — everything is wired via constructor injection (registry, queue, bus)

## Reporting Issues

Open an issue with:
- What you expected to happen
- What actually happened
- Minimal reproduction steps
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
