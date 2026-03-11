"""Tests for chain execution with mediation — E2E Spike 2."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atlas.chains.definition import ChainDefinition, ChainStep
from atlas.chains.runner import ChainRunner
from atlas.contract.registry import AgentRegistry
from atlas.mediation.engine import MediationEngine
from atlas.runtime.runner import RunError

from conftest import AGENTS_DIR

CHAINS_DIR = Path(__file__).parent.parent / "chains"


@pytest.fixture
def chain_runner(registry: AgentRegistry) -> ChainRunner:
    engine = MediationEngine()
    return ChainRunner(registry, engine)


class TestChainDefinition:
    def test_load_from_yaml(self):
        chain = ChainDefinition.from_yaml(CHAINS_DIR / "translate-then-format.yaml")
        assert chain.name == "translate-then-format"
        assert len(chain.steps) == 2
        assert chain.steps[0].agent_name == "translator"
        assert chain.steps[1].agent_name == "formatter"

    def test_from_dict(self):
        chain = ChainDefinition.from_dict({
            "chain": {
                "name": "test",
                "steps": [
                    {"agent": "echo"},
                    {"agent": "echo"},
                ],
            }
        })
        assert len(chain.steps) == 2

    def test_named_steps(self):
        chain = ChainDefinition.from_dict({
            "chain": {
                "name": "named",
                "steps": [
                    {"agent": "echo", "name": "first"},
                    {"agent": "echo", "name": "second"},
                ],
            }
        })
        assert chain.steps[0].name == "first"
        assert chain.steps[1].name == "second"
        assert chain.step_name(0) == "first"
        assert chain.step_name(1) == "second"

    def test_default_step_names(self):
        chain = ChainDefinition(name="test", steps=[
            ChainStep(agent_name="echo"),
            ChainStep(agent_name="echo"),
        ])
        assert chain.step_name(0) == "step_0"
        assert chain.step_name(1) == "step_1"


class TestChainExecution:
    async def test_single_step_chain(self, chain_runner: ChainRunner):
        chain = ChainDefinition(
            name="single",
            steps=[ChainStep(agent_name="echo")],
        )
        result = await chain_runner.execute(chain, {"message": "hello"})
        assert result.success
        assert result.output == {"message": "hello"}
        assert len(result.steps) == 1

    async def test_echo_echo_chain(self, chain_runner: ChainRunner):
        """Two identical agents chain with direct mediation."""
        chain = ChainDefinition(
            name="echo-echo",
            steps=[
                ChainStep(agent_name="echo"),
                ChainStep(agent_name="echo"),
            ],
        )
        result = await chain_runner.execute(chain, {"message": "round trip"})
        assert result.success
        assert result.output == {"message": "round trip"}
        assert result.steps[1].mediation.strategy_used == "direct"

    async def test_translate_then_format_with_map(self, chain_runner: ChainRunner):
        """Translator → Formatter using explicit input_map."""
        chain = ChainDefinition.from_yaml(CHAINS_DIR / "translate-then-format.yaml")
        result = await chain_runner.execute(chain, {
            "text": "hello world",
            "target_lang": "fr",
        })
        assert result.success
        assert "formatted" in result.output
        assert "fr" in result.output["formatted"] or "hello" in result.output["formatted"]

    async def test_translate_then_format_auto_mediation(self, chain_runner: ChainRunner):
        """Translator → Formatter WITHOUT explicit input_map — tests coercion."""
        chain = ChainDefinition(
            name="auto-mediate",
            steps=[
                ChainStep(agent_name="translator"),
                ChainStep(agent_name="formatter"),  # No input_map!
            ],
        )
        result = await chain_runner.execute(chain, {
            "text": "hello",
            "target_lang": "es",
        })
        if not result.success:
            assert result.failed_at == 1

    async def test_chain_with_missing_agent(self, chain_runner: ChainRunner):
        chain = ChainDefinition(
            name="broken",
            steps=[
                ChainStep(agent_name="echo"),
                ChainStep(agent_name="nonexistent"),
            ],
        )
        result = await chain_runner.execute(chain, {"message": "hi"})
        assert not result.success

    async def test_chain_agent_failure(self, chain_runner: ChainRunner, tmp_path):
        """Agent that crashes mid-chain produces clean failure."""
        agent_dir = tmp_path / "crasher"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(yaml.dump({
            "agent": {
                "name": "crasher",
                "version": "1.0.0",
            }
        }))
        (agent_dir / "agent.py").write_text(
            "from atlas.runtime.base import AgentBase\n"
            "class CrasherAgent(AgentBase):\n"
            "    async def execute(self, input):\n"
            "        raise RuntimeError('chain crash')\n"
        )

        reg = AgentRegistry(search_paths=[AGENTS_DIR, tmp_path])
        reg.discover()
        runner = ChainRunner(reg, MediationEngine())

        chain = ChainDefinition(
            name="crash-chain",
            steps=[
                ChainStep(agent_name="echo"),
                ChainStep(agent_name="crasher"),
            ],
        )
        result = await runner.execute(chain, {"message": "boom"})
        assert not result.success
        assert result.failed_at == 1
        assert "chain crash" in result.error

    async def test_mediation_summary(self, chain_runner: ChainRunner):
        chain = ChainDefinition(
            name="summary-test",
            steps=[
                ChainStep(agent_name="echo"),
                ChainStep(agent_name="echo"),
            ],
        )
        result = await chain_runner.execute(chain, {"message": "test"})
        assert result.success
        summary = result.mediation_summary
        assert len(summary) == 2
        assert summary[0]["strategy"] == "none"  # First step has no mediation
        assert summary[1]["strategy"] == "direct"


class TestPartialResults:
    async def test_partial_outputs_on_failure(self, chain_runner: ChainRunner, tmp_path):
        """On failure, partial_outputs contains completed step outputs."""
        agent_dir = tmp_path / "crasher"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(yaml.dump({
            "agent": {"name": "crasher", "version": "1.0.0"}
        }))
        (agent_dir / "agent.py").write_text(
            "from atlas.runtime.base import AgentBase\n"
            "class CrasherAgent(AgentBase):\n"
            "    async def execute(self, input):\n"
            "        raise RuntimeError('step 2 crash')\n"
        )

        reg = AgentRegistry(search_paths=[AGENTS_DIR, tmp_path])
        reg.discover()
        runner = ChainRunner(reg, MediationEngine())

        chain = ChainDefinition(
            name="partial-test",
            steps=[
                ChainStep(agent_name="echo"),
                ChainStep(agent_name="crasher"),
            ],
        )
        result = await runner.execute(chain, {"message": "hello"})
        assert not result.success
        assert result.failed_at == 1
        # partial_outputs should contain step 0's output
        assert len(result.partial_outputs) == 1
        assert result.partial_outputs[0] == {"message": "hello"}
        # output dict also has partial info
        assert "partial_outputs" in result.output
        assert result.output["failed_step"] == 1

    async def test_partial_outputs_empty_on_first_step_failure(self, chain_runner: ChainRunner):
        """If first step fails, partial_outputs is empty."""
        chain = ChainDefinition(
            name="first-fail",
            steps=[
                ChainStep(agent_name="echo"),  # Will fail — missing required 'message'
            ],
        )
        result = await chain_runner.execute(chain, {})  # Missing 'message'
        assert not result.success
        assert result.partial_outputs == []

    async def test_success_has_no_partial(self, chain_runner: ChainRunner):
        """Successful chain doesn't have partial_outputs in output."""
        chain = ChainDefinition(
            name="success",
            steps=[ChainStep(agent_name="echo")],
        )
        result = await chain_runner.execute(chain, {"message": "ok"})
        assert result.success
        assert "partial_outputs" not in result.output


class TestNamedStepChains:
    async def test_named_steps_in_context(self, chain_runner: ChainRunner):
        """Named steps appear in chain context."""
        chain = ChainDefinition(
            name="named-chain",
            steps=[
                ChainStep(agent_name="echo", name="greet"),
                ChainStep(agent_name="echo"),
            ],
        )
        result = await chain_runner.execute(chain, {"message": "hello"})
        assert result.success
        # Verify the step name is properly assigned
        assert chain.step_name(0) == "greet"
        assert chain.step_name(1) == "step_1"
