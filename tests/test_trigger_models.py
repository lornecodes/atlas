"""Tests for TriggerDefinition model."""

import time
from pathlib import Path

import pytest

from atlas.triggers.models import TriggerDefinition, VALID_TRIGGER_TYPES


class TestTriggerDefaults:
    def test_auto_id(self):
        t = TriggerDefinition()
        assert t.id.startswith("trigger-")
        assert len(t.id) == 16  # "trigger-" + 8 hex chars

    def test_auto_created_at(self):
        before = time.time()
        t = TriggerDefinition()
        assert t.created_at >= before

    def test_is_recurring_cron(self):
        t = TriggerDefinition(trigger_type="cron")
        assert t.is_recurring is True

    def test_is_recurring_interval(self):
        t = TriggerDefinition(trigger_type="interval")
        assert t.is_recurring is True

    def test_not_recurring_one_shot(self):
        t = TriggerDefinition(trigger_type="one_shot")
        assert t.is_recurring is False

    def test_not_recurring_webhook(self):
        t = TriggerDefinition(trigger_type="webhook")
        assert t.is_recurring is False

    def test_target_agent(self):
        t = TriggerDefinition(agent_name="echo")
        assert t.target == "echo"

    def test_target_chain(self):
        t = TriggerDefinition(chain_name="my-chain")
        assert t.target == "my-chain"


class TestTriggerValidation:
    def test_valid_cron(self):
        t = TriggerDefinition(
            trigger_type="cron", cron_expr="*/5 * * * *", agent_name="echo"
        )
        t.validate()  # should not raise

    def test_invalid_type(self):
        t = TriggerDefinition(trigger_type="bogus", agent_name="echo")
        with pytest.raises(ValueError, match="Invalid trigger_type"):
            t.validate()

    def test_no_target(self):
        t = TriggerDefinition(trigger_type="cron", cron_expr="* * * * *")
        with pytest.raises(ValueError, match="agent_name or chain_name"):
            t.validate()

    def test_both_targets(self):
        t = TriggerDefinition(
            trigger_type="cron", cron_expr="* * * * *",
            agent_name="echo", chain_name="my-chain",
        )
        with pytest.raises(ValueError, match="cannot specify both"):
            t.validate()

    def test_cron_missing_expr(self):
        t = TriggerDefinition(trigger_type="cron", agent_name="echo")
        with pytest.raises(ValueError, match="cron_expr"):
            t.validate()

    def test_interval_zero(self):
        t = TriggerDefinition(
            trigger_type="interval", interval_seconds=0, agent_name="echo"
        )
        with pytest.raises(ValueError, match="interval_seconds"):
            t.validate()

    def test_one_shot_no_fire_at(self):
        t = TriggerDefinition(trigger_type="one_shot", agent_name="echo")
        with pytest.raises(ValueError, match="fire_at"):
            t.validate()

    def test_webhook_valid(self):
        t = TriggerDefinition(trigger_type="webhook", agent_name="echo")
        t.validate()  # no extra fields required


class TestComputeNextFire:
    def test_cron_next_fire(self):
        t = TriggerDefinition(
            trigger_type="cron", cron_expr="0 9 * * *", agent_name="echo"
        )
        nf = t.compute_next_fire(now=0.0)
        assert nf > 0

    def test_interval_first_fire(self):
        now = time.time()
        t = TriggerDefinition(
            trigger_type="interval", interval_seconds=300, agent_name="echo"
        )
        nf = t.compute_next_fire(now=now)
        assert abs(nf - (now + 300)) < 1.0

    def test_interval_subsequent_fire(self):
        now = time.time()
        t = TriggerDefinition(
            trigger_type="interval", interval_seconds=300,
            agent_name="echo", last_fired=now,
        )
        nf = t.compute_next_fire(now=now)
        assert abs(nf - (now + 300)) < 1.0

    def test_one_shot(self):
        t = TriggerDefinition(
            trigger_type="one_shot", fire_at=1741000000.0, agent_name="echo"
        )
        assert t.compute_next_fire() == 1741000000.0

    def test_webhook_returns_zero(self):
        t = TriggerDefinition(trigger_type="webhook", agent_name="echo")
        assert t.compute_next_fire() == 0.0


class TestSerialization:
    def test_to_dict_roundtrip(self):
        t = TriggerDefinition(
            name="test", trigger_type="cron", cron_expr="0 9 * * *",
            agent_name="echo", priority=5, input_data={"key": "val"},
        )
        d = t.to_dict()
        t2 = TriggerDefinition.from_dict(d)
        assert t2.name == "test"
        assert t2.trigger_type == "cron"
        assert t2.cron_expr == "0 9 * * *"
        assert t2.agent_name == "echo"
        assert t2.priority == 5
        assert t2.input_data == {"key": "val"}

    def test_from_dict_with_trigger_key(self):
        d = {"trigger": {"name": "nested", "trigger_type": "webhook", "agent_name": "x"}}
        t = TriggerDefinition.from_dict(d)
        assert t.name == "nested"
        assert t.trigger_type == "webhook"

    def test_from_yaml(self, tmp_path):
        yaml_content = """\
trigger:
  name: test-cron
  trigger_type: cron
  cron_expr: "*/5 * * * *"
  agent_name: echo
  input_data:
    message: hello
"""
        p = tmp_path / "trigger.yaml"
        p.write_text(yaml_content)
        t = TriggerDefinition.from_yaml(p)
        assert t.name == "test-cron"
        assert t.trigger_type == "cron"
        assert t.cron_expr == "*/5 * * * *"
        assert t.agent_name == "echo"
        assert t.input_data == {"message": "hello"}
