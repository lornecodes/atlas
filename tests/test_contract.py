"""Tests for agent contract loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from atlas.contract.schema import ContractError, load_contract, validate_input, validate_output
from atlas.contract.types import AgentContract, SchemaSpec

from conftest import AGENTS_DIR


class TestLoadContract:
    def test_load_echo(self):
        contract = load_contract(AGENTS_DIR / "echo" / "agent.yaml")
        assert contract.name == "echo"
        assert contract.version == "1.0.0"
        assert "echo" in contract.capabilities

    def test_load_summarizer(self):
        contract = load_contract(AGENTS_DIR / "summarizer" / "agent.yaml")
        assert contract.name == "summarizer"
        assert "summarization" in contract.capabilities
        assert "text" in contract.input_schema.properties
        assert "text" in contract.input_schema.required

    def test_load_all_agents(self):
        for agent_dir in AGENTS_DIR.iterdir():
            yaml_path = agent_dir / "agent.yaml"
            if yaml_path.exists():
                contract = load_contract(yaml_path)
                assert contract.name
                assert contract.version

    def test_missing_file(self):
        with pytest.raises(ContractError, match="not found"):
            load_contract(Path("/nonexistent/agent.yaml"))

    def test_invalid_yaml(self, tmp_path):
        bad = tmp_path / "agent.yaml"
        bad.write_text("{{{{invalid yaml")
        with pytest.raises(ContractError, match="Invalid YAML"):
            load_contract(bad)

    def test_missing_required_fields(self, tmp_path):
        bad = tmp_path / "agent.yaml"
        bad.write_text(yaml.dump({"agent": {"description": "no name or version"}}))
        with pytest.raises(ContractError, match="validation failed"):
            load_contract(bad)

    def test_invalid_version_format(self, tmp_path):
        bad = tmp_path / "agent.yaml"
        bad.write_text(yaml.dump({"agent": {"name": "test", "version": "not-semver"}}))
        with pytest.raises(ContractError, match="validation failed"):
            load_contract(bad)

    def test_contract_is_frozen(self):
        contract = load_contract(AGENTS_DIR / "echo" / "agent.yaml")
        with pytest.raises(AttributeError):
            contract.name = "hacked"

    def test_execution_timeout_parsed(self, tmp_path):
        """execution_timeout is parsed from agent.yaml."""
        agent_dir = tmp_path / "timed"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(yaml.dump({
            "agent": {
                "name": "timed",
                "version": "1.0.0",
                "execution_timeout": 30.0,
            }
        }))
        contract = load_contract(agent_dir / "agent.yaml")
        assert contract.execution_timeout == 30.0

    def test_execution_timeout_default(self):
        """Default execution_timeout is 60s."""
        contract = load_contract(AGENTS_DIR / "echo" / "agent.yaml")
        assert contract.execution_timeout == 60.0


class TestInputValidation:
    def test_valid_input(self):
        contract = load_contract(AGENTS_DIR / "echo" / "agent.yaml")
        errors = validate_input(contract, {"message": "hello"})
        assert errors == []

    def test_missing_required_field(self):
        contract = load_contract(AGENTS_DIR / "echo" / "agent.yaml")
        errors = validate_input(contract, {})
        assert len(errors) > 0
        assert any("message" in e for e in errors)

    def test_wrong_type(self):
        contract = load_contract(AGENTS_DIR / "echo" / "agent.yaml")
        errors = validate_input(contract, {"message": 123})
        assert len(errors) > 0

    def test_extra_fields_allowed(self):
        contract = load_contract(AGENTS_DIR / "echo" / "agent.yaml")
        errors = validate_input(contract, {"message": "hello", "extra": "stuff"})
        assert errors == []


class TestOutputValidation:
    def test_valid_output(self):
        contract = load_contract(AGENTS_DIR / "summarizer" / "agent.yaml")
        errors = validate_output(contract, {"summary": "short", "token_count": 1})
        assert errors == []

    def test_missing_required_output(self):
        contract = load_contract(AGENTS_DIR / "summarizer" / "agent.yaml")
        errors = validate_output(contract, {"summary": "short"})
        assert len(errors) > 0

    def test_wrong_output_type(self):
        contract = load_contract(AGENTS_DIR / "summarizer" / "agent.yaml")
        errors = validate_output(contract, {"summary": "short", "token_count": "not-int"})
        assert len(errors) > 0


class TestSchemaSpec:
    def test_from_dict_none(self):
        spec = SchemaSpec.from_dict(None)
        assert spec.type == "object"
        assert spec.properties == {}

    def test_to_json_schema_round_trip(self):
        raw = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        spec = SchemaSpec.from_dict(raw)
        schema = spec.to_json_schema()
        assert schema == raw

    def test_empty_schema_accepts_anything(self):
        contract = AgentContract(name="test", version="1.0.0")
        assert validate_input(contract, {"anything": "goes"}) == []
        assert validate_output(contract, {}) == []

    def test_immutable(self):
        spec = SchemaSpec({"type": "object"})
        with pytest.raises(AttributeError, match="immutable"):
            spec.x = "bad"

    def test_equality(self):
        a = SchemaSpec({"type": "object", "properties": {"x": {"type": "string"}}})
        b = SchemaSpec({"type": "object", "properties": {"x": {"type": "string"}}})
        c = SchemaSpec({"type": "object"})
        assert a == b
        assert a != c

    def test_repr(self):
        spec = SchemaSpec({"type": "object"})
        assert "SchemaSpec" in repr(spec)
        assert "object" in repr(spec)
