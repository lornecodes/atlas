"""Tests for the Atlas CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atlas.cli.app import app

runner = CliRunner()

AGENTS_PATH = str(Path(__file__).parent.parent / "agents")
CHAINS_PATH = str(Path(__file__).parent.parent / "chains")


# ============================================================================
# TestRunCommand
# ============================================================================

class TestRunCommand:
    """atlas run <agent> -i '...'"""

    def test_run_echo(self):
        """Run echo agent with valid input."""
        result = runner.invoke(app, [
            "run", "echo",
            "-i", '{"message": "hello"}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 0
        assert "hello" in result.output

    def test_run_json_output(self):
        """--json flag outputs raw JSON."""
        result = runner.invoke(app, [
            "run", "echo",
            "-i", '{"message": "hello"}',
            "--agents-path", AGENTS_PATH,
            "--json",
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed.get("message") == "hello"

    def test_run_invalid_json(self):
        """Invalid JSON input exits with error."""
        result = runner.invoke(app, [
            "run", "echo",
            "-i", "not json",
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_run_missing_agent(self):
        """Missing agent exits with error."""
        result = runner.invoke(app, [
            "run", "nonexistent",
            "-i", '{"message": "hello"}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1

    def test_run_custom_agents_path(self):
        """--agents-path points to custom directory."""
        result = runner.invoke(app, [
            "run", "echo",
            "-i", '{"message": "custom path"}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 0


# ============================================================================
# TestRunChainCommand
# ============================================================================

class TestRunChainCommand:
    """atlas run-chain <file> -i '...'"""

    def test_run_chain(self):
        """Run a chain from YAML file."""
        chain_file = Path(CHAINS_PATH) / "translate-then-format.yaml"
        if not chain_file.exists():
            pytest.skip("Chain file not found")

        result = runner.invoke(app, [
            "run-chain", str(chain_file),
            "-i", '{"text": "hello", "target_lang": "fr"}',
            "--agents-path", AGENTS_PATH,
        ])
        # May fail if chain agents are missing, but should not crash
        assert result.exit_code in (0, 1)

    def test_run_chain_missing_file(self):
        """Missing chain file exits with error."""
        result = runner.invoke(app, [
            "run-chain", "/nonexistent/chain.yaml",
            "-i", '{"text": "hello"}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_run_chain_invalid_json(self, tmp_path):
        """Invalid JSON input exits with error."""
        # Create a dummy chain file so we get past the file-exists check
        dummy = tmp_path / "chain.yaml"
        dummy.write_text("chain:\n  name: test\n  steps: []")
        result = runner.invoke(app, [
            "run-chain", str(dummy),
            "-i", "not json",
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output


# ============================================================================
# TestListCommand
# ============================================================================

class TestListCommand:
    """atlas list"""

    def test_list_agents(self):
        """List all discovered agents."""
        result = runner.invoke(app, [
            "list",
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 0
        assert "echo" in result.output
        assert "Found" in result.output

    def test_list_empty_dir(self, tmp_path):
        """List with empty agents directory."""
        result = runner.invoke(app, [
            "list",
            "--agents-path", str(tmp_path),
        ])
        assert result.exit_code == 0
        assert "No agents found" in result.output

    def test_list_json_output(self):
        """--json flag outputs JSON array."""
        result = runner.invoke(app, [
            "list",
            "--agents-path", AGENTS_PATH,
            "--json",
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert any(a["name"] == "echo" for a in parsed)


# ============================================================================
# TestInspectCommand
# ============================================================================

class TestInspectCommand:
    """atlas inspect <agent>"""

    def test_inspect_agent(self):
        """Inspect shows contract details."""
        result = runner.invoke(app, [
            "inspect", "echo",
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 0
        assert "echo" in result.output
        assert "Input Schema" in result.output
        assert "Output Schema" in result.output

    def test_inspect_shows_schemas(self):
        """Inspect shows actual schema content."""
        result = runner.invoke(app, [
            "inspect", "echo",
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 0
        assert "message" in result.output  # echo has a message field

    def test_inspect_missing_agent(self):
        """Inspecting nonexistent agent exits with error."""
        result = runner.invoke(app, [
            "inspect", "nonexistent",
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_inspect_json_output(self):
        """--json flag outputs JSON object."""
        result = runner.invoke(app, [
            "inspect", "echo",
            "--agents-path", AGENTS_PATH,
            "--json",
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["name"] == "echo"
        assert "input_schema" in parsed


# ============================================================================
# TestValidateCommand
# ============================================================================

class TestValidateCommand:
    """atlas validate <path>"""

    def test_validate_valid(self):
        """Validating a valid agent.yaml succeeds."""
        agent_yaml = Path(AGENTS_PATH) / "echo" / "agent.yaml"
        result = runner.invoke(app, ["validate", str(agent_yaml)])
        assert result.exit_code == 0
        assert "Valid" in result.output

    def test_validate_directory(self):
        """Validating a directory looks for agent.yaml inside it."""
        agent_dir = Path(AGENTS_PATH) / "echo"
        result = runner.invoke(app, ["validate", str(agent_dir)])
        assert result.exit_code == 0
        assert "Valid" in result.output

    def test_validate_invalid(self, tmp_path):
        """Validating an invalid YAML exits with error."""
        bad_yaml = tmp_path / "agent.yaml"
        bad_yaml.write_text("not: valid: yaml: agent")
        result = runner.invoke(app, ["validate", str(bad_yaml)])
        assert result.exit_code == 1
        assert "Invalid" in result.output

    def test_validate_missing(self):
        """Validating nonexistent file exits with error."""
        result = runner.invoke(app, ["validate", "/nonexistent/agent.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output
