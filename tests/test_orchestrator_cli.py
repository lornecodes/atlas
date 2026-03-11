"""Tests for CLI orchestrator commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from atlas.cli.app import app

runner = CliRunner()
AGENTS_DIR = str(Path(__file__).parent.parent / "agents")


class TestOrchestratorList:
    def test_list_finds_orchestrators(self):
        result = runner.invoke(app, ["orchestrator", "list", "--agents-path", AGENTS_DIR])
        assert result.exit_code == 0
        assert "priority-router" in result.output

    def test_list_json(self):
        result = runner.invoke(app, ["orchestrator", "list", "--agents-path", AGENTS_DIR, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        names = [o["name"] for o in data]
        assert "priority-router" in names

    def test_list_empty_dir(self, tmp_path):
        result = runner.invoke(app, ["orchestrator", "list", "--agents-path", str(tmp_path)])
        assert result.exit_code == 0
        assert "No orchestrators found" in result.output


class TestOrchestratorInspect:
    def test_inspect_existing(self):
        result = runner.invoke(app, ["orchestrator", "inspect", "priority-router", "--agents-path", AGENTS_DIR])
        assert result.exit_code == 0
        assert "priority-router" in result.output
        assert "orchestration" in result.output.lower() or "routing" in result.output.lower()

    def test_inspect_json(self):
        result = runner.invoke(app, ["orchestrator", "inspect", "priority-router", "--agents-path", AGENTS_DIR, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "priority-router"
        assert data["type"] == "orchestrator"

    def test_inspect_not_found(self):
        result = runner.invoke(app, ["orchestrator", "inspect", "nonexistent", "--agents-path", AGENTS_DIR])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestOrchestratorSetReset:
    def test_set_no_server(self):
        """Set should fail gracefully when no server is running."""
        result = runner.invoke(app, [
            "orchestrator", "set", "priority-router",
            "--host", "localhost", "--port", "19999",
        ])
        assert result.exit_code == 1
        assert "could not connect" in result.output.lower()

    def test_reset_no_server(self):
        """Reset should fail gracefully when no server is running."""
        result = runner.invoke(app, [
            "orchestrator", "reset",
            "--host", "localhost", "--port", "19999",
        ])
        assert result.exit_code == 1
        assert "could not connect" in result.output.lower()
