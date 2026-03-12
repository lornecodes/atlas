"""Tests for security CLI commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from atlas.cli.app import app

runner = CliRunner()


class TestSecurityValidate:
    def test_validate_valid_policy(self, tmp_path):
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "container_image: test:v1\n"
            "max_memory_mb: 256\n"
            "allowed_secrets:\n"
            "  - SECRET_A\n"
        )
        result = runner.invoke(app, ["security", "validate", str(policy_file)])
        assert result.exit_code == 0
        assert "Valid security policy" in result.output
        assert "test:v1" in result.output
        assert "256MB" in result.output

    def test_validate_empty_policy(self, tmp_path):
        policy_file = tmp_path / "empty.yaml"
        policy_file.write_text("")
        result = runner.invoke(app, ["security", "validate", str(policy_file)])
        assert result.exit_code == 0
        assert "Valid security policy" in result.output

    def test_validate_missing_file(self):
        result = runner.invoke(app, ["security", "validate", "/nonexistent/policy.yaml"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()


class TestSecurityCheck:
    def test_check_agent_default_policy(self, tmp_path):
        agent_dir = tmp_path / "test-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(
            "name: test-agent\n"
            "version: '1.0.0'\n"
            "description: A test agent\n"
            "input:\n"
            "  schema:\n"
            "    type: object\n"
            "output:\n"
            "  schema:\n"
            "    type: object\n"
        )
        result = runner.invoke(app, ["security", "check", str(agent_dir)])
        assert result.exit_code == 0
        assert "test-agent" in result.output
        assert "process" in result.output

    def test_check_agent_with_permissions(self, tmp_path):
        agent_dir = tmp_path / "secure-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(
            "name: secure-agent\n"
            "version: '1.0.0'\n"
            "permissions:\n"
            "  isolation: container\n"
            "  secrets:\n"
            "    - API_KEY\n"
            "  max_memory_mb: 256\n"
            "input:\n"
            "  schema:\n"
            "    type: object\n"
            "output:\n"
            "  schema:\n"
            "    type: object\n"
        )
        result = runner.invoke(app, ["security", "check", str(agent_dir)])
        assert result.exit_code == 0
        assert "container" in result.output
        assert "API_KEY" in result.output

    def test_check_agent_with_policy(self, tmp_path):
        agent_dir = tmp_path / "capped-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(
            "name: capped-agent\n"
            "version: '1.0.0'\n"
            "permissions:\n"
            "  max_memory_mb: 2048\n"
            "input:\n"
            "  schema:\n"
            "    type: object\n"
            "output:\n"
            "  schema:\n"
            "    type: object\n"
        )
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text("max_memory_mb: 512\n")

        result = runner.invoke(app, [
            "security", "check", str(agent_dir),
            "--policy", str(policy_file),
        ])
        assert result.exit_code == 0
        assert "512MB" in result.output  # capped by policy

    def test_check_json_output(self, tmp_path):
        agent_dir = tmp_path / "json-agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text(
            "name: json-agent\n"
            "version: '1.0.0'\n"
            "input:\n"
            "  schema:\n"
            "    type: object\n"
            "output:\n"
            "  schema:\n"
            "    type: object\n"
        )
        result = runner.invoke(app, [
            "security", "check", str(agent_dir), "--json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["agent"] == "json-agent"
        assert "permissions" in data

    def test_check_missing_agent(self):
        result = runner.invoke(app, ["security", "check", "/nonexistent/agent"])
        assert result.exit_code == 1
