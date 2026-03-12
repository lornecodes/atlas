"""Tests for skill CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atlas.cli.app import app

runner = CliRunner()

SKILLS_DIR = str(Path(__file__).parent.parent / "skills")


class TestSkillList:
    def test_list_skills(self):
        result = runner.invoke(app, ["skill", "list", "--skills-path", SKILLS_DIR])
        assert result.exit_code == 0
        assert "reverse" in result.output

    def test_list_skills_json(self):
        result = runner.invoke(app, ["skill", "list", "--skills-path", SKILLS_DIR, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) >= 1
        names = [s["name"] for s in data]
        assert "reverse" in names

    def test_list_empty(self, tmp_path):
        result = runner.invoke(app, ["skill", "list", "--skills-path", str(tmp_path)])
        assert result.exit_code == 0
        assert "No skills found" in result.output


class TestSkillDescribe:
    def test_describe_skill(self):
        result = runner.invoke(app, ["skill", "describe", "reverse", "--skills-path", SKILLS_DIR])
        assert result.exit_code == 0
        assert "reverse" in result.output
        assert "1.0.0" in result.output

    def test_describe_skill_json(self):
        result = runner.invoke(app, [
            "skill", "describe", "reverse", "--skills-path", SKILLS_DIR, "--json"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "reverse"
        assert data["has_implementation"] is True
        assert "input_schema" in data

    def test_describe_not_found(self):
        result = runner.invoke(app, [
            "skill", "describe", "nonexistent", "--skills-path", SKILLS_DIR
        ])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()
