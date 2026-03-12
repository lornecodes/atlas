"""Tests for trigger CLI commands."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from atlas.cli.app import app

runner = CliRunner()


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_cli.db")


class TestTriggerCreate:
    def test_create_cron(self, db_path):
        result = runner.invoke(app, [
            "trigger", "create",
            "--type", "cron",
            "--agent", "echo",
            "--cron", "*/5 * * * *",
            "--name", "test-cron",
            "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "Created trigger" in result.output
        assert "cron" in result.output

    def test_create_interval(self, db_path):
        result = runner.invoke(app, [
            "trigger", "create",
            "--type", "interval",
            "--agent", "echo",
            "--interval", "300",
            "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "Created trigger" in result.output

    def test_create_webhook(self, db_path):
        result = runner.invoke(app, [
            "trigger", "create",
            "--type", "webhook",
            "--agent", "echo",
            "--secret", "mysecret",
            "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "Created trigger" in result.output

    def test_create_invalid_type(self, db_path):
        result = runner.invoke(app, [
            "trigger", "create",
            "--type", "bogus",
            "--agent", "echo",
            "--db", db_path,
        ])
        assert result.exit_code == 1

    def test_create_with_input(self, db_path):
        result = runner.invoke(app, [
            "trigger", "create",
            "--type", "webhook",
            "--agent", "echo",
            "--input", '{"message": "hello"}',
            "--db", db_path,
        ])
        assert result.exit_code == 0


class TestTriggerList:
    def test_list_empty(self, db_path):
        result = runner.invoke(app, [
            "trigger", "list", "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "No triggers found" in result.output

    def test_list_with_triggers(self, db_path):
        runner.invoke(app, [
            "trigger", "create",
            "--type", "webhook", "--agent", "echo",
            "--name", "test1", "--db", db_path,
        ])
        runner.invoke(app, [
            "trigger", "create",
            "--type", "webhook", "--agent", "echo",
            "--name", "test2", "--db", db_path,
        ])
        result = runner.invoke(app, [
            "trigger", "list", "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "test1" in result.output
        assert "test2" in result.output

    def test_list_json(self, db_path):
        runner.invoke(app, [
            "trigger", "create",
            "--type", "webhook", "--agent", "echo",
            "--db", db_path,
        ])
        result = runner.invoke(app, [
            "trigger", "list", "--json", "--db", db_path,
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1


class TestTriggerDelete:
    def test_delete(self, db_path):
        # Create then extract ID from output
        create_result = runner.invoke(app, [
            "trigger", "create",
            "--type", "webhook", "--agent", "echo",
            "--db", db_path,
        ])
        # Output: "Created trigger trigger-XXXXXXXX (webhook → echo)"
        trigger_id = create_result.output.split()[2]

        result = runner.invoke(app, [
            "trigger", "delete", trigger_id, "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_delete_not_found(self, db_path):
        result = runner.invoke(app, [
            "trigger", "delete", "nonexistent", "--db", db_path,
        ])
        assert result.exit_code == 1
