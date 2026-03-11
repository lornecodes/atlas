"""Tests for the pool CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from atlas.cli.app import app

runner = CliRunner()

AGENTS_PATH = str(Path(__file__).parent.parent / "agents")


class TestPoolRun:
    """atlas pool run <agent> -i '...'"""

    def test_run_echo(self):
        result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "pool hello"}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 0
        assert "pool hello" in result.output

    def test_run_json_output(self):
        result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "json"}',
            "--agents-path", AGENTS_PATH,
            "--json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "completed"
        assert data["output_data"]["message"] == "json"

    def test_run_with_priority(self):
        result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "hi"}',
            "--agents-path", AGENTS_PATH,
            "--priority", "10",
        ])
        assert result.exit_code == 0

    def test_run_invalid_json(self):
        result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", "not json",
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1
        assert "Invalid JSON" in result.output

    def test_run_missing_agent(self):
        result = runner.invoke(app, [
            "pool", "run", "nonexistent",
            "-i", '{"x": 1}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1

    def test_run_invalid_input(self):
        """Echo requires 'message' field — omitting it should fail."""
        result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"wrong_field": "oops"}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 1
        assert "failed" in result.output.lower() or "validation" in result.output.lower()

    def test_run_summarizer(self):
        result = runner.invoke(app, [
            "pool", "run", "summarizer",
            "-i", '{"text": "The quick brown fox jumped over the lazy dog."}',
            "--agents-path", AGENTS_PATH,
        ])
        assert result.exit_code == 0
        assert "summary" in result.output.lower() or "token" in result.output.lower()


class TestPoolPersistence:
    """Pool commands with --db for SQLite persistence."""

    def test_run_with_db(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "persist this"}',
            "--agents-path", AGENTS_PATH,
            "--db", db_path,
        ])
        assert result.exit_code == 0
        assert "persist this" in result.output

    def test_status_after_run(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        # Run first to create the job
        run_result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "track me"}',
            "--agents-path", AGENTS_PATH,
            "--db", db_path,
            "--json",
        ])
        assert run_result.exit_code == 0
        job_data = json.loads(run_result.output)
        job_id = job_data["id"]

        # Now check status
        status_result = runner.invoke(app, [
            "pool", "status", job_id,
            "--db", db_path,
        ])
        assert status_result.exit_code == 0
        assert job_id in status_result.output
        assert "completed" in status_result.output

    def test_status_json(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        run_result = runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "json status"}',
            "--agents-path", AGENTS_PATH,
            "--db", db_path,
            "--json",
        ])
        job_data = json.loads(run_result.output)
        job_id = job_data["id"]

        status_result = runner.invoke(app, [
            "pool", "status", job_id,
            "--db", db_path,
            "--json",
        ])
        assert status_result.exit_code == 0
        status_data = json.loads(status_result.output)
        assert status_data["status"] == "completed"

    def test_list_jobs(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        # Run two jobs
        for msg in ["first", "second"]:
            runner.invoke(app, [
                "pool", "run", "echo",
                "-i", json.dumps({"message": msg}),
                "--agents-path", AGENTS_PATH,
                "--db", db_path,
            ])

        list_result = runner.invoke(app, [
            "pool", "list",
            "--db", db_path,
        ])
        assert list_result.exit_code == 0
        assert "2 job(s)" in list_result.output

    def test_list_filter_by_status(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "ok"}',
            "--agents-path", AGENTS_PATH,
            "--db", db_path,
        ])

        list_result = runner.invoke(app, [
            "pool", "list",
            "--db", db_path,
            "--status", "completed",
        ])
        assert list_result.exit_code == 0
        assert "1 job(s)" in list_result.output

        # No pending jobs
        list_result = runner.invoke(app, [
            "pool", "list",
            "--db", db_path,
            "--status", "pending",
        ])
        assert "No jobs found" in list_result.output

    def test_list_json(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "json list"}',
            "--agents-path", AGENTS_PATH,
            "--db", db_path,
        ])

        list_result = runner.invoke(app, [
            "pool", "list",
            "--db", db_path,
            "--json",
        ])
        assert list_result.exit_code == 0
        data = json.loads(list_result.output)
        assert len(data) == 1
        assert data[0]["status"] == "completed"

    def test_status_not_found(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        # Create empty DB by running and completing a job first
        runner.invoke(app, [
            "pool", "run", "echo",
            "-i", '{"message": "setup"}',
            "--agents-path", AGENTS_PATH,
            "--db", db_path,
        ])

        result = runner.invoke(app, [
            "pool", "status", "job-nonexistent",
            "--db", db_path,
        ])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()
