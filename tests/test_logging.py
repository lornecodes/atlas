"""Tests for Atlas logging infrastructure."""

from __future__ import annotations

import logging

from atlas.logging import get_logger, configure_logging


class TestLogging:
    def test_get_logger_returns_logger(self):
        logger = get_logger("atlas.test")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "atlas.test"

    def test_logger_has_handler(self):
        get_logger("atlas.test2")
        root = logging.getLogger("atlas")
        assert len(root.handlers) > 0

    def test_configure_sets_level(self):
        configure_logging(level=logging.DEBUG)
        root = logging.getLogger("atlas")
        assert root.level == logging.DEBUG
        # Reset
        configure_logging(level=logging.INFO)

    def test_configure_custom_format(self):
        configure_logging(fmt="%(name)s - %(message)s")
        root = logging.getLogger("atlas")
        assert len(root.handlers) == 1

    def test_discovery_logs_warning_on_bad_contract(self, tmp_path, caplog):
        """Registry discovery logs ContractError at WARNING."""
        from atlas.contract.registry import AgentRegistry

        # Create invalid agent
        agent_dir = tmp_path / "bad_agent"
        agent_dir.mkdir()
        (agent_dir / "agent.yaml").write_text("agent:\n  description: missing name\n")

        reg = AgentRegistry(search_paths=[tmp_path])
        with caplog.at_level(logging.WARNING, logger="atlas"):
            reg.discover()

        assert any("Skipping invalid contract" in r.message for r in caplog.records)
