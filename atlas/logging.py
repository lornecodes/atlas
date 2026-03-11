"""Atlas logging — structured logging with consistent format."""

from __future__ import annotations

import logging
import sys

_configured = False
_default_level = logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Get a logger for an Atlas module.

    Usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)
    if not _configured:
        _setup_default()
    return logger


def configure_logging(
    level: int = logging.INFO,
    fmt: str | None = None,
) -> None:
    """Configure Atlas logging. Call once at startup."""
    global _configured, _default_level
    _default_level = level

    root = logging.getLogger("atlas")
    root.setLevel(level)

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt or "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    _configured = True


def _setup_default() -> None:
    """Set up default logging if configure_logging hasn't been called."""
    global _configured
    root = logging.getLogger("atlas")
    if not root.handlers:
        root.setLevel(_default_level)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(_default_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    _configured = True
