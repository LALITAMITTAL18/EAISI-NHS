"""Logging utilities.

Centralised logging setup so every module imports from here
rather than configuring its own handlers.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for the pipeline.

    Call this once per module::

        from nhs_proms_pipeline.utils.logging import get_logger
        logger = get_logger(__name__)

    Handlers are added only to the root logger to avoid duplicate messages.
    """
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format.

    Call this once at application startup (e.g. in the CLI entry-point).

    Args:
        level: A standard Python logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    root_logger = logging.getLogger()
    # Avoid adding multiple handlers on repeated calls
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)
