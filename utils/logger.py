"""Logging setup for the application and processing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import config


_LOGGER_NAME = "smartvision"


def get_logger(
    name: str = _LOGGER_NAME,
    log_file_path: Path = config.LOG_FILE_PATH,
) -> logging.Logger:
    """Return a configured logger that writes to the processing log.

    Args:
        name: Logger name.
        log_file_path: File receiving processing logs.

    Returns:
        A reusable logger configured once per name and file path.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved_path = Path(log_file_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    handler_key = str(resolved_path.resolve())
    already_configured = any(
        getattr(handler, "_smartvision_path", None) == handler_key
        for handler in logger.handlers
    )
    if not already_configured:
        handler = logging.FileHandler(resolved_path, encoding="utf-8")
        handler._smartvision_path = handler_key  # type: ignore[attr-defined]
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    return logger
