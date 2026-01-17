"""Motion logging helpers."""

from __future__ import annotations

import logging


LOGGER = logging.getLogger("motion")


def log_info(message: str, *args: object) -> None:
    """Log info-level motion message."""

    LOGGER.info(message, *args)


def log_warning(message: str, *args: object) -> None:
    """Log warning-level motion message."""

    LOGGER.warning(message, *args)


def log_error(message: str, *args: object) -> None:
    """Log error-level motion message."""

    LOGGER.error(message, *args)
