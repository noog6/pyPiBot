"""Application runtime entry points and lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Sequence


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Configuration for the application runtime.

    Attributes:
        prompts: Optional prompts to send to the AI assistant on startup.
    """

    prompts: list[str] = field(default_factory=list)


def run(config: AppConfig) -> int:
    """Run the application with the provided configuration.

    Args:
        config: Application configuration values.

    Returns:
        Process exit code (0 for success).
    """

    LOGGER.info("Starting Theo runtime")
    if config.prompts:
        LOGGER.info("Startup prompts: %s", ", ".join(config.prompts))
    else:
        LOGGER.info("No startup prompts provided")
    LOGGER.info("Runtime skeleton initialized")
    return 0


def build_config(prompts: Sequence[str]) -> AppConfig:
    """Construct an AppConfig from parsed prompt values.

    Args:
        prompts: Prompt strings passed from the CLI.

    Returns:
        An AppConfig instance.
    """

    return AppConfig(prompts=list(prompts))
