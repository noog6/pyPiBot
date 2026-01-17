"""Application runtime entry points and lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Sequence

from interaction import AsyncMicrophone, AudioPlayer
from storage import StorageController


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

    storage = StorageController.get_instance()
    storage_info = storage.get_storage_info()
    LOGGER.info(
        "Storage ready (run_id=%s, run_id_file=%s, db_path=%s)",
        storage_info.run_id,
        storage_info.run_id_file,
        storage_info.db_path,
    )

    try:
        microphone = AsyncMicrophone()
        microphone.start_recording()
        LOGGER.info("Microphone started")
    except Exception as exc:
        LOGGER.warning("Microphone unavailable: %s", exc)
        microphone = None

    try:
        player = AudioPlayer()
        LOGGER.info("Audio playback ready")
    except Exception as exc:
        LOGGER.warning("Audio playback unavailable: %s", exc)
        player = None

    LOGGER.info("Runtime skeleton initialized")

    if microphone:
        microphone.stop_recording()
        microphone.close()
    if player:
        player.close()

    return 0


def build_config(prompts: Sequence[str]) -> AppConfig:
    """Construct an AppConfig from parsed prompt values.

    Args:
        prompts: Prompt strings passed from the CLI.

    Returns:
        An AppConfig instance.
    """

    return AppConfig(prompts=list(prompts))
