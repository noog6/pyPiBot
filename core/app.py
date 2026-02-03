"""Application runtime entry points and lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Sequence

from config import ConfigController
from hardware import ADS1015Sensor
from interaction import AsyncMicrophone, AudioPlayer
from motion import MotionController
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
        sensor = ADS1015Sensor.get_instance()
        battery_voltage = sensor.read_battery_voltage()
        LOGGER.info("Battery voltage: %sV", battery_voltage)
    except Exception as exc:
        LOGGER.warning("ADS1015 sensor unavailable: %s", exc)
        sensor = None

    config = ConfigController.get_instance().get_config()
    audio_cfg = config.get("audio") or {}
    input_cfg = audio_cfg.get("input") or {}
    output_cfg = audio_cfg.get("output") or {}

    try:
        input_device_index = input_cfg.get("device_index")
        microphone = AsyncMicrophone(
            input_device_index=(
                int(input_device_index) if input_device_index is not None else None
            ),
            input_name_hint=input_cfg.get("device_name"),
        )
        microphone.start_recording()
        LOGGER.info("Microphone started")
    except Exception as exc:
        LOGGER.warning("Microphone unavailable: %s", exc)
        microphone = None

    try:
        output_device_index = output_cfg.get("device_index")
        player = AudioPlayer(
            output_device_index=(
                int(output_device_index) if output_device_index is not None else None
            ),
            output_name_hint=output_cfg.get("device_name"),
        )
        LOGGER.info("Audio playback ready")
    except Exception as exc:
        LOGGER.warning("Audio playback unavailable: %s", exc)
        player = None

    try:
        LOGGER.info("Starting motion controller...")
        motion_controller = MotionController.get_instance()
        motion_controller.start_control_loop()
    except Exception as exc:
        LOGGER.warning("Motion controller unavailable: %s", exc)
        motion_controller = None

    LOGGER.info("Runtime skeleton initialized")

    if motion_controller:
        motion_controller.stop_control_loop()
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
