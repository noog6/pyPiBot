"""Command-line entry point for the Theo robot runtime."""

from __future__ import annotations

import argparse
import logging
import sys

from ai import RealtimeAPI
from config import ConfigController
from core.logging import enable_file_logging, logger
from hardware import CameraController
from motion import MotionController
from storage.controller import StorageController
from services.battery_monitor import BatteryMonitor
from services.imu_monitor import ImuMonitor
from services.profile_manager import ProfileManager


def configure_logging(level_name: str) -> None:
    """Configure application logging."""

    level = logging._nameToLevel.get(level_name.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Raw command-line arguments.

    Returns:
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description="Run the realtime API with optional prompts."
    )
    parser.add_argument("--prompts", type=str, help="Prompts separated by |")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run diagnostics probes and exit.",
    )
    parser.add_argument(
        "--active-user-id",
        type=str,
        help="Override the active user profile id for this session.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Application entry point.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code.
    """

    if argv is None:
        argv = sys.argv[1:]

    config_controller = ConfigController.get_instance()
    config = config_controller.get_config()
    configure_logging(config.get("logging_level", "INFO"))
    args = parse_args(argv)
    if args.active_user_id:
        ProfileManager.get_instance().set_active_user_id(args.active_user_id)
    if args.diagnostics:
        from config.diagnostics import probe as config_probe
        from ai.diagnostics import probe as ai_probe
        from core.diagnostics import probe as core_probe
        from diagnostics.models import DiagnosticStatus
        from diagnostics.runner import format_results, run_diagnostics
        from interaction.diagnostics import probe as audio_probe
        from interaction.microphone_diagnostics import probe as microphone_probe
        from hardware.diagnostics import probe as hardware_probe
        from motion.diagnostics import probe as motion_probe
        from services.diagnostics import probe as services_probe
        from storage.diagnostics import probe as storage_probe

        results = run_diagnostics(
            [
                config_probe,
                ai_probe,
                core_probe,
                audio_probe,
                microphone_probe,
                hardware_probe,
                motion_probe,
                services_probe,
                storage_probe,
            ]
        )
        print(format_results(results))
        return 1 if any(result.status is DiagnosticStatus.FAIL for result in results) else 0

    prompts = args.prompts.split("|") if args.prompts else None
    storage_controller = StorageController.get_instance()
    if config.get("file_logging_enabled", True):
        log_file_path = storage_controller.get_log_file_path()
        enable_file_logging(log_file_path)
        logger.info("Writing logs to %s", log_file_path)

    logger.info( "··········································" )
    logger.info( ":                                        :" )
    logger.info( ":                                        :" )
    logger.info( ":                 ___  _ ___       __    :" )
    logger.info( ":      ___  __ __/ _ \\(_) _ )___  / /_   :" )
    logger.info( ":     / _ \\/ // / ___/ / _  / _ \\/ __/   :" )
    logger.info( ":    / .__/\\_, /_/  /_/____/\\___/\\__/    :" )
    logger.info( ":   /_/   /___/                          :" )
    logger.info( ":                                        :" )
    logger.info( ":                                        :" )
    logger.info( "··········································" )
    
    try:
        logger.info("Starting realtime API...")
        realtime_api_instance = RealtimeAPI(prompts)
    except Exception as exc:
        logger.exception("Realtime API startup failed: %s", exc)
        return 1

    event_bus = realtime_api_instance.get_event_bus()

    motion_controller = None
    try:
        logger.info("Starting motion controller...")
        motion_controller = MotionController.get_instance()
        motion_controller.start_control_loop()
    except Exception as exc:
        logger.warning("Motion controller unavailable: %s", exc)

    camera_instance = None
    try:
        logger.info("Starting camera controller...")
        camera_instance = CameraController.get_instance()
        logger.info("Starting vision thread...")
        camera_instance.set_realtime_instance(realtime_api_instance)
        camera_instance.start_vision_loop(vision_loop_period_ms=1000)
    except Exception as exc:
        logger.warning("Camera controller unavailable: %s", exc)

    imu_monitor = None
    imu_event_handler = None
    try:
        logger.info("Starting IMU monitor...")
        imu_monitor = ImuMonitor.get_instance()
        imu_monitor.start_loop()
        imu_event_handler = imu_monitor.create_event_bus_handler(event_bus)
        imu_monitor.register_event_handler(imu_event_handler)
    except Exception as exc:
        logger.warning("IMU monitor unavailable: %s", exc)

    battery_monitor = None
    battery_event_handler = None
    try:
        logger.info("Starting battery monitor...")
        battery_monitor = BatteryMonitor.get_instance()
        battery_monitor.start_loop()
        battery_event_handler = battery_monitor.create_event_bus_handler(event_bus)
        battery_monitor.register_event_handler(battery_event_handler)
    except Exception as exc:
        logger.warning("Battery monitor unavailable: %s", exc)

    try:
        asyncio.run(realtime_api_instance.run())
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
    except Exception as exc:
        logger.exception("An unexpected error occurred: %s", exc)
    finally:
        if camera_instance:
            camera_instance.stop_vision_loop()
        if motion_controller:
            motion_controller.stop_control_loop()
        if imu_monitor:
            if imu_event_handler:
                imu_monitor.unregister_event_handler(imu_event_handler)
            imu_monitor.stop_loop()
        if battery_monitor:
            if battery_event_handler:
                battery_monitor.unregister_event_handler(battery_event_handler)
            battery_monitor.stop_loop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
