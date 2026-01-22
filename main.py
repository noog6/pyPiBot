"""Command-line entry point for the Theo robot runtime."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from ai import RealtimeAPI
from core.logging import logger
from hardware import CameraController
from motion import MotionController


def configure_logging() -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=logging.INFO,
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

    configure_logging()
    args = parse_args(argv)
    if args.diagnostics:
        from config.diagnostics import probe as config_probe
        from ai.diagnostics import probe as ai_probe
        from core.diagnostics import probe as core_probe
        from diagnostics.models import DiagnosticStatus
        from diagnostics.runner import format_results, run_diagnostics
        from interaction.diagnostics import probe as audio_probe
        from services.diagnostics import probe as services_probe
        from storage.diagnostics import probe as storage_probe

        results = run_diagnostics(
            [config_probe, ai_probe, core_probe, audio_probe, services_probe, storage_probe]
        )
        print(format_results(results))
        return 1 if any(result.status is DiagnosticStatus.FAIL for result in results) else 0

    prompts = args.prompts.split("|") if args.prompts else None

    try:
        logger.info("Starting realtime API...")
        realtime_api_instance = RealtimeAPI(prompts)
    except Exception as exc:
        logger.exception("Realtime API startup failed: %s", exc)
        return 1

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
