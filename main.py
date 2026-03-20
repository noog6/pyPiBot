"""Command-line entry point for the Theo robot runtime."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import importlib
import importlib.util
import logging
import sys

from ai import RealtimeAPI
from ai.realtime_api import RealtimeAPIStartupError
from config import ConfigController
from core.logging import configure_websocket_library_logging, enable_file_logging, logger
from hardware import CameraController
from interaction.stderr_suppression import suppress_noisy_stderr
from motion import MotionController
from storage.controller import StorageController
from storage.diagnostics import inspect_memory_embeddings
from services.battery_monitor import BatteryMonitor
from services.imu_monitor import ImuMonitor
from services.memory_manager import MemoryManager
from services.ops_orchestrator import OpsOrchestrator
from services.profile_manager import ProfileManager
from services.system_context_coordinator import SystemContextCoordinator


def configure_logging(level_name: str) -> None:
    """Configure application logging."""

    level = logging._nameToLevel.get(level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Keep third-party websocket traces safe by redacting Bearer secrets.
    configure_websocket_library_logging()

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if importlib.util.find_spec("rich") is not None:
        rich_logging = importlib.import_module("rich.logging")
        rich_console = importlib.import_module("rich.console")
        rich_handler = rich_logging.RichHandler(
            rich_tracebacks=True,
            console=rich_console.Console(),
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root_logger.addHandler(rich_handler)
        return

    fallback_handler = logging.StreamHandler()
    fallback_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(fallback_handler)


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


def _normalize_stop_status(stopped: bool) -> str:
    return "stopped" if stopped else "timed_out"


def _stop_imu_monitor_with_status(monitor: ImuMonitor, timeout_s: float = 2.0) -> str:
    """Thin adapter for uniform stop result reporting in `main` shutdown."""

    try:
        monitor.stop_loop(timeout_s=timeout_s)
    except TypeError:
        monitor.stop_loop()
    is_alive = getattr(monitor, "is_loop_alive", None)
    if callable(is_alive):
        return _normalize_stop_status(not bool(is_alive()))
    return "stopped"


def _stop_battery_monitor_with_status(monitor: BatteryMonitor, timeout_s: float = 2.0) -> str:
    """Thin adapter for uniform stop result reporting in `main` shutdown."""

    try:
        monitor.stop_loop(timeout_s=timeout_s)
    except TypeError:
        monitor.stop_loop()
    is_alive = getattr(monitor, "is_loop_alive", None)
    if callable(is_alive):
        return _normalize_stop_status(not bool(is_alive()))
    return "stopped"


def _run_stop_with_status(stop_fn, is_alive_fn=None) -> str:
    """Normalize stop method behavior into `stopped`/`timed_out`/`interrupted`/`error`."""

    try:
        stop_result = stop_fn()
    except KeyboardInterrupt:
        return "interrupted"
    except Exception:
        logger.exception("Shutdown stop call failed")
        return "error"

    if isinstance(stop_result, str):
        return stop_result

    if callable(is_alive_fn):
        try:
            return _normalize_stop_status(not bool(is_alive_fn()))
        except Exception:
            logger.exception("Shutdown liveness check failed")
            return "error"

    return "stopped"


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

    def log_startup_status(
        component: str,
        dependency_class: str,
        status: str,
        *,
        detail: str | None = None,
        level: str = "info",
    ) -> None:
        message = (
            "startup component=%s dependency_class=%s status=%s"
            if detail is None
            else "startup component=%s dependency_class=%s status=%s detail=%s"
        )
        args: tuple[object, ...] = (
            (component, dependency_class, status)
            if detail is None
            else (component, dependency_class, status, detail)
        )
        if level == "warning":
            logger.warning(message, *args)
            return
        if level == "exception":
            logger.exception(message, *args)
            return
        logger.info(message, *args)

    prompts = args.prompts.split("|") if args.prompts else None
    storage_controller = StorageController.get_instance()
    semantic_state = inspect_memory_embeddings(config)
    logger.info(
        "Semantic memory enabled=%s",
        semantic_state["enabled"],
    )
    logger.info(
        "Semantic embeddings table available=%s",
        semantic_state["table_exists"],
    )
    runtime_session_id = f"run-{storage_controller.get_current_run_number()}"
    boot_time = datetime.now(timezone.utc).isoformat()
    memory_manager = MemoryManager.get_instance()
    semantic_startup_summary = memory_manager.get_semantic_startup_summary()
    canary_dimension_label = (
        "null"
        if semantic_startup_summary.get("canary_dimension") is None
        else str(semantic_startup_summary.get("canary_dimension"))
    )
    logger.info(
        "Semantic startup summary enabled=%s provider=%s model=%s rerank_enabled=%s "
        "background_embedding_enabled=%s provider_ready=%s readiness_reason=%s "
        "provider_timeout_s=%s startup_canary_timeout_ms=%s query_timeout_ms=%s write_timeout_ms=%s "
        "effective_timeout_budget_ms=%s max_queries_per_minute=%s max_writes_per_minute=%s "
        "canary_dimension=%s canary_embedding_emitted=%s "
        "canary_error_code=%s canary_timeout_triggered=%s canary_timeout_budget_ms=%s canary_elapsed_ms=%s "
        "canary_observed_elapsed_ms_at_timeout=%s canary_raw_error_code=%s "
        "canary_timer_start=%s canary_queue_delay_ms=%s",
        semantic_startup_summary["enabled"],
        semantic_startup_summary["provider"],
        semantic_startup_summary["provider_model"],
        semantic_startup_summary["rerank_enabled"],
        semantic_startup_summary["background_embedding_enabled"],
        semantic_startup_summary["provider_ready"],
        semantic_startup_summary["provider_readiness_reason"],
        semantic_startup_summary["provider_timeout_s"],
        semantic_startup_summary["startup_canary_timeout_ms"],
        semantic_startup_summary["query_timeout_ms"],
        semantic_startup_summary["write_timeout_ms"],
        semantic_startup_summary["effective_timeout_budget_ms"],
        semantic_startup_summary["max_queries_per_minute"],
        semantic_startup_summary["max_writes_per_minute"],
        canary_dimension_label,
        semantic_startup_summary["canary_embedding_emitted"],
        semantic_startup_summary["canary_error_code"],
        semantic_startup_summary["canary_timeout_triggered"],
        semantic_startup_summary["canary_timeout_budget_ms"],
        semantic_startup_summary["canary_elapsed_ms"],
        semantic_startup_summary["canary_observed_elapsed_ms_at_timeout"],
        semantic_startup_summary["canary_raw_error_code"],
        semantic_startup_summary["canary_timer_start"],
        semantic_startup_summary["canary_queue_delay_ms"],
    )
    logger.info(
        "semantic_runtime=%s reason=%s canary_latency_ms=%s",
        "ready" if semantic_startup_summary["provider_ready"] else "offline",
        semantic_startup_summary["provider_readiness_reason"],
        semantic_startup_summary["canary_latency_ms"],
    )
    memory_manager.set_active_session_id(runtime_session_id)
    logger.info("Assigned runtime memory session_id=%s", runtime_session_id)

    retrieval_metrics_fn = getattr(memory_manager, "get_retrieval_health_metrics", None)
    startup_retrieval_metrics = retrieval_metrics_fn() if callable(retrieval_metrics_fn) else {}
    logger.info(
        "Memory embedding queue startup pending=%s retry_blocked=%s oldest_pending_age_ms=%s",
        startup_retrieval_metrics.get("pending_count", 0),
        startup_retrieval_metrics.get("retry_blocked_count", 0),
        startup_retrieval_metrics.get("oldest_pending_age_ms", 0),
    )

    embedding_worker = memory_manager.get_embedding_worker()
    if embedding_worker is not None:
        try:
            queued = embedding_worker.backfill_recent_missing_embeddings(limit=6)
            logger.info("Queued startup memory embedding backfill count=%s", queued)
            embedding_worker.start()
        except Exception as exc:
            logger.warning("Background memory embedding worker unavailable: %s", exc)
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
    
    # Required dependencies: runtime cannot continue without websocket/session startup.
    try:
        log_startup_status("realtime_api", "required", "starting")
        log_startup_status("audio_input", "required", "starting")
        realtime_api_instance = RealtimeAPI(prompts)
        log_startup_status("audio_input", "required", "ready")
        log_startup_status("realtime_api", "required", "ready")
    except RealtimeAPIStartupError as exc:
        outcome = exc.outcome
        log_startup_status(
            outcome.component,
            outcome.dependency_class,
            outcome.status,
            detail=outcome.detail,
            level="exception",
        )
        log_startup_status(
            "realtime_api",
            "required",
            "fatal",
            detail=str(exc),
            level="exception",
        )
        return 1
    except Exception as exc:
        log_startup_status(
            "realtime_api",
            "required",
            "fatal",
            detail=str(exc),
            level="exception",
        )
        return 1

    event_bus = realtime_api_instance.get_event_bus()

    # Optional dependency: motion control failure degrades capabilities only.
    motion_controller = None
    try:
        log_startup_status("motion_controller", "optional", "starting")
        motion_controller = MotionController.get_instance()
        motion_controller.start_control_loop()
        log_startup_status("motion_controller", "optional", "ready")
    except Exception as exc:
        log_startup_status(
            "motion_controller",
            "optional",
            "warning",
            detail=str(exc),
            level="warning",
        )

    # Optional dependency: camera/vision failure is non-fatal.
    camera_instance = None
    try:
        log_startup_status("camera_controller", "optional", "starting")
        with suppress_noisy_stderr(
            "camera startup",
            env_var="THEO_CAMERA_DEBUG",
            logger=logger,
        ):
            camera_instance = CameraController.get_instance()
        camera_instance.set_realtime_instance(realtime_api_instance)
        realtime_api_instance.camera_controller = camera_instance
        camera_instance.start_vision_loop(vision_loop_period_ms=1000)
        log_startup_status("camera_controller", "optional", "ready")
    except Exception as exc:
        log_startup_status(
            "camera_controller",
            "optional",
            "warning",
            detail=str(exc),
            level="warning",
        )

    # Optional dependency: IMU telemetry failure is non-fatal.
    imu_monitor = None
    imu_event_handler = None
    try:
        log_startup_status("imu_monitor", "optional", "starting")
        imu_monitor = ImuMonitor.get_instance()
        imu_monitor.start_loop()
        imu_event_handler = imu_monitor.create_event_bus_handler(event_bus)
        imu_monitor.register_event_handler(imu_event_handler)
        log_startup_status("imu_monitor", "optional", "ready")
    except Exception as exc:
        log_startup_status(
            "imu_monitor",
            "optional",
            "warning",
            detail=str(exc),
            level="warning",
        )

    # Optional dependency: battery telemetry failure is non-fatal.
    battery_monitor = None
    battery_event_handler = None
    try:
        log_startup_status("battery_monitor", "optional", "starting")
        battery_monitor = BatteryMonitor.get_instance()
        battery_monitor.start_loop()
        battery_event_handler = battery_monitor.create_event_bus_handler(event_bus)
        battery_monitor.register_event_handler(battery_event_handler)
        log_startup_status("battery_monitor", "optional", "ready")
    except Exception as exc:
        log_startup_status(
            "battery_monitor",
            "optional",
            "warning",
            detail=str(exc),
            level="warning",
        )

    # Optional dependency: ops/system-context is additive and may be skipped.
    ops_orchestrator = None
    system_context_coordinator = None
    try:
        log_startup_status("ops_orchestrator", "optional", "starting")
        ops_orchestrator = OpsOrchestrator.get_instance()
        ops_orchestrator.set_realtime_api(realtime_api_instance)
        ops_orchestrator.set_event_bus(event_bus)
        ops_orchestrator.start_loop()
        semantic_state = "ready" if semantic_startup_summary.get("provider_ready") else "timeout"
        semantic_reason = str(semantic_startup_summary.get("provider_readiness_reason") or "unknown")
        system_context_coordinator = SystemContextCoordinator(
            realtime_api=realtime_api_instance,
            ops_orchestrator=ops_orchestrator,
            battery_monitor=battery_monitor,
            run_id=runtime_session_id,
            boot_time=boot_time,
            semantic_state=semantic_state,
            semantic_reason=semantic_reason,
        )
        system_context_coordinator.start()
        log_startup_status("ops_orchestrator", "optional", "ready")
    except Exception as exc:
        log_startup_status(
            "ops_orchestrator",
            "optional",
            "warning",
            detail=str(exc),
            level="warning",
        )

    runtime_exit_code = 0
    interrupted = False
    session_failures_before = 0
    get_session_health = getattr(realtime_api_instance, "get_session_health", None)
    if callable(get_session_health):
        try:
            baseline_health = get_session_health() or {}
            session_failures_before = int(baseline_health.get("failures", 0) or 0)
        except Exception:
            logger.debug("Unable to read baseline session health before runtime start.")
    try:
        asyncio.run(realtime_api_instance.run())
        get_session_health = getattr(realtime_api_instance, "get_session_health", None)
        if callable(get_session_health):
            session_health = get_session_health()
            session_failures = int(session_health.get("failures", 0) or 0)
            if session_failures > 0:
                runtime_exit_code = 1
                logger.error(
                    "Runtime session failed failures=%s last_failure_reason=%s",
                    session_failures,
                    session_health.get("last_failure_reason", ""),
                )
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Program terminated by user")
    except Exception as exc:
        runtime_exit_code = 1
        logger.exception("An unexpected error occurred: %s", exc)
    else:
        if runtime_exit_code == 0 and not interrupted and callable(get_session_health):
            try:
                runtime_health = get_session_health() or {}
                session_failures_after = int(runtime_health.get("failures", 0) or 0)
                if session_failures_after > session_failures_before:
                    runtime_exit_code = 1
                    failure_reason = runtime_health.get("last_failure_reason") or "unknown"
                    logger.error(
                        "Realtime session failed during runtime (failures_before=%s failures_after=%s reason=%s)",
                        session_failures_before,
                        session_failures_after,
                        failure_reason,
                    )
            except Exception:
                logger.debug("Unable to read runtime session health after realtime loop exit.")
    finally:
        # Lifecycle ownership contract (startup/shutdown):
        # - main owns RealtimeAPI, MotionController, CameraController, ImuMonitor,
        #   BatteryMonitor, and OpsOrchestrator start/stop boundaries.
        # - main owns SystemContextCoordinator and embedding worker startup/shutdown.
        # - CameraController and MotionController each own their internal thread joins;
        #   main only invokes stop methods in existing order.
        # - ImuMonitor and BatteryMonitor own their loop lifecycle internals;
        #   main collects normalized stop outcomes via thin adapters.
        shutdown_summary: dict[str, str] = {}
        if system_context_coordinator is not None:
            system_context_coordinator.stop()
            shutdown_summary["system_context_coordinator"] = "stopped"
        if embedding_worker is not None:
            try:
                embedding_worker.stop(timeout_s=0.5)
                shutdown_summary["embedding_worker"] = "stopped"
            except Exception as exc:
                logger.warning("Failed stopping memory embedding worker: %s", exc)
                shutdown_summary["embedding_worker"] = "error"
        if ops_orchestrator:
            try:
                ops_status = ops_orchestrator.stop_loop()
            except KeyboardInterrupt:
                logger.warning(
                    "Interrupted while stopping ops orchestrator; continuing shutdown."
                )
                ops_status = "interrupted"
            shutdown_summary["ops_orchestrator"] = ops_status
            if ops_status != "stopped":
                loop_alive = ops_orchestrator.is_loop_alive()
                logger.warning(
                    "Ops orchestrator shutdown incomplete (status=%s forced_shutdown_continuation=%s loop_alive=%s)",
                    ops_status,
                    ops_orchestrator.forced_shutdown_continuation(),
                    loop_alive,
                )
                if ops_status == "timed_out":
                    logger.warning(
                        "Ops orchestrator timed out. Follow-up: inspect blocked probes/ticks and verify loop thread exits cleanly before restart (loop_alive=%s).",
                        loop_alive,
                    )
        if camera_instance:
            def _stop_camera() -> None:
                with suppress_noisy_stderr(
                    "camera shutdown",
                    env_var="THEO_CAMERA_DEBUG",
                    logger=logger,
                ):
                    close_camera = getattr(camera_instance, "close", None)
                    if callable(close_camera):
                        close_camera()
                    else:
                        camera_instance.stop_vision_loop()

            shutdown_summary["camera_controller"] = _run_stop_with_status(
                _stop_camera,
                getattr(camera_instance, "is_vision_loop_alive", None),
            )
        if motion_controller:
            shutdown_summary["motion_controller"] = _run_stop_with_status(
                motion_controller.stop_control_loop,
                getattr(motion_controller, "is_control_loop_alive", None),
            )
        if imu_monitor:
            if imu_event_handler:
                imu_monitor.unregister_event_handler(imu_event_handler)
            shutdown_summary["imu_monitor"] = _run_stop_with_status(
                lambda: _stop_imu_monitor_with_status(imu_monitor),
            )
        if battery_monitor:
            if battery_event_handler:
                battery_monitor.unregister_event_handler(battery_event_handler)
            shutdown_summary["battery_monitor"] = _run_stop_with_status(
                lambda: _stop_battery_monitor_with_status(battery_monitor),
            )

        logger.info("shutdown summary=%s", shutdown_summary)

    return runtime_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
