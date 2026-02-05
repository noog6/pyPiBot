"""Realtime API controller for OpenAI realtime connections."""

from __future__ import annotations

import asyncio
import base64
from collections import deque
from datetime import datetime
from io import BytesIO
import json
import os
import random
import re
import signal
import threading
import time
from typing import Any
from urllib import request
from urllib.parse import urlparse

from config import ConfigController
from core.alert_policy import Alert, AlertPolicy
from core.budgeting import RollingWindowBudget
from core.logging import (
    logger,
    log_error,
    log_info,
    log_session_updated,
    log_tool_call,
    log_warning,
    log_ws_event,
)
from interaction import (
    AsyncMicrophone,
    AudioPlayer,
    InteractionState,
    InteractionStateManager,
)
from ai.tools import function_map, tools
from ai.reflection import ReflectionCoordinator, ReflectionContext
from ai.stimuli_coordinator import StimuliCoordinator
from ai.governance import ActionPacket, GovernanceLayer, build_tool_specs
from ai.orchestration import OrchestrationPhase, OrchestrationState
from ai.event_bus import Event, EventBus
from ai.event_injector import EventInjector
from ai.utils import (
    PREFIX_PADDING_MS,
    RUN_TIME_TABLE_LOG_JSON,
    SILENCE_DURATION_MS,
    SILENCE_THRESHOLD,
    build_session_instructions,
)
from motion import (
    MotionController,
    gesture_attention_snap,
    gesture_curious_tilt,
    gesture_idle,
    gesture_nod,
)
from motion.gesture_library import DEFAULT_GESTURES
from services.profile_manager import ProfileManager
from services.reflection_manager import ReflectionManager
from storage import StorageController

RESPONSE_DONE_REFLECTION_PROMPT = """Summarize what changed. Any anomalies? Should anything be remembered?
Return ONLY valid JSON with keys: summary, remember_memory.
Rules:
- summary: one-line run summary.
- remember_memory: null OR an object with keys {{content, tags, importance}}.
- tags: optional list of short strings.
- importance: integer 1-5.

Context:
User input: {user_input}
Assistant reply: {assistant_reply}
Tool calls: {tool_calls}
Response metadata: {response_metadata}
"""

ALLOWED_OUTBOUND_HOSTS = {"api.openai.com"}
ALLOWED_OUTBOUND_SCHEMES = {"https", "wss"}


def _validate_outbound_endpoint(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_OUTBOUND_SCHEMES:
        raise RuntimeError(
            f"Blocked outbound endpoint with non-TLS scheme: {parsed.scheme or 'missing'}"
        )
    if not parsed.hostname:
        raise RuntimeError("Blocked outbound endpoint with missing hostname.")
    if parsed.hostname not in ALLOWED_OUTBOUND_HOSTS:
        raise RuntimeError(f"Blocked outbound endpoint to untrusted host: {parsed.hostname}")
    if parsed.username or parsed.password:
        raise RuntimeError("Blocked outbound endpoint with embedded credentials.")


def _require_websockets() -> Any:
    import importlib
    import importlib.util

    if importlib.util.find_spec("websockets") is None:
        raise RuntimeError("websockets is required for RealtimeAPI")

    websockets = importlib.import_module("websockets")
    return websockets


def _resolve_websocket_exceptions(websockets: Any) -> tuple[type[BaseException], type[BaseException]]:
    connection_closed = getattr(websockets, "ConnectionClosed", None)
    connection_closed_error = getattr(websockets, "ConnectionClosedError", None)
    exceptions_module = getattr(websockets, "exceptions", None)
    if exceptions_module is not None:
        connection_closed = getattr(exceptions_module, "ConnectionClosed", connection_closed)
        connection_closed_error = getattr(
            exceptions_module, "ConnectionClosedError", connection_closed_error
        )
    if connection_closed is None or connection_closed_error is None:
        raise RuntimeError("Unsupported websockets version: missing ConnectionClosed errors.")
    return connection_closed, connection_closed_error


def log_runtime(function_or_name: str, duration: float) -> None:
    jsonl_file = RUN_TIME_TABLE_LOG_JSON
    os.makedirs(os.path.dirname(jsonl_file), exist_ok=True)
    time_record = {
        "timestamp": datetime.now().isoformat(),
        "function": function_or_name,
        "duration": f"{duration:.4f}",
    }
    with open(jsonl_file, "a", encoding="utf-8") as file:
        json.dump(time_record, file)
        file.write("\n")

    logger.info("⏰ %s() took %.4f seconds", function_or_name, duration)


def base64_encode_audio(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")


def _validate_tool_specs(tool_specs: dict[str, Any], tool_defs: list[dict[str, Any]]) -> None:
    required_fields = ("tier", "reversible", "cost_hint", "safety_tags")
    tool_names = {tool.get("name") for tool in tool_defs}
    missing_specs = sorted(tool_names - set(tool_specs.keys()))
    if missing_specs:
        log_warning(
            "Governance tool_specs missing entries for: %s",
            ", ".join(missing_specs),
        )
    unknown_specs = sorted(set(tool_specs.keys()) - tool_names)
    if unknown_specs:
        log_warning(
            "Governance tool_specs include unknown tools: %s",
            ", ".join(unknown_specs),
        )
    for tool_name, spec in tool_specs.items():
        if not isinstance(spec, dict):
            log_warning("Governance tool_specs for %s should be a mapping.", tool_name)
            continue
        missing_fields = [field for field in required_fields if field not in spec]
        if missing_fields:
            log_warning(
                "Governance tool_specs for %s missing fields: %s",
                tool_name,
                ", ".join(missing_fields),
            )
        tier = spec.get("tier")
        if tier is None:
            log_warning("Governance tool_specs for %s missing tier value.", tool_name)
        elif int(tier) not in (0, 1, 2, 3):
            log_warning(
                "Governance tool_specs for %s has unexpected tier %s.",
                tool_name,
                tier,
            )


class RealtimeAPI:
    """Realtime OpenAI API client."""

    def __init__(self, prompts: list[str] | None = None) -> None:
        self.prompts = prompts
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
        config = ConfigController.get_instance().get_config()
        audio_cfg = config.get("audio") or {}
        input_cfg = audio_cfg.get("input") or {}
        output_cfg = audio_cfg.get("output") or {}
        self._audio_input_device_name = input_cfg.get("device_name")
        self._audio_output_device_name = output_cfg.get("device_name")
        self.exit_event = asyncio.Event()
        self.mic = AsyncMicrophone(
            input_device_name=self._audio_input_device_name,
            debug_list_devices=False,
        )
        self.audio_player: AudioPlayer | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ready_event = threading.Event()

        self.assistant_reply = ""
        self._assistant_reply_accum = ""
        self._audio_accum = bytearray()
        self._audio_accum_bytes_target = 9600
        self.response_in_progress = False
        self.function_call: dict[str, Any] | None = None
        self.function_call_args = ""
        self.mic_send_suppress_until = 0.0
        self.rate_limits: dict[str, Any] | None = None
        self.response_start_time: float | None = None
        self.websocket = None
        self.profile_manager = ProfileManager.get_instance()
        self.state_manager = InteractionStateManager()
        self.state_manager.set_gesture_handler(self._handle_state_gesture)
        self.state_manager.set_earcon_handler(self._handle_state_earcon)
        self._speaking_started = False
        self._gesture_last_fired: dict[str, float] = {}
        self._last_gesture_time = 0.0
        self._last_interaction_state = self.state_manager.state
        self._gesture_global_cooldown_s = 10.0
        self._pending_image_stimulus: dict[str, Any] | None = None
        self._pending_image_flush_after_playback = False
        self._last_user_input_text: str | None = None
        self._last_user_input_time: float | None = None
        self._last_user_input_source: str | None = None
        self._last_outgoing_event_type: str | None = None
        self._last_outgoing_response_origin: str | None = None
        self._tool_call_records: list[dict[str, Any]] = []
        self._last_tool_call_results: list[dict[str, Any]] = []
        self._last_response_metadata: dict[str, Any] = {}
        self._injection_response_cooldown_s = float(
            config.get("injection_response_cooldown_s", 0.0)
        )
        self._max_injection_responses_per_minute = int(
            config.get("max_injection_responses_per_minute", 0)
        )
        self._injection_response_timestamps: deque[float] = deque()
        self._injection_response_triggers = config.get("injection_response_triggers") or {}
        self._injection_response_trigger_timestamps: dict[str, deque[float]] = {}
        self._image_response_mode = str(config.get("image_response_mode", "respond")).lower()
        self._image_response_enabled = self._image_response_mode != "catalog_only"
        self._reflection_enabled = bool(config.get("reflection_enabled", False))
        self._reflection_min_interval_s = float(config.get("reflection_min_interval_s", 300.0))
        ops_cfg = config.get("ops") or {}
        budgets_cfg = ops_cfg.get("budgets") or {}
        self._ai_call_budget = RollingWindowBudget(
            int(budgets_cfg.get("ai_calls_per_minute", 0)),
            60.0,
            name="ai_calls",
        )
        self._alert_policy = AlertPolicy.from_config(config)
        governance_cfg = config.get("governance") or {}
        _validate_tool_specs(governance_cfg.get("tool_specs") or {}, tools)
        tool_specs = build_tool_specs(governance_cfg.get("tool_specs") or {})
        self._governance = GovernanceLayer(tool_specs, config)
        self._pending_action: ActionPacket | None = None
        self._pending_action_staging: dict[str, Any] | None = None
        self._approval_timeout_s = float(config.get("approval_timeout_s", 90.0))
        self._tool_definitions = {tool["name"]: tool for tool in tools}
        self._storage = StorageController.get_instance()
        self._presented_actions: set[str] = set()
        self._stop_words = [
            word.strip().lower()
            for word in (config.get("stop_words") or [])
            if isinstance(word, str) and word.strip()
        ]
        self._stop_word_cooldown_s = float(config.get("stop_word_cooldown_s", 0.0))
        self._tool_execution_disabled_until = 0.0
        self._reflection_coordinator = ReflectionCoordinator(
            api_key=self.api_key,
            enabled=self._reflection_enabled,
            min_interval_s=self._reflection_min_interval_s,
            storage=self._storage,
            budget=self._ai_call_budget,
        )
        self._reflection_enqueued = False
        self._response_done_reflection_task: asyncio.Task | None = None
        self._stimuli_coordinator = StimuliCoordinator(
            debounce_window_s=float(config.get("injection_debounce_window_s", 0.4)),
            cooldown_s=self._injection_response_cooldown_s,
            emit_callback=self.maybe_request_response,
        )
        self._gesture_cooldowns_s = {
            "gesture_attention_snap": 10.0,
            "gesture_curious_tilt": 6.0,
            "gesture_nod": 8.0,
            "gesture_idle": 8.0,
        }
        self.orchestration_state = OrchestrationState()
        self.event_bus = EventBus()
        self._event_injector = EventInjector(
            self.event_bus,
            ready_event=self.ready_event,
            is_ready=self.is_ready_for_injections,
            inject_callback=self.inject_event,
        )
        self._session_connection_attempts = 0
        self._session_connections = 0
        self._session_reconnects = 0
        self._session_failures = 0
        self._session_connected = False
        self._last_connect_time: float | None = None
        self._last_disconnect_reason: str | None = None
        self._last_failure_reason: str | None = None

    def get_event_bus(self) -> EventBus:
        return self.event_bus

    def _allow_ai_call(self, reason: str, *, bypass: bool = False) -> bool:
        if bypass:
            return True
        if self._ai_call_budget.allow():
            return True
        self._emit_alert(
            Alert(
                key="budget_ai_calls",
                message=f"AI call budget exhausted ({reason}).",
                severity="warning",
                metadata={"reason": reason},
            )
        )
        return False

    def _record_ai_call(self) -> None:
        self._ai_call_budget.record()

    def _emit_alert(self, alert: Alert) -> None:
        self._alert_policy.emit(self.event_bus, alert)

    def is_ready_for_injections(self) -> bool:
        return (
            self.ready_event.is_set()
            and self.websocket is not None
            and self.loop is not None
            and self.loop.is_running()
        )

    def get_session_health(self) -> dict[str, Any]:
        return {
            "connected": self._session_connected,
            "ready": self.ready_event.is_set(),
            "connection_attempts": self._session_connection_attempts,
            "connections": self._session_connections,
            "reconnects": self._session_reconnects,
            "failures": self._session_failures,
            "last_connect_time": self._last_connect_time or 0.0,
            "last_disconnect_reason": self._last_disconnect_reason or "",
            "last_failure_reason": self._last_failure_reason or "",
        }

    def _note_connection_attempt(self) -> None:
        self._session_connection_attempts += 1

    def _note_connected(self) -> None:
        self._session_connections += 1
        self._session_connected = True
        self._last_connect_time = time.time()
        self._last_disconnect_reason = None

    def _note_disconnect(self, reason: str) -> None:
        self._session_connected = False
        self._last_disconnect_reason = reason

    def _note_failure(self, reason: str) -> None:
        self._session_failures += 1
        self._last_failure_reason = reason

    def _note_reconnect(self) -> None:
        self._session_reconnects += 1

    def _track_outgoing_event(
        self,
        event: dict[str, Any],
        *,
        origin: str | None = None,
    ) -> None:
        event_type = event.get("type")
        if not event_type:
            return
        self._last_outgoing_event_type = str(event_type)
        if event_type == "response.create":
            self._last_outgoing_response_origin = origin

    def inject_event(self, event: Event) -> None:
        message, request_response = self._format_event_for_injection(event)
        if event.request_response is not None:
            request_response = event.request_response
        self._log_injection_event(event, request_response)
        self._send_text_message(message, request_response=request_response)

    def _log_injection_event(self, event: Event, request_response: bool) -> None:
        metadata = event.metadata or {}
        parts = [
            "injection:",
            f"source={event.source}",
        ]
        if event.kind:
            parts.append(f"kind={event.kind}")
        severity = metadata.get("severity")
        if severity:
            parts.append(f"severity={severity}")
        reason = metadata.get("reason") or metadata.get("event_type")
        if reason:
            parts.append(f"reason={reason}")
        parts.append(f"create_response={'true' if request_response else 'false'}")
        log_info(" ".join(parts))

    def _send_text_message(self, message: str, request_response: bool = True) -> None:
        if not self.loop:
            logger.debug("Unable to send message; event loop unavailable.")
            return
        future = asyncio.run_coroutine_threadsafe(
            self.send_text_message_to_conversation(
                message,
                request_response=request_response,
            ),
            self.loop,
        )

        def _on_complete(task) -> None:
            try:
                task.result()
            except Exception as exc:
                logger.warning("Failed to send queued message: %s", exc)

        future.add_done_callback(_on_complete)

    def _format_event_for_injection(self, event: Event) -> tuple[str, bool]:
        if event.content:
            return event.content, True
        if event.source == "imu":
            metadata = event.metadata
            return (
                "IMU event: "
                f"{metadata.get('event_type')} ({metadata.get('severity')}) "
                f"details={metadata.get('details')}",
                False,
            )
        if event.source == "battery":
            metadata = event.metadata
            percent = float(metadata.get("percent_of_range", 0.0)) * 100
            voltage = float(metadata.get("voltage", 0.0))
            event_type = metadata.get("event_type")
            if event_type == "clear":
                return (
                    "Battery warning cleared: "
                    f"{voltage:.2f}V ({percent:.1f}% of range)",
                    False,
                )
            severity = metadata.get("severity", "info")
            return (
                "Battery voltage: "
                f"{voltage:.2f}V ({percent:.1f}% of range) "
                f"severity={severity}",
                severity != "info",
            )
        return f"{event.source} event: {event.metadata}", True

    def _handle_state_gesture(self, state: InteractionState) -> None:
        """Hook for gesture cues on state transitions."""
        previous_state = self._last_interaction_state
        self._last_interaction_state = state

        if state == InteractionState.SPEAKING:
            logger.debug("Gesture cue skipped for state: %s", state.value)
            return

        gesture_name: str | None = None
        delay_ms = 0
        if state == InteractionState.LISTENING:
            gesture_name = "gesture_attention_snap"
        elif state == InteractionState.THINKING:
            gesture_name = "gesture_curious_tilt"
            delay_ms = random.randint(150, 300)
        elif state == InteractionState.IDLE and previous_state == InteractionState.SPEAKING:
            gesture_name = "gesture_nod"

        if not gesture_name:
            logger.debug("Gesture cue ignored for state: %s", state.value)
            return

        try:
            controller = MotionController.get_instance()
        except Exception as exc:
            logger.warning("Gesture cue skipped: motion controller unavailable (%s).", exc)
            return

        if not controller.is_control_loop_alive():
            logger.info(
                "Gesture cue skipped: motion controller not running (state=%s).",
                state.value,
            )
            return

        try:
            servos = controller.servo_registry.get_servos()
        except Exception as exc:
            logger.warning("Gesture cue skipped: servo registry unavailable (%s).", exc)
            return

        if not servos:
            logger.warning("Gesture cue skipped: servo registry not ready.")
            return

        if controller.is_moving():
            logger.debug("Gesture cue skipped: motion active for state %s.", state.value)
            return

        with controller._queue_lock:
            queued_actions = list(controller.action_queue)

        if queued_actions:
            logger.debug(
                "Gesture cue skipped: action queue not empty (%s items).",
                len(queued_actions),
            )
            return

        now = time.monotonic()
        global_elapsed = now - self._last_gesture_time
        if global_elapsed < self._gesture_global_cooldown_s:
            logger.debug(
                "Gesture cue skipped: global cooldown %.2fs remaining for %s.",
                self._gesture_global_cooldown_s - global_elapsed,
                state.value,
            )
            return

        per_cooldown = self._gesture_cooldowns_s.get(gesture_name, 0.0)
        last_fired = self._gesture_last_fired.get(gesture_name, 0.0)
        per_elapsed = now - last_fired
        if per_elapsed < per_cooldown:
            logger.debug(
                "Gesture cue skipped: %s cooldown %.2fs remaining for %s.",
                gesture_name,
                per_cooldown - per_elapsed,
                state.value,
            )
            return

        action = None
        if gesture_name == "gesture_attention_snap":
            action = gesture_attention_snap(delay_ms=delay_ms)
        elif gesture_name == "gesture_curious_tilt":
            action = gesture_curious_tilt(delay_ms=delay_ms)
        elif gesture_name == "gesture_nod":
            action = gesture_nod(delay_ms=delay_ms)
        elif gesture_name == "gesture_idle":
            action = gesture_idle(delay_ms=delay_ms)

        if action is None:
            logger.warning("Gesture cue skipped: no action built for %s.", gesture_name)
            return

        controller.add_action_to_queue(action)
        self._gesture_last_fired[gesture_name] = now
        self._last_gesture_time = now
        logger.info(
            "Gesture cue emitted: state=%s gesture=%s delay_ms=%s global_cd=%.2fs "
            "gesture_cd=%.2fs",
            state.value,
            gesture_name,
            delay_ms,
            self._gesture_global_cooldown_s,
            per_cooldown,
        )

    def _handle_state_earcon(self, state: InteractionState) -> None:
        """Hook for earcon cues on state transitions."""
        logger.debug("Earcon cue for state: %s", state.value)

    def _record_user_input(self, text: str, *, source: str) -> None:
        clean_text = text.strip()
        if not clean_text:
            return
        self._last_user_input_text = clean_text
        self._last_user_input_time = time.monotonic()
        self._last_user_input_source = source

    def _find_stop_word(self, text: str) -> str | None:
        if not text or not self._stop_words:
            return None
        lowered = text.lower()
        for word in self._stop_words:
            if " " in word:
                if word in lowered:
                    return word
                continue
            if re.search(rf"\\b{re.escape(word)}\\b", lowered):
                return word
        return None

    def _tool_execution_cooldown_remaining(self) -> float:
        remaining = self._tool_execution_disabled_until - time.monotonic()
        return remaining if remaining > 0 else 0.0

    async def _handle_stop_word(
        self,
        text: str,
        websocket: Any,
        *,
        source: str,
    ) -> bool:
        stop_word = self._find_stop_word(text)
        if not stop_word:
            return False
        now = time.monotonic()
        if self._stop_word_cooldown_s > 0:
            self._tool_execution_disabled_until = max(
                self._tool_execution_disabled_until,
                now + self._stop_word_cooldown_s,
            )
        else:
            self._tool_execution_disabled_until = max(self._tool_execution_disabled_until, now)
        if self._pending_action:
            await self._reject_tool_call(
                self._pending_action,
                f"Stop word '{stop_word}' detected; tool execution paused.",
                websocket,
                staging=self._pending_action_staging,
                status="cancelled",
            )
            self._clear_pending_action()
        else:
            cooldown_msg = ""
            if self._stop_word_cooldown_s > 0:
                cooldown_msg = f" for {int(self._stop_word_cooldown_s)}s"
            await self.send_assistant_message(
                f"Stop word '{stop_word}' detected. Pending actions cancelled and tool use paused{cooldown_msg}.",
                websocket,
            )
        return True

    def _extract_transcript(self, event: dict[str, Any]) -> str | None:
        transcript = event.get("transcript")
        if isinstance(transcript, str) and transcript.strip():
            return transcript
        item = event.get("item", {})
        item_transcript = item.get("transcript")
        if isinstance(item_transcript, str) and item_transcript.strip():
            return item_transcript
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                content_transcript = content.get("transcript")
                if isinstance(content_transcript, str) and content_transcript.strip():
                    return content_transcript
                if content.get("type") == "input_text" and content.get("text"):
                    return content["text"]
        return None

    def _maybe_enqueue_reflection(self, trigger: str) -> None:
        if self._reflection_enqueued:
            logger.debug(
                "Reflection enqueue skipped: already enqueued (trigger=%s).",
                trigger,
            )
            return
        context = ReflectionContext(
            last_user_input=self._last_user_input_text,
            assistant_reply=self._assistant_reply_accum,
            tool_calls=list(self._tool_call_records),
            response_metadata={
                **self._last_response_metadata,
                "trigger": trigger,
                "last_user_input_source": self._last_user_input_source,
                "last_user_input_time": self._last_user_input_time,
            },
        )
        logger.debug(
            "Reflection enqueue requested (trigger=%s, tool_calls=%d, reply_len=%d).",
            trigger,
            len(self._tool_call_records),
            len(self._assistant_reply_accum),
        )
        self._last_tool_call_results = list(self._tool_call_records)
        self._reflection_coordinator.enqueue_reflection(context)
        self._reflection_enqueued = True

    def _clip_text(self, text: str | None, limit: int = 1200) -> str:
        if not text:
            return "(none)"
        return text if len(text) <= limit else f"{text[:limit]}…"

    def _find_gesture_definition(self, gesture_name: str) -> dict[str, Any] | None:
        for definition in DEFAULT_GESTURES:
            if definition.name == gesture_name:
                total_duration = sum(frame.duration_ms for frame in definition.frames)
                return {
                    "name": definition.name,
                    "total_duration_ms": total_duration,
                    "frame_count": len(definition.frames),
                }
        return None

    def _stage_action(self, action: ActionPacket) -> dict[str, Any]:
        tool_def = self._tool_definitions.get(action.tool_name)
        errors: list[str] = []
        warnings: list[str] = []
        bounds_checks: list[dict[str, Any]] = []

        if tool_def is None:
            warnings.append("Tool definition not found; skipped schema validation.")
        else:
            schema = tool_def.get("parameters") or {}
            properties = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            provided = set(action.tool_args.keys())

            missing = sorted(required - provided)
            if missing:
                errors.append(f"Missing required args: {', '.join(missing)}.")

            extra = sorted(provided - set(properties.keys()))
            if extra:
                warnings.append(f"Unknown args will be ignored: {', '.join(extra)}.")

            for key, spec in properties.items():
                if key not in action.tool_args:
                    continue
                value = action.tool_args[key]
                expected = spec.get("type")
                if expected and not self._validate_arg_type(expected, value):
                    errors.append(f"Arg '{key}' expected type {expected}.")
                minimum = spec.get("minimum")
                maximum = spec.get("maximum")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if minimum is not None or maximum is not None:
                        within = True
                        if minimum is not None and value < minimum:
                            within = False
                        if maximum is not None and value > maximum:
                            within = False
                        bounds_checks.append(
                            {
                                "field": key,
                                "value": value,
                                "min": minimum,
                                "max": maximum,
                                "within": within,
                            }
                        )
                        if not within:
                            errors.append(
                                f"Arg '{key}' out of bounds ({minimum}..{maximum})."
                            )

        motion_info = None
        if action.tool_name.startswith("gesture_"):
            gesture = self._find_gesture_definition(action.tool_name)
            motion_info = {
                "gesture": action.tool_name,
                "duration_ms": gesture["total_duration_ms"] if gesture else None,
                "frame_count": gesture["frame_count"] if gesture else None,
                "delay_ms": action.tool_args.get("delay_ms", 0),
                "intensity": action.tool_args.get("intensity", 1.0),
                "safe_servo_bounds": {"pan": [-90.0, 90.0], "tilt": [-45.0, 45.0]},
            }

        explanation = self._describe_staged_action(action, motion_info)

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "bounds_checks": bounds_checks,
            "explanation": explanation,
            "motion": motion_info,
        }

    def _validate_arg_type(self, expected: str, value: Any) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        return True

    def _normalize_dry_run_flag(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _extract_dry_run_flag(self, args: dict[str, Any]) -> bool:
        if "dry_run" not in args:
            return False
        return self._normalize_dry_run_flag(args.pop("dry_run"))

    def _describe_staged_action(
        self, action: ActionPacket, motion_info: dict[str, Any] | None
    ) -> str:
        if action.tool_name.startswith("read_") or action.tool_name.startswith("get_"):
            return f"Would read data via {action.tool_name}."
        if action.tool_name.startswith("recall_"):
            return f"Would fetch stored memories via {action.tool_name}."
        if action.tool_name.startswith("gesture_") and motion_info:
            duration = motion_info.get("duration_ms")
            return (
                f"Would queue gesture '{motion_info['gesture']}' "
                f"for ~{duration}ms within safe servo bounds."
            )
        return f"Would call {action.tool_name} with the proposed arguments."

    def _build_approval_prompt(self, action: ActionPacket) -> str:
        packet = self._format_action_packet(action)
        if action.tier >= 3:
            return (
                "Approval required for this action.\n"
                f"{packet}\n"
                "Approve? Reply exactly: 'Yes, do it now' / no / modify."
            )
        return f"Approval required for this action.\n{packet}\nApprove? (yes / no / modify)"

    def _format_action_packet(self, action: ActionPacket) -> str:
        alternatives = ", ".join(action.alternatives) if action.alternatives else "(none)"
        return (
            "Action packet:\n"
            f"- what: {action.what}\n"
            f"- why: {action.why}\n"
            f"- impact: {action.impact}\n"
            f"- rollback: {action.rollback}\n"
            f"- cost: {action.cost}\n"
            f"- confidence: {action.confidence:.2f}\n"
            f"- alternatives: {alternatives}"
        )

    def _clear_pending_action(self) -> None:
        if self._pending_action:
            self._presented_actions.discard(self._pending_action.id)
        self._pending_action = None
        self._pending_action_staging = None

    async def _maybe_handle_approval_response(
        self, text: str, websocket: Any
    ) -> bool:
        if not self._pending_action:
            return False
        if await self._handle_stop_word(text, websocket, source="approval_response"):
            return True
        cooldown_remaining = self._tool_execution_cooldown_remaining()
        if cooldown_remaining > 0:
            await self._reject_tool_call(
                self._pending_action,
                f"Tool execution paused for {cooldown_remaining:.0f}s due to stop word.",
                websocket,
                staging=self._pending_action_staging,
                status="cancelled",
            )
            self._clear_pending_action()
            return True
        now = time.monotonic()
        action = self._pending_action
        if action.expiry_ts is not None and now > action.expiry_ts:
            await self.send_assistant_message(
                "Approval window expired. Please ask again if you still want this.",
                websocket,
            )
            self._clear_pending_action()
            return True

        normalized = text.strip().lower()
        if not normalized:
            return False

        requires_phrase = action.tier >= 3
        approved = False
        if requires_phrase:
            approved = normalized in {"yes, do it now", "yes do it now"}
        else:
            approved = normalized in {"yes", "y", "approve", "ok", "okay"}

        if approved:
            staging = self._pending_action_staging or self._stage_action(action)
            await self._execute_action(action, staging, websocket)
            self._clear_pending_action()
            return True

        if normalized in {"no", "n", "deny", "cancel"}:
            await self._reject_tool_call(
                action,
                "User declined approval.",
                websocket,
                status="cancelled",
            )
            self._clear_pending_action()
            return True

        if normalized in {"modify", "change"}:
            await self.send_assistant_message(
                "Okay. Tell me what to change and I will restage the action.",
                websocket,
            )
            self._clear_pending_action()
            return True

        await self.send_assistant_message(
            "Please reply with: yes / no / modify.",
            websocket,
        )
        return True

    def _build_response_done_prompt(self, trigger: str) -> str:
        tool_calls = self._clip_text(
            json.dumps(self._tool_call_records, ensure_ascii=False)
        )
        response_metadata = self._clip_text(
            json.dumps(
                {
                    **self._last_response_metadata,
                    "trigger": trigger,
                    "last_user_input_source": self._last_user_input_source,
                    "last_user_input_time": self._last_user_input_time,
                },
                ensure_ascii=False,
            )
        )
        return RESPONSE_DONE_REFLECTION_PROMPT.format(
            user_input=self._clip_text(self._last_user_input_text),
            assistant_reply=self._clip_text(self._assistant_reply_accum),
            tool_calls=tool_calls,
            response_metadata=response_metadata,
        )

    def _call_openai_prompt(self, prompt: str) -> str:
        endpoint = "https://api.openai.com/v1/chat/completions"
        _validate_outbound_endpoint(endpoint)
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a concise internal auditor."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 220,
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
        response_payload = json.loads(body)
        return response_payload["choices"][0]["message"]["content"].strip()

    def _parse_response_done_payload(self, raw_response: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            logger.debug("response.done reflection returned non-JSON payload.")
            return None
        if not isinstance(payload, dict):
            logger.debug("response.done reflection payload is not an object.")
            return None
        return payload

    async def _run_response_done_reflection(self, trigger: str) -> None:
        if not self.api_key:
            logger.debug("Skipping response.done reflection: missing API key.")
            return
        if not self._allow_ai_call("response_done_reflection"):
            logger.info("Skipping response.done reflection: AI call budget exhausted.")
            return
        prompt = self._build_response_done_prompt(trigger)
        try:
            raw_response = await asyncio.to_thread(self._call_openai_prompt, prompt)
        except Exception as exc:  # noqa: BLE001 - guard background call
            logger.warning("response.done reflection failed: %s", exc)
            return
        self._record_ai_call()
        payload = self._parse_response_done_payload(raw_response)
        if not payload:
            return
        summary = payload.get("summary")
        if isinstance(summary, str) and summary.strip():
            logger.info("response.done summary: %s", summary.strip())
        remember_payload = payload.get("remember_memory")
        if not isinstance(remember_payload, dict):
            return
        content = remember_payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        tags = remember_payload.get("tags")
        if not isinstance(tags, list):
            tags = None
        importance = remember_payload.get("importance", 3)
        if not isinstance(importance, int):
            importance = 3
        remember_func = function_map.get("remember_memory")
        if remember_func is None:
            logger.warning("response.done reflection skipped: remember_memory unavailable.")
            return
        result = await remember_func(
            content=content.strip(),
            tags=tags,
            importance=importance,
        )
        logger.info("response.done remember_memory stored: %s", result.get("memory_id"))

    def _enqueue_response_done_reflection(self, trigger: str) -> None:
        if self._response_done_reflection_task and not self._response_done_reflection_task.done():
            logger.debug("response.done reflection skipped: task already running.")
            return

        async def _runner() -> None:
            try:
                await self._run_response_done_reflection(trigger)
            except Exception as exc:  # noqa: BLE001 - safety for background task
                logger.warning("response.done reflection task failed: %s", exc)

        self._response_done_reflection_task = asyncio.create_task(_runner())

    def _on_playback_complete(self) -> None:
        logger.info("Playback complete -> restarting mic")

        self.mic.stop_receiving()
        self.mic_send_suppress_until = time.monotonic() + 1.2

        if self.websocket:
            try:
                asyncio.create_task(
                    self.websocket.send(json.dumps({"type": "input_audio_buffer.clear"}))
                )
            except Exception:
                logger.exception("Failed to send input_audio_buffer.clear")

        self.mic.start_recording()
        if self._pending_image_flush_after_playback and self._pending_image_stimulus:
            self._pending_image_flush_after_playback = False
            self._schedule_pending_image_flush("playback complete")

    async def run(self) -> None:
        websockets = _require_websockets()
        _, ConnectionClosedError = _resolve_websocket_exceptions(websockets)

        self.loop = asyncio.get_running_loop()
        self.ready_event.clear()
        self._event_injector.start()
        pending_disconnect_reason: str | None = None

        def _playback_complete_from_thread() -> None:
            if self.loop:
                self.loop.call_soon_threadsafe(self._on_playback_complete)

        self.audio_player = AudioPlayer(
            on_playback_complete=_playback_complete_from_thread,
            output_device_name=self._audio_output_device_name,
        )

        try:
            while True:
                try:
                    self._note_connection_attempt()
                    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
                    _validate_outbound_endpoint(url)
                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    async with websockets.connect(
                        url,
                        additional_headers=headers,
                        close_timeout=120,
                        ping_interval=30,
                        ping_timeout=10,
                    ) as websocket:
                        log_info("✅ Connected to the server.", style="bold green")
                        self._note_connected()
                        pending_disconnect_reason = None

                        await self.initialize_session(websocket)
                        ws_task = asyncio.create_task(self.process_ws_messages(websocket))

                        logger.info(
                            "Conversation started. Speak freely, and the assistant will respond."
                        )

                        self.websocket = websocket

                        self.loop.add_signal_handler(signal.SIGTERM, self.shutdown_handler)

                        if self.prompts:
                            await self.send_initial_prompts(websocket)
                        else:
                            self.mic.start_recording()
                            logger.info("Recording started. Listening for speech...")

                        await self.send_audio_loop(websocket)

                        await ws_task

                    break
                except ConnectionClosedError as exc:
                    pending_disconnect_reason = str(exc)
                    self._note_disconnect(pending_disconnect_reason)
                    if "keepalive ping timeout" in str(exc):
                        logger.warning(
                            "WebSocket connection lost due to keepalive ping timeout. Reconnecting..."
                        )
                        self._note_reconnect()
                        await asyncio.sleep(1)
                        continue
                    logger.exception("WebSocket connection closed unexpectedly.")
                    self._note_failure(str(exc))
                    break
                except Exception as exc:
                    logger.exception("An unexpected error occurred: %s", exc)
                    pending_disconnect_reason = str(exc)
                    self._note_failure(str(exc))
                    break
                finally:
                    if self.audio_player:
                        self.audio_player.close()
                    self.mic.stop_recording()
                    self.mic.close()
                    self.websocket = None
                    self.ready_event.clear()
                    if pending_disconnect_reason:
                        self._note_disconnect(pending_disconnect_reason)
        finally:
            self._event_injector.stop()

    async def initialize_session(self, websocket: Any) -> None:
        profile_context = self.profile_manager.get_profile_context()
        reflection_manager = ReflectionManager.get_instance()
        recent_lessons = reflection_manager.get_recent_lessons()
        lessons_block = None
        if recent_lessons:
            lesson_lines = "\n".join(f"- {lesson}" for lesson in recent_lessons)
            lessons_block = (
                "Lessons learned:\n"
                f"{lesson_lines}\n"
                "Use these lessons when planning next actions."
            )
        instructions = build_session_instructions(
            profile_context.to_instruction_block(),
            lessons_block,
        )
        session_update = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": SILENCE_THRESHOLD,
                            "prefix_padding_ms": PREFIX_PADDING_MS,
                            "silence_duration_ms": SILENCE_DURATION_MS,
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "voice": "ballad",
                    },
                },
                "instructions": instructions,
                "tools": tools,
            },
        }
        log_ws_event("Outgoing", session_update)
        await websocket.send(json.dumps(session_update))

    async def process_ws_messages(self, websocket: Any) -> None:
        websockets = _require_websockets()
        ConnectionClosed, _ = _resolve_websocket_exceptions(websockets)
        while True:
            try:
                message = await websocket.recv()
                event = json.loads(message)
                log_ws_event("Incoming", event)
                await self.handle_event(event, websocket)
            except ConnectionClosed:
                log_warning("⚠️ WebSocket connection lost.")
                self._note_disconnect("websocket connection closed")
                break

    async def handle_event(self, event: dict[str, Any], websocket: Any) -> None:
        event_type = event.get("type")
        if event_type == "response.created":
            origin = "unknown"
            if self._last_outgoing_event_type == "response.create":
                origin = self._last_outgoing_response_origin or "unknown"
            log_info(f"response.created: origin={origin}")
            self.orchestration_state.transition(
                OrchestrationPhase.PLAN,
                reason="response created",
            )
            if self.audio_player:
                self.audio_player.start_response()
            self._audio_accum.clear()
            self.mic.start_receiving()
            self.response_in_progress = True
            self._speaking_started = False
            self._assistant_reply_accum = ""
            self._tool_call_records = []
            self._last_tool_call_results = []
            self._last_response_metadata = {}
            self._reflection_enqueued = False
            self.state_manager.update_state(InteractionState.THINKING, "response created")
        elif event_type == "response.output_item.added":
            await self.handle_output_item_added(event)
        elif event_type == "response.function_call_arguments.delta":
            self.function_call_args += event.get("delta", "")
        elif event_type == "response.function_call_arguments.done":
            await self.handle_function_call(event, websocket)
        elif event_type == "response.text.delta":
            delta = event.get("delta", "")
            self.assistant_reply += delta
            self._assistant_reply_accum += delta
            self.state_manager.update_state(InteractionState.SPEAKING, "text output")
        elif event_type == "response.output_audio.delta":
            audio_data = base64.b64decode(event["delta"])
            self._audio_accum.extend(audio_data)

            if not self._speaking_started:
                self._speaking_started = True
                self.state_manager.update_state(InteractionState.SPEAKING, "audio output")

            if len(self._audio_accum) >= self._audio_accum_bytes_target:
                if self.audio_player:
                    self.audio_player.play_audio(bytes(self._audio_accum))
                self._audio_accum.clear()
        elif event_type == "response.output_audio.done":
            await self.handle_audio_response_done()
            self.state_manager.update_state(InteractionState.IDLE, "audio output done")
        elif event_type == "response.output_audio_transcript.delta":
            delta = event.get("delta", "")
            self.assistant_reply += delta
            self._assistant_reply_accum += delta
        elif event_type == "response.output_audio_transcript.done":
            await self.handle_transcribe_response_done()
            self.state_manager.update_state(
                InteractionState.IDLE,
                "audio transcript done",
            )
        elif event_type == "response.done":
            await self.handle_response_done(event)
        elif event_type == "response.completed":
            await self.handle_response_completed(event)
        elif event_type == "error":
            await self.handle_error(event, websocket)
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = self._extract_transcript(event)
            if transcript:
                self._record_user_input(transcript, source="input_audio_transcription")
                if await self._handle_stop_word(
                    transcript,
                    websocket,
                    source="input_audio_transcription",
                ):
                    return
                if await self._maybe_handle_approval_response(transcript, websocket):
                    return
        elif event_type == "input_audio_buffer.speech_started":
            logger.info("Speech detected, listening...")
            self.orchestration_state.transition(
                OrchestrationPhase.SENSE,
                reason="speech started",
            )
            self.state_manager.update_state(InteractionState.LISTENING, "speech started")
        elif event_type == "input_audio_buffer.speech_stopped":
            await self.handle_speech_stopped(websocket)
            self.state_manager.update_state(InteractionState.THINKING, "speech stopped")
        elif event_type == "rate_limits.updated":
            rl = {r["name"]: r for r in event.get("rate_limits", [])}
            self.rate_limits = rl

            req = rl.get("requests", {})
            tok = rl.get("tokens", {})
            logger.info(
                "Rate limits: requests %s/%s reset=%ss | tokens %s/%s reset=%ss",
                req.get("remaining"),
                req.get("limit"),
                req.get("reset_seconds"),
                tok.get("remaining"),
                tok.get("limit"),
                tok.get("reset_seconds"),
            )
        elif event_type == "session.updated":
            log_session_updated(event, full_payload=True)
            if not self.ready_event.is_set():
                logger.info("Realtime API ready to accept injections.")
                self.ready_event.set()

    async def handle_output_item_added(self, event: dict[str, Any]) -> None:
        item = event.get("item", {})
        if item.get("type") == "function_call":
            self.function_call = item
            self.function_call_args = ""

    async def handle_function_call(self, event: dict[str, Any], websocket: Any) -> None:
        if self.function_call:
            function_name = self.function_call.get("name")
            call_id = self.function_call.get("call_id")
            logger.info(
                "Function call: %s with args: %s",
                function_name,
                self.function_call_args,
            )
            self.orchestration_state.transition(
                OrchestrationPhase.ACT,
                reason=f"function_call {function_name}",
            )
            try:
                args = json.loads(self.function_call_args) if self.function_call_args else {}
            except json.JSONDecodeError:
                args = {}
            dry_run_requested = self._extract_dry_run_flag(args)
            action = self._governance.build_action_packet(
                function_name,
                call_id,
                args,
                reason=f"function_call {function_name}",
            )
            cooldown_remaining = self._tool_execution_cooldown_remaining()
            if cooldown_remaining > 0:
                await self._reject_tool_call(
                    action,
                    f"Tool execution paused for {cooldown_remaining:.0f}s due to stop word.",
                    websocket,
                    status="cancelled",
                )
                return
            staging = self._stage_action(action)
            if dry_run_requested:
                await self._send_dry_run_output(action, staging, websocket)
                return
            if not staging["valid"]:
                await self._reject_tool_call(
                    action,
                    "Argument validation failed.",
                    websocket,
                    staging=staging,
                    status="invalid_arguments",
                )
                return
            decision = self._governance.review(action)
            log_info(f"🛡️ Governance decision: {decision.status} ({decision.reason}) {action.summary()}")
            if decision.approved:
                await self._execute_action(action, staging, websocket)
            elif decision.needs_confirmation:
                action.requires_confirmation = True
                self._pending_action = action
                self._pending_action_staging = staging
                action.expiry_ts = time.monotonic() + self._approval_timeout_s
                await self._request_tool_confirmation(action, decision.reason, websocket, staging)
            else:
                await self._reject_tool_call(
                    action,
                    decision.reason,
                    websocket,
                    staging=staging,
                    status="denied",
                )

    async def execute_function_call(
        self,
        function_name: str,
        call_id: str,
        args: dict[str, Any],
        websocket: Any,
        *,
        action: ActionPacket | None = None,
        staging: dict[str, Any] | None = None,
    ) -> None:
        if function_name in function_map:
            try:
                result = await function_map[function_name](**args)
                log_tool_call(function_name, args, result)
            except Exception as exc:
                error_message = f"Error executing function '{function_name}': {exc}"
                log_error(error_message)
                result = {"error": error_message}
                await self.send_error_message_to_assistant(error_message, websocket)
        else:
            error_message = (
                f"Function '{function_name}' not found. Add to function_map in tools.py."
            )
            log_error(error_message)
            result = {"error": error_message}
            await self.send_error_message_to_assistant(error_message, websocket)

        self._tool_call_records.append(
            {
                "name": function_name,
                "call_id": call_id,
                "args": args,
                "result": result,
                "action_packet": action.to_payload() if action else None,
                "staging": staging,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self._last_tool_call_results = list(self._tool_call_records)

        output_payload: dict[str, Any] = {"result": result}
        if action:
            output_payload["action_packet"] = action.to_payload()
        if staging is not None:
            output_payload["staging"] = staging

        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(output_payload),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="tool_output")
        await websocket.send(json.dumps(response_create_event))

        self.function_call = None
        self.function_call_args = ""

    async def _execute_action(
        self,
        action: ActionPacket,
        staging: dict[str, Any],
        websocket: Any,
    ) -> None:
        if not staging.get("valid", True):
            await self._reject_tool_call(
                action,
                "Argument validation failed.",
                websocket,
                staging=staging,
                status="invalid_arguments",
            )
            return
        if action.tier > 1 and action.id not in self._presented_actions:
            await self._present_action(
                action,
                staging,
                websocket,
                reason="pre-execution presentation",
            )
        await self.execute_function_call(
            action.tool_name,
            action.id,
            action.tool_args,
            websocket,
            action=action,
            staging=staging,
        )
        self._governance.record_execution(action)
        self._presented_actions.discard(action.id)

    async def _request_tool_confirmation(
        self,
        action: ActionPacket,
        reason: str,
        websocket: Any,
        staging: dict[str, Any],
    ) -> None:
        summary = action.summary()
        tool_metadata = self._governance.describe_tool(action.tool_name)
        message = self._build_approval_prompt(action)
        await self.send_assistant_message(message, websocket)
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": "awaiting_confirmation",
                        "message": message,
                        "tool": action.tool_name,
                        "tool_metadata": tool_metadata,
                        "action_packet": action.to_payload(),
                        "staging": staging,
                        "summary": summary,
                        "reason": reason,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="tool_output")
        await websocket.send(json.dumps(response_create_event))
        self._presented_actions.add(action.id)
        self.function_call = None
        self.function_call_args = ""

    async def _present_action(
        self,
        action: ActionPacket,
        staging: dict[str, Any],
        websocket: Any,
        *,
        reason: str | None = None,
    ) -> None:
        summary = action.summary()
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": "presented",
                        "tool": action.tool_name,
                        "tool_metadata": self._governance.describe_tool(action.tool_name),
                        "action_packet": action.to_payload(),
                        "staging": staging,
                        "summary": summary,
                        "reason": reason,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="tool_output")
        await websocket.send(json.dumps(response_create_event))
        self._presented_actions.add(action.id)

    async def _send_dry_run_output(
        self,
        action: ActionPacket,
        staging: dict[str, Any],
        websocket: Any,
    ) -> None:
        summary = action.summary()
        dry_run_payload = {
            "requested": True,
            "summary": summary,
            "valid": staging.get("valid", True),
            "staging": staging,
        }
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": "dry_run",
                        "message": "Dry-run requested; no execution performed.",
                        "tool": action.tool_name,
                        "tool_metadata": self._governance.describe_tool(action.tool_name),
                        "action_packet": action.to_payload(),
                        "staging": staging,
                        "summary": summary,
                        "dry_run": dry_run_payload,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="tool_output")
        await websocket.send(json.dumps(response_create_event))
        self.function_call = None
        self.function_call_args = ""

    async def _reject_tool_call(
        self,
        action: ActionPacket,
        reason: str,
        websocket: Any,
        *,
        staging: dict[str, Any] | None = None,
        status: str = "denied",
    ) -> None:
        packet = self._format_action_packet(action)
        message = f"Tool execution not run.\n{packet}\nReason: {reason}."
        await self.send_assistant_message(message, websocket)
        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": action.id,
                "output": json.dumps(
                    {
                        "status": status,
                        "message": message,
                        "tool": action.tool_name,
                        "tool_metadata": self._governance.describe_tool(action.tool_name),
                        "action_packet": action.to_payload(),
                        "staging": staging,
                    }
                ),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        self._track_outgoing_event(function_call_output)
        await websocket.send(json.dumps(function_call_output))
        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="tool_output")
        await websocket.send(json.dumps(response_create_event))
        self._presented_actions.discard(action.id)
        self.function_call = None
        self.function_call_args = ""

    async def send_image_to_assistant(self, new_image: Any) -> None:
        if self.websocket:
            bytes_buffer = BytesIO()
            new_image.save(bytes_buffer, format="JPEG", quality=55, optimize=True)
            encoded_image = base64.b64encode(bytes_buffer.getvalue()).decode("utf-8")
            image_item = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded_image}",
                        }
                    ],
                },
            }
            log_ws_event("Image", image_item)
            await self.websocket.send(json.dumps(image_item))
            if self._image_response_enabled:
                await self._stimuli_coordinator.enqueue(
                    trigger="image_message",
                    metadata={"image_format": "jpeg"},
                    priority=self._get_injection_priority("image_message"),
                )
            else:
                logger.debug("Skipping image injection response (mode=%s).", self._image_response_mode)
        else:
            log_warning("Unable to send image to assistant, websocket not available")

    async def send_assistant_message(self, message: str, websocket: Any) -> None:
        assistant_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": message}],
            },
        }
        log_ws_event("Outgoing", assistant_item)
        await websocket.send(json.dumps(assistant_item))

    async def send_error_message_to_assistant(self, error_message: str, websocket: Any) -> None:
        error_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": error_message}],
            },
        }
        log_ws_event("Outgoing", error_item)
        await websocket.send(json.dumps(error_item))

    async def handle_transcribe_response_done(self) -> None:
        if self.assistant_reply:
            log_info(f"Assistant Response: {self.assistant_reply}", style="bold blue")
            self.assistant_reply = ""

        self.response_in_progress = False
        self._maybe_enqueue_reflection("response transcript done")
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response transcript done")
        logger.info("Finished handle_transcribe_response_done()")

    async def handle_audio_response_done(self) -> None:
        if self.response_start_time is not None:
            response_end_time = time.perf_counter()
            response_duration = response_end_time - self.response_start_time
            log_runtime("realtime_api_response", response_duration)
            self.response_start_time = None

        log_info("Assistant audio response complete.", style="bold blue")
        self.response_in_progress = False
        if self._audio_accum:
            if self.audio_player:
                self.audio_player.play_audio(bytes(self._audio_accum))
            self._audio_accum.clear()
        if self.audio_player:
            self.audio_player.close_response()
        if self._pending_image_stimulus:
            if self.audio_player:
                self._pending_image_flush_after_playback = True
            else:
                await self._flush_pending_image_stimulus("audio response done")

    async def handle_response_done(self, event: dict[str, Any] | None = None) -> None:
        self.response_in_progress = False
        self.state_manager.update_state(InteractionState.IDLE, "response done")
        logger.info("Received response.done event.")
        if event:
            self._last_response_metadata = {
                "event_type": event.get("type"),
                "response": event.get("response"),
                "rate_limits": self.rate_limits,
            }
        self.orchestration_state.transition(
            OrchestrationPhase.REFLECT,
            reason="response done",
        )
        self._enqueue_response_done_reflection("response done")
        self.orchestration_state.transition(
            OrchestrationPhase.IDLE,
            reason="response done reflection",
        )
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response done")

    async def handle_response_completed(self, event: dict[str, Any] | None = None) -> None:
        self.response_in_progress = False
        self.state_manager.update_state(InteractionState.IDLE, "response completed")
        logger.info("Received response.completed event.")
        if event:
            self._last_response_metadata = {
                "event_type": event.get("type"),
                "response": event.get("response"),
                "rate_limits": self.rate_limits,
            }
        self.orchestration_state.transition(
            OrchestrationPhase.REFLECT,
            reason="response completed",
        )
        self._maybe_enqueue_reflection("response completed")
        self.orchestration_state.transition(
            OrchestrationPhase.IDLE,
            reason="reflection enqueued",
        )
        if self._pending_image_stimulus and not self._pending_image_flush_after_playback:
            await self._flush_pending_image_stimulus("response completed")

    async def handle_error(self, event: dict[str, Any], websocket: Any) -> None:
        error_message = event.get("error", {}).get("message", "")
        log_error(f"Error: {error_message}")
        if "buffer is empty" in error_message:
            logger.info("Received 'buffer is empty' error, no audio data sent.")
        elif "Conversation already has an active response" in error_message:
            logger.info("Received 'active response' error, adjusting response flow.")
            self.response_in_progress = True
        else:
            logger.error("Unhandled error: %s", error_message)

    async def handle_speech_stopped(self, websocket: Any) -> None:
        self.mic.stop_recording()
        logger.info("Speech ended, processing...")
        self.response_start_time = time.perf_counter()

    async def send_initial_prompts(self, websocket: Any) -> None:
        logger.info("Sending %s prompts: %s", len(self.prompts), self.prompts)
        content = [{"type": "input_text", "text": prompt} for prompt in self.prompts]
        if self.prompts:
            self._record_user_input(self.prompts[-1], source="startup_prompt")
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": content,
            },
        }
        log_ws_event("Outgoing", event)
        self._track_outgoing_event(event)
        await websocket.send(json.dumps(event))

        if not self._allow_ai_call("startup_prompt"):
            logger.info("Skipping startup response: AI call budget exhausted.")
            return
        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="prompt")
        await websocket.send(json.dumps(response_create_event))
        self._record_ai_call()

    async def send_text_message_to_conversation(
        self,
        text_message: str,
        request_response: bool = True,
    ) -> None:
        self._record_user_input(text_message, source="text_message")
        if await self._handle_stop_word(text_message, self.websocket, source="text_message"):
            return
        if await self._maybe_handle_approval_response(text_message, self.websocket):
            return
        self.orchestration_state.transition(
            OrchestrationPhase.SENSE,
            reason="text message",
        )
        text_event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text_message}],
            },
        }
        log_ws_event("Outgoing", text_event)
        self._track_outgoing_event(text_event)
        await self.websocket.send(json.dumps(text_event))
        if request_response:
            await self._stimuli_coordinator.enqueue(
                trigger="text_message",
                metadata={"text_length": len(text_message)},
                priority=self._get_injection_priority("text_message"),
            )

    async def maybe_request_response(self, trigger: str, metadata: dict[str, Any]) -> None:
        if not self.websocket:
            log_warning("Skipping injected response (%s): websocket unavailable.", trigger)
            return

        trigger_config = self._injection_response_triggers.get(trigger, {})
        trigger_cooldown_s = float(
            trigger_config.get("cooldown_s", self._injection_response_cooldown_s)
        )
        trigger_max_per_minute = int(
            trigger_config.get("max_per_minute", self._max_injection_responses_per_minute)
        )
        trigger_priority = int(trigger_config.get("priority", 0))
        trigger_timestamps = self._injection_response_trigger_timestamps.setdefault(
            trigger, deque()
        )

        if self.response_in_progress:
            if trigger == "image_message":
                self._queue_pending_image_stimulus(trigger, metadata)
                return
            logger.info("Skipping injected response (%s): response already in progress.", trigger)
            return

        if self.state_manager.state not in (InteractionState.IDLE, InteractionState.LISTENING):
            logger.info(
                "Skipping injected response (%s): invalid state %s.",
                trigger,
                self.state_manager.state.value,
            )
            return

        if trigger_cooldown_s > 0.0:
            now = time.monotonic()
            last_ts = trigger_timestamps[-1] if trigger_timestamps else None
            if last_ts is not None and now - last_ts < trigger_cooldown_s:
                logger.info(
                    "Skipping injected response (%s): trigger cooldown %.2fs remaining.",
                    trigger,
                    trigger_cooldown_s - (now - last_ts),
                )
                return

        if trigger_max_per_minute > 0:
            now = time.monotonic()
            while trigger_timestamps and now - trigger_timestamps[0] > 60:
                trigger_timestamps.popleft()
            if len(trigger_timestamps) >= trigger_max_per_minute:
                logger.info(
                    "Skipping injected response (%s): trigger rate limit %s/minute reached.",
                    trigger,
                    trigger_max_per_minute,
                )
                return

        bypass_global_limits = trigger_priority > 0

        if self._injection_response_cooldown_s > 0.0 and not bypass_global_limits:
            now = time.monotonic()
            last_ts = (
                self._injection_response_timestamps[-1]
                if self._injection_response_timestamps
                else None
            )
            if last_ts is not None and now - last_ts < self._injection_response_cooldown_s:
                logger.info(
                    "Skipping injected response (%s): cooldown %.2fs remaining.",
                    trigger,
                    self._injection_response_cooldown_s - (now - last_ts),
                )
                return

        if self._max_injection_responses_per_minute > 0 and not bypass_global_limits:
            now = time.monotonic()
            while self._injection_response_timestamps and now - self._injection_response_timestamps[0] > 60:
                self._injection_response_timestamps.popleft()
            if len(self._injection_response_timestamps) >= self._max_injection_responses_per_minute:
                logger.info(
                    "Skipping injected response (%s): rate limit %s/minute reached.",
                    trigger,
                    self._max_injection_responses_per_minute,
                )
                return

        if self.rate_limits:
            for limit_name in ("requests", "responses"):
                rate_limit = self.rate_limits.get(limit_name)
                if not rate_limit:
                    continue
                remaining = rate_limit.get("remaining")
                if remaining is not None and remaining <= 0:
                    logger.info(
                        "Skipping injected response (%s): %s rate limit exhausted.",
                        trigger,
                        limit_name,
                    )
                    return

        bypass_budget = trigger == "text_message"
        if not self._allow_ai_call(f"injection:{trigger}", bypass=bypass_budget):
            logger.info("Skipping injected response (%s): AI call budget exhausted.", trigger)
            return
        response_metadata = {
            "trigger": trigger,
            "priority": str(trigger_priority),
            "stimulus": json.dumps(metadata, sort_keys=True),
        }
        response_create_event = {
            "type": "response.create",
            "response": {"metadata": response_metadata},
        }
        log_info(
            f"Requesting injected response for {trigger} with metadata {response_metadata}."
        )
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="injection")
        await self.websocket.send(json.dumps(response_create_event))
        now = time.monotonic()
        self._injection_response_timestamps.append(now)
        trigger_timestamps.append(now)
        self._record_ai_call()

    def _get_injection_priority(self, trigger: str) -> int:
        trigger_config = self._injection_response_triggers.get(trigger, {})
        return int(trigger_config.get("priority", 0))

    def _queue_pending_image_stimulus(self, trigger: str, metadata: dict[str, Any]) -> None:
        self._pending_image_stimulus = {"trigger": trigger, "metadata": metadata}
        self._pending_image_flush_after_playback = False
        logger.info("Queued pending image stimulus while response is in progress.")

    def _schedule_pending_image_flush(self, reason: str) -> None:
        if not self._pending_image_stimulus:
            return
        if self.loop and self.loop.is_running():
            self.loop.create_task(self._flush_pending_image_stimulus(reason))
        else:
            logger.warning(
                "Unable to flush pending image stimulus after %s: event loop unavailable.",
                reason,
            )

    async def _flush_pending_image_stimulus(self, reason: str) -> None:
        if not self._pending_image_stimulus:
            return
        if self.response_in_progress:
            logger.info(
                "Deferring pending image stimulus flush after %s: response in progress.",
                reason,
            )
            return
        pending = self._pending_image_stimulus
        self._pending_image_stimulus = None
        self._pending_image_flush_after_playback = False
        logger.info("Flushing pending image stimulus after %s.", reason)
        await self.maybe_request_response(pending["trigger"], pending["metadata"])

    async def send_audio_loop(self, websocket: Any) -> None:
        try:
            while not self.exit_event.is_set():
                if self.mic.is_receiving:
                    await asyncio.sleep(0.05)
                    continue

                now = time.monotonic()
                if now < self.mic_send_suppress_until:
                    self.mic.drain_queue()
                    await asyncio.sleep(0.05)
                    continue

                audio_data = self.mic.get_audio_data()
                if audio_data:
                    base64_audio = base64_encode_audio(audio_data)
                    if base64_audio:
                        audio_event = {
                            "type": "input_audio_buffer.append",
                            "audio": base64_audio,
                        }
                        await websocket.send(json.dumps(audio_event))
                    else:
                        logger.debug("Failed to encode audio data for sending")

                await asyncio.sleep(0.03)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Closing the connection.")
        finally:
            self.exit_event.set()
            await websocket.close()

    def shutdown_handler(self) -> None:
        logger.info("Received SIGTERM. Initiating graceful shutdown...")
        if self.loop:
            for task in asyncio.all_tasks(self.loop):
                task.cancel()
