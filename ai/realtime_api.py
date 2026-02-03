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
import signal
import threading
import time
from typing import Any
from urllib import request

from config import ConfigController
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


class RealtimeAPI:
    """Realtime OpenAI API client."""

    def __init__(self, prompts: list[str] | None = None) -> None:
        self.prompts = prompts
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
        self.exit_event = asyncio.Event()
        self.mic = AsyncMicrophone(input_name_hint="default", debug_list_devices=False)
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
        config = ConfigController.get_instance().get_config()
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
        self._storage = StorageController.get_instance()
        self._reflection_coordinator = ReflectionCoordinator(
            api_key=self.api_key,
            enabled=self._reflection_enabled,
            min_interval_s=self._reflection_min_interval_s,
            storage=self._storage,
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

    def get_event_bus(self) -> EventBus:
        return self.event_bus

    def is_ready_for_injections(self) -> bool:
        return (
            self.ready_event.is_set()
            and self.websocket is not None
            and self.loop is not None
            and self.loop.is_running()
        )

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
        self._send_text_message(message, request_response=request_response)

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
            "https://api.openai.com/v1/chat/completions",
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
        prompt = self._build_response_done_prompt(trigger)
        try:
            raw_response = await asyncio.to_thread(self._call_openai_prompt, prompt)
        except Exception as exc:  # noqa: BLE001 - guard background call
            logger.warning("response.done reflection failed: %s", exc)
            return
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

        def _playback_complete_from_thread() -> None:
            if self.loop:
                self.loop.call_soon_threadsafe(self._on_playback_complete)

        self.audio_player = AudioPlayer(
            on_playback_complete=_playback_complete_from_thread,
            output_name_hint="softvol",
        )

        try:
            while True:
                try:
                    url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
                    headers = {"Authorization": f"Bearer {self.api_key}"}

                    async with websockets.connect(
                        url,
                        additional_headers=headers,
                        close_timeout=120,
                        ping_interval=30,
                        ping_timeout=10,
                    ) as websocket:
                        log_info("✅ Connected to the server.", style="bold green")

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
                    if "keepalive ping timeout" in str(exc):
                        logger.warning(
                            "WebSocket connection lost due to keepalive ping timeout. Reconnecting..."
                        )
                        await asyncio.sleep(1)
                        continue
                    logger.exception("WebSocket connection closed unexpectedly.")
                    break
                except Exception as exc:
                    logger.exception("An unexpected error occurred: %s", exc)
                    break
                finally:
                    if self.audio_player:
                        self.audio_player.close()
                    self.mic.stop_recording()
                    self.mic.close()
                    self.websocket = None
                    self.ready_event.clear()
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
            await self.execute_function_call(function_name, call_id, args, websocket)

    async def execute_function_call(
        self, function_name: str, call_id: str, args: dict[str, Any], websocket: Any
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
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self._last_tool_call_results = list(self._tool_call_records)

        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
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

        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        self._track_outgoing_event(response_create_event, origin="prompt")
        await websocket.send(json.dumps(response_create_event))

    async def send_text_message_to_conversation(
        self,
        text_message: str,
        request_response: bool = True,
    ) -> None:
        self.orchestration_state.transition(
            OrchestrationPhase.SENSE,
            reason="text message",
        )
        self._record_user_input(text_message, source="text_message")
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
