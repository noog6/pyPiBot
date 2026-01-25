"""Realtime API controller for OpenAI realtime connections."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from io import BytesIO
import json
import os
import random
import signal
import time
from typing import Any

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

        self.assistant_reply = ""
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
        self._gesture_cooldowns_s = {
            "gesture_attention_snap": 10.0,
            "gesture_curious_tilt": 6.0,
            "gesture_nod": 8.0,
            "gesture_idle": 8.0,
        }

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

    async def run(self) -> None:
        websockets = _require_websockets()
        _, ConnectionClosedError = _resolve_websocket_exceptions(websockets)

        self.loop = asyncio.get_running_loop()

        def _playback_complete_from_thread() -> None:
            if self.loop:
                self.loop.call_soon_threadsafe(self._on_playback_complete)

        self.audio_player = AudioPlayer(
            on_playback_complete=_playback_complete_from_thread,
            output_name_hint="default",
        )

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

    async def initialize_session(self, websocket: Any) -> None:
        profile_context = self.profile_manager.get_profile_context()
        instructions = build_session_instructions(profile_context.to_instruction_block())
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
            if self.audio_player:
                self.audio_player.start_response()
            self._audio_accum.clear()
            self.mic.start_receiving()
            self.response_in_progress = True
            self._speaking_started = False
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
        elif event_type == "response.output_audio_transcript.done":
            await self.handle_transcribe_response_done()
            self.state_manager.update_state(
                InteractionState.IDLE,
                "audio transcript done",
            )
        elif event_type == "error":
            await self.handle_error(event, websocket)
        elif event_type == "input_audio_buffer.speech_started":
            logger.info("Speech detected, listening...")
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
            log_session_updated(event, full_payload=False)

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

        function_call_output = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result),
            },
        }
        log_ws_event("Outgoing", function_call_output)
        await websocket.send(json.dumps(function_call_output))
        await websocket.send(json.dumps({"type": "response.create"}))

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

        logger.info("Finished handle_transcribe_response_done()")

    async def handle_audio_response_done(self) -> None:
        if self.response_start_time is not None:
            response_end_time = time.perf_counter()
            response_duration = response_end_time - self.response_start_time
            log_runtime("realtime_api_response", response_duration)
            self.response_start_time = None

        log_info("Assistant audio response complete.", style="bold blue")
        if self._audio_accum:
            if self.audio_player:
                self.audio_player.play_audio(bytes(self._audio_accum))
            self._audio_accum.clear()
        if self.audio_player:
            self.audio_player.close_response()

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
        event = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": content,
            },
        }
        log_ws_event("Outgoing", event)
        await websocket.send(json.dumps(event))

        response_create_event = {"type": "response.create"}
        log_ws_event("Outgoing", response_create_event)
        await websocket.send(json.dumps(response_create_event))

    async def send_text_message_to_conversation(self, text_message: str) -> None:
        text_event = {
            "type": "conversation.item.added",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text_message}],
            },
        }
        await self.websocket.send(json.dumps(text_event))

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
