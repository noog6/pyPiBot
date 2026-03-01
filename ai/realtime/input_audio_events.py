"""Input-audio realtime event handlers."""

from __future__ import annotations

import time
from typing import Any

from core.logging import logger, log_ws_event
from interaction import InteractionState
from ai.orchestration import OrchestrationPhase


class InputAudioEventHandlers:
    """Facade-backed handlers for input-audio speech and transcript events."""

    def __init__(self, api: Any) -> None:
        self._api = api

    async def handle_input_audio_transcription_partial(
        self, event: dict[str, Any], websocket: Any
    ) -> None:
        _ = websocket
        event_type = str(event.get("type") or "")
        partial_text = event.get("delta")
        if not isinstance(partial_text, str) or not partial_text.strip():
            partial_text = self._api._extract_transcript(event) or ""
        transcript_input_event_key = self._api._resolve_input_event_key(event)
        with self._api._utterance_context_scope(
            turn_id=self._api._current_turn_id_or_unknown(),
            input_event_key=transcript_input_event_key,
        ):
            self._api._log_user_transcript(partial_text, final=False, event_type=event_type)

    async def handle_input_audio_buffer_speech_started(
        self, event: dict[str, Any], websocket: Any
    ) -> None:
        _ = event
        logger.info("Speech detected, listening...")
        manager = getattr(self._api, "_micro_ack_manager", None)
        talk_over_active = self._api.state_manager.state == InteractionState.SPEAKING or self._api._audio_playback_busy
        if manager is not None:
            manager.on_user_speech_started()
            if talk_over_active:
                manager.mark_talk_over_incident()

            manager.cancel_all(reason="speech_active")
        if talk_over_active:
            self._api._clear_all_pending_response_creates(reason="talk_over_abort")
            turn_id = self._api._current_turn_id_or_unknown()
            input_event_key = str(getattr(self._api, "_current_input_event_key", "") or "").strip()
            self._api._clear_pending_response_contenders(
                turn_id=turn_id,
                input_event_key=input_event_key,
                reason="talk_over_abort",
            )
            if bool(getattr(self._api, "_response_in_flight", False)):
                cancel_event = {"type": "response.cancel"}
                log_ws_event("Outgoing", cancel_event)
                self._api._track_outgoing_event(cancel_event, origin="talk_over_abort")
                try:
                    transport = self._api._get_or_create_transport()
                    await transport.send_json(websocket, cancel_event)
                except Exception as exc:
                    logger.debug("talk_over_abort_cancel_failed turn_id=%s error=%s", turn_id, exc)
        self._api._utterance_counter += 1
        next_turn_id = self._api._next_response_turn_id()
        with self._api._utterance_context_scope(
            turn_id=next_turn_id,
            input_event_key="",
            utterance_seq=self._api._utterance_counter,
        ):
            pass
        self._api._active_utterance = {
            "utterance_id": self._api._utterance_counter,
            "t_start": time.monotonic(),
            "t_stop": None,
            "duration_ms": None,
            "rms_estimate": None,
            "peak_estimate": None,
            "transcript": "",
            "transcript_len": 0,
            "confirmation_candidate": False,
            "decision": "unclear",
            "suppressed": False,
        }
        self._api._log_utterance_envelope("input_audio_buffer.speech_started")
        if self._api._has_active_confirmation_token():
            self._api._confirmation_speech_active = True
            self._api._confirmation_asr_pending = True
            self._api._mark_confirmation_activity(reason="speech_started")
            logger.info(
                "Speech started while awaiting confirmation; confirmation mode remains active."
            )
        else:
            self._api.orchestration_state.transition(
                OrchestrationPhase.SENSE,
                reason="speech started",
            )
        self._api.state_manager.update_state(InteractionState.LISTENING, "speech started")

    async def handle_input_audio_buffer_speech_stopped(
        self, event: dict[str, Any], websocket: Any
    ) -> None:
        _ = event
        manager = getattr(self._api, "_micro_ack_manager", None)
        if manager is not None:
            manager.on_user_speech_ended()
        if self._api._active_utterance is not None:
            self._api._active_utterance["t_stop"] = time.monotonic()
            self._api._active_utterance["duration_ms"] = (
                self._api._active_utterance["t_stop"] - self._api._active_utterance["t_start"]
            ) * 1000.0
            self._api._refresh_utterance_audio_levels()
            self._api._log_utterance_envelope("input_audio_buffer.speech_stopped")
        if self._api._has_active_confirmation_token():
            self._api._confirmation_speech_active = False
            self._api._confirmation_asr_pending = True
            self._api._mark_confirmation_activity(reason="speech_stopped")
        await self._api.handle_speech_stopped(websocket)
        current_turn_id = self._api._current_turn_id_or_unknown()
        self._api._maybe_schedule_micro_ack(
            turn_id=current_turn_id,
            category=self._api._micro_ack_category_for_reason("speech_stopped"),
            channel="voice",
            action=self._api._canonical_utterance_key(
                turn_id=current_turn_id,
                input_event_key=self._api._active_input_event_key_for_turn(current_turn_id),
            ),
            reason="speech_stopped",
            expected_delay_ms=700,
        )
        self._api.state_manager.update_state(InteractionState.THINKING, "speech stopped")

    async def handle_input_audio_buffer_committed(
        self, event: dict[str, Any], websocket: Any
    ) -> None:
        _ = event
        _ = websocket
        self._api._refresh_utterance_audio_levels()
        self._api._log_utterance_envelope("input_audio_buffer.committed")
