"""Unit tests for realtime inbound event dispatch."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

from ai.realtime.event_router import EventRouter
from ai.function_call_accumulator import FunctionCallAccumulator
from ai.realtime_api import RealtimeAPI


def _attach_function_call_accumulator(api: RealtimeAPI) -> None:
    api._function_call_accumulator = FunctionCallAccumulator(
        on_function_call_item=api._on_function_call_item_added,
        on_assistant_message_item=api._on_assistant_output_item_added,
        on_arguments_done=api.handle_function_call,
    )



def test_dispatch_invokes_registered_handler() -> None:
    calls: list[tuple[str, dict[str, Any], Any]] = []

    async def _fallback(event: dict[str, Any], websocket: Any) -> None:
        calls.append(("fallback", event, websocket))

    async def _handler(event: dict[str, Any], websocket: Any) -> None:
        calls.append(("handler", event, websocket))

    router = EventRouter(fallback=_fallback)
    router.register("response.created", _handler)

    async def _run() -> None:
        await router.dispatch({"type": "response.created", "id": "evt-1"}, websocket="ws")

    asyncio.run(_run())

    assert calls == [("handler", {"type": "response.created", "id": "evt-1"}, "ws")]


def test_unknown_event_uses_fallback_stably() -> None:
    fallback_calls: list[str] = []

    async def _fallback(event: dict[str, Any], websocket: Any) -> None:
        fallback_calls.append(str(event.get("type") or "unknown"))

    router = EventRouter(fallback=_fallback)

    async def _run() -> None:
        await router.dispatch({"type": "unregistered.one"}, websocket=None)
        await router.dispatch({"type": "unregistered.two"}, websocket=None)

    asyncio.run(_run())

    assert fallback_calls == ["unregistered.one", "unregistered.two"]


def test_handler_exception_triggers_exception_callback_once() -> None:
    exception_calls: list[tuple[str, str]] = []

    async def _fallback(event: dict[str, Any], websocket: Any) -> None:
        raise AssertionError("fallback should not execute for registered handler")

    def _on_exception(event_type: str, exc: Exception) -> None:
        exception_calls.append((event_type, str(exc)))

    async def _handler(event: dict[str, Any], websocket: Any) -> None:
        raise RuntimeError("boom")

    router = EventRouter(fallback=_fallback, on_exception=_on_exception)
    router.register("response.done", _handler)

    async def _run() -> None:
        try:
            await router.dispatch({"type": "response.done"}, websocket=None)
        except RuntimeError:
            return
        raise AssertionError("dispatch should re-raise handler exceptions")

    asyncio.run(_run())

    assert exception_calls == [("response.done", "boom")]


def test_realtime_api_configure_event_router_registers_conversation_item_added() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    api._event_router = EventRouter(fallback=_fallback)
    api._input_audio_events = type(
        "InputAudioHandlers",
        (),
        {
            "handle_input_audio_buffer_speech_started": AsyncMock(return_value=None),
            "handle_input_audio_buffer_speech_stopped": AsyncMock(return_value=None),
            "handle_input_audio_buffer_committed": AsyncMock(return_value=None),
            "handle_input_audio_transcription_partial": AsyncMock(return_value=None),
        },
    )()
    _attach_function_call_accumulator(api)

    api._configure_event_router()

    assert api._event_router._handlers["conversation.item.added"] == api._handle_response_lifecycle_event




def test_realtime_api_configure_event_router_registers_input_audio_handlers_from_module() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    class _InputAudioHandlers:
        async def handle_input_audio_buffer_speech_started(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            return None

        async def handle_input_audio_buffer_speech_stopped(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            return None

        async def handle_input_audio_buffer_committed(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            return None

        async def handle_input_audio_transcription_partial(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            return None

    api._event_router = EventRouter(fallback=_fallback)
    api._input_audio_events = _InputAudioHandlers()
    _attach_function_call_accumulator(api)

    api._configure_event_router()

    assert (
        api._event_router._handlers["input_audio_buffer.speech_started"]
        == api._input_audio_events.handle_input_audio_buffer_speech_started
    )
    assert (
        api._event_router._handlers["input_audio_buffer.speech_stopped"]
        == api._input_audio_events.handle_input_audio_buffer_speech_stopped
    )
    assert (
        api._event_router._handlers["input_audio_buffer.committed"]
        == api._input_audio_events.handle_input_audio_buffer_committed
    )
    assert (
        api._event_router._handlers["conversation.item.input_audio_transcription.partial"]
        == api._input_audio_events.handle_input_audio_transcription_partial
    )


def test_realtime_api_handle_event_triggers_recovery_when_input_audio_handler_raises() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    event = {"type": "input_audio_buffer.speech_started", "id": "evt-err"}
    websocket = object()

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    class _InputAudioHandlers:
        async def handle_input_audio_buffer_speech_started(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            raise RuntimeError("input handler exploded")

        async def handle_input_audio_buffer_speech_stopped(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            return None

        async def handle_input_audio_buffer_committed(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            return None

        async def handle_input_audio_transcription_partial(
            self, event: dict[str, Any], websocket: Any
        ) -> None:
            return None

    api._event_router = EventRouter(fallback=_fallback)
    api._input_audio_events = _InputAudioHandlers()
    _attach_function_call_accumulator(api)
    api._configure_event_router()
    api._maybe_handle_confirmation_decision_timeout = AsyncMock(return_value=None)
    api._recover_from_event_handler_error = AsyncMock(return_value=None)

    with (
        patch("ai.realtime_api.logger.exception") as mock_exception,
        patch("ai.realtime_api.logger.error") as mock_error,
    ):
        asyncio.run(api.handle_event(event, websocket))

    api._recover_from_event_handler_error.assert_awaited_once_with(
        "input_audio_buffer.speech_started",
        websocket,
    )
    assert mock_exception.call_count == 1
    mock_error.assert_called_once_with(
        "EVENT_HANDLER_ERROR event=%s",
        "input_audio_buffer.speech_started",
    )


def test_realtime_api_handle_event_legacy_processes_assistant_conversation_item_added() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._current_input_event_key = "input-1"
    api._is_active_response_guarded = lambda: False
    api._cancel_micro_ack = Mock()
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._mark_first_assistant_utterance_observed_if_needed = Mock()
    api._set_response_delivery_state = Mock()
    api.state_manager = type("StateManager", (), {"update_state": Mock()})()
    _attach_function_call_accumulator(api)

    event = {
        "type": "conversation.item.added",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Hello"},
                {"type": "audio", "transcript": " world"},
            ],
        },
    }

    asyncio.run(api._handle_event_legacy(event, websocket=None))

    assert api.assistant_reply == "Hello world"
    assert api._assistant_reply_accum == "Hello world"
    api._mark_first_assistant_utterance_observed_if_needed.assert_called_once_with("Hello world")
    api._set_response_delivery_state.assert_called_once_with(
        turn_id="turn-1",
        input_event_key="input-1",
        state="delivered",
    )



def test_realtime_api_handle_event_legacy_processes_assistant_response_output_item_added() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._active_response_input_event_key = "input-active"
    api._current_input_event_key = "input-fallback"
    api._is_active_response_guarded = lambda: False
    api._cancel_micro_ack = Mock()
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._mark_first_assistant_utterance_observed_if_needed = Mock()
    api._set_response_delivery_state = Mock()
    api.state_manager = type("StateManager", (), {"update_state": Mock()})()
    _attach_function_call_accumulator(api)

    event = {
        "type": "response.output_item.added",
        "item": {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Ready"},
                {"type": "audio", "transcript": " now"},
            ],
        },
    }

    asyncio.run(api._handle_event_legacy(event, websocket=None))

    assert api.assistant_reply == "Ready now"
    assert api._assistant_reply_accum == "Ready now"
    api._mark_first_assistant_utterance_observed_if_needed.assert_called_once_with("Ready now")
    api._set_response_delivery_state.assert_called_once_with(
        turn_id="turn-1",
        input_event_key="input-active",
        state="delivered",
    )


def test_realtime_api_prompt_origin_binding_sets_active_key_before_output_processing() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._active_input_event_key_by_turn_id = {}

    api._bind_active_input_event_key_for_turn(
        turn_id="turn-1",
        input_event_key="synthetic_prompt_1",
    )

    assert api._active_input_event_key_for_turn("turn-1") == "synthetic_prompt_1"


def test_realtime_api_handle_event_dispatches_via_event_router_once() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    event = {"type": "response.created", "id": "evt-123"}
    websocket = object()

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    router = EventRouter(fallback=_fallback)
    router.dispatch = AsyncMock(return_value=None)
    api._event_router = router
    api._maybe_handle_confirmation_decision_timeout = AsyncMock(return_value=None)

    asyncio.run(api.handle_event(event, websocket))

    router.dispatch.assert_awaited_once_with(event, websocket)


def test_realtime_api_handle_event_triggers_recovery_when_router_dispatch_raises() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    event = {"type": "response.done", "id": "evt-err"}
    websocket = object()

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    router = EventRouter(fallback=_fallback)
    router.dispatch = AsyncMock(side_effect=RuntimeError("router exploded"))
    api._event_router = router
    api._maybe_handle_confirmation_decision_timeout = AsyncMock(return_value=None)
    api._recover_from_event_handler_error = AsyncMock(return_value=None)

    with (
        patch("ai.realtime_api.logger.exception") as mock_exception,
        patch("ai.realtime_api.logger.error") as mock_error,
    ):
        asyncio.run(api.handle_event(event, websocket))

    router.dispatch.assert_awaited_once_with(event, websocket)
    api._recover_from_event_handler_error.assert_awaited_once_with("response.done", websocket)
    assert mock_exception.call_count == 1
    mock_error.assert_called_once_with("EVENT_HANDLER_ERROR event=%s", "response.done")


def test_realtime_api_configure_event_router_registers_function_call_accumulator_handlers() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)

    async def _fallback(_event: dict[str, Any], _websocket: Any) -> None:
        return None

    api._event_router = EventRouter(fallback=_fallback)
    api._input_audio_events = type(
        "InputAudioHandlers",
        (),
        {
            "handle_input_audio_buffer_speech_started": AsyncMock(return_value=None),
            "handle_input_audio_buffer_speech_stopped": AsyncMock(return_value=None),
            "handle_input_audio_buffer_committed": AsyncMock(return_value=None),
            "handle_input_audio_transcription_partial": AsyncMock(return_value=None),
        },
    )()
    _attach_function_call_accumulator(api)

    api._configure_event_router()

    assert api._event_router._handlers["response.output_item.added"] == api._handle_output_item_added_event
    assert (
        api._event_router._handlers["response.function_call_arguments.done"]
        == api._handle_function_call_arguments_done_event
    )

