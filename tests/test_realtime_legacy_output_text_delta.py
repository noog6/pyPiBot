from __future__ import annotations

import asyncio
from unittest.mock import patch

from interaction import InteractionState
from ai.realtime_api import RealtimeAPI


class _StateManagerStub:
    def __init__(self) -> None:
        self.transitions: list[tuple[InteractionState, str]] = []

    def update_state(self, state: InteractionState, reason: str) -> None:
        self.transitions.append((state, reason))


def test_handle_event_legacy_response_output_text_delta_matches_text_delta_behavior() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api.state_manager = _StateManagerStub()

    marks: list[str] = []
    cancel_calls: list[tuple[str, str]] = []

    api._is_active_response_guarded = lambda: False
    api._current_turn_id_or_unknown = lambda: "turn-1"
    api._cancel_micro_ack = lambda turn_id, reason: cancel_calls.append((turn_id, reason))
    api._mark_first_assistant_utterance_observed_if_needed = lambda delta: marks.append(delta)

    with patch("ai.realtime_api.logger.debug") as mock_debug:
        asyncio.run(
            api._handle_event_legacy(
                {"type": "response.output_text.delta", "delta": "hello"},
                websocket=None,
            )
        )

    assert cancel_calls == [("turn-1", "response_started")]
    assert marks == ["hello"]
    assert api.assistant_reply == "hello"
    assert api._assistant_reply_accum == "hello"
    assert api.state_manager.transitions == [(InteractionState.SPEAKING, "text output")]
    mock_debug.assert_any_call(
        "assistant_content_event_received event_type=response.output_text.delta delta_len=%s",
        5,
    )
