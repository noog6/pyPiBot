from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from ai.realtime_api import RealtimeAPI


class _FakeEventDispatcher:
    def __init__(self) -> None:
        self.ordered_events: list[tuple[str, str, str]] = []
        self.audible_delta_count = 0
        self._cancelled_response_ids: set[str] = set()
        self._first_audible_response_id: str | None = None

    def emit(self, event_type: str, *, response_id: str) -> None:
        self.ordered_events.append(("sent", event_type, response_id))

    def handle(self, event_type: str, *, response_id: str) -> None:
        self.ordered_events.append(("handled", event_type, response_id))

        if event_type == "response.cancel":
            self._cancelled_response_ids.add(response_id)
            return

        if event_type != "response.output_audio.delta":
            return

        if response_id in self._cancelled_response_ids:
            return

        self.audible_delta_count += 1
        if self._first_audible_response_id is None:
            self._first_audible_response_id = response_id

    def first_audible_delta_response_id(self) -> str | None:
        return self._first_audible_response_id


class _FakeWebsocket:
    def __init__(self) -> None:
        self.sent_payloads: list[dict[str, object]] = []

    async def send(self, payload: str) -> None:
        self.sent_payloads.append(json.loads(payload))


def _load_stream_fixture() -> dict[str, object]:
    fixture_path = Path(__file__).parent / "fixtures" / "realtime_stream_single_turn.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def test_realtime_lifecycle_invariants_single_turn_fixture() -> None:
    payload = _load_stream_fixture()
    events = payload["events"]

    response_to_canonical: dict[str, str] = {}
    audio_started_response_ids_by_key: dict[str, set[str]] = defaultdict(set)
    assistant_done_count_by_utterance: dict[str, int] = defaultdict(int)
    explicit_multipart_by_utterance: dict[str, bool] = defaultdict(bool)

    for event in events:
        event_type = event["type"]
        response_id = event.get("response_id")
        canonical_key = event.get("canonical_key")
        utterance_id = event.get("utterance_id")
        metadata = event.get("metadata") or {}

        if response_id and canonical_key:
            response_to_canonical[response_id] = canonical_key

        if event_type == "response.audio_started":
            resolved_key = canonical_key or response_to_canonical.get(str(response_id), "")
            assert resolved_key, f"audio_started missing canonical mapping for response_id={response_id}"
            audio_started_response_ids_by_key[resolved_key].add(str(response_id))

        if event_type == "response.done" and event.get("origin") == "assistant_message":
            assert utterance_id, f"assistant done missing utterance_id for response_id={response_id}"
            assistant_done_count_by_utterance[utterance_id] += 1
            explicit_multipart_by_utterance[utterance_id] = bool(metadata.get("explicit_multipart", False))

    # Invariant 1: at most one distinct response reaches audio_started=true for the same canonical key.
    for canonical_key, response_ids in audio_started_response_ids_by_key.items():
        assert len(response_ids) <= 1, (
            "expected <=1 audio_started response per canonical key "
            f"canonical_key={canonical_key} response_ids={sorted(response_ids)}"
        )

    # Invariant 2: assistant response count per utterance is exactly 1 unless explicit_multipart=true.
    for utterance_id, done_count in assistant_done_count_by_utterance.items():
        if explicit_multipart_by_utterance[utterance_id]:
            assert done_count >= 1
        else:
            assert done_count == 1, (
                "expected a single final assistant response for non-multipart utterance "
                f"utterance_id={utterance_id} done_count={done_count}"
            )


def test_guarded_server_auto_cancel_precedes_first_accepted_audio_delta() -> None:
    dispatcher = _FakeEventDispatcher()

    guarded_response_id = "resp_server_auto_guarded"

    dispatcher.emit("response.cancel", response_id=guarded_response_id)
    dispatcher.handle("response.cancel", response_id=guarded_response_id)

    dispatcher.handle("response.output_audio.delta", response_id=guarded_response_id)

    accepted_response_id = "resp_assistant_final"
    dispatcher.handle("response.output_audio.delta", response_id=accepted_response_id)

    cancel_event_index = dispatcher.ordered_events.index(("handled", "response.cancel", guarded_response_id))
    first_accepted_delta_index = dispatcher.ordered_events.index(
        ("handled", "response.output_audio.delta", accepted_response_id)
    )

    assert cancel_event_index < first_accepted_delta_index
    assert dispatcher.first_audible_delta_response_id() == accepted_response_id
    assert dispatcher.audible_delta_count == 1


def test_startup_prompt_speech_started_and_assistant_message_preserve_turn_key_boundaries() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_turn_counter = 0
    api._current_response_turn_id = None
    api._current_input_event_key = None
    api._synthetic_input_event_counter = 0
    api._active_input_event_key_by_turn_id = {}
    api._current_run_id = lambda: "run-lifecycle-test"

    startup_response_create = {"type": "response.create", "response": {"metadata": {}}}
    startup_turn_id = api._resolve_response_create_turn_id(
        origin="prompt",
        response_create_event=startup_response_create,
    )
    startup_input_event_key = api._ensure_response_create_correlation(
        response_create_event=startup_response_create,
        origin="prompt",
        turn_id=startup_turn_id,
    )

    # Simulate speech_started lifecycle branch (deterministic direct methods only).
    api._utterance_counter = 0
    api._utterance_counter += 1
    user_turn_id = api._next_response_turn_id()
    with api._utterance_context_scope(turn_id=user_turn_id, input_event_key="", utterance_seq=api._utterance_counter):
        pass

    transcript_event = {
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": "item-user-audio-1",
    }
    user_input_event_key = api._resolve_input_event_key(transcript_event)
    api._active_input_event_key_by_turn_id[startup_turn_id] = startup_input_event_key
    api._active_input_event_key_by_turn_id[user_turn_id] = user_input_event_key

    startup_canonical_key = api._canonical_utterance_key(
        turn_id=startup_turn_id,
        input_event_key=startup_input_event_key,
    )
    user_canonical_key = api._canonical_utterance_key(
        turn_id=user_turn_id,
        input_event_key=user_input_event_key,
    )
    user_canonical_with_startup_key = api._canonical_utterance_key(
        turn_id=user_turn_id,
        input_event_key=startup_input_event_key,
    )

    assert startup_turn_id == "turn_1"
    assert user_turn_id == "turn_2"
    assert user_turn_id != startup_turn_id
    assert startup_input_event_key.startswith("synthetic_prompt_")
    assert startup_input_event_key != user_input_event_key
    assert user_canonical_key != startup_canonical_key
    assert user_canonical_key != user_canonical_with_startup_key
    assert set(api._active_input_event_key_by_turn_id.keys()) == {startup_turn_id, user_turn_id}
    assert api._active_input_event_key_by_turn_id[startup_turn_id] == startup_input_event_key
    assert api._active_input_event_key_by_turn_id[user_turn_id] == user_input_event_key

    websocket = _FakeWebsocket()
    scheduled_responses: list[dict[str, object]] = []

    async def _fake_send_response_create(_websocket, event, *, origin, utterance_context, **_kwargs) -> None:
        scheduled_responses.append(
            {
                "origin": origin,
                "turn_id": utterance_context.turn_id,
                "input_event_key": utterance_context.input_event_key,
                "metadata": event["response"]["metadata"],
            }
        )

    api._send_response_create = _fake_send_response_create
    api._current_run_id = lambda: "run-lifecycle-test"
    api._pending_confirmation_token = None
    api._is_awaiting_confirmation_phase = lambda: False

    class _FakeTransport:
        async def send_json(self, _websocket, _payload):
            return None

    api._get_or_create_transport = lambda: _FakeTransport()

    # Guard applies when assistant_message has no explicit parent key.
    asyncio.run(api.send_assistant_message("guard me", websocket, utterance_context=None))
    assert len(scheduled_responses) == 0

    # With current turn context (from speech_started + transcript key), assistant_message is allowed.
    assistant_context = api._build_utterance_context(turn_id=user_turn_id, input_event_key=user_input_event_key)
    asyncio.run(api.send_assistant_message("speak", websocket, utterance_context=assistant_context))

    assert len(scheduled_responses) == 1
    assert scheduled_responses[0]["origin"] == "assistant_message"
    assert scheduled_responses[0]["turn_id"] == user_turn_id
    assert scheduled_responses[0]["input_event_key"] == user_input_event_key


def test_send_assistant_message_suppresses_generic_same_turn_when_owned_by_tool_followup() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_turn_counter = 0
    api._current_response_turn_id = "turn_1"
    api._current_input_event_key = "item_1"
    api._synthetic_input_event_counter = 0
    api._active_input_event_key_by_turn_id = {"turn_1": "item_1"}
    api._tool_followup_state_by_canonical_key = {"run-lifecycle-test:turn_1:item_1:tool:call_1": "created"}
    api._canonical_response_lifecycle_state = {}
    api._active_response_id = None
    api._active_response_origin = "unknown"
    api._active_response_consumes_canonical_slot = True
    api._response_obligations = {}
    api._preference_recall_suppressed_turns = set()
    api._pending_server_auto_input_event_keys = []
    api._pending_confirmation_token = None
    api._is_awaiting_confirmation_phase = lambda: False
    api._current_run_id = lambda: "run-lifecycle-test"

    class _FakeTransport:
        async def send_json(self, _websocket, _payload):
            return None

    api._get_or_create_transport = lambda: _FakeTransport()

    scheduled_responses: list[dict[str, object]] = []

    async def _fake_send_response_create(_websocket, event, *, origin, utterance_context, **_kwargs) -> None:
        scheduled_responses.append(
            {
                "origin": origin,
                "turn_id": utterance_context.turn_id,
                "input_event_key": utterance_context.input_event_key,
                "metadata": event["response"]["metadata"],
            }
        )

    api._send_response_create = _fake_send_response_create
    outcome_calls: list[dict[str, str]] = []

    def _record_outcome(*, input_event_key: str, turn_id: str, outcome: str, reason: str | None = None, details: str | None = None) -> None:
        outcome_calls.append(
            {
                "input_event_key": input_event_key,
                "turn_id": turn_id,
                "outcome": outcome,
                "reason": reason or "",
                "details": details or "",
            }
        )

    api._mark_transcript_response_outcome = _record_outcome

    websocket = _FakeWebsocket()
    context = api._build_utterance_context(turn_id="turn_1", input_event_key="item_1")

    asyncio.run(api.send_assistant_message("generic chatter", websocket, utterance_context=context))

    assert len(websocket.sent_payloads) == 0
    assert len(scheduled_responses) == 0
    assert len(outcome_calls) == 1
    assert outcome_calls[0]["outcome"] == "response_not_scheduled"
    assert outcome_calls[0]["reason"] == "same_turn_already_owned"
    assert "owner_reason=tool_followup_owned" in outcome_calls[0]["details"]


def test_send_assistant_message_allows_allowlisted_trigger_under_same_turn_ownership() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_turn_counter = 0
    api._current_response_turn_id = "turn_1"
    api._current_input_event_key = "item_1"
    api._synthetic_input_event_counter = 0
    api._active_input_event_key_by_turn_id = {"turn_1": "item_1"}
    api._tool_followup_state_by_canonical_key = {"run-lifecycle-test:turn_1:item_1:tool:call_1": "created"}
    api._canonical_response_lifecycle_state = {}
    api._active_response_id = None
    api._active_response_origin = "unknown"
    api._active_response_consumes_canonical_slot = True
    api._response_obligations = {}
    api._preference_recall_suppressed_turns = set()
    api._pending_server_auto_input_event_keys = []
    api._pending_confirmation_token = None
    api._is_awaiting_confirmation_phase = lambda: False
    api._current_run_id = lambda: "run-lifecycle-test"

    class _FakeTransport:
        async def send_json(self, _websocket, _payload):
            return None

    api._get_or_create_transport = lambda: _FakeTransport()

    scheduled_responses: list[dict[str, object]] = []

    async def _fake_send_response_create(_websocket, event, *, origin, utterance_context, **_kwargs) -> None:
        scheduled_responses.append(
            {
                "origin": origin,
                "turn_id": utterance_context.turn_id,
                "input_event_key": utterance_context.input_event_key,
                "metadata": event["response"]["metadata"],
            }
        )

    api._send_response_create = _fake_send_response_create
    websocket = _FakeWebsocket()
    context = api._build_utterance_context(turn_id="turn_1", input_event_key="item_1")

    asyncio.run(
        api.send_assistant_message(
            "repair",
            websocket,
            utterance_context=context,
            response_metadata={"trigger": "turn_contract_exact_phrase_repair"},
        )
    )

    assert len(scheduled_responses) == 1
    assert scheduled_responses[0]["origin"] == "assistant_message"


def test_lifecycle_state_coordinator_parity_with_realtime_api_methods() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._current_run_id = lambda: "run-parity"
    api._pending_confirmation_token = None
    api._is_awaiting_confirmation_phase = lambda: False
    api._has_active_confirmation_token = lambda: False

    canonical = api._canonical_utterance_key(turn_id=" turn_1 ", input_event_key=" evt_1 ")
    assert canonical == "run-parity:turn_1:evt_1"

    obligation = api._response_obligation_key(turn_id="turn_1", input_event_key="evt_1")
    assert obligation == canonical

    assert api._can_release_queued_response_create("assistant_message", {"source": "assistant"}) is True

    api._pending_confirmation_token = "tok"
    api._is_awaiting_confirmation_phase = lambda: True
    api._has_active_confirmation_token = lambda: True

    assert api._can_release_queued_response_create("assistant_message", {"source": "assistant"}) is False
    assert api._can_release_queued_response_create("text_message", {"source": "assistant"}) is True
    assert api._can_release_queued_response_create(
        "assistant_message", {"source": "assistant", "approval_flow": "yes"}
    ) is True
