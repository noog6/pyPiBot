from __future__ import annotations

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


def test_startup_prompt_and_user_audio_use_distinct_turn_and_canonical_keys() -> None:
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

    user_turn_id = "turn_2"
    user_input_event_key = "item-user-audio-1"
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

    assert user_turn_id != startup_turn_id
    assert startup_input_event_key.startswith("synthetic_prompt_")
    assert user_canonical_key != startup_canonical_key
    assert user_canonical_key != user_canonical_with_startup_key
    assert set(api._active_input_event_key_by_turn_id.keys()) == {startup_turn_id, user_turn_id}
    assert api._active_input_event_key_by_turn_id[startup_turn_id] != api._active_input_event_key_by_turn_id[user_turn_id]
