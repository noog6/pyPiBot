from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest


class _FixtureEventDispatcher:
    def __init__(self) -> None:
        self.ordered_events: list[tuple[str, str]] = []
        self._cancelled_response_ids: set[str] = set()
        self._first_accepted_audio_delta_index: int | None = None
        self._cancel_event_index: int | None = None

    def dispatch(self, event: dict[str, object]) -> None:
        event_type = str(event["type"])
        response_id = str(event.get("response_id") or "")

        self.ordered_events.append((event_type, response_id))
        event_index = len(self.ordered_events) - 1

        if event_type == "response.cancel":
            self._cancelled_response_ids.add(response_id)
            if self._cancel_event_index is None:
                self._cancel_event_index = event_index
            return

        if event_type != "response.output_audio.delta":
            return

        if response_id in self._cancelled_response_ids:
            return

        if self._first_accepted_audio_delta_index is None:
            self._first_accepted_audio_delta_index = event_index

    def cancel_precedes_first_accepted_audio_delta(self) -> bool:
        if self._cancel_event_index is None or self._first_accepted_audio_delta_index is None:
            return False
        return self._cancel_event_index < self._first_accepted_audio_delta_index


def _load_fixture() -> dict[str, object]:
    fixture_path = Path(__file__).parent / "fixtures" / "realtime_server_auto_turn_lifecycle_stream.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario", _load_fixture()["scenarios"], ids=lambda s: s["name"])
def test_server_auto_turn_lifecycle_stream_scenarios(scenario: dict[str, object]) -> None:
    events = scenario["events"]
    expected = scenario["expected"]

    dispatcher = _FixtureEventDispatcher()
    created_by_origin: Counter[str] = Counter()
    assistant_done_count_by_utterance: dict[str, int] = defaultdict(int)
    canonical_progression: list[str] = []

    for event in events:
        dispatcher.dispatch(event)

        event_type = str(event["type"])
        origin = str(event.get("origin") or "")
        canonical_key = str(event.get("canonical_key") or "")
        utterance_id = str(event.get("utterance_id") or "")

        if event_type == "response.created":
            created_by_origin[origin] += 1

        if event_type == "response.done" and origin == "assistant_message":
            assistant_done_count_by_utterance[utterance_id] += 1

        if canonical_key and (not canonical_progression or canonical_progression[-1] != canonical_key):
            canonical_progression.append(canonical_key)

    expected_created = expected["response_created_by_origin"]
    for origin, count in expected_created.items():
        assert created_by_origin[origin] == count

    expected_cancel_before_audio = bool(expected["cancel_before_first_accepted_audio_delta"])
    assert dispatcher.cancel_precedes_first_accepted_audio_delta() is expected_cancel_before_audio

    expected_assistant_counts = expected["final_assistant_response_count_by_utterance"]
    for utterance_id, count in expected_assistant_counts.items():
        assert assistant_done_count_by_utterance[utterance_id] == count

    assert canonical_progression == expected["canonical_progression"]

    # The fallback scenario specifically must never schedule a second canonical-slot response.
    if scenario["name"] == "already_audio_started_fallback_no_second_slot":
        assert created_by_origin["assistant_message"] == 0
