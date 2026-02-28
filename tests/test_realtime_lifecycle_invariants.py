from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


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
