from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI


def test_no_text_concatenation_on_replace() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._stale_response_ids_set = {"resp-old"}

    assert api._should_drop_stale_response_event(
        {"type": "response.output_text.delta", "response_id": "resp-old", "delta": "that"}
    ) is True
    assert api._should_drop_stale_response_event(
        {"type": "response.text.delta", "response_id": "resp-old", "delta": "Your"}
    ) is True
