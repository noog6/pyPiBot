from __future__ import annotations

import sys
import types
from types import SimpleNamespace

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai import realtime_api as realtime_api_module
from ai.quiet_intent import QuietIntentSelector
from ai.realtime_api import RealtimeAPI
from interaction import InteractionState


def _build_quiet_intent_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._quiet_intent_selector = QuietIntentSelector()
    api._latest_quiet_intent_decision = None
    api._latest_quiet_intent_log_fingerprint = None
    api._last_user_input_time = 0.0
    api._last_user_input_text = ""
    api.response_in_progress = False
    api._response_in_flight = False
    api._latest_ops_severity = "unknown"
    api._current_run_id = lambda: "run-quiet-intent"
    api._current_turn_id_or_unknown = lambda: "turn-quiet-intent"
    api.get_continuity_brief = lambda **_kwargs: SimpleNamespace(stance="idle")
    return api


def test_generic_do_you_know_prompt_does_not_raise_curiosity_signal() -> None:
    api = _build_quiet_intent_api()
    api._last_user_input_text = "Theo, do you know of anything that is gray?"

    flags = RealtimeAPI._classify_recent_utterance_flags(api)

    assert "curiosity_signal" not in flags


def test_explicit_curiosity_language_still_raises_curiosity_signal() -> None:
    api = _build_quiet_intent_api()
    api._last_user_input_text = "I'm curious how this works and why it changed"

    flags = RealtimeAPI._classify_recent_utterance_flags(api)

    assert "curiosity_signal" in flags


def test_quiet_intent_info_logging_dedupes_low_value_repeats(monkeypatch) -> None:
    api = _build_quiet_intent_api()
    api._last_user_input_text = "Can you assist with this query"
    api._last_user_input_time = 0.0

    log_records: list[tuple[str, str]] = []

    def _capture_info(message: str, *args) -> None:
        log_records.append(("info", message % args))

    def _capture_debug(message: str, *args) -> None:
        log_records.append(("debug", message % args))

    monkeypatch.setattr(realtime_api_module.logger, "info", _capture_info)
    monkeypatch.setattr(realtime_api_module.logger, "debug", _capture_debug)

    attention = SimpleNamespace(active=False)

    RealtimeAPI._refresh_quiet_intent(api, state=InteractionState.IDLE, attention=attention)
    RealtimeAPI._refresh_quiet_intent(api, state=InteractionState.THINKING, attention=attention)

    info_messages = [line for level, line in log_records if level == "info"]
    debug_messages = [line for level, line in log_records if level == "debug"]
    assert len(info_messages) == 1
    assert "quiet_intent_decision mode=observer" in info_messages[0]
    assert any("quiet_intent_decision_unchanged" in line for line in debug_messages)

    api._latest_ops_severity = "critical"
    RealtimeAPI._refresh_quiet_intent(api, state=InteractionState.THINKING, attention=attention)

    info_messages = [line for level, line in log_records if level == "info"]
    assert len(info_messages) == 2
    assert "quiet_intent_decision mode=sentinel" in info_messages[-1]
