"""Tests for user transcript run logging controls."""

from __future__ import annotations

from ai.realtime_api import RealtimeAPI


def _make_api_stub(*, enabled: bool = True, redact_enabled: bool = True) -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._log_user_transcripts_enabled = enabled
    api._log_user_transcript_partials_enabled = False
    api._log_user_transcript_partials_min_chars_delta = 8
    api._log_user_transcript_redact_enabled = redact_enabled
    api._last_logged_partial_user_transcript = ""
    api._current_run_id = lambda: "run-123"
    return api


def test_user_transcript_logs_when_enabled(monkeypatch) -> None:
    api = _make_api_stub(enabled=True)
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._log_user_transcript(
        "hello there",
        final=True,
        event_type="conversation.item.input_audio_transcription.completed",
    )

    assert len(captured) == 1
    assert '[USER] transcript final: "hello there"' in captured[0]
    assert '"run_id": "run-123"' in captured[0]


def test_user_transcript_does_not_log_when_disabled(monkeypatch) -> None:
    api = _make_api_stub(enabled=False)
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._log_user_transcript(
        "hidden",
        final=True,
        event_type="conversation.item.input_audio_transcription.completed",
    )

    assert captured == []


def test_user_transcript_redaction_masks_email_and_phone(monkeypatch) -> None:
    api = _make_api_stub(enabled=True, redact_enabled=True)
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._log_user_transcript(
        "Reach me at test.user@example.com or +1 (555) 123-4567",
        final=True,
        event_type="conversation.item.input_audio_transcription.completed",
    )

    assert len(captured) == 1
    assert "<redacted_email>" in captured[0]
    assert "<redacted_phone>" in captured[0]
    assert "test.user@example.com" not in captured[0]
    assert "555" not in captured[0]


def test_utterance_info_summary_classifies_transcript_null(monkeypatch) -> None:
    api = _make_api_stub(enabled=True)
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._reset_utterance_info_summary()
    api._mark_utterance_info_summary(
        speech_started_seen=True,
        speech_stopped_seen=True,
        commit_seen=True,
        transcript_present=False,
    )

    api._emit_utterance_info_summary(anchor="transcript_completed_empty")

    assert len(captured) == 1
    assert "UTTERANCE_INFO_SUMMARY" in captured[0]
    assert "anchor=transcript_completed_empty" in captured[0]
    assert "transcript_present=False" in captured[0]
    assert "asr_error_present=False" in captured[0]


def test_utterance_info_summary_classifies_asr_error(monkeypatch) -> None:
    api = _make_api_stub(enabled=True)
    captured: list[str] = []

    def _capture(message, *args):
        captured.append(message % args)

    monkeypatch.setattr("ai.realtime_api.logger.info", _capture)

    api._reset_utterance_info_summary()
    api._mark_utterance_info_summary(
        speech_started_seen=True,
        speech_stopped_seen=True,
        asr_error_present=True,
        transcript_present=False,
    )

    api._emit_utterance_info_summary(anchor="transcript_failed")

    assert len(captured) == 1
    assert "UTTERANCE_INFO_SUMMARY" in captured[0]
    assert "anchor=transcript_failed" in captured[0]
    assert "transcript_present=False" in captured[0]
    assert "asr_error_present=True" in captured[0]
