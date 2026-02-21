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
