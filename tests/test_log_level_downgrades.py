"""Regression tests for noisy log lines downgraded to DEBUG."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core import logging as core_logging


def _sample_session_updated_event() -> dict[str, object]:
    return {
        "type": "session.updated",
        "event_id": "evt_123",
        "session": {
            "id": "sess_123",
            "model": "gpt-test",
            "output_modalities": ["audio"],
            "tool_choice": "auto",
            "max_output_tokens": 512,
            "truncation": "disabled",
            "audio": {
                "input": {
                    "format": {"type": "pcm16", "rate": 24000},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.3,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                    },
                },
                "output": {
                    "format": {"type": "pcm16", "rate": 24000},
                    "voice": "alloy",
                    "speed": 1.0,
                },
            },
            "tools": [{"name": "tool_a", "type": "function", "parameters": {"properties": {}}}],
            "instructions": "# SOUL\nLong instructions payload",
        },
    }


@pytest.fixture
def realtime_logger_scope() -> logging.Logger:
    logger = core_logging.logger
    original_level = logger.level
    original_propagate = logger.propagate
    logger.propagate = True
    yield logger
    logger.propagate = original_propagate
    logger.setLevel(original_level)


def test_session_updated_payload_dump_hidden_at_info(
    realtime_logger_scope: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    realtime_logger_scope.setLevel(logging.INFO)
    caplog.set_level(logging.INFO)

    core_logging.log_session_updated(_sample_session_updated_event(), full_payload=True)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("SESSION_UPDATED |") for message in messages)
    assert core_logging.MARK_SUMMARY not in messages
    assert core_logging.MARK_PAYLOAD not in messages


def test_session_updated_payload_dump_visible_at_debug(
    realtime_logger_scope: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    realtime_logger_scope.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)

    core_logging.log_session_updated(_sample_session_updated_event(), full_payload=True)

    messages = [record.getMessage() for record in caplog.records]
    assert any(message.startswith("SESSION_UPDATED |") for message in messages)
    assert core_logging.MARK_SUMMARY in messages
    assert core_logging.MARK_PAYLOAD in messages


def test_ws_wire_trace_and_conversation_bookkeeping_are_debug(
    realtime_logger_scope: logging.Logger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    realtime_logger_scope.setLevel(logging.INFO)
    caplog.set_level(logging.INFO)
    core_logging.log_ws_event("Outgoing", {"type": "response.create"})
    core_logging.log_ws_event("Incoming", {"type": "conversation.item.done"})

    assert not caplog.records

    caplog.clear()
    realtime_logger_scope.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG)
    core_logging.log_ws_event("Outgoing", {"type": "response.create"})
    core_logging.log_ws_event("Incoming", {"type": "conversation.item.done"})

    messages = [record.getMessage() for record in caplog.records]
    assert any("➡️ ⬆️ - Out response.create" in message for message in messages)
    assert any("📝 ⬇️ - In conversation.item.done" in message for message in messages)


def test_camera_and_motion_emitters_use_debug_level() -> None:
    camera_source = Path("hardware/camera_controller.py").read_text(encoding="utf-8")
    motion_source = Path("motion/motion_controller.py").read_text(encoding="utf-8")

    assert 'logger.debug("[CAMERA] skipped (motion active)")' in camera_source
    assert "log_debug(" in motion_source
    assert "[MOTION] 'pan' servo move completed" in motion_source
    assert "[MOTION] 'tilt' servo move completed" in motion_source
