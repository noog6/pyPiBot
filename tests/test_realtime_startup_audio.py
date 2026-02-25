"""Focused startup tests for audio dependency handling."""

from __future__ import annotations

import pytest

from ai.realtime_api import RealtimeAPI, RealtimeAPIStartupError


def test_initialize_microphone_strict_mode_raises_typed_startup_error(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._audio_input_device_name = "default"

    def _boom() -> object:
        raise RuntimeError("PyAudio is required for AsyncMicrophone")

    monkeypatch.setattr(api, "_create_microphone", _boom)

    with pytest.raises(RealtimeAPIStartupError) as exc_info:
        api._initialize_microphone(allow_failure=False)

    outcome = exc_info.value.outcome
    assert outcome.component == "audio_input"
    assert outcome.dependency_class == "required"
    assert outcome.status == "fatal"
    assert outcome.detail == "PyAudio is required for AsyncMicrophone"


def test_initialize_microphone_degraded_mode_returns_fallback(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._audio_input_device_name = "default"

    def _boom() -> object:
        raise RuntimeError("input device missing")

    monkeypatch.setattr(api, "_create_microphone", _boom)

    mic = api._initialize_microphone(allow_failure=True)

    mic.start_recording()
    mic.start_receiving()
    mic.stop_receiving()
    mic.stop_recording()

    assert mic.get_audio_data() is None
    assert mic.drain_queue() == 0
    assert mic.is_recording is False
    assert mic.is_receiving is False
