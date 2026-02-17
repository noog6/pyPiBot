"""Tests for guarded websocket close behavior."""

from __future__ import annotations

import asyncio

from ai.realtime_api import RealtimeAPI


class _ExitEvent:
    def __init__(self) -> None:
        self.set_count = 0

    def set(self) -> None:
        self.set_count += 1


class _BlockingWebSocket:
    def __init__(self) -> None:
        self.close_attempts = 0

    async def close(self) -> None:
        self.close_attempts += 1
        await asyncio.sleep(10)


def test_close_guard_allows_single_timeout_log_for_concurrent_shutdown_paths(monkeypatch) -> None:
    warnings: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )
    async def _run_test() -> None:
        api = RealtimeAPI.__new__(RealtimeAPI)
        api.websocket = _BlockingWebSocket()
        api.exit_event = _ExitEvent()
        api._ws_close_lock = asyncio.Lock()
        api._ws_close_started = False
        api._ws_close_done = False
        api.loop = asyncio.get_running_loop()

        api._request_shutdown()
        await api._close_websocket("audio loop exiting", websocket=api.websocket, timeout_s=0.02)
        await asyncio.sleep(0.05)

        assert api.websocket.close_attempts == 1
        timeout_warnings = [
            message for message in warnings if "Timed out closing WebSocket" in message
        ]
        assert len(timeout_warnings) == 1

    asyncio.run(_run_test())


class _MicStub:
    def __init__(self) -> None:
        self.stop_receiving_calls = 0
        self.start_recording_calls = 0

    def stop_receiving(self) -> None:
        self.stop_receiving_calls += 1

    def start_recording(self) -> None:
        self.start_recording_calls += 1


class _EventStub:
    def __init__(self, set_state: bool) -> None:
        self._set = set_state

    def is_set(self) -> bool:
        return self._set


def test_playback_complete_skips_mic_restart_during_shutdown(monkeypatch) -> None:
    info_logs: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.info",
        lambda message, *args: info_logs.append(message % args if args else message),
    )

    api = RealtimeAPI.__new__(RealtimeAPI)
    api.exit_event = _EventStub(True)
    api.mic = _MicStub()
    api.websocket = None
    api._audio_playback_busy = True
    api._response_create_queue = []
    api._pending_image_flush_after_playback = False
    api._pending_image_stimulus = None

    api._on_playback_complete()

    assert api.mic.stop_receiving_calls == 0
    assert api.mic.start_recording_calls == 0
    assert "Playback complete during shutdown -> skipping mic restart" in info_logs


def test_close_websocket_uses_configured_timeout_by_default(monkeypatch) -> None:
    warnings: list[str] = []

    monkeypatch.setattr(
        "ai.realtime_api.logger.warning",
        lambda message, *args: warnings.append(message % args),
    )

    async def _run_test() -> None:
        api = RealtimeAPI.__new__(RealtimeAPI)
        api.websocket = _BlockingWebSocket()
        api._websocket_close_timeout_s = 0.02
        api._ws_close_lock = asyncio.Lock()
        api._ws_close_started = False
        api._ws_close_done = False

        await api._close_websocket("configured-timeout", websocket=api.websocket)

        assert api.websocket.close_attempts == 1
        assert any("Timed out closing WebSocket (configured-timeout)." in message for message in warnings)

    asyncio.run(_run_test())
