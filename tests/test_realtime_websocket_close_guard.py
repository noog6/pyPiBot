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
