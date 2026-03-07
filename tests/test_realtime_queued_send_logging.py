"""Regression tests for queued message send logging during shutdown."""

from __future__ import annotations

from collections.abc import Callable
import sys
import types

sys.modules.setdefault("audioop", types.SimpleNamespace(rms=lambda *_args, **_kwargs: 0, max=lambda *_args, **_kwargs: 0))

import ai.realtime_api as realtime_api
from ai.realtime_api import RealtimeAPI


class _DoneFuture:
    def __init__(self, result_factory: Callable[[], object]) -> None:
        self._result_factory = result_factory

    def add_done_callback(self, callback):
        callback(self)

    def result(self, timeout: float | None = None):
        return self._result_factory()


class _ShutdownStub:
    def __init__(self, *, shutdown_requested: bool, websocket_state: str) -> None:
        self._shutdown_requested = shutdown_requested
        self._websocket_state = websocket_state

    def is_shutdown_requested(self) -> bool:
        return self._shutdown_requested

    async def websocket_close_state(self) -> str:
        return self._websocket_state


class _ExpectedSendFailure(Exception):
    pass


class _UnexpectedSendFailure(Exception):
    pass


def _build_api(*, shutdown_requested: bool, websocket_state: str) -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.loop = object()
    api._shutdown = _ShutdownStub(
        shutdown_requested=shutdown_requested,
        websocket_state=websocket_state,
    )
    api._shutdown_coordinator = lambda: api._shutdown

    async def _send(*args, **kwargs):
        return None

    api.send_text_message_to_conversation = _send
    return api


def test_send_text_message_suppresses_warning_when_shutdown_requested() -> None:
    api = _build_api(shutdown_requested=True, websocket_state="open")
    info_logs: list[str] = []
    warning_logs: list[str] = []

    original_info = realtime_api.logger.info
    original_warning = realtime_api.logger.warning
    original_rcts = realtime_api.asyncio.run_coroutine_threadsafe

    realtime_api.logger.info = lambda message, *args: info_logs.append(message % args if args else message)
    realtime_api.logger.warning = lambda message, *args: warning_logs.append(message % args if args else message)
    realtime_api.asyncio.run_coroutine_threadsafe = lambda coro, _loop: (coro.close() if hasattr(coro, "close") else None) or _DoneFuture(lambda: None)
    try:
        api._send_text_message("ignored")
    finally:
        realtime_api.logger.info = original_info
        realtime_api.logger.warning = original_warning
        realtime_api.asyncio.run_coroutine_threadsafe = original_rcts

    assert not warning_logs
    assert any("queued_message_dropped_during_shutdown" in entry for entry in info_logs)


def test_send_text_message_downgrades_closed_socket_failure_to_info() -> None:
    api = _build_api(shutdown_requested=False, websocket_state="closed")
    info_logs: list[str] = []
    warning_logs: list[str] = []

    original_info = realtime_api.logger.info
    original_warning = realtime_api.logger.warning
    original_rcts = realtime_api.asyncio.run_coroutine_threadsafe

    call_index = {"value": 0}

    def _fake_run_coroutine_threadsafe(coro, _loop):
        if hasattr(coro, "close"):
            coro.close()
        call_index["value"] += 1
        if call_index["value"] == 1:
            return _DoneFuture(lambda: (_ for _ in ()).throw(_ExpectedSendFailure()))
        return _DoneFuture(lambda: "closed")

    realtime_api.logger.info = lambda message, *args: info_logs.append(message % args if args else message)
    realtime_api.logger.warning = lambda message, *args: warning_logs.append(message % args if args else message)
    realtime_api.asyncio.run_coroutine_threadsafe = _fake_run_coroutine_threadsafe
    try:
        api._send_text_message("queued")
    finally:
        realtime_api.logger.info = original_info
        realtime_api.logger.warning = original_warning
        realtime_api.asyncio.run_coroutine_threadsafe = original_rcts

    assert not warning_logs
    assert any("queued_message_dropped_during_shutdown" in entry for entry in info_logs)
    assert any("exception=_ExpectedSendFailure" in entry for entry in info_logs)


def test_send_text_message_keeps_warning_for_unexpected_failure() -> None:
    api = _build_api(shutdown_requested=False, websocket_state="open")
    warning_logs: list[str] = []

    original_warning = realtime_api.logger.warning
    original_rcts = realtime_api.asyncio.run_coroutine_threadsafe

    call_index = {"value": 0}

    def _fake_run_coroutine_threadsafe(coro, _loop):
        if hasattr(coro, "close"):
            coro.close()
        call_index["value"] += 1
        if call_index["value"] == 1:
            return _DoneFuture(lambda: (_ for _ in ()).throw(_UnexpectedSendFailure()))
        return _DoneFuture(lambda: "open")

    realtime_api.logger.warning = lambda message, *args: warning_logs.append(message % args if args else message)
    realtime_api.asyncio.run_coroutine_threadsafe = _fake_run_coroutine_threadsafe
    try:
        api._send_text_message("queued")
    finally:
        realtime_api.logger.warning = original_warning
        realtime_api.asyncio.run_coroutine_threadsafe = original_rcts

    assert any("queued_message_send_failed" in entry for entry in warning_logs)
    assert any("exception=_UnexpectedSendFailure" in entry for entry in warning_logs)
