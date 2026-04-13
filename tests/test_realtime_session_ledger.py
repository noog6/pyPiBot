"""Tests for realtime session-ledger readiness persistence."""

from __future__ import annotations

import asyncio
import threading

from ai.realtime_api import RealtimeAPI


class _FakeStorage:
    def __init__(self) -> None:
        self.running_marks: list[str] = []

    def get_canonical_run_id(self) -> str:
        return "run-9"

    def mark_session_running(self, run_id: str) -> None:
        self.running_marks.append(run_id)


def test_session_updated_marks_run_running_once() -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.ready_event = threading.Event()
    api._storage = _FakeStorage()
    api._ensure_startup_injection_timeout_task = lambda: None
    api._should_process_response_event_ingress = lambda event, source: True

    asyncio.run(api._handle_event_legacy({"type": "session.updated"}, websocket=None))
    asyncio.run(api._handle_event_legacy({"type": "session.updated"}, websocket=None))

    assert api.ready_event.is_set()
    assert api._storage.running_marks == ["run-9"]
