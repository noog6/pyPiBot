from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.realtime_api import RealtimeAPI


def _make_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._response_create_runtime = SimpleNamespace(
        send_response_create=AsyncMock(return_value=True),
        drain_response_create_queue=AsyncMock(return_value=None),
    )
    return api


def test_send_response_create_delegates_to_runtime() -> None:
    api = _make_api()

    result = asyncio.run(
        api._send_response_create(
            object(),
            {"type": "response.create"},
            origin="assistant_message",
            debug_context={"k": "v"},
        )
    )

    assert result is True
    api._response_create_runtime.send_response_create.assert_awaited_once()


def test_drain_response_create_queue_delegates_to_runtime() -> None:
    api = _make_api()

    asyncio.run(api._drain_response_create_queue(source_trigger="response_done"))

    api._response_create_runtime.drain_response_create_queue.assert_awaited_once_with(
        source_trigger="response_done"
    )
