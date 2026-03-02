
from __future__ import annotations

import pytest

from unittest.mock import AsyncMock, patch

from ai.realtime_api import RealtimeAPI


def test_find_stop_word_delegates_to_runtime() -> None:
    api = RealtimeAPI(prompts=[])
    with patch("ai.realtime_api.preference_recall_runtime._find_stop_word", return_value="halt") as mocked:
        result = api._find_stop_word("please halt now")
    mocked.assert_called_once_with(api, "please halt now")
    assert result == "halt"


@pytest.mark.asyncio
async def test_preference_recall_intent_delegates_to_runtime() -> None:
    api = RealtimeAPI(prompts=[])
    websocket = object()
    with patch(
        "ai.realtime_api.preference_recall_runtime._maybe_handle_preference_recall_intent",
        new=AsyncMock(return_value=True),
    ) as mocked:
        result = await api._maybe_handle_preference_recall_intent("text", websocket, source="test")
    mocked.assert_awaited_once_with(api, "text", websocket, source="test")
    assert result is True
