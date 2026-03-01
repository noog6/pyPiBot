"""Websocket transport wrappers for realtime API."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Callable


class RealtimeTransport:
    """Narrow websocket connect/send/recv facade."""

    def __init__(
        self,
        *,
        connect_fn: Callable[..., Any],
        validate_outbound_endpoint: Callable[[str], None],
    ) -> None:
        self._connect_fn = connect_fn
        self._validate_outbound_endpoint = validate_outbound_endpoint

    @asynccontextmanager
    async def connect(
        self,
        *,
        url: str,
        headers: dict[str, str],
        close_timeout: float,
        ping_interval: float,
        ping_timeout: float,
    ) -> AsyncIterator[Any]:
        self._validate_outbound_endpoint(url)
        async with self._connect_fn(
            url,
            additional_headers=headers,
            close_timeout=close_timeout,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        ) as websocket:
            yield websocket

    async def send_json(self, websocket: Any, payload: dict[str, Any]) -> None:
        await websocket.send(json.dumps(payload))

    async def recv_json(self, websocket: Any) -> dict[str, Any]:
        return json.loads(await websocket.recv())
