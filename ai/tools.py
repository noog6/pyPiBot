"""Tool definitions for realtime API calls."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


ToolFn = Callable[..., Awaitable[Any]]

function_map: dict[str, ToolFn] = {}

tools: list[dict[str, Any]] = []
