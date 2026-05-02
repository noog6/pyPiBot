"""AI package utilities."""

from __future__ import annotations

from typing import Any

__all__ = ["RealtimeAPI"]


def __getattr__(name: str) -> Any:
    if name == "RealtimeAPI":
        from ai.realtime_api import RealtimeAPI

        return RealtimeAPI
    raise AttributeError(name)
