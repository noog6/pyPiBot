"""Rolling-window budget utilities."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Deque


class RollingWindowBudget:
    """Track a rolling-window budget for events."""

    def __init__(self, limit: int, window_s: float, *, name: str = "") -> None:
        self._limit = int(limit)
        self._window_s = float(window_s)
        self._name = name
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def name(self) -> str:
        return self._name

    def allow(self, now: float | None = None) -> bool:
        if self._limit <= 0:
            return True
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._prune(now)
            return len(self._timestamps) < self._limit

    def record(self, now: float | None = None) -> None:
        if self._limit <= 0:
            return
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._prune(now)
            self._timestamps.append(now)

    def remaining(self, now: float | None = None) -> int | None:
        if self._limit <= 0:
            return None
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._prune(now)
            return max(self._limit - len(self._timestamps), 0)

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
