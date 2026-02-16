"""Persistent stores for research cache and budgets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any


class ResearchCacheStore:
    """Simple JSON file cache keyed by SHA-256 of a scope+key pair."""

    def __init__(self, base_dir: str, ttl_hours: int = 24) -> None:
        self._base_dir = Path(base_dir)
        self._ttl_s = max(300, int(ttl_hours) * 3600)

    def get(self, scope: str, key: str) -> dict[str, Any] | None:
        path = self._path(scope, key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            ts = float(payload.get("timestamp", 0.0))
            if time.time() - ts > self._ttl_s:
                return None
            data = payload.get("data")
            if isinstance(data, dict):
                return data
        except Exception:
            return None
        return None

    def set(self, scope: str, key: str, data: dict[str, Any]) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": time.time(), "data": data}
        self._path(scope, key).write_text(json.dumps(payload), encoding="utf-8")

    def _path(self, scope: str, key: str) -> Path:
        digest = hashlib.sha256(f"{scope}:{key}".encode("utf-8")).hexdigest()
        return self._base_dir / f"{digest}.json"


class ResearchBudgetTracker:
    """Daily research budget tracker with simple persistent JSON state."""

    def __init__(self, state_file: str, daily_limit: int) -> None:
        self._state_file = Path(state_file)
        self._daily_limit = max(0, int(daily_limit))

    def get_remaining(self) -> int:
        state = self._load_state()
        return max(0, self._daily_limit - int(state.get("count", 0)))

    def can_spend(self, units: int = 1) -> bool:
        if self._daily_limit <= 0:
            return True
        return self.get_remaining() >= max(1, int(units))

    def spend(self, units: int = 1) -> int:
        amount = max(1, int(units))
        state = self._load_state()
        state["count"] = int(state.get("count", 0)) + amount
        self._save_state(state)
        return max(0, self._daily_limit - state["count"]) if self._daily_limit > 0 else 999999

    def _load_state(self) -> dict[str, Any]:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if not self._state_file.exists():
            return {"date": today, "count": 0}
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
            if payload.get("date") != today:
                return {"date": today, "count": 0}
            return {"date": today, "count": int(payload.get("count", 0))}
        except Exception:
            return {"date": today, "count": 0}

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(state), encoding="utf-8")
