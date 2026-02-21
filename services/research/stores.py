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

