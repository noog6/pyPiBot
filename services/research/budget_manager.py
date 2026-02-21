"""Research budget manager with persisted daily caps and optional rate limiting."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


class ResearchBudgetManager:
    """Manage persisted daily budget state with optional audit metadata."""

    def __init__(
        self,
        state_file: str,
        daily_limit: int,
        *,
        rate_limit_per_minute: int | None = None,
    ) -> None:
        self._state_file = Path(state_file)
        self._daily_limit = max(0, int(daily_limit))
        self._rate_limit_per_minute = (
            max(1, int(rate_limit_per_minute)) if rate_limit_per_minute is not None else None
        )
        self._window_started_at_s = 0.0
        self._window_count = 0

    def can_spend(self, units: int = 1) -> bool:
        amount = max(1, int(units))
        if not self._within_rate_limit():
            return False
        if self._daily_limit <= 0:
            return True
        return self.current_state()["remaining"] >= amount

    def spend_if_allowed(
        self,
        units: int = 1,
        *,
        audit_payload: dict[str, Any] | None = None,
    ) -> bool:
        amount = max(1, int(units))
        if not self.can_spend(amount):
            return False

        state = self._load_state()
        state["count"] = int(state.get("count", 0)) + amount
        state["updated_at_ts"] = int(time.time())
        if audit_payload:
            state["last_audit"] = {
                "request_fingerprint": audit_payload.get("request_fingerprint"),
                "research_id": audit_payload.get("research_id"),
                "source": audit_payload.get("source"),
                "prompt_preview": audit_payload.get("prompt_preview"),
                "provider": audit_payload.get("provider"),
            }
        self._save_state(state)
        self._record_rate_limit_spend()
        return True

    def current_state(self) -> dict[str, Any]:
        state = self._load_state()
        count = int(state.get("count", 0))
        remaining = 999999 if self._daily_limit <= 0 else max(0, self._daily_limit - count)
        return {
            "date": state.get("date"),
            "count": count,
            "remaining": remaining,
            "daily_limit": self._daily_limit,
            "last_audit": state.get("last_audit") if isinstance(state.get("last_audit"), dict) else None,
        }

    def _today_utc(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load_state(self) -> dict[str, Any]:
        today = self._today_utc()
        if not self._state_file.exists():
            return {"date": today, "count": 0}
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            return {"date": today, "count": 0}

        if payload.get("date") != today:
            return {"date": today, "count": 0}

        return {
            "date": today,
            "count": int(payload.get("count", 0)),
            "updated_at_ts": int(payload.get("updated_at_ts", 0)),
            "last_audit": payload.get("last_audit"),
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(json.dumps(state), encoding="utf-8")

    def _within_rate_limit(self) -> bool:
        if self._rate_limit_per_minute is None:
            return True
        now = time.time()
        if now - self._window_started_at_s >= 60:
            self._window_started_at_s = now
            self._window_count = 0
        return self._window_count < self._rate_limit_per_minute

    def _record_rate_limit_spend(self) -> None:
        if self._rate_limit_per_minute is None:
            return
        now = time.time()
        if now - self._window_started_at_s >= 60:
            self._window_started_at_s = now
            self._window_count = 0
        self._window_count += 1
