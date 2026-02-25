"""Research budget manager backed by SQLite budget state and usage audit rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

from core.logging import logger as LOGGER
from storage.factories import create_research_budget_store
from storage.research_budget import ResearchBudgetStorage, UsageEvent, UsageReservation


class ResearchBudgetManager:
    """Manage persisted daily budget state with optional audit metadata."""

    def __init__(
        self,
        state_file: str,
        daily_limit: int,
        *,
        rate_limit_per_minute: int | None = None,
        storage: ResearchBudgetStorage | None = None,
    ) -> None:
        self._legacy_state_file = Path(state_file)
        self._daily_limit = max(0, int(daily_limit))
        self._rate_limit_per_minute = (
            max(1, int(rate_limit_per_minute)) if rate_limit_per_minute is not None else None
        )
        self._window_started_at_s = 0.0
        self._window_count = 0
        self._storage = storage or create_research_budget_store()
        digest = hashlib.sha256(str(self._legacy_state_file).encode("utf-8")).hexdigest()[:12]
        self._budget_key = f"research_daily_fetch:{digest}"
        self._legacy_json_present = self._legacy_state_file.exists()
        self._migration_status = "skipped"
        self._migration_reason = "not_attempted"
        self._migrate_legacy_json_once()

    def startup_status(self) -> dict[str, Any]:
        """Return initialization metadata used for operator-facing startup logs."""

        return {
            "daily_limit": self._daily_limit,
            "authority": "sqlite_storage_controller",
            "legacy_json": "present" if self._legacy_json_present else "absent",
            "migration": self._migration_status,
            "migration_reason": self._migration_reason,
        }

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

        spent_at_ts = int(time.time())
        if self._daily_limit <= 0:
            self._storage.append_usage(
                date_utc=self._today_utc(),
                spent_at_ts=spent_at_ts,
                units=amount,
                request_fingerprint=self._audit_field(audit_payload, "request_fingerprint"),
                research_id=self._audit_field(audit_payload, "research_id"),
                source=self._audit_field(audit_payload, "source"),
                prompt_preview=self._audit_field(audit_payload, "prompt_preview"),
                provider=self._audit_field(audit_payload, "provider"),
                metadata=None,
            )
            self._record_rate_limit_spend()
            return True

        result = self._storage.spend_budget(
            key=self._budget_key,
            units=amount,
            daily_limit=self._daily_limit,
            usage_event=UsageEvent(
                spent_at_ts=spent_at_ts,
                request_fingerprint=self._audit_field(audit_payload, "request_fingerprint"),
                research_id=self._audit_field(audit_payload, "research_id"),
                source=self._audit_field(audit_payload, "source"),
                prompt_preview=self._audit_field(audit_payload, "prompt_preview"),
                provider=self._audit_field(audit_payload, "provider"),
            ),
        )
        if not result.allowed:
            return False
        self._record_rate_limit_spend()
        return True

    def spend_with_override(
        self,
        units: int = 1,
        *,
        audit_payload: dict[str, Any] | None = None,
        decision_source: str | None = None,
    ) -> bool:
        """Record an explicitly authorized over-budget spend audit row.

        This path bypasses remaining-budget checks and persists explicit authorization
        metadata for operator auditing.
        """

        amount = max(1, int(units))
        if not self._within_rate_limit():
            return False

        spent_at_ts = int(time.time())
        usage_metadata = self._storage.build_usage_metadata(
            over_budget_approved=True,
            decision_source=decision_source,
        )
        self._storage.append_usage(
            date_utc=self._today_utc(),
            spent_at_ts=spent_at_ts,
            units=amount,
            request_fingerprint=self._audit_field(audit_payload, "request_fingerprint"),
            research_id=self._audit_field(audit_payload, "research_id"),
            source=self._audit_field(audit_payload, "source"),
            prompt_preview=self._audit_field(audit_payload, "prompt_preview"),
            provider=self._audit_field(audit_payload, "provider"),
            metadata=usage_metadata,
        )
        self._record_rate_limit_spend()
        return True

    def reserve_execution(
        self,
        units: int = 1,
        *,
        audit_payload: dict[str, Any] | None = None,
        over_budget_approved: bool = False,
        decision_source: str | None = None,
    ) -> UsageReservation | None:
        """Reserve budget before provider work and persist a started audit row."""

        amount = max(1, int(units))
        if not self._within_rate_limit():
            return None

        started_at_ts = int(time.time())
        metadata = self._storage.build_usage_metadata(
            {"execution_status": "started"},
            over_budget_approved=over_budget_approved,
            decision_source=decision_source,
        )
        usage_event = UsageEvent(
            spent_at_ts=started_at_ts,
            request_fingerprint=self._audit_field(audit_payload, "request_fingerprint"),
            research_id=self._audit_field(audit_payload, "research_id"),
            source=self._audit_field(audit_payload, "source"),
            prompt_preview=self._audit_field(audit_payload, "prompt_preview"),
            provider=self._audit_field(audit_payload, "provider"),
            metadata=metadata,
        )

        if over_budget_approved:
            usage_id = self._storage.append_usage(
                date_utc=self._today_utc(),
                spent_at_ts=started_at_ts,
                units=amount,
                request_fingerprint=usage_event.request_fingerprint,
                research_id=usage_event.research_id,
                source=usage_event.source,
                prompt_preview=usage_event.prompt_preview,
                provider=usage_event.provider,
                metadata=usage_event.metadata,
            )
            self._record_rate_limit_spend()
            return UsageReservation(
                usage_id=usage_id,
                date_utc=self._today_utc(),
                spent_at_ts=started_at_ts,
                units=amount,
                key=self._budget_key,
                charged_to_budget=False,
            )

        if self._daily_limit <= 0:
            usage_id = self._storage.append_usage(
                date_utc=self._today_utc(),
                spent_at_ts=started_at_ts,
                units=amount,
                request_fingerprint=usage_event.request_fingerprint,
                research_id=usage_event.research_id,
                source=usage_event.source,
                prompt_preview=usage_event.prompt_preview,
                provider=usage_event.provider,
                metadata=usage_event.metadata,
            )
            self._record_rate_limit_spend()
            return UsageReservation(
                usage_id=usage_id,
                date_utc=self._today_utc(),
                spent_at_ts=started_at_ts,
                units=amount,
                key=self._budget_key,
                charged_to_budget=False,
            )

        reservation = self._storage.reserve_budget_usage(
            key=self._budget_key,
            units=amount,
            daily_limit=self._daily_limit,
            usage_event=usage_event,
        )
        if reservation is None:
            return None
        self._record_rate_limit_spend()
        return reservation

    def finalize_execution(
        self,
        reservation: UsageReservation,
        *,
        execution_status: str,
        refund: bool,
    ) -> bool:
        return self._storage.finalize_budget_usage(
            reservation=reservation,
            status=execution_status,
            refund=refund and self._daily_limit > 0,
        )

    def current_state(self) -> dict[str, Any]:
        date_utc = self._today_utc()
        if self._daily_limit <= 0:
            return {
                "date": date_utc,
                "count": 0,
                "remaining": 999999,
                "daily_limit": self._daily_limit,
                "last_audit": self._latest_audit_for_date(date_utc),
            }

        now_ts = int(time.time())
        state = self._storage.get_or_init_daily_state(
            key=self._budget_key,
            daily_limit=self._daily_limit,
            now_ts=now_ts,
            date_utc=date_utc,
        )
        remaining = int(state.remaining)
        return {
            "date": state.date_utc,
            "count": max(0, int(state.limit) - remaining),
            "remaining": remaining,
            "daily_limit": self._daily_limit,
            "last_audit": self._latest_audit_for_date(state.date_utc),
        }

    def _today_utc(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _latest_audit_for_date(self, date_utc: str) -> dict[str, Any] | None:
        usage_rows = self._storage.get_usage_for_date(date_utc)
        if not usage_rows:
            return None
        latest = usage_rows[-1]
        return {
            "request_fingerprint": latest.request_fingerprint,
            "research_id": latest.research_id,
            "source": latest.source,
            "prompt_preview": latest.prompt_preview,
            "provider": latest.provider,
        }

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

    @staticmethod
    def _audit_field(audit_payload: dict[str, Any] | None, key: str) -> str | None:
        if not isinstance(audit_payload, dict):
            return None
        value = audit_payload.get(key)
        return str(value) if value is not None else None

    def _migrate_legacy_json_once(self) -> None:
        if self._daily_limit <= 0:
            self._migration_reason = "daily_limit_non_positive"
            return
        if not self._legacy_json_present:
            self._migration_reason = "legacy_json_absent"
            return

        today = self._today_utc()
        existing = self._storage.get_state(self._budget_key)
        if existing is not None and existing.date_utc == today:
            self._migration_reason = "already_initialized_for_today"
            return

        try:
            payload = json.loads(self._legacy_state_file.read_text(encoding="utf-8"))
        except Exception:
            self._migration_reason = "legacy_json_invalid"
            return

        if str(payload.get("date")) != today:
            self._migration_reason = "legacy_json_date_mismatch"
            return

        count = max(0, int(payload.get("count", 0)))
        remaining = max(0, self._daily_limit - count)
        self._storage.upsert_state(
            key=self._budget_key,
            date_utc=today,
            remaining=remaining,
            limit=self._daily_limit,
            updated_at_ts=int(time.time()),
        )
        self._migration_status = "migrated"
        self._migration_reason = "legacy_json_loaded"
        LOGGER.info("research_budget_migration migrated=true source=json")
