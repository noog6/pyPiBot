"""Battery-specific injection decision helpers."""

from __future__ import annotations

import time
from typing import Any

from ai.event_bus import Event
from core.logging import logger


class BatteryInjectionPolicy:
    def __init__(self, api: Any) -> None:
        self._api = api

    def is_battery_status_query(self, text: str) -> bool:
        lowered = text.strip().lower()
        if not lowered:
            return False
        query_tokens = (
            "battery",
            "battery level",
            "charge",
            "charging",
            "voltage",
            "power level",
            "low battery",
            "how's battery",
            "hows battery",
            "how is battery",
        )
        return any(token in lowered for token in query_tokens)

    def is_query_context_active(self) -> bool:
        if self._api._last_user_battery_query_time is None:
            return False
        return (
            time.monotonic() - self._api._last_user_battery_query_time
            <= self._api._battery_query_context_window_s
        )

    def is_safety_override(self, event: Event) -> bool:
        metadata = event.metadata or {}
        severity = str(metadata.get("severity", "")).strip().lower()
        if severity != "critical":
            return False
        percent = float(metadata.get("percent_of_range", 1.0)) * 100.0
        return percent <= self._api._battery_redline_percent

    def should_request_response(self, event: Event, *, fallback: bool = False) -> bool:
        metadata = event.metadata or {}
        severity = str(metadata.get("severity", "info"))
        event_type = str(metadata.get("event_type", "status"))
        transition = str(metadata.get("transition", "steady"))

        if "battery" in getattr(self._api, "_suppressed_topics", set()) and not self.is_safety_override(event):
            logger.info("battery_response_suppressed reason=topic_suppression")
            return False

        if event_type == "clear" or severity == "info":
            return self.is_query_context_active()
        if not self._api._battery_response_enabled:
            return self.is_query_context_active()

        if self.is_query_context_active():
            return True

        if severity == "critical" and self._api._battery_response_allow_critical:
            if self._api._battery_response_require_transition:
                return not transition.startswith("steady_")
            return True

        if severity == "warning" and self._api._battery_response_allow_warning:
            if self._api._battery_response_require_transition:
                return transition in {"enter_warning", "enter_critical", "delta_drop"}
            return transition in {"enter_warning", "enter_critical", "delta_drop"}

        return fallback and not transition.startswith("steady_")
