"""Battery-specific injection decision helpers."""

from __future__ import annotations

import time
from typing import Any

from ai.event_bus import Event
from ai.governance_spine import GovernanceDecision
from core.logging import logger

BATTERY_PRIORITY_ALLOW_CRITICAL = 20
BATTERY_PRIORITY_ALLOW_WARNING = 25
BATTERY_PRIORITY_ALLOW_QUERY_CONTEXT = 30
BATTERY_PRIORITY_ALLOW_QUERY_CONTEXT_SOFT = 35
BATTERY_PRIORITY_ALLOW_FALLBACK = 45
BATTERY_PRIORITY_SUPPRESS_WARNING = 55
BATTERY_PRIORITY_SUPPRESS_QUERY_CONTEXT = 60
BATTERY_PRIORITY_SUPPRESS_CRITICAL_TRANSITION = 65
BATTERY_PRIORITY_SUPPRESS_POLICY_DISABLED = 70
BATTERY_PRIORITY_SUPPRESS_POLICY_BLOCKED = 75
BATTERY_PRIORITY_SUPPRESS_TOPIC = 80


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

    def response_decision(self, event: Event, *, fallback: bool = False) -> GovernanceDecision:
        metadata = event.metadata or {}
        severity = str(metadata.get("severity", "info"))
        event_type = str(metadata.get("event_type", "status"))
        transition = str(metadata.get("transition", "steady"))

        # Adapter mapping is intentionally conservative at this seam:
        # suppress = hard policy block; allow = eligible for response creation.
        # GovernanceDecision.priority remains seam-local observability metadata
        # unless/until an explicit cross-system arbiter consumes it.
        if "battery" in getattr(self._api, "_suppressed_topics", set()) and not self.is_safety_override(event):
            logger.info("battery_response_suppressed reason=topic_suppression")
            return GovernanceDecision(
                decision="suppress",
                reason_code="topic_suppression",
                subsystem="battery",
                priority=BATTERY_PRIORITY_SUPPRESS_TOPIC,
                metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
            )

        if event_type == "clear" or severity == "info":
            active = self.is_query_context_active()
            return GovernanceDecision(
                decision="allow" if active else "suppress",
                reason_code="query_context_active" if active else "query_context_inactive",
                subsystem="battery",
                priority=(BATTERY_PRIORITY_ALLOW_QUERY_CONTEXT_SOFT if active else BATTERY_PRIORITY_SUPPRESS_QUERY_CONTEXT),
                metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
            )
        if not self._api._battery_response_enabled:
            active = self.is_query_context_active()
            return GovernanceDecision(
                decision="allow" if active else "suppress",
                reason_code="policy_disabled_with_query_context" if active else "policy_disabled",
                subsystem="battery",
                priority=(BATTERY_PRIORITY_ALLOW_QUERY_CONTEXT_SOFT if active else BATTERY_PRIORITY_SUPPRESS_POLICY_DISABLED),
                metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
            )

        if self.is_query_context_active():
            return GovernanceDecision(
                decision="allow",
                reason_code="query_context_active",
                subsystem="battery",
                priority=BATTERY_PRIORITY_ALLOW_QUERY_CONTEXT,
                metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
            )

        if severity == "critical" and self._api._battery_response_allow_critical:
            if self._api._battery_response_require_transition:
                allowed = not transition.startswith("steady_")
                return GovernanceDecision(
                    decision="allow" if allowed else "suppress",
                    reason_code="critical_transition_allowed" if allowed else "critical_transition_required",
                    subsystem="battery",
                    priority=(BATTERY_PRIORITY_ALLOW_CRITICAL if allowed else BATTERY_PRIORITY_SUPPRESS_CRITICAL_TRANSITION),
                    metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
                )
            return GovernanceDecision(
                decision="allow",
                reason_code="critical_allowed",
                subsystem="battery",
                priority=BATTERY_PRIORITY_ALLOW_CRITICAL,
                metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
            )

        if severity == "warning" and self._api._battery_response_allow_warning:
            allowed = transition in {"enter_warning", "enter_critical", "delta_drop"}
            return GovernanceDecision(
                decision="allow" if allowed else "suppress",
                reason_code="warning_transition_allowed" if allowed else "warning_transition_blocked",
                subsystem="battery",
                priority=(BATTERY_PRIORITY_ALLOW_WARNING if allowed else BATTERY_PRIORITY_SUPPRESS_WARNING),
                metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
            )

        fallback_allowed = fallback and not transition.startswith("steady_")
        return GovernanceDecision(
            decision="allow" if fallback_allowed else "suppress",
            reason_code="fallback_transition_allowed" if fallback_allowed else "policy_blocked",
            subsystem="battery",
            priority=(BATTERY_PRIORITY_ALLOW_FALLBACK if fallback_allowed else BATTERY_PRIORITY_SUPPRESS_POLICY_BLOCKED),
            metadata={"fallback": fallback, "event_type": event_type, "severity": severity, "transition": transition},
        )

    def should_request_response(self, event: Event, *, fallback: bool = False) -> bool:
        return self.response_decision(event, fallback=fallback).decision == "allow"
