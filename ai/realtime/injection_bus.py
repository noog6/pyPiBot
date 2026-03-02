"""Central stimulus and response injection policy bus."""

from __future__ import annotations

import json
import time
from collections import deque
from typing import Any

from core.logging import log_info, log_warning, logger
from interaction import InteractionState

from .injections import InjectionCoordinator


class InjectionBus:
    def __init__(self, api: Any, coordinator: InjectionCoordinator) -> None:
        self._api = api
        self._coordinator = coordinator

    @property
    def coordinator(self) -> InjectionCoordinator:
        return self._coordinator

    def can_accept_external_stimulus(
        self,
        source: str,
        kind: str,
        *,
        priority: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        self._api._expire_confirmation_awaiting_decision_timeout()
        phase = getattr(self._api.orchestration_state, "phase", None)
        phase_name = getattr(phase, "value", str(phase))
        normalized_source = str(source).lower()
        normalized_kind = self.normalize_external_stimulus_kind(kind, metadata)
        normalized_priority = (priority or "").lower()

        pending_action = getattr(self._api, "_pending_confirmation_token", None)
        response_in_progress = bool(getattr(self._api, "response_in_progress", False))

        if (
            pending_action is not None
            and self._api._is_awaiting_confirmation_phase()
            and not self._api._is_allowed_awaiting_confirmation_stimulus(
                normalized_source,
                normalized_kind,
                normalized_priority,
            )
        ):
            return False, "awaiting_confirmation_policy"

        if response_in_progress:
            return False, "response_in_progress"

        state = getattr(self._api.state_manager, "state", None)
        if state not in (InteractionState.IDLE, InteractionState.LISTENING):
            state_name = getattr(state, "value", str(state))
            return False, f"interaction_state={state_name}"

        return True, f"phase={phase_name}"

    def normalize_external_stimulus_kind(
        self,
        kind: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if metadata:
            severity = metadata.get("severity")
            if isinstance(severity, str) and severity.strip():
                return severity.strip().lower()
        return str(kind).strip().lower() or "unknown"

    def startup_injection_gate_active(self) -> bool:
        return self._coordinator.gate_active()

    def startup_gate_is_critical_allowed(self, source: str, kind: str, priority: str) -> bool:
        normalized_source = str(source).strip().lower()
        normalized_kind = str(kind).strip().lower()
        normalized_priority = str(priority).strip().lower()
        return self._api._is_allowed_awaiting_confirmation_stimulus(
            normalized_source,
            normalized_kind,
            normalized_priority,
        )

    def maybe_defer_startup_injection(
        self,
        *,
        source: str,
        kind: str,
        priority: str,
        payload: dict[str, Any],
    ) -> bool:
        if not self.startup_injection_gate_active():
            return False
        if self.startup_gate_is_critical_allowed(source, kind, priority):
            return False
        deferred = self._coordinator.should_defer(payload, source)
        if deferred:
            self._coordinator.enqueue(payload)
        return deferred

    def release_startup_injection_gate(self, *, reason: str) -> None:
        self._coordinator.release(reason)

    async def startup_injection_timeout_release(self) -> None:
        await self._coordinator._timeout_release()

    def ensure_startup_injection_timeout_task(self) -> None:
        self._coordinator.schedule_timeout(getattr(self._api, "loop", None))

    async def maybe_request_response(self, trigger: str, metadata: dict[str, Any]) -> None:
        if await self._api._should_defer_sensor_response(trigger, metadata):
            return
        if not self._api.websocket:
            log_warning("Skipping injected response (%s): websocket unavailable.", trigger)
            return
        if self._api._has_active_confirmation_token() and not self._api._is_user_confirmation_trigger(trigger, metadata):
            logger.info(
                "Suppressing duplicate non-user response request trigger=%s phase=%s pending_action=%s",
                trigger,
                getattr(getattr(self._api.orchestration_state, "phase", None), "value", "unknown"),
                self._api._pending_action is not None,
            )
            return

        trigger_config = self._api._injection_response_triggers.get(trigger, {})
        trigger_cooldown_s = float(trigger_config.get("cooldown_s", self._api._injection_response_cooldown_s))
        trigger_max_per_minute = int(trigger_config.get("max_per_minute", self._api._max_injection_responses_per_minute))
        trigger_priority = int(trigger_config.get("priority", 0))
        trigger_timestamps = self._api._injection_response_trigger_timestamps.setdefault(trigger, deque())
        bypass_limits = bool(metadata.get("bypass_limits", False))

        if self._api.response_in_progress and not bypass_limits:
            if trigger == "image_message":
                self._api._queue_pending_image_stimulus(trigger, metadata)
                return
            logger.info("Skipping injected response (%s): response already in progress.", trigger)
            return

        if self._api.state_manager.state not in (InteractionState.IDLE, InteractionState.LISTENING) and not bypass_limits:
            logger.info("Skipping injected response (%s): invalid state %s.", trigger, self._api.state_manager.state.value)
            return

        if trigger_cooldown_s > 0.0 and not bypass_limits:
            now = time.monotonic()
            last_ts = trigger_timestamps[-1] if trigger_timestamps else None
            if last_ts is not None and now - last_ts < trigger_cooldown_s:
                logger.info(
                    "Skipping injected response (%s): trigger cooldown %.2fs remaining.",
                    trigger,
                    trigger_cooldown_s - (now - last_ts),
                )
                return

        if trigger_max_per_minute > 0 and not bypass_limits:
            now = time.monotonic()
            while trigger_timestamps and now - trigger_timestamps[0] > 60:
                trigger_timestamps.popleft()
            if len(trigger_timestamps) >= trigger_max_per_minute:
                logger.info("Skipping injected response (%s): trigger rate limit %s/minute reached.", trigger, trigger_max_per_minute)
                return

        bypass_global_limits = trigger_priority > 0 or bypass_limits

        if self._api._injection_response_cooldown_s > 0.0 and not bypass_global_limits:
            now = time.monotonic()
            last_ts = self._api._injection_response_timestamps[-1] if self._api._injection_response_timestamps else None
            if last_ts is not None and now - last_ts < self._api._injection_response_cooldown_s:
                logger.info(
                    "Skipping injected response (%s): cooldown %.2fs remaining.",
                    trigger,
                    self._api._injection_response_cooldown_s - (now - last_ts),
                )
                return

        if self._api._max_injection_responses_per_minute > 0 and not bypass_global_limits:
            now = time.monotonic()
            while self._api._injection_response_timestamps and now - self._api._injection_response_timestamps[0] > 60:
                self._api._injection_response_timestamps.popleft()
            if len(self._api._injection_response_timestamps) >= self._api._max_injection_responses_per_minute:
                logger.info("Skipping injected response (%s): rate limit %s/minute reached.", trigger, self._api._max_injection_responses_per_minute)
                return

        if self._api.rate_limits:
            for limit_name in ("requests", "responses"):
                rate_limit = self._api.rate_limits.get(limit_name)
                if not rate_limit:
                    continue
                remaining = rate_limit.get("remaining")
                if remaining is not None and remaining <= 0:
                    logger.info("Skipping injected response (%s): %s rate limit exhausted.", trigger, limit_name)
                    return

        bypass_budget = trigger == "text_message"
        if not self._api._allow_ai_call(f"injection:{trigger}", bypass=bypass_budget):
            logger.info("Skipping injected response (%s): AI call budget exhausted.", trigger)
            return
        response_metadata = {
            "trigger": trigger,
            "priority": str(trigger_priority),
            "stimulus": json.dumps(metadata, sort_keys=True),
            "origin": "injection",
            "sensor_aggregation_metrics": json.dumps(self._api._sensor_event_aggregation_metrics, sort_keys=True),
        }
        if bool(metadata.get("safety_override", False)):
            response_metadata["safety_override"] = "true"
        if self._api._has_active_confirmation_token():
            response_metadata["approval_flow"] = "true"
        memory_brief_note = self._api._consume_pending_memory_brief_note()
        response_create_event = {"type": "response.create", "response": {"metadata": response_metadata}}
        log_info(f"Requesting injected response for {trigger} with metadata {response_metadata}.")
        sent_now = await self._api._send_response_create(
            self._api.websocket,
            response_create_event,
            origin="injection",
            record_ai_call=True,
            memory_brief_note=memory_brief_note,
        )
        if sent_now:
            now = time.monotonic()
            self._api._injection_response_timestamps.append(now)
            trigger_timestamps.append(now)
