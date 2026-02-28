"""Micro-ack scheduling for low-latency conversational feedback."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import random
import time
from typing import Callable


class MicroAckCategory(str, Enum):
    START_OF_WORK = "start_of_work"
    LATENCY_MASK = "latency_mask"
    SAFETY_GATE = "safety_gate"
    FAILURE_FALLBACK = "failure_fallback"


@dataclass(frozen=True)
class MicroAckConfig:
    enabled: bool = True
    delay_ms: int = 450
    expected_wait_threshold_ms: int = 700
    long_wait_second_ack_ms: int = 4000
    global_cooldown_ms: int = 10000
    global_cooldown_scope: str = "channel"
    per_turn_max: int = 1
    anti_chatter_after_assistant_audio_end_ms: int = 1500
    talk_over_risk_window_ms: int = 6000
    dedupe_ttl_ms: int = 8000
    channel_enabled: dict[str, bool] | None = None
    channel_cooldown_ms: dict[str, int] | None = None
    category_cooldown_ms: dict[str, int] | None = None


@dataclass(frozen=True)
class MicroAckContext:
    category: MicroAckCategory
    channel: str
    turn_id: str
    run_id: str | None = None
    session_id: str | None = None
    intent: str | None = None
    action: str | None = None
    tool_call_id: str | None = None


@dataclass
class _ScheduledMicroAck:
    context: MicroAckContext
    reason: str
    handle: asyncio.TimerHandle


class MicroAckManager:
    """Handles scheduling, cancellation, and rate limits for micro-ack utterances."""

    def __init__(
        self,
        *,
        config: MicroAckConfig,
        on_emit: Callable[[MicroAckContext, str, str], None],
        on_log: Callable[
            [
                str,
                str,
                str,
                int | None,
                str | None,
                str | None,
                str | None,
                str | None,
                str | None,
                str | None,
                str | None,
            ],
            None,
        ],
        suppression_reason: Callable[[], str | None],
        now_fn: Callable[[], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._on_emit = on_emit
        self._on_log = on_log
        self._suppression_reason = suppression_reason
        self._now = now_fn or time.monotonic
        _ = rng
        self._emitted_at_by_dedupe_key: dict[str, float] = {}
        self._emitted_scope_history: dict[str, deque[float]] = {}
        self._scheduled: dict[str, _ScheduledMicroAck] = {}
        self._scheduled_reason: dict[str, str] = {}
        self._last_micro_ack_ts: float | None = None
        self._last_micro_ack_ts_by_channel: dict[str, float] = {}
        self._last_micro_ack_ts_by_category: dict[str, float] = {}
        self._last_user_speech_started_ts: float | None = None
        self._last_user_speech_ended_ts: float | None = None
        self._last_assistant_audio_end_ts: float | None = None
        self._talk_over_incident_ts: deque[float] = deque()

        self._phrases_by_category: dict[MicroAckCategory, tuple[tuple[str, str], ...]] = {
            MicroAckCategory.START_OF_WORK: (
                ("start_of_work_one_sec", "One moment while I check."),
                ("start_of_work_on_it", "Let me check."),
            ),
            MicroAckCategory.LATENCY_MASK: (
                ("latency_mask_hmm", "Hmm…"),
                ("latency_mask_let_me_think", "Let me think."),
                ("latency_mask_checking_that", "Checking that…"),
            ),
            MicroAckCategory.SAFETY_GATE: (
                ("safety_gate_hold_on", "Hold on—I should verify that first."),
                ("safety_gate_careful", "I should be careful here—checking."),
            ),
            MicroAckCategory.FAILURE_FALLBACK: (
                ("failure_fallback_retry", "One sec—I can try another way."),
                ("failure_fallback_recover", "Give me a moment to recover that."),
            ),
        }

    def _log(
        self,
        event: str,
        turn_id: str,
        reason: str,
        delay_ms: int | None,
        context: MicroAckContext | None,
        suppression_source: str | None = None,
    ) -> None:
        dedupe_fingerprint = self._dedupe_fingerprint(context) if context else None
        self._on_log(
            event,
            turn_id,
            reason,
            delay_ms,
            (
                context.category.value
                if context and isinstance(context.category, MicroAckCategory)
                else (str(context.category) if context else None)
            ),
            context.channel if context else None,
            context.intent if context else None,
            context.action if context else None,
            context.tool_call_id if context else None,
            dedupe_fingerprint,
            suppression_source,
        )

    def on_user_speech_started(self) -> None:
        self._last_user_speech_started_ts = self._now()

    def on_user_speech_ended(self) -> None:
        self._last_user_speech_ended_ts = self._now()

    def mark_assistant_audio_ended(self) -> None:
        self._last_assistant_audio_end_ts = self._now()

    def mark_talk_over_incident(self) -> None:
        ts = self._now()
        self._talk_over_incident_ts.append(ts)
        window_s = max(0.1, self._config.talk_over_risk_window_ms / 1000.0)
        while self._talk_over_incident_ts and ts - self._talk_over_incident_ts[0] > window_s:
            self._talk_over_incident_ts.popleft()

    def has_recent_talk_over_incident(self) -> bool:
        now = self._now()
        window_s = max(0.1, self._config.talk_over_risk_window_ms / 1000.0)
        while self._talk_over_incident_ts and now - self._talk_over_incident_ts[0] > window_s:
            self._talk_over_incident_ts.popleft()
        return bool(self._talk_over_incident_ts)

    def maybe_schedule(
        self,
        *,
        context: MicroAckContext,
        reason: str,
        loop: asyncio.AbstractEventLoop,
        expected_delay_ms: int | None = None,
    ) -> None:
        turn_id = context.turn_id
        dedupe_key = self._dedupe_key(context)
        scope_key = self._scope_key(context)
        self._expire_ttl_state()
        if not self._config.enabled:
            self._log("suppressed", turn_id, "disabled", None, context)
            return
        if expected_delay_ms is not None and expected_delay_ms < self._config.expected_wait_threshold_ms:
            self._log("suppressed", turn_id, "predicted_fast_response", expected_delay_ms, context)
            return
        if not self._is_channel_enabled(context.channel):
            self._log("suppressed", turn_id, "channel_disabled", None, context)
            return
        scope_history = self._emitted_scope_history.get(scope_key, deque())
        if len(scope_history) >= self._config.per_turn_max:
            self._log("suppressed", turn_id, "already_emitted", None, context)
            return
        if dedupe_key in self._emitted_at_by_dedupe_key:
            self._log("suppressed", turn_id, "duplicate_within_ttl", None, context)
            return
        now = self._now()
        category_key = self._category_key(context.category)
        category_last_ts = self._last_micro_ack_ts_by_category.get(category_key)
        if category_last_ts is not None:
            elapsed_category_ms = (now - category_last_ts) * 1000.0
            category_cooldown_ms = self._category_cooldown_ms(context.category)
            if elapsed_category_ms < category_cooldown_ms:
                self._log("suppressed", turn_id, "category_cooldown", int(elapsed_category_ms), context)
                return
        if dedupe_key in self._scheduled:
            self._log("suppressed", turn_id, "already_scheduled", None, context)
            return
        suppress = self._suppression_reason()
        if suppress:
            self._log("suppressed", turn_id, suppress, None, context, self._suppression_source_for_reason(suppress))
            return

        delay_s = max(0.01, self._config.delay_ms / 1000.0)
        handle = loop.call_later(delay_s, self._emit_if_allowed, context, reason, loop)
        self._scheduled[dedupe_key] = _ScheduledMicroAck(context=context, reason=reason, handle=handle)
        self._scheduled_reason[dedupe_key] = reason
        self._log("scheduled", turn_id, reason, self._config.delay_ms, context)

    def cancel(self, *, turn_id: str, reason: str) -> None:
        keys_for_turn = [key for key, item in self._scheduled.items() if item.context.turn_id == turn_id]
        for key in keys_for_turn:
            scheduled = self._scheduled.pop(key, None)
            self._scheduled_reason.pop(key, None)
            if scheduled is None:
                continue
            scheduled.handle.cancel()
            self._log("cancelled", turn_id, reason, None, scheduled.context)

    def cancel_all(self, *, reason: str) -> None:
        for key in list(self._scheduled.keys()):
            scheduled = self._scheduled.pop(key, None)
            self._scheduled_reason.pop(key, None)
            if scheduled is None:
                continue
            scheduled.handle.cancel()
            self._log("cancelled", scheduled.context.turn_id, reason, None, scheduled.context)

    def _emit_if_allowed(self, context: MicroAckContext, reason: str, loop: asyncio.AbstractEventLoop) -> None:
        turn_id = context.turn_id
        dedupe_key = self._dedupe_key(context)
        scope_key = self._scope_key(context)
        self._scheduled.pop(dedupe_key, None)
        self._scheduled_reason.pop(dedupe_key, None)
        self._expire_ttl_state()

        suppress = self._suppression_reason()
        if suppress:
            self._log("suppressed", turn_id, suppress, None, context, self._suppression_source_for_reason(suppress))
            return

        scope_history = self._emitted_scope_history.get(scope_key, deque())
        if len(scope_history) >= self._config.per_turn_max:
            self._log("suppressed", turn_id, "already_emitted", None, context)
            return
        if dedupe_key in self._emitted_at_by_dedupe_key:
            self._log("suppressed", turn_id, "duplicate_within_ttl", None, context)
            return

        now = self._now()
        channel_last_ts = self._last_micro_ack_ts_by_channel.get(context.channel)
        if channel_last_ts is not None:
            elapsed_channel_ms = (now - channel_last_ts) * 1000.0
            channel_cooldown_ms = self._channel_cooldown_ms(context.channel)
            if elapsed_channel_ms < channel_cooldown_ms:
                self._log("suppressed", turn_id, "channel_cooldown", int(elapsed_channel_ms), context, "cooldown")
                return

        if self._global_cooldown_scope() == "all" and self._last_micro_ack_ts is not None:
            elapsed_ms = (now - self._last_micro_ack_ts) * 1000.0
            if elapsed_ms < self._config.global_cooldown_ms:
                self._log("suppressed", turn_id, "cooldown", int(elapsed_ms), context, "cooldown")
                return
        category_key = self._category_key(context.category)
        category_last_ts = self._last_micro_ack_ts_by_category.get(category_key)
        if category_last_ts is not None:
            elapsed_category_ms = (now - category_last_ts) * 1000.0
            category_cooldown_ms = self._category_cooldown_ms(context.category)
            if elapsed_category_ms < category_cooldown_ms:
                self._log("suppressed", turn_id, "category_cooldown", int(elapsed_category_ms), context, "cooldown")
                return

        phrase_id, phrase = self._select_phrase(context=context, dedupe_key=dedupe_key)
        emitted_at = self._now()
        self._emitted_at_by_dedupe_key[dedupe_key] = emitted_at
        scope_history.append(emitted_at)
        self._emitted_scope_history[scope_key] = scope_history
        self._last_micro_ack_ts = now
        self._last_micro_ack_ts_by_channel[context.channel] = now
        self._last_micro_ack_ts_by_category[category_key] = now
        self._on_emit(context, phrase_id, phrase)
        self._log("emitted", turn_id, phrase_id, None, context)

        if len(scope_history) < self._config.per_turn_max and self._config.long_wait_second_ack_ms > 0:
            delay_s = self._config.long_wait_second_ack_ms / 1000.0
            handle = loop.call_later(delay_s, self._emit_if_allowed, context, "long_wait", loop)
            self._scheduled[dedupe_key] = _ScheduledMicroAck(context=context, reason="long_wait", handle=handle)
            self._scheduled_reason[dedupe_key] = "long_wait"
            self._log("scheduled", turn_id, "long_wait", self._config.long_wait_second_ack_ms, context)

    def _expire_ttl_state(self) -> None:
        now = self._now()
        ttl_s = max(0.1, self._config.dedupe_ttl_ms / 1000.0)
        for key, emitted_at in list(self._emitted_at_by_dedupe_key.items()):
            if now - emitted_at > ttl_s:
                self._emitted_at_by_dedupe_key.pop(key, None)

        for scope_key, history in list(self._emitted_scope_history.items()):
            while history and now - history[0] > ttl_s:
                history.popleft()
            if not history:
                self._emitted_scope_history.pop(scope_key, None)

    @staticmethod
    def _scope_key(context: MicroAckContext) -> str:
        return f"turn={context.turn_id}|channel={context.channel}"

    @staticmethod
    def _dedupe_key(context: MicroAckContext) -> str:
        return (
            f"channel={context.channel}|category={context.category}|run={context.run_id or ''}|"
            f"session={context.session_id or ''}|intent={context.intent or ''}|"
            f"action={context.action or ''}|tool_call_id={context.tool_call_id or ''}"
        )

    def _dedupe_fingerprint(self, context: MicroAckContext) -> str:
        digest = hashlib.blake2s(self._dedupe_key(context).encode("utf-8"), digest_size=4)
        return digest.hexdigest()

    @staticmethod
    def _suppression_source_for_reason(reason: str) -> str:
        normalized = str(reason or "").strip().lower()
        if normalized in {"cooldown", "channel_cooldown", "category_cooldown"}:
            return "cooldown"
        if normalized.startswith("confirmation"):
            return "confirmation"
        if normalized in {"anti_chatter", "talk_over_risk", "speech_active", "listening_state"}:
            return "baseline"
        return "policy"

    def _select_phrase(self, *, context: MicroAckContext, dedupe_key: str) -> tuple[str, str]:
        phrases = self._phrases_by_category.get(
            context.category,
            self._phrases_by_category[MicroAckCategory.LATENCY_MASK],
        )
        if len(phrases) == 1:
            return phrases[0]
        selector = sum(ord(ch) for ch in dedupe_key) % len(phrases)
        return phrases[selector]

    def suppression_baseline_reason(self) -> str | None:
        now = self._now()
        if self._last_user_speech_started_ts and (
            self._last_user_speech_ended_ts is None
            or self._last_user_speech_started_ts > self._last_user_speech_ended_ts
        ):
            return "speech_active"
        if self.has_recent_talk_over_incident():
            return "talk_over_risk"
        if self._last_assistant_audio_end_ts is not None:
            elapsed_ms = (now - self._last_assistant_audio_end_ts) * 1000.0
            if elapsed_ms < self._config.anti_chatter_after_assistant_audio_end_ms:
                return "anti_chatter"
        return None

    def _is_channel_enabled(self, channel: str) -> bool:
        mapping = self._config.channel_enabled or {}
        return bool(mapping.get(channel, True))

    def _channel_cooldown_ms(self, channel: str) -> int:
        mapping = self._config.channel_cooldown_ms or {}
        configured = int(mapping.get(channel, self._config.global_cooldown_ms))
        return max(0, configured)

    def _global_cooldown_scope(self) -> str:
        configured = str(getattr(self._config, "global_cooldown_scope", "channel")).strip().lower()
        if configured in {"all", "channel"}:
            return configured
        return "channel"

    @staticmethod
    def _category_key(category: MicroAckCategory | str) -> str:
        if isinstance(category, MicroAckCategory):
            return category.value
        return str(category)

    def _category_cooldown_ms(self, category: MicroAckCategory | str) -> int:
        mapping = self._config.category_cooldown_ms or {}
        configured = int(mapping.get(self._category_key(category), 0))
        return max(0, configured)
