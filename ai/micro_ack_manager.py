"""Micro-ack scheduling for low-latency conversational feedback."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import random
import time
from typing import Callable


@dataclass(frozen=True)
class MicroAckConfig:
    enabled: bool = True
    delay_ms: int = 450
    expected_wait_threshold_ms: int = 700
    long_wait_second_ack_ms: int = 4000
    global_cooldown_ms: int = 10000
    per_turn_max: int = 1
    anti_chatter_after_assistant_audio_end_ms: int = 1500
    talk_over_risk_window_ms: int = 6000


class MicroAckManager:
    """Handles scheduling, cancellation, and rate limits for micro-ack utterances."""

    def __init__(
        self,
        *,
        config: MicroAckConfig,
        on_emit: Callable[[str, str, str], None],
        on_log: Callable[[str, str, str, int | None], None],
        suppression_reason: Callable[[], str | None],
        now_fn: Callable[[], float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._on_emit = on_emit
        self._on_log = on_log
        self._suppression_reason = suppression_reason
        self._now = now_fn or time.monotonic
        self._rng = rng or random.Random()

        self._emitted_counts: dict[str, int] = {}
        self._scheduled: dict[str, asyncio.TimerHandle] = {}
        self._scheduled_reason: dict[str, str] = {}
        self._last_micro_ack_ts: float | None = None
        self._last_user_speech_started_ts: float | None = None
        self._last_user_speech_ended_ts: float | None = None
        self._last_assistant_audio_end_ts: float | None = None
        self._talk_over_incident_ts: deque[float] = deque()

        self._phrases: tuple[tuple[str, str], ...] = (
            ("one_sec_checking", "One sec—checking."),
            ("hmm", "Hmm…"),
            ("let_me_think", "Let me think."),
            ("checking_that", "Checking that…"),
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
        turn_id: str,
        reason: str,
        loop: asyncio.AbstractEventLoop,
        expected_delay_ms: int | None = None,
    ) -> None:
        if not self._config.enabled:
            self._on_log("suppressed", turn_id, "disabled", None)
            return
        if expected_delay_ms is not None and expected_delay_ms < self._config.expected_wait_threshold_ms:
            self._on_log("suppressed", turn_id, "predicted_fast_response", expected_delay_ms)
            return
        if self._emitted_counts.get(turn_id, 0) >= self._config.per_turn_max:
            self._on_log("suppressed", turn_id, "already_emitted", None)
            return
        if turn_id in self._scheduled:
            self._on_log("suppressed", turn_id, "already_scheduled", None)
            return
        suppress = self._suppression_reason()
        if suppress:
            self._on_log("suppressed", turn_id, suppress, None)
            return

        delay_s = max(0.01, self._config.delay_ms / 1000.0)
        handle = loop.call_later(delay_s, self._emit_if_allowed, turn_id, reason, loop)
        self._scheduled[turn_id] = handle
        self._scheduled_reason[turn_id] = reason
        self._on_log("scheduled", turn_id, reason, self._config.delay_ms)

    def cancel(self, *, turn_id: str, reason: str) -> None:
        handle = self._scheduled.pop(turn_id, None)
        self._scheduled_reason.pop(turn_id, None)
        if handle is None:
            return
        handle.cancel()
        self._on_log("cancelled", turn_id, reason, None)

    def cancel_all(self, *, reason: str) -> None:
        for turn_id in list(self._scheduled.keys()):
            self.cancel(turn_id=turn_id, reason=reason)

    def _emit_if_allowed(self, turn_id: str, reason: str, loop: asyncio.AbstractEventLoop) -> None:
        self._scheduled.pop(turn_id, None)
        self._scheduled_reason.pop(turn_id, None)

        suppress = self._suppression_reason()
        if suppress:
            self._on_log("suppressed", turn_id, suppress, None)
            return

        if self._emitted_counts.get(turn_id, 0) >= self._config.per_turn_max:
            self._on_log("suppressed", turn_id, "already_emitted", None)
            return

        now = self._now()
        if self._last_micro_ack_ts is not None:
            elapsed_ms = (now - self._last_micro_ack_ts) * 1000.0
            if elapsed_ms < self._config.global_cooldown_ms:
                self._on_log("suppressed", turn_id, "cooldown", int(elapsed_ms))
                return

        phrase_id, phrase = self._rng.choice(self._phrases)
        self._emitted_counts[turn_id] = self._emitted_counts.get(turn_id, 0) + 1
        self._last_micro_ack_ts = now
        self._on_emit(turn_id, phrase_id, phrase)
        self._on_log("emitted", turn_id, phrase_id, None)

        if self._emitted_counts[turn_id] < self._config.per_turn_max and self._config.long_wait_second_ack_ms > 0:
            delay_s = self._config.long_wait_second_ack_ms / 1000.0
            handle = loop.call_later(delay_s, self._emit_if_allowed, turn_id, "long_wait", loop)
            self._scheduled[turn_id] = handle
            self._scheduled_reason[turn_id] = "long_wait"
            self._on_log("scheduled", turn_id, "long_wait", self._config.long_wait_second_ack_ms)

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
