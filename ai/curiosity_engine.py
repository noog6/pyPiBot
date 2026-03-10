"""Tiny curiosity scoring engine for low-rate, governed interest signals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class CuriosityCandidate:
    source: str
    reason_code: str
    score: float
    dedupe_key: str
    created_at: float
    expires_at: float
    suggested_followup: str | None = None


@dataclass(frozen=True)
class CuriosityDecision:
    outcome: str
    reason: str
    candidate: CuriosityCandidate | None = None


class CuriosityEngine:
    """Deterministic, bounded curiosity candidate evaluator."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_recent_candidates: int = 24,
        dedupe_window_s: float = 30.0,
        candidate_ttl_s: float = 120.0,
        record_threshold: float = 0.40,
        surface_threshold: float = 0.60,
        surface_cooldown_s: float = 90.0,
    ) -> None:
        self.enabled = bool(enabled)
        self._max_recent_candidates = max(4, int(max_recent_candidates))
        self._dedupe_window_s = max(1.0, float(dedupe_window_s))
        self._candidate_ttl_s = max(5.0, float(candidate_ttl_s))
        self._record_threshold = max(0.0, float(record_threshold))
        self._surface_threshold = max(self._record_threshold, float(surface_threshold))
        self._surface_cooldown_s = max(1.0, float(surface_cooldown_s))
        self._recent_candidates: deque[CuriosityCandidate] = deque(maxlen=self._max_recent_candidates)
        self._last_seen_by_dedupe_key: dict[str, float] = {}
        self._last_surface_at: float = 0.0

    def _prune_expired(self, *, now: float) -> None:
        while self._recent_candidates and self._recent_candidates[0].expires_at <= now:
            self._recent_candidates.popleft()
        expired_keys = [key for key, ts in self._last_seen_by_dedupe_key.items() if (now - ts) > self._dedupe_window_s]
        for key in expired_keys:
            self._last_seen_by_dedupe_key.pop(key, None)

    def build_conversation_candidate(
        self,
        *,
        topic_anchor: str,
        repetition_count: int,
        now: float | None = None,
    ) -> CuriosityCandidate:
        ts = float(now if now is not None else time.monotonic())
        normalized_anchor = str(topic_anchor or "").strip().lower()
        score = min(0.85, 0.34 + (0.12 * max(0, repetition_count - 1)))
        followup = (
            f"User keeps returning to '{normalized_anchor}'. Consider one brief follow-up question."
            if score >= self._surface_threshold
            else None
        )
        return CuriosityCandidate(
            source="conversation",
            reason_code="repeated_topic",
            score=score,
            dedupe_key=f"conversation:repeated_topic:{normalized_anchor}",
            created_at=ts,
            expires_at=ts + self._candidate_ttl_s,
            suggested_followup=followup,
        )

    def evaluate(
        self,
        *,
        candidate: CuriosityCandidate,
        arbitration_block_reason: str | None = None,
        suppress_surface: bool = False,
        now: float | None = None,
    ) -> CuriosityDecision:
        ts = float(now if now is not None else time.monotonic())
        self._prune_expired(now=ts)
        if not self.enabled:
            return CuriosityDecision(outcome="ignore", reason="engine_disabled")
        if candidate.score < self._record_threshold:
            return CuriosityDecision(outcome="ignore", reason="below_threshold")
        last_seen = self._last_seen_by_dedupe_key.get(candidate.dedupe_key)
        if isinstance(last_seen, float) and (ts - last_seen) <= self._dedupe_window_s:
            return CuriosityDecision(outcome="ignore", reason="deduped_recent")

        self._last_seen_by_dedupe_key[candidate.dedupe_key] = ts
        self._recent_candidates.append(candidate)

        if suppress_surface:
            return CuriosityDecision(outcome="record", reason="suppressed_busy_turn", candidate=candidate)
        if arbitration_block_reason:
            return CuriosityDecision(outcome="record", reason=arbitration_block_reason, candidate=candidate)
        if candidate.score < self._surface_threshold:
            return CuriosityDecision(outcome="record", reason="record_only", candidate=candidate)
        if self._last_surface_at > 0.0 and (ts - self._last_surface_at) <= self._surface_cooldown_s:
            return CuriosityDecision(outcome="record", reason="cooldown_active", candidate=candidate)

        self._last_surface_at = ts
        return CuriosityDecision(outcome="surface", reason="surface_eligible", candidate=candidate)

    def recent_candidates(self, *, now: float | None = None) -> list[CuriosityCandidate]:
        ts = float(now if now is not None else time.monotonic())
        self._prune_expired(now=ts)
        return list(self._recent_candidates)
