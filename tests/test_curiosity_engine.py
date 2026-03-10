from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.curiosity_engine import CuriosityCandidate, CuriosityEngine


def _candidate(*, score: float = 0.5, dedupe_key: str = "conversation:repeated_topic:python") -> CuriosityCandidate:
    return CuriosityCandidate(
        source="conversation",
        reason_code="repeated_topic",
        score=score,
        dedupe_key=dedupe_key,
        created_at=10.0,
        expires_at=130.0,
        suggested_followup="brief follow-up" if score >= 0.6 else None,
    )


def test_curiosity_candidate_below_threshold_ignored() -> None:
    engine = CuriosityEngine(record_threshold=0.4, surface_threshold=0.6)

    decision = engine.evaluate(candidate=_candidate(score=0.2), now=20.0)

    assert decision.outcome == "ignore"
    assert decision.reason == "below_threshold"


def test_curiosity_duplicate_recent_candidate_deduped() -> None:
    engine = CuriosityEngine(record_threshold=0.4, surface_threshold=0.6, dedupe_window_s=30.0)

    first = engine.evaluate(candidate=_candidate(score=0.55), now=20.0)
    second = engine.evaluate(candidate=_candidate(score=0.55), now=30.0)

    assert first.outcome == "record"
    assert second.outcome == "ignore"
    assert second.reason == "deduped_recent"


def test_curiosity_cooldown_suppresses_repeated_surface() -> None:
    engine = CuriosityEngine(record_threshold=0.4, surface_threshold=0.6, surface_cooldown_s=90.0)

    first = engine.evaluate(candidate=_candidate(score=0.7, dedupe_key="k1"), now=20.0)
    second = engine.evaluate(candidate=_candidate(score=0.7, dedupe_key="k2"), now=60.0)

    assert first.outcome == "surface"
    assert second.outcome == "record"
    assert second.reason == "cooldown_active"


def test_curiosity_obligation_or_confirmation_prevents_surface() -> None:
    engine = CuriosityEngine(record_threshold=0.4, surface_threshold=0.6)

    decision = engine.evaluate(candidate=_candidate(score=0.8), arbitration_block_reason="obligation_open", now=20.0)

    assert decision.outcome == "record"
    assert decision.reason == "obligation_open"


def test_curiosity_surfacing_non_recursive_and_bounded() -> None:
    engine = CuriosityEngine(max_recent_candidates=4, dedupe_window_s=1.0)

    engine.evaluate(candidate=_candidate(score=0.8, dedupe_key="k1"), now=10.0)
    engine.evaluate(candidate=_candidate(score=0.8, dedupe_key="k2"), now=120.0)
    engine.evaluate(candidate=_candidate(score=0.8, dedupe_key="k3"), now=240.0)
    engine.evaluate(candidate=_candidate(score=0.8, dedupe_key="k4"), now=360.0)
    engine.evaluate(candidate=_candidate(score=0.8, dedupe_key="k5"), now=480.0)

    recent = engine.recent_candidates(now=490.0)

    assert len(recent) <= 4
    assert all(item.reason_code == "repeated_topic" for item in recent)


def test_curiosity_record_only_when_surface_suppressed() -> None:
    engine = CuriosityEngine(record_threshold=0.4, surface_threshold=0.6)

    decision = engine.evaluate(candidate=_candidate(score=0.8), suppress_surface=True, now=20.0)

    assert decision.outcome == "record"
    assert decision.reason == "suppressed_busy_turn"


def test_curiosity_builds_repeated_topic_candidate() -> None:
    engine = CuriosityEngine(record_threshold=0.4, surface_threshold=0.6)

    candidate = engine.build_conversation_candidate(topic_anchor="battery", repetition_count=3, now=10.0)

    assert candidate.source == "conversation"
    assert candidate.reason_code == "repeated_topic"
    assert candidate.score > 0.4
    assert candidate.dedupe_key == "conversation:repeated_topic:battery"
