"""ASR trust snapshot and verify-on-risk heuristics."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "i", "in", "is",
    "it", "know", "me", "my", "of", "on", "or", "that", "the", "to", "what", "where", "who", "you",
    "your", "was", "were", "with", "there", "hey", "theo",
}

_VISUAL_TERMS = {"pants", "shirt", "hat", "wearing", "wear", "color", "see"}
_GREETING_TERMS = {"hi", "hello", "hey", "yo", "sup", "howdy"}
_CLEAR_MOTION_TERMS = {"look", "center", "left", "right", "up", "down", "back"}
_CURRENT_VISUAL_PATTERNS = (
    r"\bwhat\s+do\s+you\s+see\s*(right\s+now|now)?\b",
    r"\bwhat'?s\s+in\s+front\s+of\s+you\b",
    r"\b(?:tell|let\s+me\s+know)\s+me\s+what\s+you\s+see\s+in\s+front\s+of\s+you\b",
    r"\bwhat\s+are\s+you\s+looking\s+at\s+(?:right\s+now|now)\b",
    r"\bcan\s+you\s+look\s+and\s+tell\s+me\s+what(?:'?s|\s+is)?\s+(?:there|in\s+front\s+of\s+you)\b",
    r"\blook\s+at\s+what'?s\s+in\s+front\s+of\s+you\s+and\s+describe\s+it\b",
    r"\btake\s+a\s+look\b",
    r"\bcan\s+you\s+check\b.*\b(now|right\s+now)\b",
    r"\blook\s+(now|right\s+now)\b",
)

_VISUAL_ACTION_TERMS = {"see", "look", "looking", "check", "describe"}
_VISUAL_TARGET_TERMS = {"front", "there"}
_VISUAL_PRESENT_TERMS = {"now", "current", "currently", "right"}
_VISUAL_NEGATIVE_PATTERNS = (
    r"\b(earlier|before|previously|yesterday|last\s+(?:time|night|week))\b",
    r"\bwhat\s+did\s+you\s+see\b",
)


def extract_topic_anchors(text: str, *, top_k: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z0-9']+", (text or "").lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [token for token, _ in ranked[:top_k]]


def _contains_rare_terms(text: str) -> bool:
    tokens = re.findall(r"\S+", text or "")
    if any(any(ord(ch) > 127 for ch in token) for token in tokens):
        return True
    long_tokens = sum(1 for token in tokens if len(token) >= 16)
    digit_heavy = sum(1 for token in tokens if sum(c.isdigit() for c in token) >= 4)
    return long_tokens > 0 or digit_heavy >= 2


def _likely_proper_noun(text: str) -> bool:
    return any(re.match(r"^[A-Z][a-z]+$", token) for token in re.findall(r"\b\w+\b", text or ""))


@dataclass(slots=True)
class UtteranceTrustSnapshot:
    run_id: str
    turn_id: str
    input_event_key: str
    transcript_text: str
    utterance_duration_ms: int | None
    word_count: int
    asr_confidence: float | None
    asr_avg_logprob: float | None
    asr_no_speech_prob: float | None
    short_utterance: bool
    very_short_text: bool
    contains_rare_terms: bool
    likely_proper_noun: bool
    topic_anchors: list[str]
    visual_question: bool


def build_utterance_trust_snapshot(
    *,
    run_id: str,
    turn_id: str,
    input_event_key: str,
    transcript_text: str,
    utterance_duration_ms: int | None,
    asr_meta: dict[str, Any] | None,
    short_utterance_ms: int,
    transcript_max_chars: int = 160,
) -> UtteranceTrustSnapshot:
    cleaned = " ".join((transcript_text or "").split())
    words = re.findall(r"\b\w+\b", cleaned)
    meta = asr_meta if isinstance(asr_meta, dict) else {}
    confidence = meta.get("confidence") if isinstance(meta.get("confidence"), (int, float)) else None
    avg_logprob = meta.get("avg_logprob") if isinstance(meta.get("avg_logprob"), (int, float)) else None
    no_speech_prob = meta.get("no_speech_prob") if isinstance(meta.get("no_speech_prob"), (int, float)) else None
    anchors = extract_topic_anchors(cleaned)
    token_set = set(re.findall(r"[a-zA-Z0-9']+", cleaned.lower()))
    visual_question = bool(token_set & _VISUAL_TERMS)
    duration = int(utterance_duration_ms) if isinstance(utterance_duration_ms, (int, float)) else None
    return UtteranceTrustSnapshot(
        run_id=run_id,
        turn_id=turn_id,
        input_event_key=input_event_key,
        transcript_text=cleaned[:transcript_max_chars],
        utterance_duration_ms=duration,
        word_count=len(words),
        asr_confidence=float(confidence) if confidence is not None else None,
        asr_avg_logprob=float(avg_logprob) if avg_logprob is not None else None,
        asr_no_speech_prob=float(no_speech_prob) if no_speech_prob is not None else None,
        short_utterance=duration is not None and duration < short_utterance_ms,
        very_short_text=len(words) <= 2,
        contains_rare_terms=_contains_rare_terms(cleaned),
        likely_proper_noun=_likely_proper_noun(cleaned),
        topic_anchors=anchors,
        visual_question=visual_question,
    )


def should_clarify(
    *,
    transcript_text: str,
    snapshot: UtteranceTrustSnapshot,
    min_confidence: float,
    known_domain: bool = False,
    tool_risk: str = "low",
    camera_available: bool = False,
    camera_recent: bool = True,
) -> tuple[bool, str]:
    token_set = set(re.findall(r"[a-zA-Z0-9']+", (transcript_text or "").lower()))
    greeting_only = bool(token_set) and token_set <= _GREETING_TERMS
    clear_motion_command = {"look", "center"} <= token_set or bool(token_set & _CLEAR_MOTION_TERMS and "look" in token_set)

    if snapshot.asr_confidence is not None and snapshot.asr_confidence < min_confidence:
        return True, "low_conf"
    if snapshot.short_utterance and snapshot.very_short_text:
        return True, "short_utterance"
    if snapshot.very_short_text and not known_domain and not greeting_only and not clear_motion_command:
        return True, "low_semantic_confidence"
    if snapshot.contains_rare_terms and not known_domain:
        return True, "rare_terms"
    if snapshot.visual_question and (not camera_available or not camera_recent):
        return True, "visual_unavailable"
    if tool_risk in {"high", "critical"} and snapshot.asr_confidence is not None and snapshot.asr_confidence < 0.8:
        return True, "high_risk_confirmation"
    return False, "none"


def is_current_visual_question(text: str) -> bool:
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return False
    for pattern in _VISUAL_NEGATIVE_PATTERNS:
        if re.search(pattern, normalized):
            return False
    for pattern in _CURRENT_VISUAL_PATTERNS:
        if re.search(pattern, normalized):
            return True
    token_set = set(re.findall(r"[a-zA-Z0-9']+", normalized))
    has_visual_action = bool(token_set & _VISUAL_ACTION_TERMS)
    has_current_target = bool(token_set & _VISUAL_TARGET_TERMS)
    has_present_hint = bool(token_set & _VISUAL_PRESENT_TERMS) or "in front of you" in normalized
    if has_visual_action and has_current_target and has_present_hint:
        return True
    return False


def topic_mismatch_detected(transcript_text: str, response_text: str, *, threshold: float = 0.2) -> bool:
    transcript_anchors = set(extract_topic_anchors(transcript_text))
    response_anchors = set(extract_topic_anchors(response_text))
    if not transcript_anchors or not response_anchors:
        return False
    union = transcript_anchors | response_anchors
    overlap = transcript_anchors & response_anchors
    jaccard = len(overlap) / len(union)
    has_favorite_drift = "favorite" in response_text.lower() and "favorite" not in transcript_text.lower()
    return jaccard <= threshold and has_favorite_drift
