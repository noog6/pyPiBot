"""ASR trust snapshot and verify-on-risk heuristics."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from services.research.intent import has_research_intent

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from", "how", "i", "in", "is",
    "it", "know", "me", "my", "of", "on", "or", "that", "the", "to", "what", "where", "who", "you",
    "your", "was", "were", "with", "there", "hey",
}

_VISUAL_TERMS = {"pants", "shirt", "hat", "wearing", "wear", "color", "see"}
_GREETING_TERMS = {"hi", "hello", "hey", "yo", "sup", "howdy"}
_PHATIC_ACKNOWLEDGEMENTS = {
    "thanks",
    "thank you",
    "ok",
    "okay",
    "got it",
    "gotcha",
    "cool",
    "nice",
    "sounds good",
    "all good",
    "perfect",
    "alright",
}
_CLEAR_MOTION_TERMS = {"look", "center", "left", "right", "up", "down", "back"}
_COMPOUND_MOTION_TERMS = {"move", "turn", "look", "return", "go"}
_DIRECTION_TERMS = {"center", "left", "right", "up", "down", "back", "front", "forward"}
_VISUAL_PHRASE_PATTERNS = (
    re.compile(r"\bwhat do you see\b"),
    re.compile(r"\bcan you see\b"),
    re.compile(r"\bwhat you see\b"),
    re.compile(r"\bwhat(?:'s| is) in (?:my|the|this|that)\b"),
    re.compile(r"\b(?:look|looking) at (?:this|that|it|me)\b"),
    re.compile(r"\bin front of you\b"),
)


def _assistant_name_tokens(assistant_name: str | None) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9']+", str(assistant_name or "").lower()))


def extract_topic_anchors(text: str, *, top_k: int = 5, assistant_name: str | None = None) -> list[str]:
    stopwords = _STOPWORDS | _assistant_name_tokens(assistant_name)
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z0-9']+", (text or "").lower()):
        if token in stopwords or len(token) < 2:
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


def _normalize_acknowledgement_text(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9']+", (text or "").lower()))


def is_phatic_acknowledgement_text(text: str) -> bool:
    normalized = _normalize_acknowledgement_text(text)
    return normalized in _PHATIC_ACKNOWLEDGEMENTS


def _is_visual_question(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    if has_research_intent(normalized):
        return False

    token_set = set(re.findall(r"[a-zA-Z0-9']+", normalized))
    if bool(token_set & (_VISUAL_TERMS - {"see"})):
        return True
    return any(pattern.search(normalized) is not None for pattern in _VISUAL_PHRASE_PATTERNS)


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
    assistant_name: str | None = None,
) -> UtteranceTrustSnapshot:
    cleaned = " ".join((transcript_text or "").split())
    words = re.findall(r"\b\w+\b", cleaned)
    meta = asr_meta if isinstance(asr_meta, dict) else {}
    confidence = meta.get("confidence") if isinstance(meta.get("confidence"), (int, float)) else None
    avg_logprob = meta.get("avg_logprob") if isinstance(meta.get("avg_logprob"), (int, float)) else None
    no_speech_prob = meta.get("no_speech_prob") if isinstance(meta.get("no_speech_prob"), (int, float)) else None
    anchors = extract_topic_anchors(cleaned, assistant_name=assistant_name)
    visual_question = _is_visual_question(cleaned)
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
    compound_motion_visual_request = (
        snapshot.visual_question
        and camera_available
        and bool(token_set & _COMPOUND_MOTION_TERMS)
        and bool(token_set & _DIRECTION_TERMS)
    )

    if is_phatic_acknowledgement_text(transcript_text):
        return False, "phatic_acknowledgement"
    if snapshot.asr_confidence is not None and snapshot.asr_confidence < min_confidence:
        return True, "low_conf"
    if snapshot.short_utterance and snapshot.very_short_text:
        return True, "short_utterance"
    if snapshot.very_short_text and not known_domain and not greeting_only and not clear_motion_command:
        return True, "low_semantic_confidence"
    if snapshot.contains_rare_terms and not known_domain:
        return True, "rare_terms"
    if snapshot.visual_question and (not camera_available or not camera_recent):
        if compound_motion_visual_request:
            return False, "compound_motion_visual_defer"
        return True, "visual_unavailable"
    if tool_risk in {"high", "critical"} and snapshot.asr_confidence is not None and snapshot.asr_confidence < 0.8:
        return True, "high_risk_confirmation"
    return False, "none"


def topic_mismatch_detected(
    transcript_text: str,
    response_text: str,
    *,
    threshold: float = 0.2,
    assistant_name: str | None = None,
) -> bool:
    transcript_anchors = set(extract_topic_anchors(transcript_text, assistant_name=assistant_name))
    response_anchors = set(extract_topic_anchors(response_text, assistant_name=assistant_name))
    if not transcript_anchors or not response_anchors:
        return False
    union = transcript_anchors | response_anchors
    overlap = transcript_anchors & response_anchors
    jaccard = len(overlap) / len(union)
    has_favorite_drift = "favorite" in response_text.lower() and "favorite" not in transcript_text.lower()
    return jaccard <= threshold and has_favorite_drift
