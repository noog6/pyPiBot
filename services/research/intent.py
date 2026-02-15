"""Intent helpers for research-triggering user messages."""

from __future__ import annotations

RESEARCH_INTENT_PATTERNS = (
    "look up",
    "search online",
    "find datasheet",
    "what does the datasheet say",
)


def has_research_intent(text: str) -> bool:
    """Return True when text appears to request web-style lookup."""

    normalized = text.strip().lower()
    if not normalized:
        return False
    return any(pattern in normalized for pattern in RESEARCH_INTENT_PATTERNS)
