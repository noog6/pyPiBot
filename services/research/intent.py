"""Intent helpers for research-triggering user messages."""

from __future__ import annotations

import re

RESEARCH_INTENT_PATTERNS = (
    "look up",
    "search the web",
    "search online",
    "search for",
    "find spec",
    "find specs",
    "find pinout",
    "find data sheet",
    "find datasheet",
    "check the datasheet",
    "read the datasheet",
    "what does the datasheet say",
)

RESEARCH_INTENT_REGEXES = (
    re.compile(r"\b(can you|please|could you)?\s*(search|look up|look for|find)\b.*\b(online|web|internet)\b"),
    re.compile(r"\b(datasheet|data\s*sheet|specs?|pinout|manual)\b"),
)


def has_research_intent(text: str) -> bool:
    """Return True when text appears to request web-style lookup."""

    normalized = text.strip().lower()
    if not normalized:
        return False
    if any(pattern in normalized for pattern in RESEARCH_INTENT_PATTERNS):
        return True
    return any(regex.search(normalized) is not None for regex in RESEARCH_INTENT_REGEXES)
