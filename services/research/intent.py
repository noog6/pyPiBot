"""Intent helpers for research-triggering user messages."""

from __future__ import annotations

import re

RESEARCH_INTENT_PATTERNS = (
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

_SHORT_DIRECTIONAL_LOOKUP_TAILS = {
    "",
    "please",
    "now",
    "up",
    "a bit",
    "abit",
    "a little",
    "slightly",
    "more",
    "higher",
}

_POSITIONAL_MOTION_LOOKUP_TAILS = (
    re.compile(r"^(?:at|to)\s+(?:me|us|you|him|her|them)\b"),
    re.compile(r"^(?:at|to)\s+(?:the\s+)?(?:ceiling|sky|roof|top)\b"),
    re.compile(r"^(?:at|to)\s+(?:the\s+)?(?:left|right|center|centre|middle|ceiling|sky|roof|top)\b"),
)


def _has_searchable_look_up_tail(normalized: str) -> bool:
    match = re.match(r"^(?:can you|could you|would you|please)\s+look up\b(?P<tail>.*)$", normalized)
    if match is None:
        match = re.match(r"^look up\b(?P<tail>.*)$", normalized)
    if match is None:
        return False

    tail = re.sub(r"^[\s,!.?-]+", "", str(match.group("tail") or "").strip())
    if not tail:
        return False
    if tail in _SHORT_DIRECTIONAL_LOOKUP_TAILS:
        return False
    if re.match(r"^(?:and|then)\b", tail):
        return False
    if any(pattern.match(tail) for pattern in _POSITIONAL_MOTION_LOOKUP_TAILS):
        return False
    return True


def has_research_intent(text: str) -> bool:
    """Return True when text appears to request web-style lookup."""

    normalized = text.strip().lower()
    if not normalized:
        return False
    if _has_searchable_look_up_tail(normalized):
        return True
    if any(pattern in normalized for pattern in RESEARCH_INTENT_PATTERNS):
        return True
    return any(regex.search(normalized) is not None for regex in RESEARCH_INTENT_REGEXES)
