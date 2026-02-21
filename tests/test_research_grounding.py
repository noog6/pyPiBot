"""Tests for truth-from-packet research grounding helpers."""

from __future__ import annotations

import pytest

from services.research.grounding import build_research_grounding_explanation
from services.research.models import ResearchPacket


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"content_fetch_status": "skipped", "content_fetch_skip_reason": "firecrawl_disabled"}, "content fetching is disabled in config"),
        ({"content_fetch_status": "skipped", "content_fetch_skip_reason": "firecrawl_missing_key"}, "PDF ingestion requires FIRECRAWL_API_KEY but it is missing"),
        ({"content_fetch_status": "skipped", "content_fetch_skip_reason": "allowlist_blocked"}, "content fetching blocked by allowlist policy"),
        ({"content_fetch_status": "skipped", "content_fetch_skip_reason": "no_sources"}, "no sources returned to fetch"),
    ],
)
def test_grounding_explanation_for_skipped_reasons(metadata: dict[str, str], expected: str) -> None:
    packet = ResearchPacket(status="ok", answer_summary="summary", metadata=metadata)
    explanation = build_research_grounding_explanation(packet)

    assert "I found sources but did not fetch/parse their contents in this run" in explanation
    assert expected in explanation
    assert "don't have Firecrawl" not in explanation


def test_grounding_explanation_for_failed_fetch() -> None:
    packet = ResearchPacket(
        status="ok",
        answer_summary="summary",
        metadata={"content_fetch_status": "failed", "content_fetch_error": "TimeoutError"},
    )

    explanation = build_research_grounding_explanation(packet)

    assert explanation == "I attempted to fetch content but it failed (TimeoutError)."
    assert "don't have Firecrawl" not in explanation


def test_grounding_explanation_for_ok_fetch_with_markdown_facts() -> None:
    packet = ResearchPacket(
        status="ok",
        answer_summary="summary",
        extracted_facts=["Fact A"],
        metadata={"content_fetch_status": "ok", "content_fetch_markdown": "# heading"},
    )

    explanation = build_research_grounding_explanation(packet)

    assert "From the fetched source content" in explanation
