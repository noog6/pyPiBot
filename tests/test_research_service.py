"""Tests for the research subsystem scaffold."""

from __future__ import annotations

from services.research import NullResearchService, ResearchRequest


def test_null_research_service_returns_disabled_packet() -> None:
    service = NullResearchService()

    packet = service.request_research(
        ResearchRequest(
            prompt="Tell me about battery chemistry.",
            context={"topic": "battery", "depth": "brief"},
        )
    )

    assert packet.schema == "research_packet_v1"
    assert packet.status == "disabled"
    assert packet.answer_summary == "Research subsystem disabled"
    assert packet.extracted_facts == []
    assert packet.sources == []
    assert "research_disabled" in packet.safety_notes
    assert packet.metadata["reason"] == "research_disabled"
    assert packet.metadata["context_keys"] == ["depth", "topic"]
