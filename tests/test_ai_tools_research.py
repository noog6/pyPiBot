"""Tests for research tool bridge in ai.tools."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import ai.tools as ai_tools
from services.research.models import ResearchPacket


class _FakeResearchService:
    def __init__(self, packet: ResearchPacket) -> None:
        self.packet = packet
        self.queries: list[str] = []

    def request_research(self, request):
        self.queries.append(request.prompt)
        return self.packet


def test_perform_research_tool_registered() -> None:
    assert "perform_research" in ai_tools.function_map
    assert any(tool.get("name") == "perform_research" for tool in ai_tools.tools)


def test_perform_research_routes_through_packet_flow(monkeypatch, tmp_path: Path) -> None:
    packet = ResearchPacket(
        status="ok",
        answer_summary="summary",
        extracted_facts=["fact"],
        sources=[{"title": "src", "url": "https://example.com"}],
        safety_notes=["note"],
        metadata={"provider": "fake", "content_fetch_status": "ok"},
    )
    service = _FakeResearchService(packet)
    monkeypatch.setattr(ai_tools, "_research_service", service)
    run_dir = tmp_path / "log" / "42"
    monkeypatch.setattr(
        ai_tools,
        "resolve_research_transcript_run_context",
        lambda: (run_dir, 42),
    )

    result = asyncio.run(ai_tools.perform_research("find a datasheet", {"source": "tool"}))

    assert service.queries == ["find a datasheet"]
    assert result["status"] == "ok"
    assert result["answer_summary"] == "summary"
    assert result["extracted_facts"] == ["fact"]
    assert result["sources"] == [{"title": "src", "url": "https://example.com"}]
    assert result["metadata"] == {"provider": "fake", "content_fetch_status": "ok"}
    assert "From the fetched source content" in result["grounding_explanation"]
    assert result["transcript_path"] is not None
    transcript_path = Path(result["transcript_path"])
    assert transcript_path.exists()
    assert transcript_path.parent == run_dir / "research"

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "42"
    assert payload["request"] == {"query": "find a datasheet", "source": "tool"}
    assert payload["packet"]["answer_summary"] == result["answer_summary"]
    assert payload["packet"]["extracted_facts"] == result["extracted_facts"]
    assert payload["packet"]["sources"] == result["sources"]
    assert payload["packet"]["metadata"] == result["metadata"]


def test_perform_research_disabled_returns_standardized_unavailable_contract(monkeypatch, tmp_path: Path) -> None:
    packet = ResearchPacket(
        status="disabled",
        answer_summary="Research subsystem disabled",
        extracted_facts=[],
        sources=[],
        safety_notes=["research_disabled"],
        metadata={"provider": "null_research", "reason": "research_disabled"},
    )
    service = _FakeResearchService(packet)
    monkeypatch.setattr(ai_tools, "_research_service", service)
    run_dir = tmp_path / "log" / "disabled"
    monkeypatch.setattr(ai_tools, "resolve_research_transcript_run_context", lambda: (run_dir, 99))

    result = asyncio.run(ai_tools.perform_research("find uptime status", {"source": "tool"}))

    assert result["status"] == "disabled"
    assert result["reason_code"] == "research_disabled"
    assert result["message"] == "Research subsystem disabled"
    assert result["retryable"] is False
    assert result["provider"] == "null_research"
    assert "Enable research in config" in result["user_action"]
    assert result["tool_result_has_distinct_info"] is True
    assert result["details"]["metadata"]["reason"] == "research_disabled"
    assert result["details"]["safety_notes"] == ["research_disabled"]
