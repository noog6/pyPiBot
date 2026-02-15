"""Tests for research tool bridge in ai.tools."""

from __future__ import annotations

import asyncio
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


class _FakeStorage:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def get_storage_info(self):
        return type("StorageInfo", (), {"run_dir": self._run_dir, "run_id": 42})()


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
        metadata={"provider": "fake"},
    )
    service = _FakeResearchService(packet)
    monkeypatch.setattr(ai_tools, "_research_service", service)
    monkeypatch.setattr(ai_tools.StorageController, "get_instance", lambda: _FakeStorage(tmp_path))

    result = asyncio.run(ai_tools.perform_research("find a datasheet", {"source": "tool"}))

    assert service.queries == ["find a datasheet"]
    assert result["status"] == "ok"
    assert result["answer_summary"] == "summary"
    assert result["extracted_facts"] == ["fact"]
    assert result["sources"] == [{"title": "src", "url": "https://example.com"}]
    assert result["metadata"] == {"provider": "fake"}
    assert result["transcript_path"] is not None
    assert Path(result["transcript_path"]).exists()
