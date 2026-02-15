"""Tests for research transcript artifact writer."""

from __future__ import annotations

import json
from pathlib import Path

from services.research.models import ResearchPacket, ResearchRequest
from services.research.research_transcript import write_research_transcript


def test_write_research_transcript_creates_json_and_md(tmp_path: Path) -> None:
    run_dir = tmp_path / "log" / "12"
    request = ResearchRequest(prompt="find imu datasheet", context={"mode": "quick", "round": 1})
    packet = ResearchPacket(
        status="ok",
        answer_summary="summary",
        extracted_facts=["fact"],
        sources=[{"url": "https://example.com/a"}, {"href": "https://example.com/b"}],
        safety_notes=["safe"],
        metadata={"foo": "bar"},
    )

    json_path = write_research_transcript(
        run_dir=run_dir,
        run_id=12,
        request=request,
        packet=packet,
        research_id="research_fixedid1234",
    )

    assert json_path is not None
    assert json_path.exists()
    md_path = json_path.with_suffix(".md")
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["research_id"] == "research_fixedid1234"
    assert payload["run_id"] == "12"
    assert payload["request"]["query"] == "find imu datasheet"
    assert payload["request"]["mode"] == "quick"
    assert payload["packet"]["schema"] == "research_packet_v1"
    assert payload["packet"]["status"] == "ok"
    assert payload["packet"]["sources"][0]["url"] == "https://example.com/a"

    md_body = md_path.read_text(encoding="utf-8")
    assert "https://example.com/a" in md_body
    assert "https://example.com/b" in md_body
