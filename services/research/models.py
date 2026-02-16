"""Data models for research service packets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RESEARCH_PACKET_SCHEMA = "research_packet_v1"


@dataclass(frozen=True)
class ResearchRequest:
    """Research input request scaffold."""

    prompt: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchPacket:
    """Structured packet returned by research providers."""

    schema: str = RESEARCH_PACKET_SCHEMA
    status: str = "disabled"
    answer_summary: str = "Research subsystem disabled"
    extracted_facts: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_realtime_payload(self) -> dict[str, Any]:
        """Return the only fields that may be passed to realtime flow."""

        return {
            "answer_summary": self.answer_summary,
            "extracted_facts": list(self.extracted_facts),
            "sources": [dict(source) for source in self.sources],
            "safety_notes": list(self.safety_notes),
        }
