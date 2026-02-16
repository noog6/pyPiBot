"""Research service interface and null implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.logging import logger as LOGGER

from services.research.models import ResearchPacket, ResearchRequest


class ResearchService(ABC):
    """Interface for research subsystem providers."""

    @abstractmethod
    def request_research(self, request: ResearchRequest) -> ResearchPacket:
        """Build a structured research packet from the provided request."""


class NullResearchService(ResearchService):
    """Safe default implementation that does not perform any network activity."""

    def request_research(self, request: ResearchRequest) -> ResearchPacket:
        packet = ResearchPacket(
            status="disabled",
            answer_summary="Research subsystem disabled",
            extracted_facts=[],
            sources=[],
            safety_notes=["research_disabled"],
            metadata={
                "reason": "research_disabled",
                "prompt_length": len(request.prompt),
                "context_keys": sorted(request.context.keys()),
            },
        )
        LOGGER.info(
            "[Research] Returning null research packet",
            extra={
                "event": "research_packet_built",
                "research_status": packet.status,
                "research_schema": packet.schema,
            },
        )
        return packet

    def build_packet(self, request: ResearchRequest) -> ResearchPacket:
        """Backward-compatible alias for older call sites."""

        return self.request_research(request)
