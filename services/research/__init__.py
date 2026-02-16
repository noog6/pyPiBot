"""Research subsystem service scaffolding."""

from services.research.intent import RESEARCH_INTENT_PATTERNS, has_research_intent
from services.research.firecrawl_client import FirecrawlClient
from services.research.models import RESEARCH_PACKET_SCHEMA, ResearchPacket, ResearchRequest
from services.research.openai_service import OpenAIResearchService, build_openai_service_or_null
from services.research.service import NullResearchService, ResearchService
from services.research.stores import ResearchBudgetTracker, ResearchCacheStore

__all__ = [
    "RESEARCH_INTENT_PATTERNS",
    "has_research_intent",
    "RESEARCH_PACKET_SCHEMA",
    "ResearchPacket",
    "ResearchRequest",
    "ResearchService",
    "NullResearchService",
    "OpenAIResearchService",
    "FirecrawlClient",
    "ResearchBudgetTracker",
    "ResearchCacheStore",
    "build_openai_service_or_null",
]
