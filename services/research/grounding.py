"""Helpers for grounding research narration to packet truth."""

from __future__ import annotations

from services.research.models import ResearchPacket

_SKIP_REASON_MESSAGES: dict[str, str] = {
    "firecrawl_disabled": "content fetching is disabled in config",
    "firecrawl_key_missing": "content fetching is enabled but key is missing",
    "allowlist_blocked": "content fetching blocked by allowlist policy",
    "domain_not_allowed": "content fetching blocked by allowlist policy",
    "pdf_unsupported": "only PDF sources were found and PDF parsing is disabled",
    "no_sources": "no sources returned to fetch",
}


def get_content_fetch_state(packet: ResearchPacket) -> tuple[str, str | None, str | None]:
    """Return (status, skip_reason, failure_name) from packet metadata."""

    metadata = packet.metadata or {}
    status = str(metadata.get("content_fetch_status") or "skipped").strip().lower()
    if status not in {"ok", "skipped", "failed"}:
        status = "skipped"
    skip_reason_raw = metadata.get("content_fetch_skip_reason")
    skip_reason = str(skip_reason_raw).strip().lower() if skip_reason_raw else None
    failure_raw = metadata.get("content_fetch_error")
    failure_name = str(failure_raw).strip() if failure_raw else None
    return status, skip_reason, failure_name


def build_research_grounding_explanation(packet: ResearchPacket) -> str:
    """Build a short auditable explanation for research narration."""

    status, skip_reason, failure_name = get_content_fetch_state(packet)
    if status == "ok":
        if packet.extracted_facts:
            return "From the fetched source content, I extracted these facts."
        return "From the fetched source content, here is a concise summary."
    if status == "failed":
        error_name = failure_name or "UnknownError"
        return f"I attempted to fetch content but it failed ({error_name})."

    human_reason = _SKIP_REASON_MESSAGES.get(skip_reason or "")
    if human_reason is None:
        human_reason = "content fetching was skipped for this run"
    return f"I found sources but did not fetch/parse their contents in this run: {human_reason}."
