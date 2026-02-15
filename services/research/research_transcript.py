"""Helpers for writing auditable research transcript artifacts per run."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from typing import Any

from core.logging import logger

from services.research.models import ResearchPacket, ResearchRequest


_REDACTED = "***redacted***"


def _utc_stamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%SZ")


def _sanitize_context(context: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in context.items():
        normalized = str(key).lower()
        if any(token in normalized for token in ("api_key", "token", "secret", "password")):
            sanitized[key] = _REDACTED
        else:
            sanitized[key] = value
    return sanitized


def _packet_dict(packet: ResearchPacket | dict[str, Any] | None) -> dict[str, Any] | None:
    if packet is None:
        return None
    if is_dataclass(packet):
        return asdict(packet)
    return dict(packet)


def _extract_source_urls(packet_payload: dict[str, Any] | None) -> list[str]:
    if not packet_payload:
        return []
    sources = packet_payload.get("sources")
    if not isinstance(sources, list):
        return []
    urls: list[str] = []
    for source in sources:
        if isinstance(source, str):
            source_url = source.strip()
        elif isinstance(source, dict):
            source_url = (
                str(source.get("url") or source.get("href") or source.get("source") or "").strip()
            )
        else:
            source_url = ""
        if source_url:
            urls.append(source_url)
    return urls


def write_research_transcript(
    *,
    run_dir: Path,
    run_id: int,
    request: ResearchRequest,
    packet: ResearchPacket | dict[str, Any] | None,
    research_id: str | None = None,
    now: datetime | None = None,
) -> Path | None:
    """Persist structured research transcript artifacts and return the JSON path."""

    created_at = (now or datetime.now(timezone.utc)).isoformat()
    timestamp = _utc_stamp(now)
    transcript_id = research_id or f"research_{uuid.uuid4().hex}"
    short_id = transcript_id.split("_")[-1][:8]

    packet_payload = _packet_dict(packet)
    source_urls = _extract_source_urls(packet_payload)
    request_context = _sanitize_context(dict(request.context))
    request_payload: dict[str, Any] = {"query": request.prompt}
    request_payload.update(request_context)

    transcript_payload = {
        "research_id": transcript_id,
        "run_id": str(run_id),
        "created_at": created_at,
        "request": request_payload,
        "packet": packet_payload,
    }

    research_dir = run_dir / "research"
    base_name = f"research_{timestamp}_{short_id}"
    json_path = research_dir / f"{base_name}.json"
    md_path = research_dir / f"{base_name}.md"

    try:
        research_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(transcript_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = [
            "# Research Transcript",
            "",
            f"- research_id: `{transcript_id}`",
            f"- run_id: `{run_id}`",
            f"- created_at: `{created_at}`",
            f"- query: `{request.prompt}`",
            "",
            "## Sources",
        ]
        if source_urls:
            lines.extend(f"- {url}" for url in source_urls)
        else:
            lines.append("- (none)")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("research transcript saved: %s sources=%s", json_path, len(source_urls))
        return json_path
    except Exception as exc:  # noqa: BLE001 - writing transcripts must not crash runtime
        logger.warning("research transcript write failed: %s", exc)
        return None

