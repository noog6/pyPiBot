"""Tool definitions for realtime API calls."""

from __future__ import annotations

import time
import uuid
import asyncio
import re
from typing import Any, Awaitable, Callable

from config import ConfigController
from services.imu_monitor import ImuMonitor
from services.memory_manager import MemoryManager, MemoryScope
from services.profile_manager import ProfileManager
from services.research import ResearchRequest, build_openai_service_or_null
from services.research.grounding import (
    build_research_grounding_explanation,
    build_unverified_sources_only_response,
    requires_unverified_sources_only_response,
)
from services.research.research_transcript import write_research_transcript
from services.research.research_transcript import resolve_research_transcript_run_context
from services.research.service import ResearchService
from services import tool_runtime


ToolFn = Callable[..., Awaitable[Any]]

function_map: dict[str, ToolFn] = {}

tools: list[dict[str, Any]] = []
_research_service: ResearchService | None = None


def _get_research_service() -> ResearchService:
    global _research_service
    if _research_service is None:
        config = ConfigController.get_instance().get_config()
        _research_service = build_openai_service_or_null(config)
    return _research_service


async def read_battery_voltage() -> dict[str, Any]:
    """Return the current LiPo battery voltage via the ADS1015 sensor."""
    return tool_runtime.read_battery_voltage()


async def read_environment() -> dict[str, Any]:
    """Return the current onboard air pressure and temperature."""
    return tool_runtime.read_environment()


async def read_imu_data() -> dict[str, Any]:
    """Return the latest IMU orientation and sensor readings."""

    monitor = ImuMonitor.get_instance()
    sample = monitor.get_latest_sample()
    if sample is None:
        return {
            "status": "no_data",
            "message": "IMU has not produced a sample yet.",
        }

    events = monitor.get_recent_events(limit=5)
    event_items = [
        {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "severity": event.severity,
            "details": event.details,
        }
        for event in events
    ]
    orientation = {
        "roll_deg": sample.roll,
        "pitch_deg": sample.pitch,
        "yaw_deg": sample.yaw,
    }
    summary = (
        "IMU readings: "
        f"roll {sample.roll:.2f}°, pitch {sample.pitch:.2f}°, yaw {sample.yaw:.2f}°. "
        f"Accel {tuple(round(val, 3) for val in sample.accel)}. "
        f"Gyro {tuple(round(val, 3) for val in sample.gyro)}. "
        f"Mag {tuple(round(val, 3) for val in sample.mag)}."
    )
    return {
        "status": "ok",
        "summary": summary,
        "timestamp": sample.timestamp,
        "orientation": orientation,
        "accel": sample.accel,
        "gyro": sample.gyro,
        "mag": sample.mag,
        "recent_events": event_items,
        "units": {
            "orientation": "degrees",
            "accel": "raw",
            "gyro": "raw",
            "mag": "raw",
        },
    }


async def enqueue_idle_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue an idle gesture action on the motion controller."""
    return tool_runtime.enqueue_idle_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_nod_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a nod gesture action on the motion controller."""
    return tool_runtime.enqueue_nod_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_no_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a head shake gesture action on the motion controller."""
    return tool_runtime.enqueue_no_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_look_around_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a casual look around gesture action on the motion controller."""
    return tool_runtime.enqueue_look_around_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_look_up_gesture(delay_ms: int = 0, intensity: float = 1.0) -> dict[str, Any]:
    """Queue a look up gesture action on the motion controller."""
    return tool_runtime.enqueue_look_up_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_look_left_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a look left gesture action on the motion controller."""
    return tool_runtime.enqueue_look_left_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_look_right_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a look right gesture action on the motion controller."""
    return tool_runtime.enqueue_look_right_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_look_down_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a look down gesture action on the motion controller."""
    return tool_runtime.enqueue_look_down_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_look_center_gesture(delay_ms: int = 0) -> dict[str, Any]:
    """Queue a look center gesture action on the motion controller."""
    return tool_runtime.enqueue_look_center_gesture(delay_ms=delay_ms)


async def enqueue_curious_tilt_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a curious tilt gesture action on the motion controller."""
    return tool_runtime.enqueue_curious_tilt_gesture(delay_ms=delay_ms, intensity=intensity)


async def enqueue_attention_snap_gesture(
    delay_ms: int = 0, intensity: float = 1.0
) -> dict[str, Any]:
    """Queue a quick attention snap gesture action on the motion controller."""
    return tool_runtime.enqueue_attention_snap_gesture(delay_ms=delay_ms, intensity=intensity)


async def update_user_profile(
    name: str | None = None,
    preferences: dict[str, Any] | None = None,
    favorites: list[str] | None = None,
) -> dict[str, Any]:
    """Update the active user profile with provided fields."""

    manager = ProfileManager.get_instance()
    profile = manager.update_active_profile_fields(
        name=name,
        preferences=preferences,
        favorites=favorites,
        last_seen=int(time.time() * 1000),
    )
    return {
        "user_id": profile.user_id,
        "name": profile.name,
        "preferences": profile.preferences,
        "favorites": profile.favorites,
        "last_seen": profile.last_seen,
    }


async def get_output_volume() -> dict[str, Any]:
    """Return the current output audio volume."""
    return tool_runtime.get_output_volume()


async def set_output_volume(percent: int, emergency: bool = False) -> dict[str, Any]:
    """Set the output audio volume within safe bounds."""
    return tool_runtime.set_output_volume(percent=percent, emergency=emergency)


def _truncate_memory_content(content: str, *, max_chars: int = 160) -> str:
    normalized = " ".join(str(content or "").split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1]}…"


def _derive_memory_confidence(*, influence_score: float, threshold: float) -> str:
    if influence_score >= max(threshold + 2.0, 3.0):
        return "High"
    if influence_score >= max(threshold + 0.5, 1.5):
        return "Medium"
    return "Low"


def _tokenize_for_evidence(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9']+", str(value).lower()) if token}


def _derive_memory_confidence_from_evidence(
    *,
    lexical_exact_match: bool,
    lexical_score: float,
    semantic_score: float,
    influence_score: float,
    influence_threshold: float,
    rewrite_applied: str,
    exclusion_reason: str | None = None,
) -> str:
    """Map retrieval evidence to deterministic confidence tiers."""

    semantic_high_threshold = 0.8
    semantic_medium_threshold = 0.55
    lexical_high_threshold = 0.85

    excluded_by_threshold = (
        exclusion_reason == "below_influence_threshold"
        or (influence_threshold > 0.0 and influence_score < influence_threshold)
    )
    fallback_rewrite_used = rewrite_applied != "none"

    if not excluded_by_threshold:
        if lexical_exact_match or lexical_score >= lexical_high_threshold:
            return "High"
        if semantic_score >= semantic_high_threshold:
            return "High"

    if semantic_score >= semantic_medium_threshold:
        return "Medium"
    if fallback_rewrite_used:
        return "Low"

    # Backward-compatible fallback when only influence score is available.
    return _derive_memory_confidence(influence_score=influence_score, threshold=influence_threshold)


def build_recall_memory_cards(
    *,
    query: str | None,
    memories: list[dict[str, Any]],
    trace: dict[str, Any] | None,
    max_cards: int = 3,
) -> list[dict[str, str]]:
    """Create user-facing memory cards without exposing internal IDs."""

    ranking_summary = trace.get("ranking_summary") if isinstance(trace, dict) else []
    ranked_selected = [
        item
        for item in (ranking_summary if isinstance(ranking_summary, list) else [])
        if isinstance(item, dict) and item.get("selected")
    ]
    threshold = 0.0
    if isinstance(trace, dict):
        thresholds = trace.get("thresholds_used") if isinstance(trace.get("thresholds_used"), dict) else {}
        threshold = float(thresholds.get("influence_threshold", 0.0) or 0.0)
    rewrite_applied = str(trace.get("rewrite_applied", "none")) if isinstance(trace, dict) else "none"

    query_reason = "You asked to recall related details."
    normalized_query = " ".join(str(query or "").split())
    if normalized_query:
        query_reason = f"It matches your query about {normalized_query[:60]!r}."
    query_tokens = _tokenize_for_evidence(normalized_query)

    cards: list[dict[str, str]] = []
    for idx, memory in enumerate(memories[: max(1, max_cards)]):
        score = 0.0
        lexical_score = 0.0
        semantic_score = 0.0
        exclusion_reason = None
        if idx < len(ranked_selected):
            selected = ranked_selected[idx]
            score = float(selected.get("influence_score", 0.0) or 0.0)
            lexical_score = float(selected.get("score_lexical", 0.0) or 0.0)
            semantic_score = float(selected.get("score_semantic", 0.0) or 0.0)
            exclusion_reason = str(selected.get("exclusion_reason")) if selected.get("exclusion_reason") else None

        memory_tokens = _tokenize_for_evidence(str(memory.get("content", "")))
        lexical_exact_match = bool(query_tokens and memory_tokens.intersection(query_tokens))
        evidence_hint = f"semantic similarity={semantic_score:.2f}"
        if lexical_exact_match and query_tokens:
            matched_token = sorted(memory_tokens.intersection(query_tokens))[0]
            evidence_hint = f"lexical exact match on '{matched_token}'"
        elif lexical_score > 0.0:
            evidence_hint = f"lexical score={lexical_score:.2f}"

        cards.append(
            {
                "memory": _truncate_memory_content(str(memory.get("content", ""))),
                "why_relevant": f"{query_reason} Evidence: {evidence_hint}.",
                "confidence": _derive_memory_confidence_from_evidence(
                    lexical_exact_match=lexical_exact_match,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    influence_score=score,
                    influence_threshold=threshold,
                    rewrite_applied=rewrite_applied,
                    exclusion_reason=exclusion_reason,
                ),
            }
        )
    return cards


def render_memory_cards_for_assistant(cards: list[dict[str, str]], *, total_memories: int) -> str:
    if not cards:
        return ""
    lines: list[str] = []
    for card in cards:
        lines.append("Relevant memory:")
        lines.append(f'- "{card.get("memory", "")}"')
        lines.append("Why it's relevant:")
        lines.append(f'- "{card.get("why_relevant", "")}"')
        lines.append(f'Confidence: {card.get("confidence", "Low")}')
    if total_memories > len(cards):
        lines.append(f"+ {total_memories - len(cards)} more related memories")
    return "\n".join(lines)


async def remember_memory(
    content: str,
    tags: list[str] | None = None,
    importance: int = 3,
    scope: str = MemoryScope.USER_GLOBAL.value,
    source: str = "manual_tool",
    pinned: bool = False,
) -> dict[str, Any]:
    """Store a memory entry for later recall."""

    manager = MemoryManager.get_instance()
    entry = manager.remember_memory(
        content=content,
        tags=tags,
        importance=importance,
        scope=scope,
        source=source,
        pinned=bool(pinned),
    )
    return {
        "memory_id": entry.memory_id,
        "content": entry.content,
        "tags": entry.tags,
        "importance": entry.importance,
        "source": entry.source,
        "pinned": entry.pinned,
        "needs_review": entry.needs_review,
    }


async def recall_memories(
    query: str | None = None,
    limit: int = 5,
    scope: str = MemoryScope.USER_GLOBAL.value,
) -> dict[str, Any]:
    """Recall stored memories based on a query."""

    manager = MemoryManager.get_instance()
    memories, trace = manager.recall_memories_with_trace(query=query, limit=limit, scope=scope)
    memory_payload = [
        {
            "content": memory.content,
            "tags": list(memory.tags),
            "importance": memory.importance,
            "source": memory.source,
            "pinned": memory.pinned,
            "needs_review": memory.needs_review,
        }
        for memory in memories
    ]
    cards = build_recall_memory_cards(query=query, memories=memory_payload, trace=trace)
    payload: dict[str, Any] = {
        "memories": memory_payload,
        "memory_cards": cards,
        "memory_cards_text": render_memory_cards_for_assistant(cards, total_memories=len(memory_payload)),
    }
    if trace is not None:
        payload["trace"] = trace
    return payload


async def forget_memory(memory_id: int) -> dict[str, Any]:
    """Delete a stored memory by id."""

    manager = MemoryManager.get_instance()
    removed = manager.forget_memory(memory_id=memory_id)
    return {"removed": removed, "memory_id": memory_id}


async def perform_research(query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a web research request through the ResearchService packet flow."""

    request = ResearchRequest(prompt=query, context=dict(context or {}))
    service = _get_research_service()
    packet = await asyncio.to_thread(service.request_research, request)
    research_id = f"research_{uuid.uuid4().hex}"

    run_dir, run_id = resolve_research_transcript_run_context()
    transcript_path = write_research_transcript(
        run_dir=run_dir,
        run_id=run_id,
        request=request,
        packet=packet,
        research_id=research_id,
    )

    payload = packet.to_realtime_payload()
    answer_summary = payload["answer_summary"]
    extracted_facts = payload["extracted_facts"]
    if requires_unverified_sources_only_response(packet):
        answer_summary = build_unverified_sources_only_response(packet)
        extracted_facts = []

    return {
        "research_id": research_id,
        "schema": packet.schema,
        "status": packet.status,
        "answer_summary": answer_summary,
        "grounding_explanation": build_research_grounding_explanation(packet),
        "extracted_facts": extracted_facts,
        "sources": payload["sources"],
        "safety_notes": payload["safety_notes"],
        "metadata": dict(packet.metadata),
        "transcript_path": str(transcript_path) if transcript_path is not None else None,
    }


tools.append(
    {
        "type": "function",
        "name": "inspect_current_view",
        "description": (
            "Use this as the primary visual-semantic tool when the user asks what Theo can "
            "currently see (for example: what do you see, what am I holding, can you see it now, "
            "look at this/my hand). This takes an explicit fresh look at the current camera view "
            "and returns structured capture status metadata for the turn. This tool does not "
            "move or recenter the camera. If the user asks to center/recenter first, call the "
            "appropriate motion tool (for example gesture_look_center) before calling "
            "inspect_current_view."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": [],
        },
    }
)

tools.append(
    {
        "type": "function",
        "name": "read_battery_voltage",
        "description": (
            "Fetch the current voltage of the onboard 2S LiPo battery. "
            "Safe operating range is 7.0V to 8.4V. If the reading is within "
            "0.5V of the minimum voltage, complain about it; being near the max is fine."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["read_battery_voltage"] = read_battery_voltage

tools.append(
    {
        "type": "function",
        "name": "read_environment",
        "description": (
            "Fetch Theo's onboard environment readings. "
            "Includes air pressure/ambient temperature from the LPS22HB sensor (inside Theo), "
            "plus the SoC core CPU temperature via vcgencmd (often higher than ambient). "
            "This is onboard data, not external weather. "
            "Return values in hPa and Celsius."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["read_environment"] = read_environment

tools.append(
    {
        "type": "function",
        "name": "read_imu_data",
        "description": (
            "Fetch the latest IMU orientation (roll/pitch/yaw) and raw accel/gyro/mag readings. "
            "Return a human-readable summary string plus structured values."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["read_imu_data"] = read_imu_data

tools.append(
    {
        "type": "function",
        "name": "gesture_idle",
        "description": (
            "Queue a gentle idle gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_idle"] = enqueue_idle_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_nod",
        "description": (
            "Queue a nod gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_nod"] = enqueue_nod_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_no",
        "description": (
            "Queue a head shake (no) gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_no"] = enqueue_no_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_look_around",
        "description": (
            "Queue a casual look around gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_look_around"] = enqueue_look_around_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_look_up",
        "description": (
            "Tilt the camera all the way up while keeping the current pan. "
            "Use for requests like 'look up' or 'look all the way up'. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_look_up"] = enqueue_look_up_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_look_left",
        "description": (
            "Pan the camera all the way left while keeping the current tilt. "
            "Use for requests like 'look left' or 'look all the way left'. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_look_left"] = enqueue_look_left_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_look_right",
        "description": (
            "Pan the camera all the way right while keeping the current tilt. "
            "Use for requests like 'look right' or 'look all the way right'. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_look_right"] = enqueue_look_right_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_look_down",
        "description": (
            "Tilt the camera all the way down while keeping the current pan. "
            "Use for requests like 'look down' or 'look all the way down'. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_look_down"] = enqueue_look_down_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_look_center",
        "description": (
            "Return the camera to a neutral, centered pan/tilt position. "
            "Use for requests like 'look back to center' or 'return to neutral'. "
            "This is motion-only setup and does not inspect/describe objects by itself; for "
            "semantic visual questions use inspect_current_view as a separate follow-up step. "
            "Provide an optional delay in milliseconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": [],
        },
    }
)

function_map["gesture_look_center"] = enqueue_look_center_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_curious_tilt",
        "description": (
            "Queue a curious tilt gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_curious_tilt"] = enqueue_curious_tilt_gesture

tools.append(
    {
        "type": "function",
        "name": "gesture_attention_snap",
        "description": (
            "Queue a quick attention snap gesture on the pan/tilt rig. "
            "Provide an optional delay in milliseconds and intensity (1.0 is normal)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_ms": {"type": "integer", "minimum": 0, "default": 0},
                "intensity": {"type": "number", "minimum": 0.1, "maximum": 2.0, "default": 1.0},
            },
            "required": [],
        },
    }
)

function_map["gesture_attention_snap"] = enqueue_attention_snap_gesture

tools.append(
    {
        "type": "function",
        "name": "update_user_profile",
        "description": (
            "Update the active user profile with personal details like name, "
            "preferences, or favorites."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "preferences": {"type": "object"},
                "favorites": {"type": "array", "items": {"type": "string"}},
            },
            "required": [],
        },
    }
)

function_map["update_user_profile"] = update_user_profile

tools.append(
    {
        "type": "function",
        "name": "get_output_volume",
        "description": "Read the current output audio volume.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["get_output_volume"] = get_output_volume

tools.append(
    {
        "type": "function",
        "name": "set_output_volume",
        "description": (
            "Set the output audio volume. Volume percent must be between 1 and 100. "
            "Changes are rate-limited to once per second unless emergency is true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "percent": {"type": "integer", "minimum": 1, "maximum": 100},
                "emergency": {"type": "boolean", "default": False},
            },
            "required": ["percent"],
        },
    }
)

function_map["set_output_volume"] = set_output_volume

tools.append(
    {
        "type": "function",
        "name": "remember_memory",
        "description": (
            "Store a durable memory about the user, preferences, or facts worth reusing. "
            "Only store when the user provides stable, repeatable facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                "scope": {
                    "type": "string",
                    "enum": [MemoryScope.USER_GLOBAL.value, MemoryScope.SESSION_LOCAL.value],
                    "default": MemoryScope.USER_GLOBAL.value,
                },
                "pinned": {"type": "boolean", "default": False},
            },
            "required": ["content"],
        },
    }
)

function_map["remember_memory"] = remember_memory

tools.append(
    {
        "type": "function",
        "name": "recall_memories",
        "description": (
            "Fetch relevant stored memories when the user asks about prior facts, "
            "preferences, or context that might have been saved."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                "scope": {
                    "type": "string",
                    "enum": [MemoryScope.USER_GLOBAL.value, MemoryScope.SESSION_LOCAL.value],
                    "default": MemoryScope.USER_GLOBAL.value,
                },
            },
            "required": [],
        },
    }
)

function_map["recall_memories"] = recall_memories

tools.append(
    {
        "type": "function",
        "name": "forget_memory",
        "description": "Remove a stored memory when the user asks to delete or forget it.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer", "minimum": 1},
            },
            "required": ["memory_id"],
        },
    }
)

function_map["forget_memory"] = forget_memory

tools.append(
    {
        "type": "function",
        "name": "perform_research",
        "description": (
            "Perform a web lookup using Theo's research service and return a structured "
            "research packet summary with facts and sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "context": {"type": "object"},
            },
            "required": ["query"],
        },
    }
)

function_map["perform_research"] = perform_research
