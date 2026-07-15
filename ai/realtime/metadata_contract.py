"""Provider metadata normalization for realtime response.create payloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


PROVIDER_METADATA_MAX_PROPERTIES = 16
PROVIDER_METADATA_MAX_KEY_LENGTH = 64
PROVIDER_METADATA_MAX_VALUE_LENGTH = 512

MANDATORY_PROVIDER_CORRELATION = (
    "turn_id",
    "input_event_key",
    "parent_input_event_key",
    "parent_turn_id",
    "tool_call_id",
)
REQUIRED_DELIVERABLE_ENVELOPE = (
    "turn_id",
    "input_event_key",
    "parent_input_event_key",
    "tool_followup",
    "tool_followup_release",
    "consumes_canonical_slot",
    "explicit_multipart",
    "local_runtime_followthrough",
    "followthrough_step_output_policy",
    "followthrough_post_completion_reason",
    "followthrough_required_tool_name",
    "followthrough_required_tool_already_executed",
    "followthrough_context_id",
    "metadata_schema_version",
    "blocked_by_response_id",
    "tool_call_id",
)
GENERIC_ENVELOPE = (
    "turn_id",
    "input_event_key",
    "parent_input_event_key",
    "parent_turn_id",
    "tool_call_id",
    "tool_followup",
    "tool_followup_release",
    "consumes_canonical_slot",
    "explicit_multipart",
    "local_runtime_followthrough",
    "followthrough_step_output_policy",
    "followthrough_post_completion_reason",
    "followthrough_required_tool_name",
    "followthrough_required_tool_already_executed",
    "followthrough_context_id",
    "metadata_schema_version",
)
OPTIONAL_OPTIMIZATION = (
    "tool_followup_suppress_if_parent_covered",
    "tool_followup_required_tool_name",
    "tool_followup_step_output_policy",
    "tool_followup_post_completion_reason",
    "followthrough_required_tool_execution",
    "followthrough_dispatch_source",
    "tool_result_has_distinct_info",
    "tool_followup_status_only",
    "tool_followup_silent_audio",
    "tool_followup_silent_user_facing_output",
)
DIAGNOSTIC_ONLY = (
    "tool_name",
    "tool_followup_tool_choice",
    "tool_followup_tool_choice_reason",
    "followthrough_runtime_contract_version",
    "followthrough_runtime_step_available",
    "followthrough_runtime_step_id",
    "followthrough_runtime_tool_name",
    "followthrough_runtime_tool_args",
    "gesture_motion_status",
    "tool_followup_no_create",
    "tool_followup_no_create_reason",
    "tool_followup_create_suppressed",
    "tool_followup_create_suppression_reason",
    "initiative_reason_codes",
    "initiative_posture",
    "initiative_confidence_band",
)
LOCAL_ONLY = ("followthrough_catchup_payload",)
_ALIAS_TO_CANONICAL = {
    "tool_followup_step_output_policy": "followthrough_step_output_policy",
    "tool_followup_post_completion_reason": "followthrough_post_completion_reason",
    "tool_followup_required_tool_name": "followthrough_required_tool_name",
}

_PRIORITY_GROUPS = (
    REQUIRED_DELIVERABLE_ENVELOPE,
    GENERIC_ENVELOPE,
    OPTIONAL_OPTIMIZATION,
    DIAGNOSTIC_ONLY,
)
_PRIORITY_RANK: dict[str, int] = {
    key: rank for rank, group in enumerate(_PRIORITY_GROUPS) for key in group
}


@dataclass(frozen=True)
class MetadataNormalizationResult:
    metadata: dict[str, str]
    dropped: list[str] = field(default_factory=list)
    externalized: list[str] = field(default_factory=list)
    oversized: list[dict[str, Any]] = field(default_factory=list)


def _stringify_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _is_required_deliverable(metadata: Mapping[str, Any]) -> bool:
    for key in (
        "followthrough_step_output_policy",
        "tool_followup_step_output_policy",
    ):
        if str(metadata.get(key) or "").strip().lower() == "required_deliverable":
            return True
    for key in (
        "followthrough_post_completion_reason",
        "tool_followup_post_completion_reason",
    ):
        if str(metadata.get(key) or "").strip().lower() == "required_deliverable_owed":
            return True
    return False


def _collapse_aliases(metadata: dict[str, str], dropped: list[str]) -> None:
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if alias in metadata and canonical in metadata:
            metadata.pop(alias, None)
            dropped.append(alias)


def normalize_provider_metadata(
    metadata: Mapping[str, Any],
    *,
    store_local_context: Callable[[dict[str, Any]], str] | None = None,
) -> MetadataNormalizationResult:
    """Return provider-safe metadata: <=16 string pairs, keys<=64, values<=512.

    Verbose structured orchestration state is externalized behind a compact
    ``followthrough_context_id`` when a local context store is supplied. The
    minimal required-deliverable envelope is intentionally <=16 fields so a
    key labelled required cannot be dropped just because every candidate is set.
    """
    normalized: dict[str, str] = {}
    dropped: list[str] = []
    externalized: list[str] = []
    oversized: list[dict[str, Any]] = []
    local_payloads: dict[str, str] = {}

    for raw_key, raw_value in metadata.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if len(key) > PROVIDER_METADATA_MAX_KEY_LENGTH:
            dropped.append(key)
            oversized.append(
                {
                    "field": key,
                    "kind": "key",
                    "observed": len(key),
                    "limit": PROVIDER_METADATA_MAX_KEY_LENGTH,
                }
            )
            continue
        value = _stringify_metadata_value(raw_value)
        if key in LOCAL_ONLY or len(value) > PROVIDER_METADATA_MAX_VALUE_LENGTH:
            oversized.append(
                {
                    "field": key,
                    "kind": "value",
                    "observed": len(value),
                    "limit": PROVIDER_METADATA_MAX_VALUE_LENGTH,
                }
            )
            if key == "followthrough_catchup_payload" and value and store_local_context is not None:
                local_payloads[key] = value
                externalized.append(key)
                continue
            dropped.append(key)
            continue
        normalized[key] = value

    if local_payloads and store_local_context is not None:
        context_id = store_local_context({"schema_version": "followthrough_context.v1", **local_payloads})
        if context_id:
            normalized["metadata_schema_version"] = "provider_metadata.v1"
            normalized["followthrough_context_id"] = context_id[:PROVIDER_METADATA_MAX_VALUE_LENGTH]

    _collapse_aliases(normalized, dropped)
    envelope = REQUIRED_DELIVERABLE_ENVELOPE if _is_required_deliverable(normalized) else GENERIC_ENVELOPE
    original_order = [str(key or "").strip() for key in metadata.keys()]

    def sort_key(item: tuple[str, str]) -> tuple[int, int]:
        key, _value = item
        if key in envelope:
            return (0, envelope.index(key))
        if key in OPTIONAL_OPTIMIZATION:
            return (1, OPTIONAL_OPTIMIZATION.index(key))
        if key in DIAGNOSTIC_ONLY:
            return (2, DIAGNOSTIC_ONLY.index(key))
        return (3, original_order.index(key) if key in original_order else 10_000)

    if len(normalized) > PROVIDER_METADATA_MAX_PROPERTIES:
        kept_items = sorted(normalized.items(), key=sort_key)[:PROVIDER_METADATA_MAX_PROPERTIES]
        kept = {key for key, _ in kept_items}
        for key in tuple(normalized.keys()):
            if key not in kept:
                dropped.append(key)
        normalized = dict(kept_items)

    for key, value in list(normalized.items()):
        if len(value) <= PROVIDER_METADATA_MAX_VALUE_LENGTH:
            continue
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
        normalized[key] = f"oversized:{digest}"
        oversized.append(
            {
                "field": key,
                "kind": "value",
                "observed": len(value),
                "limit": PROVIDER_METADATA_MAX_VALUE_LENGTH,
            }
        )

    return MetadataNormalizationResult(metadata=normalized, dropped=dropped, externalized=externalized, oversized=oversized)
