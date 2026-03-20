"""Pure semantic-owner arbitration for response terminal handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SemanticOwnerAction = Literal["retain_execution", "reassign_parent"]
SemanticOwnerCandidateId = Literal["semantic_owner_execution", "semantic_owner_parent"]


@dataclass(frozen=True)
class SemanticOwnerDecision:
    semantic_owner_canonical_key: str
    selected_candidate_id: SemanticOwnerCandidateId
    reason_code: str
    action: SemanticOwnerAction
    execution_canonical_key: str
    parent_canonical_key: str | None = None
    parent_turn_id: str | None = None
    parent_input_event_key: str | None = None


_AUTHORITY_REASON_RETAIN_EMPTY = "execution_canonical_unavailable"


def decide_semantic_owner(
    *,
    execution_canonical_key: str,
    selected: bool,
    selection_reason: str,
    origin: str,
    parent_turn_id: str | None,
    parent_input_event_key: str | None,
    parent_canonical_key: str | None,
    parent_canonical_exists: bool,
) -> SemanticOwnerDecision:
    execution_key = str(execution_canonical_key or "").strip()
    normalized_reason = str(selection_reason or "").strip().lower()
    normalized_origin = str(origin or "").strip().lower()
    normalized_parent_turn_id = str(parent_turn_id or "").strip() or None
    normalized_parent_input_event_key = str(parent_input_event_key or "").strip() or None
    normalized_parent_key = str(parent_canonical_key or "").strip() or None

    def retain(reason_code: str) -> SemanticOwnerDecision:
        return SemanticOwnerDecision(
            semantic_owner_canonical_key=execution_key,
            selected_candidate_id="semantic_owner_execution",
            reason_code=reason_code,
            action="retain_execution",
            execution_canonical_key=execution_key,
            parent_canonical_key=normalized_parent_key,
            parent_turn_id=normalized_parent_turn_id,
            parent_input_event_key=normalized_parent_input_event_key,
        )

    if not execution_key:
        return SemanticOwnerDecision(
            semantic_owner_canonical_key="",
            selected_candidate_id="semantic_owner_execution",
            reason_code=_AUTHORITY_REASON_RETAIN_EMPTY,
            action="retain_execution",
            execution_canonical_key="",
            parent_canonical_key=normalized_parent_key,
            parent_turn_id=normalized_parent_turn_id,
            parent_input_event_key=normalized_parent_input_event_key,
        )
    if not selected:
        return retain("terminal_not_selected")
    if normalized_reason != "normal":
        return retain("terminal_reason_ineligible")
    if normalized_origin != "tool_output":
        return retain("origin_ineligible")
    if not normalized_parent_turn_id or not normalized_parent_input_event_key:
        return retain("parent_lineage_unavailable")
    if normalized_parent_input_event_key.startswith("tool:"):
        return retain("parent_input_tool_prefixed")
    if not normalized_parent_key:
        return retain("parent_canonical_unavailable")
    if normalized_parent_key == execution_key:
        return retain("parent_matches_execution")
    if not parent_canonical_exists:
        return retain("parent_canonical_missing")
    return SemanticOwnerDecision(
        semantic_owner_canonical_key=normalized_parent_key,
        selected_candidate_id="semantic_owner_parent",
        reason_code="parent_promoted_from_tool_output",
        action="reassign_parent",
        execution_canonical_key=execution_key,
        parent_canonical_key=normalized_parent_key,
        parent_turn_id=normalized_parent_turn_id,
        parent_input_event_key=normalized_parent_input_event_key,
    )
