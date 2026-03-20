from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.semantic_owner_arbitration import decide_semantic_owner


def test_retain_execution_when_no_valid_parent_exists() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::tool:call",
        selected=True,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id=None,
        parent_input_event_key=None,
        parent_canonical_key=None,
        parent_canonical_exists=False,
    )

    assert decision.semantic_owner_canonical_key == "turn::tool:call"
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "parent_lineage_unavailable"
    assert decision.action == "retain_execution"


def test_retain_execution_when_parent_canonical_missing() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::tool:call",
        selected=True,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key="turn::item_parent",
        parent_canonical_exists=False,
    )

    assert decision.semantic_owner_canonical_key == "turn::tool:call"
    assert decision.reason_code == "parent_canonical_missing"


def test_retain_execution_when_parent_equals_execution() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::item_parent",
        selected=True,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key="turn::item_parent",
        parent_canonical_exists=True,
    )

    assert decision.semantic_owner_canonical_key == "turn::item_parent"
    assert decision.reason_code == "parent_matches_execution"


def test_reassign_parent_when_valid_tool_output_parent_promotion_exists() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::tool:call",
        selected=True,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key="turn::item_parent",
        parent_canonical_exists=True,
    )

    assert decision.semantic_owner_canonical_key == "turn::item_parent"
    assert decision.selected_candidate_id == "semantic_owner_parent"
    assert decision.reason_code == "parent_promoted_from_tool_output"
    assert decision.action == "reassign_parent"


def test_preserves_tool_prefixed_parent_exclusion_behavior() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::tool:call",
        selected=True,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="tool:call_parent",
        parent_canonical_key="turn::tool:call_parent",
        parent_canonical_exists=True,
    )

    assert decision.semantic_owner_canonical_key == "turn::tool:call"
    assert decision.reason_code == "parent_input_tool_prefixed"
