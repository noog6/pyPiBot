from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.semantic_owner_arbitration import decide_semantic_owner


def test_retain_execution_when_execution_canonical_missing() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="",
        selected=True,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key="turn::item_parent",
        parent_canonical_exists=True,
    )

    assert decision.semantic_owner_canonical_key == ""
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "execution_canonical_unavailable"
    assert decision.action == "retain_execution"


def test_retain_execution_when_terminal_not_selected() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::tool:call",
        selected=False,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key="turn::item_parent",
        parent_canonical_exists=True,
    )

    assert decision.semantic_owner_canonical_key == "turn::tool:call"
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "terminal_not_selected"
    assert decision.action == "retain_execution"


def test_retain_execution_when_selection_reason_not_normal() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::tool:call",
        selected=True,
        selection_reason="cancelled",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key="turn::item_parent",
        parent_canonical_exists=True,
    )

    assert decision.semantic_owner_canonical_key == "turn::tool:call"
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "terminal_reason_ineligible"
    assert decision.action == "retain_execution"


def test_retain_execution_when_origin_not_tool_output() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::item_parent",
        selected=True,
        selection_reason="normal",
        origin="assistant_message",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key="turn::item_parent",
        parent_canonical_exists=True,
    )

    assert decision.semantic_owner_canonical_key == "turn::item_parent"
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "origin_ineligible"
    assert decision.action == "retain_execution"


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
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "parent_canonical_missing"
    assert decision.action == "retain_execution"


def test_retain_execution_when_parent_canonical_is_unavailable() -> None:
    decision = decide_semantic_owner(
        execution_canonical_key="turn::tool:call",
        selected=True,
        selection_reason="normal",
        origin="tool_output",
        parent_turn_id="turn",
        parent_input_event_key="item_parent",
        parent_canonical_key=None,
        parent_canonical_exists=False,
    )

    assert decision.semantic_owner_canonical_key == "turn::tool:call"
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "parent_canonical_unavailable"
    assert decision.action == "retain_execution"


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
    assert decision.selected_candidate_id == "semantic_owner_execution"
    assert decision.reason_code == "parent_matches_execution"
    assert decision.action == "retain_execution"


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
