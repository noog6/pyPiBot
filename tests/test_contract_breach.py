from __future__ import annotations

import os
import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.contract_breach import (
    ContractBreachSnapshot,
    CorrectiveActionKind,
    detect_contract_breach,
)


def test_detect_contract_breach_returns_none_when_core_facts_missing() -> None:
    artifact = detect_contract_breach(
        ContractBreachSnapshot(
            source_seam="",
            turn_id="",
            response_id="resp_1",
            origin="tool_output",
            canonical_key="",
            reason_code="unknown",
            is_terminal_event=True,
            is_empty_done=True,
            pending_tool_followup=True,
        )
    )

    assert artifact is None


def test_detect_contract_breach_detects_empty_tool_followup_done() -> None:
    artifact = detect_contract_breach(
        ContractBreachSnapshot(
            source_seam="response_terminal_handlers",
            turn_id="turn_1",
            response_id="resp_1",
            origin="tool_output",
            canonical_key="turn_1::item_1",
            reason_code="tool_followup_precedence",
            is_terminal_event=True,
            is_empty_done=True,
            pending_tool_followup=True,
        )
    )

    assert artifact is not None
    assert artifact.breach_type.value == "EMPTY_TOOL_FOLLOWUP_DONE"
    assert artifact.recommended_action is CorrectiveActionKind.HOLD_FOLLOWUP


def test_detect_contract_breach_fingerprint_is_stable() -> None:
    snapshot = ContractBreachSnapshot(
        source_seam="response_terminal_handlers",
        turn_id="turn_1",
        response_id="resp_1",
        origin="tool_output",
        canonical_key="turn_1::item_1",
        reason_code="tool_followup_precedence",
        is_terminal_event=True,
        is_empty_done=True,
        pending_tool_followup=True,
    )

    first = detect_contract_breach(snapshot)
    second = detect_contract_breach(snapshot)

    assert first is not None and second is not None
    assert first.fingerprint == second.fingerprint


def test_detect_contract_breach_recommended_action_deterministic_for_same_turn_owner_case() -> None:
    artifact = detect_contract_breach(
        ContractBreachSnapshot(
            source_seam="response_create_runtime",
            turn_id="turn_2",
            response_id="none",
            origin="tool_output",
            canonical_key="turn_2::tool:call_1",
            reason_code="same_turn_already_owned",
            is_tool_followup=True,
            create_action="DROP",
        )
    )

    assert artifact is not None
    assert artifact.recommended_action is CorrectiveActionKind.DEFER_CLOSE


def test_detect_contract_breach_suppresses_expected_tool_followthrough_handoff() -> None:
    artifact = detect_contract_breach(
        ContractBreachSnapshot(
            source_seam="response_terminal_handlers",
            turn_id="turn_3",
            response_id="resp_mid_chain",
            origin="tool_output",
            canonical_key="turn_3::tool:call_middle",
            reason_code="tool_followup_precedence",
            is_terminal_event=True,
            selected_deliverable=False,
            followthrough_chain_remaining=True,
        )
    )

    assert artifact is None


def test_detect_contract_breach_keeps_followthrough_remaining_breach_for_unexpected_reason() -> None:
    artifact = detect_contract_breach(
        ContractBreachSnapshot(
            source_seam="response_terminal_handlers",
            turn_id="turn_3",
            response_id="resp_mid_chain",
            origin="tool_output",
            canonical_key="turn_3::tool:call_middle",
            reason_code="normal",
            is_terminal_event=True,
            selected_deliverable=False,
            followthrough_chain_remaining=True,
        )
    )

    assert artifact is not None
    assert artifact.breach_type.value == "FOLLOWTHROUGH_REMAINING_WITH_NON_DELIVERABLE_OUTPUT"
    assert artifact.recommended_action is CorrectiveActionKind.REOPEN_FOLLOWTHROUGH_LEDGER
