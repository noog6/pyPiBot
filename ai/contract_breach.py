"""Observational contract breach detection primitives for realtime seams."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib


class CorrectiveActionKind(str, Enum):
    NONE = "NONE"
    DEFER_CLOSE = "DEFER_CLOSE"
    HOLD_FOLLOWUP = "HOLD_FOLLOWUP"
    SUPPRESS_REDUNDANT_OUTPUT = "SUPPRESS_REDUNDANT_OUTPUT"
    RETRY_ONCE = "RETRY_ONCE"
    REOPEN_FOLLOWTHROUGH_LEDGER = "REOPEN_FOLLOWTHROUGH_LEDGER"


class ContractBreachType(str, Enum):
    EMPTY_TOOL_FOLLOWUP_DONE = "EMPTY_TOOL_FOLLOWUP_DONE"
    FOLLOWTHROUGH_REMAINING_WITH_NON_DELIVERABLE_OUTPUT = "FOLLOWTHROUGH_REMAINING_WITH_NON_DELIVERABLE_OUTPUT"
    TOOL_FOLLOWUP_BLOCKED_BY_SAME_TURN_OWNER = "TOOL_FOLLOWUP_BLOCKED_BY_SAME_TURN_OWNER"


@dataclass(frozen=True)
class ContractBreachSnapshot:
    source_seam: str
    turn_id: str
    response_id: str
    origin: str
    canonical_key: str
    reason_code: str
    is_terminal_event: bool = False
    selected_deliverable: bool | None = None
    is_empty_done: bool | None = None
    pending_tool_followup: bool | None = None
    followthrough_chain_remaining: bool | None = None
    is_tool_followup: bool | None = None
    create_action: str | None = None


@dataclass(frozen=True)
class ContractBreachArtifact:
    breach_type: ContractBreachType
    severity: str
    canonical_key: str
    turn_id: str
    response_id: str
    origin: str
    evidence: tuple[str, ...]
    recommended_action: CorrectiveActionKind
    fingerprint: str
    source_seam: str
    reason_code: str


def _fingerprint(parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def detect_contract_breach(snapshot: ContractBreachSnapshot) -> ContractBreachArtifact | None:
    turn_id = str(snapshot.turn_id or "").strip()
    response_id = str(snapshot.response_id or "").strip()
    canonical_key = str(snapshot.canonical_key or "").strip()
    source_seam = str(snapshot.source_seam or "").strip()
    if not turn_id or not canonical_key or not source_seam:
        return None

    origin = str(snapshot.origin or "unknown").strip() or "unknown"
    reason_code = str(snapshot.reason_code or "unknown").strip() or "unknown"

    # Dual-evidence terminal breach: empty tool-output done while followup is still pending.
    if (
        snapshot.is_terminal_event
        and origin == "tool_output"
        and snapshot.is_empty_done is True
        and snapshot.pending_tool_followup is True
    ):
        evidence = (
            "terminal_event=response.done",
            "origin=tool_output",
            "is_empty_done=true",
            "pending_tool_followup=true",
        )
        fp = _fingerprint((ContractBreachType.EMPTY_TOOL_FOLLOWUP_DONE.value, source_seam, canonical_key, turn_id, response_id, reason_code, *evidence))
        return ContractBreachArtifact(
            breach_type=ContractBreachType.EMPTY_TOOL_FOLLOWUP_DONE,
            severity="INFO",
            canonical_key=canonical_key,
            turn_id=turn_id,
            response_id=response_id,
            origin=origin,
            evidence=evidence,
            recommended_action=CorrectiveActionKind.HOLD_FOLLOWUP,
            fingerprint=fp,
            source_seam=source_seam,
            reason_code=reason_code,
        )

    # Dual-evidence terminal breach: unresolved followthrough + non-deliverable output.
    if (
        snapshot.is_terminal_event
        and snapshot.followthrough_chain_remaining is True
        and snapshot.selected_deliverable is False
    ):
        evidence = (
            "terminal_event=response.done",
            "followthrough_chain_remaining=true",
            "selected_deliverable=false",
        )
        fp = _fingerprint((ContractBreachType.FOLLOWTHROUGH_REMAINING_WITH_NON_DELIVERABLE_OUTPUT.value, source_seam, canonical_key, turn_id, response_id, reason_code, *evidence))
        return ContractBreachArtifact(
            breach_type=ContractBreachType.FOLLOWTHROUGH_REMAINING_WITH_NON_DELIVERABLE_OUTPUT,
            severity="INFO",
            canonical_key=canonical_key,
            turn_id=turn_id,
            response_id=response_id,
            origin=origin,
            evidence=evidence,
            recommended_action=CorrectiveActionKind.REOPEN_FOLLOWTHROUGH_LEDGER,
            fingerprint=fp,
            source_seam=source_seam,
            reason_code=reason_code,
        )

    # Response-create seam: tool followup lost to parent owner arbitration.
    if (
        source_seam == "response_create_runtime"
        and snapshot.is_tool_followup is True
        and origin == "tool_output"
        and snapshot.create_action in {"DROP", "BLOCK"}
        and reason_code == "same_turn_already_owned"
    ):
        evidence = (
            "seam=response_create_runtime",
            "origin=tool_output",
            "tool_followup=true",
            f"create_action={snapshot.create_action}",
            "reason_code=same_turn_already_owned",
        )
        fp = _fingerprint((ContractBreachType.TOOL_FOLLOWUP_BLOCKED_BY_SAME_TURN_OWNER.value, source_seam, canonical_key, turn_id, response_id, reason_code, *evidence))
        return ContractBreachArtifact(
            breach_type=ContractBreachType.TOOL_FOLLOWUP_BLOCKED_BY_SAME_TURN_OWNER,
            severity="INFO",
            canonical_key=canonical_key,
            turn_id=turn_id,
            response_id=response_id,
            origin=origin,
            evidence=evidence,
            recommended_action=CorrectiveActionKind.DEFER_CLOSE,
            fingerprint=fp,
            source_seam=source_seam,
            reason_code=reason_code,
        )

    return None
