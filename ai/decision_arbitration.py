"""First-pass deterministic arbitration seam for do-now/defer/refuse choices."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from core.logging import logger


class ArbitrationAction(str, Enum):
    DO_NOW = "do_now"
    DEFER = "defer"
    REFUSE = "refuse"


@dataclass(frozen=True)
class ArbitrationCandidate:
    candidate_id: str
    action: ArbitrationAction
    reason_code: str
    priority: int
    defer_until: float | None = None


@dataclass(frozen=True)
class ArbitrationDecision:
    action: ArbitrationAction
    reason_code: str
    selected_candidate_id: str
    defer_until: float | None = None


_ACTION_TIE_BREAK_ORDER: dict[ArbitrationAction, int] = {
    ArbitrationAction.REFUSE: 0,
    ArbitrationAction.DEFER: 1,
    ArbitrationAction.DO_NOW: 2,
}


def decide_arbitration(
    *,
    policy_name: str,
    candidates: Iterable[ArbitrationCandidate],
    default_candidate: ArbitrationCandidate,
) -> ArbitrationDecision:
    """Select a deterministic arbitration decision envelope.

    First pass is intentionally narrow: candidate priority desc, then action order,
    then candidate_id asc for stable tie-breaking.
    """

    normalized_policy_name = str(policy_name or "unknown_policy").strip() or "unknown_policy"
    normalized_candidates = [candidate for candidate in candidates if isinstance(candidate, ArbitrationCandidate)]

    selected = default_candidate
    if normalized_candidates:
        selected = sorted(
            normalized_candidates,
            key=lambda candidate: (
                -int(candidate.priority),
                _ACTION_TIE_BREAK_ORDER.get(candidate.action, 99),
                str(candidate.candidate_id or ""),
            ),
        )[0]

    decision = ArbitrationDecision(
        action=selected.action,
        reason_code=selected.reason_code,
        selected_candidate_id=selected.candidate_id,
        defer_until=selected.defer_until,
    )
    logger.debug(
        "arbitration_decision policy=%s action=%s reason_code=%s selected_candidate_id=%s defer_until=%s candidate_count=%s",
        normalized_policy_name,
        decision.action.value,
        decision.reason_code,
        decision.selected_candidate_id,
        "" if decision.defer_until is None else decision.defer_until,
        len(normalized_candidates),
    )
    return decision
