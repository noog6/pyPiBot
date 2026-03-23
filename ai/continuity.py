"""Deterministic presence/continuity bookkeeping seam."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import time
import re
from typing import Callable, Literal

TurnSettlementState = Literal[
    "settled",
    "awaiting_tool",
    "followthrough_remaining",
    "unresolved_followup",
    "recently_closed_only",
    "active_items_only",
]

logger = logging.getLogger(__name__)

ContinuityKind = Literal[
    "ongoing",
    "unresolved",
    "commitment",
    "blocker",
    "constraint",
    "recently_closed",
]
ContinuityStatus = Literal["active", "blocked", "pending", "resolved", "expired"]
ContinuityPriority = Literal["low", "medium", "high"]
CompoundStepKind = Literal["gesture", "diagnostics", "observation", "followup", "report"]
CompoundStepStatus = Literal["pending", "active", "completed"]
# Deterministic continuity stance labels used for bookkeeping and inspection.
ContinuityStance = Literal[
    "idle",
    "assisting_observation",
    "assisting_execution",
    "assisting_query",
    "awaiting_user",
    "awaiting_tool",
    "awaiting_perception",
    "recovering_context",
]

_MAX_ITEMS_PER_BUCKET = 3
_MAX_LOG_SUMMARY_CHARS = 72
_MAX_LOG_DETAIL_CHARS = 48
_CONTINUITY_BRIEF_LOG_COOLDOWN_S = 10.0
_TOOL_COMMITMENT_PREFIXES = ("gesture_", "move", "look", "center", "turn", "navigate")
_MAX_COMPOUND_STEPS = 8
_ACTION_PATTERNS = (
    re.compile(r"\bmove\b", re.IGNORECASE),
    re.compile(r"\blook\b", re.IGNORECASE),
    re.compile(r"\bcenter\b", re.IGNORECASE),
    re.compile(r"\bturn\b", re.IGNORECASE),
    re.compile(r"\bgo to\b", re.IGNORECASE),
    re.compile(r"\brotate\b", re.IGNORECASE),
    re.compile(r"\bpoint\b", re.IGNORECASE),
)
_OBSERVATION_PATTERNS = (
    re.compile(r"\bwhat do you see\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) in\b", re.IGNORECASE),
    re.compile(r"\bdescribe\b", re.IGNORECASE),
    re.compile(r"\bidentify\b", re.IGNORECASE),
    re.compile(r"\bcan you see\b", re.IGNORECASE),
    re.compile(r"\bdo you see\b", re.IGNORECASE),
)
_STATUS_CHECK_PATTERNS = (
    re.compile(r"\bcan you still hear me\b", re.IGNORECASE),
    re.compile(r"\bdo you still hear me\b", re.IGNORECASE),
    re.compile(r"\bare you still listening\b", re.IGNORECASE),
)
_QUERY_REQUEST_PATTERNS = (
    re.compile(r"\bdo you know\b", re.IGNORECASE),
    re.compile(r"\bcan you check\s+(?:the\s+)?(?:battery|voltage|temperature|air pressure|pressure|environment)\b", re.IGNORECASE),
    re.compile(r"\bwhat(?:'s| is) your\s+(?:battery|battery voltage|temperature|air pressure|pressure|status)\b", re.IGNORECASE),
    re.compile(r"\btell me your current\s+(?:battery|battery voltage|temperature|air pressure|pressure|status)\b", re.IGNORECASE),
    re.compile(r"\bread_(?:battery_voltage|environment)\b", re.IGNORECASE),
    re.compile(r"\b(?:battery voltage|air pressure|temperature|environment(?:al)? status)\b", re.IGNORECASE),
)
_TOOL_REQUIRED_PATTERNS = (
    re.compile(r"\bcamera\b", re.IGNORECASE),
    re.compile(r"\bimage\b", re.IGNORECASE),
    re.compile(r"\bphoto\b", re.IGNORECASE),
    re.compile(r"\bscreenshot\b", re.IGNORECASE),
    re.compile(r"\bcheck\b", re.IGNORECASE),
    re.compile(r"\buse the tool\b", re.IGNORECASE),
)
_FOLLOW_UP_PATTERNS = (
    re.compile(r"\b(tell me\b.*)", re.IGNORECASE),
    re.compile(r"\b(let me know\b.*)", re.IGNORECASE),
    re.compile(r"\b(answer\b.*)", re.IGNORECASE),
    re.compile(r"\b(whether\b.*)", re.IGNORECASE),
    re.compile(r"\b(what\b.*\?)", re.IGNORECASE),
)
_COMPOUND_SPLIT_RE = re.compile(r"\s*(?:,|;|\band then\b|\bthen\b)\s*", re.IGNORECASE)
_FOLLOWUP_ONLY_PREFIX_RE = re.compile(r"^(?:then\s+)?(?:tell me|let me know|report back|report|say)\b", re.IGNORECASE)
_SILENT_RE = re.compile(r"\bsilently\b", re.IGNORECASE)
_DIAGNOSTIC_STEP_RE = re.compile(r"\b(?:check|run|read|query)\b.*\b(?:diagnostic|diagnostics|status|battery|temperature|pressure|environment)\b", re.IGNORECASE)
_REPORT_STEP_RE = re.compile(r"\b(?:tell me|let me know|report back|report|say)\b", re.IGNORECASE)
_OBSERVATION_STEP_RE = re.compile(r"\b(?:observe|describe|identify|what do you see|what(?:'s| is) in|do you see|can you see)\b", re.IGNORECASE)
_GESTURE_STEP_RE = re.compile(r"\b(?:look|move|center|turn|rotate|point|go to|navigate)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ContinuityItem:
    id: str
    kind: ContinuityKind
    summary: str
    status: ContinuityStatus
    priority: ContinuityPriority
    source: str
    detail: str = ""
    expires_after_turns: int | None = None


@dataclass(frozen=True)
class CompoundContinuityStep:
    step_id: str
    kind: CompoundStepKind
    summary: str
    status: CompoundStepStatus
    source_detail: str = ""


@dataclass(frozen=True)
class CompoundContinuityState:
    request_id: str
    summary: str
    steps: tuple[CompoundContinuityStep, ...]
    active_step_index: int | None
    completed_step_ids: tuple[str, ...]
    final_followup_pending: bool
    recent_completed_step_id: str | None = None
    next_pending_step_id: str | None = None


@dataclass(frozen=True)
class ContinuityTurnSettlement:
    """Read-only turn settlement classification derived from continuity state.

    This observational helper is intended for diagnostics only. It must not be
    treated as an authority surface for arbitration, scheduling, or persistence.
    """

    settlement_state: TurnSettlementState
    settlement_detail: str
    has_current_items: bool
    has_commitments: bool
    has_unresolved: bool
    has_blockers: bool
    has_recently_closed: bool


@dataclass(frozen=True)
class ContinuityBrief:
    run_id: str
    turn_id: str
    stance: ContinuityStance
    ongoing: tuple[ContinuityItem, ...] = ()
    unresolved: tuple[ContinuityItem, ...] = ()
    commitments: tuple[ContinuityItem, ...] = ()
    blockers: tuple[ContinuityItem, ...] = ()
    constraints: tuple[ContinuityItem, ...] = ()
    recently_closed: tuple[ContinuityItem, ...] = ()
    current: tuple[ContinuityItem, ...] = ()
    generated_reason: str = ""
    stance_detail: str = ""
    compound_request: CompoundContinuityState | None = None


@dataclass(frozen=True)
class ContinuityBriefLogFingerprint:
    stance: ContinuityStance
    stance_detail: str
    settlement_state: TurnSettlementState
    settlement_detail: str
    counts: tuple[tuple[str, int], ...]
    current_items: tuple[tuple[str, str, str, str], ...]
    recently_closed_items: tuple[tuple[str, str, str, str], ...]
    compound_signature: tuple[str, ...] = ()


class ContinuityLedger:
    """Small in-memory continuity ledger.

    This seam only records deterministic bookkeeping. It does not arbitrate,
    schedule, or execute behavior.
    """

    def __init__(
        self,
        *,
        brief_log_cooldown_s: float = _CONTINUITY_BRIEF_LOG_COOLDOWN_S,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stance: ContinuityStance = "idle"
        self._items: dict[str, ContinuityItem] = {}
        self._compound_state: CompoundContinuityState | None = None
        self._brief_log_cooldown_s = max(0.0, float(brief_log_cooldown_s))
        self._time_source = time_source
        self._last_brief_log_fingerprint: ContinuityBriefLogFingerprint | None = None
        self._last_brief_log_ts: float | None = None

    def update_from_event(self, event_type: str, **payload: object) -> None:
        normalized = str(event_type or "").strip().lower()
        if not normalized:
            return
        if normalized == "transcript_final":
            self._apply_transcript_final(payload)
        elif normalized == "tool_call_started":
            self._apply_tool_call_started(payload)
        elif normalized == "tool_result_received":
            self._apply_tool_result_received(payload)
        elif normalized == "response_done":
            self._apply_response_done(payload)
        else:
            return
        self._expire_items()
        logger.info(
            "continuity_event_applied event=%s stance=%s active_items=%s",
            normalized,
            self._stance,
            len(self._items),
        )
        self._log_inspection_summary(event_type=normalized, payload=payload)

    def inspect_state(
        self,
        run_id: str = "",
        turn_id: str = "",
        event_type: str = "inspection",
    ) -> dict[str, object]:
        brief = self._build_brief_projection(run_id=run_id, turn_id=turn_id, reason=event_type)
        settlement = self.build_turn_settlement(brief)
        return {
            "run_id": run_id,
            "turn_id": turn_id,
            "event_type": event_type,
            "stance": brief.stance,
            "stance_detail": brief.stance_detail,
            "settlement_state": settlement.settlement_state,
            "settlement_detail": settlement.settlement_detail,
            "ongoing": len(brief.ongoing),
            "unresolved": len(brief.unresolved),
            "commitments": len(brief.commitments),
            "blockers": len(brief.blockers),
            "constraints": len(brief.constraints),
            "recently_closed": len(brief.recently_closed),
            "current": len(brief.current),
            "current_items": self._project_items(brief.current),
            "recently_closed_items": self._project_items(brief.recently_closed),
            "compound_request": self._project_compound_state(brief.compound_request),
        }

    def _log_inspection_summary(self, *, event_type: str, payload: dict[str, object]) -> None:
        if event_type not in {"transcript_final", "tool_call_started", "tool_result_received", "response_done"}:
            return
        summary = self.inspect_state(
            run_id=self._clean_text(payload.get("run_id")),
            turn_id=self._clean_text(payload.get("turn_id")),
            event_type=event_type,
        )
        logger.info(
            "continuity_inspection_summary run_id=%s turn_id=%s event_type=%s stance=%s stance_detail=%s settlement=%s settlement_detail=%s ongoing=%s unresolved=%s commitments=%s blockers=%s constraints=%s recently_closed=%s current=%s compound_request=%s current_items=%s recently_closed_items=%s",
            summary["run_id"] or "",
            summary["turn_id"] or "",
            summary["event_type"],
            summary["stance"],
            summary["stance_detail"] or "",
            summary["settlement_state"],
            summary["settlement_detail"] or "",
            summary["ongoing"],
            summary["unresolved"],
            summary["commitments"],
            summary["blockers"],
            summary["constraints"],
            summary["recently_closed"],
            summary["current"],
            summary["compound_request"],
            summary["current_items"],
            summary["recently_closed_items"],
        )

    def build_brief(self, run_id: str, turn_id: str, reason: str) -> ContinuityBrief:
        brief = self._build_brief_projection(run_id=run_id, turn_id=turn_id, reason=reason)
        self._log_brief_if_material(run_id=run_id, turn_id=turn_id, brief=brief)
        return brief

    def _log_brief_if_material(self, *, run_id: str, turn_id: str, brief: ContinuityBrief) -> None:
        fingerprint = self._brief_log_fingerprint(brief)
        now = float(self._time_source())
        last_fingerprint = self._last_brief_log_fingerprint
        last_ts = self._last_brief_log_ts
        cooldown_elapsed_s = 0.0 if last_ts is None else max(0.0, now - last_ts)
        fingerprint_changed = fingerprint != last_fingerprint
        cooldown_expired = (
            last_ts is None
            or self._brief_log_cooldown_s <= 0
            or cooldown_elapsed_s >= self._brief_log_cooldown_s
        )
        if not fingerprint_changed and not cooldown_expired:
            return

        reminder_due = not fingerprint_changed and cooldown_expired and last_ts is not None
        self._last_brief_log_fingerprint = fingerprint
        self._last_brief_log_ts = now
        logger.info(
            "continuity_brief_built run_id=%s turn_id=%s stance=%s ongoing=%s blockers=%s commitments=%s current=%s closed=%s compound_steps=%s reason=%s settlement=%s reminder=%s fingerprint_changed=%s cooldown_s=%s cooldown_elapsed_s=%s",
            run_id,
            turn_id,
            brief.stance,
            len(brief.ongoing),
            len(brief.blockers),
            len(brief.commitments),
            len(brief.current),
            len(brief.recently_closed),
            0 if brief.compound_request is None else len(brief.compound_request.steps),
            brief.generated_reason or "none",
            fingerprint.settlement_state,
            reminder_due,
            fingerprint_changed,
            self._brief_log_cooldown_s,
            cooldown_elapsed_s,
        )

    def _brief_log_fingerprint(self, brief: ContinuityBrief) -> ContinuityBriefLogFingerprint:
        settlement = self.build_turn_settlement(brief)
        compound_signature = ()
        if brief.compound_request is not None:
            compound_signature = tuple(
                f"{step.step_id}:{step.kind}:{step.status}:{self._trim_text(step.summary, 32)}"
                for step in brief.compound_request.steps
            ) + (
                f"active={brief.compound_request.active_step_index}",
                f"recent={brief.compound_request.recent_completed_step_id or ''}",
                f"next={brief.compound_request.next_pending_step_id or ''}",
                f"followup={brief.compound_request.final_followup_pending}",
            )
        return ContinuityBriefLogFingerprint(
            stance=brief.stance,
            stance_detail=brief.stance_detail,
            settlement_state=settlement.settlement_state,
            settlement_detail=settlement.settlement_detail,
            counts=(
                ("current", len(brief.current)),
                ("ongoing", len(brief.ongoing)),
                ("unresolved", len(brief.unresolved)),
                ("commitments", len(brief.commitments)),
                ("blockers", len(brief.blockers)),
                ("constraints", len(brief.constraints)),
                ("recently_closed", len(brief.recently_closed)),
            ),
            current_items=self._brief_item_signature(brief.current),
            recently_closed_items=self._brief_item_signature(brief.recently_closed),
            compound_signature=compound_signature,
        )

    @staticmethod
    def _brief_item_signature(items: tuple[ContinuityItem, ...]) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(
            (
                item.kind,
                item.status,
                ContinuityLedger._trim_text(item.summary, _MAX_LOG_SUMMARY_CHARS),
                ContinuityLedger._trim_text(item.detail, _MAX_LOG_DETAIL_CHARS),
            )
            for item in items[:_MAX_ITEMS_PER_BUCKET]
        )

    def _build_brief_projection(self, *, run_id: str, turn_id: str, reason: str) -> ContinuityBrief:
        """Project the current continuity ledger into the shared brief shape."""
        ongoing = self._bucket("ongoing", {"active", "pending"})
        unresolved = self._bucket("unresolved", {"active", "pending", "blocked"})
        commitments = self._bucket("commitment", {"active", "pending", "blocked"})
        blockers = self._bucket("blocker", {"active", "pending", "blocked"})
        constraints = self._bucket("constraint", {"active", "pending", "blocked"})
        recently_closed = self._bucket("recently_closed", {"resolved"})
        current = self._current_projection(ongoing, unresolved, commitments, blockers, constraints)
        return ContinuityBrief(
            run_id=run_id,
            turn_id=turn_id,
            stance=self._stance,
            ongoing=ongoing,
            unresolved=unresolved,
            commitments=commitments,
            blockers=blockers,
            constraints=constraints,
            recently_closed=recently_closed,
            current=current,
            generated_reason=str(reason or "").strip(),
            stance_detail=self._stance_detail(
                stance=self._stance,
                current=current,
                blockers=blockers,
                unresolved=unresolved,
                commitments=commitments,
            ),
            compound_request=self._compound_state,
        )

    def build_turn_settlement(self, brief: ContinuityBrief) -> ContinuityTurnSettlement:
        """Return a compact observational settlement view for a continuity brief."""
        return self.classify_turn_settlement(
            current=brief.current,
            blockers=brief.blockers,
            commitments=brief.commitments,
            unresolved=brief.unresolved,
            recently_closed=brief.recently_closed,
        )

    @staticmethod
    def classify_turn_settlement(
        *,
        current: tuple[ContinuityItem, ...],
        blockers: tuple[ContinuityItem, ...],
        commitments: tuple[ContinuityItem, ...],
        unresolved: tuple[ContinuityItem, ...],
        recently_closed: tuple[ContinuityItem, ...],
    ) -> ContinuityTurnSettlement:
        """Classify turn settlement from existing continuity buckets only."""
        has_current_items = bool(current)
        has_blockers = bool(blockers)
        has_commitments = bool(commitments)
        has_unresolved = bool(unresolved)
        has_recently_closed = bool(recently_closed)

        if has_blockers:
            first_blocker = blockers[0]
            detail = first_blocker.detail or first_blocker.summary
            state = "awaiting_tool"
        elif has_commitments and has_unresolved:
            first_unresolved = unresolved[0]
            detail = first_unresolved.detail or first_unresolved.summary
            state = "unresolved_followup"
        elif has_commitments:
            first_commitment = commitments[0]
            detail = first_commitment.detail or first_commitment.summary
            state = "followthrough_remaining"
        elif has_unresolved:
            first_unresolved = unresolved[0]
            detail = first_unresolved.detail or first_unresolved.summary
            state = "unresolved_followup"
        elif has_current_items:
            first_current = current[0]
            detail = first_current.detail or f"current={first_current.kind}:{first_current.status}"
            state = "active_items_only"
        elif has_recently_closed:
            first_closed = recently_closed[0]
            detail = first_closed.detail or first_closed.summary
            state = "recently_closed_only"
        else:
            detail = "no_open_continuity_items"
            state = "settled"

        return ContinuityTurnSettlement(
            settlement_state=state,
            settlement_detail=detail,
            has_current_items=has_current_items,
            has_commitments=has_commitments,
            has_unresolved=has_unresolved,
            has_blockers=has_blockers,
            has_recently_closed=has_recently_closed,
        )

    def commitment_summary_for_tool(self, tool_name: str, transcript: str | None = None) -> str | None:
        normalized_tool = self._clean_text(tool_name).lower()
        if normalized_tool.startswith(_TOOL_COMMITMENT_PREFIXES):
            return self._clean_text(transcript) or f"Complete requested action via {normalized_tool}"
        return None

    def _apply_transcript_final(self, payload: dict[str, object]) -> None:
        transcript = self._clean_text(payload.get("text"))
        source = self._clean_text(payload.get("source")) or "transcript_final"
        if not transcript:
            self._stance = "recovering_context"
            self._compound_state = None
            return

        is_action_request = self._is_action_request(transcript)
        followup_summary = self._extract_followup_summary(transcript, requires_action=is_action_request)
        self._compound_state = self._derive_compound_state(transcript, source=source, unresolved_summary=followup_summary)

        self._set_item(
            ContinuityItem(
                id="request:current",
                kind="ongoing",
                summary=transcript,
                status="active",
                priority=self._priority_for_text(transcript),
                source=source,
                detail="origin=user_transcript",
                expires_after_turns=4,
            )
        )
        if is_action_request:
            self._set_item(
                ContinuityItem(
                    id="commitment:current",
                    kind="commitment",
                    summary=transcript,
                    status="pending",
                    priority=self._priority_for_text(transcript),
                    source=source,
                    detail="origin=user_request",
                    expires_after_turns=4,
                )
            )
        else:
            self._remove_item("commitment:current")
        if followup_summary:
            self._set_item(
                ContinuityItem(
                    id="question:followup",
                    kind="unresolved",
                    summary=followup_summary,
                    status="pending",
                    priority="medium",
                    source=source,
                    detail="opened_by=transcript_final",
                    expires_after_turns=4,
                )
            )
        else:
            self._remove_item("question:followup")
        self._stance = self._stance_for_transcript(transcript, is_action_request=is_action_request)

    def _apply_tool_call_started(self, payload: dict[str, object]) -> None:
        tool_name = self._clean_text(payload.get("tool_name")) or "tool"
        call_id = self._clean_text(payload.get("call_id")) or tool_name
        self._set_item(
            ContinuityItem(
                id=f"blocker:tool:{call_id}",
                kind="blocker",
                summary=f"Waiting for tool result: {tool_name}",
                status="blocked",
                priority="medium",
                source="tool_call_started",
                detail=f"tool={tool_name} call_id={call_id}",
                expires_after_turns=2,
            )
        )
        commitment_summary = self._clean_text(payload.get("commitment_summary"))
        if commitment_summary:
            self._set_item(
                ContinuityItem(
                    id="commitment:current",
                    kind="commitment",
                    summary=commitment_summary,
                    status="active",
                    priority="medium",
                    source="tool_call_started",
                    detail=f"origin=tool_followthrough tool={tool_name} call_id={call_id}",
                    expires_after_turns=4,
                )
            )
        self._advance_compound_state_on_tool_start(tool_name=tool_name)
        self._stance = "awaiting_tool"

    def _apply_tool_result_received(self, payload: dict[str, object]) -> None:
        call_id = self._clean_text(payload.get("call_id"))
        tool_name = self._clean_text(payload.get("tool_name"))
        if call_id:
            blocker_id = f"blocker:tool:{call_id}"
            blocker = self._items.get(blocker_id)
            if blocker is not None:
                self._set_item(
                    replace(
                        blocker,
                        kind="recently_closed",
                        status="resolved",
                        expires_after_turns=1,
                    )
                )
        if "commitment:current" in self._items:
            self._set_item(
                replace(
                    self._items["commitment:current"],
                    status="active",
                    source="tool_result_received",
                )
            )
        self._advance_compound_state_on_tool_result(tool_name=tool_name)
        self._stance = "idle"

    def _apply_response_done(self, payload: dict[str, object]) -> None:
        close_ongoing = self._as_bool(payload.get("close_ongoing"))
        close_commitment = self._as_bool(payload.get("close_commitment"))
        close_unresolved = self._as_bool(payload.get("close_unresolved"))
        keep_ongoing = self._as_bool(payload.get("keep_ongoing"))

        if close_unresolved and "question:followup" in self._items:
            self._close_item("question:followup")
        if close_commitment and "commitment:current" in self._items:
            self._close_item("commitment:current")
        for item_id, item in list(self._items.items()):
            if item.kind == "blocker":
                self._close_item(item_id)
            elif item.kind == "ongoing":
                if close_ongoing and not keep_ongoing:
                    self._remove_item(item_id)
        self._advance_compound_state_on_response_done(
            close_commitment=close_commitment,
            close_unresolved=close_unresolved,
        )
        self._stance = "idle" if close_ongoing or close_commitment or close_unresolved else "awaiting_user"

    def _advance_compound_state_on_tool_start(self, *, tool_name: str) -> None:
        state = self._compound_state
        if state is None:
            return
        active_idx = state.active_step_index
        normalized_tool = tool_name.lower()
        if active_idx is not None and active_idx < len(state.steps):
            active_step = state.steps[active_idx]
            if active_step.kind == "diagnostics" or (
                active_step.kind == "gesture" and normalized_tool.startswith(_TOOL_COMMITMENT_PREFIXES)
            ):
                return
        idx = self._first_pending_step_index(state)
        if idx is None:
            return
        step = state.steps[idx]
        if step.kind == "report":
            return
        if step.kind == "diagnostics" or normalized_tool.startswith(_TOOL_COMMITMENT_PREFIXES):
            self._compound_state = self._replace_compound_step_status(state, idx, "active")

    def _advance_compound_state_on_tool_result(self, *, tool_name: str) -> None:
        state = self._compound_state
        if state is None:
            return
        idx = state.active_step_index
        if idx is None:
            idx = self._first_pending_step_index(state)
            if idx is None:
                return
        step = state.steps[idx]
        normalized_tool = tool_name.lower()
        clear_match = (
            step.kind == "diagnostics"
            or (step.kind == "gesture" and normalized_tool.startswith(_TOOL_COMMITMENT_PREFIXES))
        )
        if clear_match:
            self._compound_state = self._replace_compound_step_status(state, idx, "completed")

    def _advance_compound_state_on_response_done(self, *, close_commitment: bool, close_unresolved: bool) -> None:
        state = self._compound_state
        if state is None:
            return
        updated = state
        if close_commitment:
            idx = updated.active_step_index
            if idx is None:
                idx = self._first_pending_step_index(updated, include_report=False)
            if idx is not None:
                step = updated.steps[idx]
                if step.kind != "report":
                    updated = self._replace_compound_step_status(updated, idx, "completed")
        if close_unresolved and updated.final_followup_pending:
            idx = self._first_pending_step_index(updated, kinds={"report"})
            if idx is not None:
                updated = self._replace_compound_step_status(updated, idx, "completed")
        self._compound_state = None if self._compound_state_resolved(updated) else updated

    def _replace_compound_step_status(
        self,
        state: CompoundContinuityState,
        step_index: int,
        status: CompoundStepStatus,
    ) -> CompoundContinuityState:
        updated_steps = list(state.steps)
        target = updated_steps[step_index]
        updated_steps[step_index] = replace(target, status=status)
        if status == "completed":
            for idx, step in enumerate(updated_steps):
                if idx != step_index and step.status == "active":
                    updated_steps[idx] = replace(step, status="completed")
        elif status == "active":
            for idx, step in enumerate(updated_steps):
                if idx != step_index and step.status == "active":
                    updated_steps[idx] = replace(step, status="pending")
        for idx, step in enumerate(updated_steps):
            if step.status == "pending" and idx < step_index and status == "completed":
                continue
            if status == "completed" and idx > step_index and step.status == "pending":
                break
        next_pending_index = next((idx for idx, step in enumerate(updated_steps) if step.status == "pending"), None)
        if status == "completed" and next_pending_index is not None:
            next_step = updated_steps[next_pending_index]
            if next_step.kind != "report":
                updated_steps[next_pending_index] = replace(next_step, status="active")
        active_step_index = next((idx for idx, step in enumerate(updated_steps) if step.status == "active"), None)
        completed_ids = tuple(step.step_id for step in updated_steps if step.status == "completed")
        next_pending_id = next((step.step_id for step in updated_steps if step.status == "pending"), None)
        recent_completed = target.step_id if status == "completed" else state.recent_completed_step_id
        final_followup_pending = any(step.kind == "report" and step.status != "completed" for step in updated_steps)
        return replace(
            state,
            steps=tuple(updated_steps),
            active_step_index=active_step_index,
            completed_step_ids=completed_ids,
            final_followup_pending=final_followup_pending,
            recent_completed_step_id=recent_completed,
            next_pending_step_id=next_pending_id,
        )

    def _compound_state_resolved(self, state: CompoundContinuityState) -> bool:
        return bool(state.steps) and all(step.status == "completed" for step in state.steps)

    def _first_pending_step_index(
        self,
        state: CompoundContinuityState,
        *,
        include_report: bool = True,
        kinds: set[str] | None = None,
    ) -> int | None:
        for idx, step in enumerate(state.steps):
            if step.status != "pending":
                continue
            if not include_report and step.kind == "report":
                continue
            if kinds is not None and step.kind not in kinds:
                continue
            return idx
        return None

    def _close_item(self, item_id: str) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        self._set_item(
            replace(
                item,
                kind="recently_closed",
                status="resolved",
                expires_after_turns=1,
            )
        )

    def _expire_items(self) -> None:
        expired_ids: list[str] = []
        for item_id, item in self._items.items():
            if item.expires_after_turns is None:
                continue
            remaining = int(item.expires_after_turns) - 1
            if remaining < 0:
                expired_ids.append(item_id)
                continue
            self._items[item_id] = replace(item, expires_after_turns=remaining)
        for item_id in expired_ids:
            self._items.pop(item_id, None)

    def _bucket(self, kind: str, statuses: set[str]) -> tuple[ContinuityItem, ...]:
        ranked = [item for item in self._items.values() if item.kind == kind and item.status in statuses]
        ranked.sort(key=lambda item: (self._priority_rank(item.priority), item.id))
        return tuple(ranked[:_MAX_ITEMS_PER_BUCKET])

    def _current_projection(
        self,
        ongoing: tuple[ContinuityItem, ...],
        unresolved: tuple[ContinuityItem, ...],
        commitments: tuple[ContinuityItem, ...],
        blockers: tuple[ContinuityItem, ...],
        constraints: tuple[ContinuityItem, ...],
    ) -> tuple[ContinuityItem, ...]:
        current = [*blockers, *commitments, *unresolved, *ongoing, *constraints]
        return tuple(current[:_MAX_ITEMS_PER_BUCKET])

    @staticmethod
    def _project_items(items: tuple[ContinuityItem, ...]) -> tuple[dict[str, str], ...]:
        return tuple(
            {
                "kind": item.kind,
                "status": item.status,
                "id": item.id,
                "summary": ContinuityLedger._trim_text(item.summary, _MAX_LOG_SUMMARY_CHARS),
                "detail": ContinuityLedger._trim_text(item.detail, _MAX_LOG_DETAIL_CHARS),
            }
            for item in items[:_MAX_ITEMS_PER_BUCKET]
        )

    def _project_compound_state(
        self,
        state: CompoundContinuityState | None,
    ) -> dict[str, object] | None:
        if state is None:
            return None
        active_step = state.steps[state.active_step_index] if state.active_step_index is not None and state.active_step_index < len(state.steps) else None
        next_step = next((step for step in state.steps if step.step_id == state.next_pending_step_id), None)
        recent_step = next((step for step in state.steps if step.step_id == state.recent_completed_step_id), None)
        return {
            "request_id": state.request_id,
            "summary": self._trim_text(state.summary, _MAX_LOG_SUMMARY_CHARS),
            "substeps_total": len(state.steps),
            "substeps_completed": len(state.completed_step_ids),
            "active_step_index": state.active_step_index,
            "active_substep": None if active_step is None else self._trim_text(active_step.summary, 48),
            "recently_completed_substep": None if recent_step is None else self._trim_text(recent_step.summary, 48),
            "next_substep": None if next_step is None else self._trim_text(next_step.summary, 48),
            "final_followup_pending": state.final_followup_pending,
            "steps": tuple(
                {
                    "step_id": step.step_id,
                    "kind": step.kind,
                    "status": step.status,
                    "summary": self._trim_text(step.summary, 48),
                    "source_detail": self._trim_text(step.source_detail, 32),
                }
                for step in state.steps[:_MAX_COMPOUND_STEPS]
            ),
        }

    @staticmethod
    def _trim_text(value: str, limit: int) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[: max(0, limit - 1)].rstrip()}…"

    @staticmethod
    def _stance_detail(
        *,
        stance: ContinuityStance,
        current: tuple[ContinuityItem, ...],
        blockers: tuple[ContinuityItem, ...],
        unresolved: tuple[ContinuityItem, ...],
        commitments: tuple[ContinuityItem, ...],
    ) -> str:
        if stance == "awaiting_tool" and blockers:
            return blockers[0].detail or blockers[0].summary
        if stance == "awaiting_user" and unresolved:
            return unresolved[0].detail or unresolved[0].summary
        if stance == "assisting_execution" and commitments:
            return commitments[0].detail or commitments[0].summary
        if stance == "assisting_query":
            return "read_query_detected"
        if stance == "idle" and current:
            return f"current={current[0].kind}:{current[0].status}"
        return ""

    def _set_item(self, item: ContinuityItem) -> None:
        self._items[item.id] = item

    def _remove_item(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    @staticmethod
    def _clean_text(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _as_bool(value: object) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes"}

    @staticmethod
    def _priority_rank(priority: ContinuityPriority) -> int:
        return {"high": 0, "medium": 1, "low": 2}[priority]

    @staticmethod
    def _priority_for_text(text: str) -> ContinuityPriority:
        lowered = text.lower()
        if any(token in lowered for token in ("urgent", "now", "immediately")):
            return "high"
        return "medium"

    def _is_action_request(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _ACTION_PATTERNS)

    def _is_observation_request(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _OBSERVATION_PATTERNS)

    def _is_status_check_request(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _STATUS_CHECK_PATTERNS)

    def _is_query_request(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _QUERY_REQUEST_PATTERNS)

    def _tool_needed_by_transcript(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in _TOOL_REQUIRED_PATTERNS)

    def _extract_followup_summary(self, text: str, *, requires_action: bool) -> str | None:
        if not requires_action:
            return None
        if "?" in text:
            parts = re.split(r"\?+", text, maxsplit=1)
            candidate = parts[0].strip()
            return f"{candidate}?" if candidate else None
        for pattern in _FOLLOW_UP_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        return None

    def _stance_for_transcript(self, text: str, *, is_action_request: bool) -> ContinuityStance:
        if is_action_request:
            return "assisting_execution"
        if self._is_observation_request(text):
            return "assisting_observation"
        if self._is_status_check_request(text):
            return "assisting_observation"
        if self._is_query_request(text):
            return "assisting_query"
        if self._tool_needed_by_transcript(text):
            return "awaiting_tool"
        if len(text.split()) <= 1:
            return "recovering_context"
        return "idle"

    def _derive_compound_state(
        self,
        transcript: str,
        *,
        source: str,
        unresolved_summary: str | None,
    ) -> CompoundContinuityState | None:
        clauses = self._split_compound_request(transcript)
        if len(clauses) <= 1:
            return None
        steps: list[CompoundContinuityStep] = []
        followup_recorded = False
        for idx, clause in enumerate(clauses[:_MAX_COMPOUND_STEPS]):
            step = self._build_compound_step(
                clause,
                step_index=idx,
                source=source,
                unresolved_summary=unresolved_summary,
                followup_recorded=followup_recorded,
            )
            if step is None:
                continue
            followup_recorded = followup_recorded or step.kind == "report"
            steps.append(step)
        if len(steps) <= 1:
            return None
        active_step_index = next((idx for idx, step in enumerate(steps) if step.status == "active"), None)
        next_pending_step_id = next((step.step_id for step in steps if step.status == "pending"), None)
        final_followup_pending = any(step.kind == "report" and step.status != "completed" for step in steps)
        return CompoundContinuityState(
            request_id="request:current",
            summary=transcript,
            steps=tuple(steps),
            active_step_index=active_step_index,
            completed_step_ids=(),
            final_followup_pending=final_followup_pending,
            recent_completed_step_id=None,
            next_pending_step_id=next_pending_step_id,
        )

    def _split_compound_request(self, transcript: str) -> tuple[str, ...]:
        normalized = re.sub(r"\s+", " ", transcript).strip()
        coarse_parts = [part.strip(" .") for part in _COMPOUND_SPLIT_RE.split(normalized) if part.strip(" .")]
        parts: list[str] = []
        for part in coarse_parts:
            parts.extend(self._split_on_bare_and(part))
        return tuple(parts[:_MAX_COMPOUND_STEPS])

    def _split_on_bare_and(self, clause: str) -> list[str]:
        normalized = clause.strip(" .")
        if not normalized:
            return []
        for match in re.finditer(r"\band\b", normalized, flags=re.IGNORECASE):
            left = normalized[: match.start()].strip(" ,.")
            right = normalized[match.end() :].strip(" ,.")
            if not left or not right:
                continue
            if self._should_split_bare_and(left, right):
                return [*self._split_on_bare_and(left), *self._split_on_bare_and(right)]
        return [normalized]

    def _should_split_bare_and(self, left: str, right: str) -> bool:
        allowed = {"gesture", "diagnostics", "report"}
        left_kind = self._classify_compound_step_kind(left, unresolved_summary=None)
        right_kind = self._classify_compound_step_kind(right, unresolved_summary=None)
        return left_kind in allowed and right_kind in allowed

    def _build_compound_step(
        self,
        clause: str,
        *,
        step_index: int,
        source: str,
        unresolved_summary: str | None,
        followup_recorded: bool,
    ) -> CompoundContinuityStep | None:
        normalized = clause.strip()
        if not normalized:
            return None
        kind = self._classify_compound_step_kind(normalized, unresolved_summary=unresolved_summary)
        if kind == "report" and followup_recorded:
            return None
        return CompoundContinuityStep(
            step_id=f"step_{step_index + 1}",
            kind=kind,
            summary=normalized if normalized.endswith((".", "?")) else f"{normalized}.",
            status="active" if step_index == 0 else "pending",
            source_detail=f"origin={source}",
        )

    def _classify_compound_step_kind(self, clause: str, *, unresolved_summary: str | None) -> CompoundStepKind:
        lowered = clause.lower().strip()
        if unresolved_summary and (lowered in unresolved_summary.lower() or _FOLLOWUP_ONLY_PREFIX_RE.search(clause)):
            return "report"
        if _DIAGNOSTIC_STEP_RE.search(clause):
            return "diagnostics"
        if _REPORT_STEP_RE.search(clause):
            return "report"
        if _OBSERVATION_STEP_RE.search(clause):
            return "observation"
        if _GESTURE_STEP_RE.search(clause):
            return "gesture"
        return "followup"
