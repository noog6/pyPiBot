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
_STRUCTURED_SAY_PATTERN = r"say\s+(?:whether|what|if|when|where)\b"
_FOLLOWUP_ONLY_PREFIX_RE = re.compile(
    rf"^(?:then\s+)?(?:tell me|let me know|report back|report|{_STRUCTURED_SAY_PATTERN})",
    re.IGNORECASE,
)
_SILENT_RE = re.compile(r"\bsilently\b", re.IGNORECASE)
_DIAGNOSTIC_STEP_RE = re.compile(
    r"\b(?:"
    r"(?:check|run|read|query)\b.*\b(?:diagnostic|diagnostics|status|battery|temperature|pressure|environment)\b"
    r"|"
    r"(?:take|get|do)\b.*\bdiagnostic(?:s)?\b.*\b(?:reading|check)\b"
    r")",
    re.IGNORECASE,
)
_REPORT_STEP_RE = re.compile(
    rf"\b(?:tell me|let me know|report back|report|{_STRUCTURED_SAY_PATTERN})",
    re.IGNORECASE,
)
_OBSERVATION_STEP_RE = re.compile(r"\b(?:observe|describe|identify|what do you see|what(?:'s| is) in|do you see|can you see)\b", re.IGNORECASE)
_GESTURE_STEP_RE = re.compile(
    r"\b(?:look(?:ing)?|move|center|turn(?:ing)?|rotate|point(?:ing)?|go to|navigate|back to center|return to center|come back to center|left|right|up|down)\b",
    re.IGNORECASE,
)
_GESTURE_DIRECTION_ONLY_STEP_RE = re.compile(
    r"^(?:to\s+the\s+)?(?:left|right|up|down|center|centre|middle)\.?$",
    re.IGNORECASE,
)
_REPORT_STATUS_TOKENS = ("done", "finished", "ready", "centered", "complete", "completed", "settled")
_REPORT_VISUAL_CUE_TOKENS = (
    "see",
    "look",
    "what do you see",
    "what is it",
    "what's that",
    "holding",
    "what am i holding",
    "what i'm holding",
    "in my hand",
    "object",
    "image",
    "photo",
    "camera",
    "scene",
)
_REPORT_AUDITORY_CUE_TOKENS = ("hear", "sound", "audio", "noise", "listening")
_REPORT_DESCRIBE_CUES = ("describe", "what do you see", "what you see", "what's there", "what is there")
_REPORT_IDENTIFY_CUES = ("identify", "what is it", "what's that", "what am i holding", "what i'm holding", "in my hand")
_REPORT_VERIFY_CUES = ("whether", "can you see", "do you see", "can you hear", "do you hear", "is it", "are you")
_STRUCTURED_SAY_RE = re.compile(rf"\b{_STRUCTURED_SAY_PATTERN}", re.IGNORECASE)
_REQUEST_INTENT_RE = re.compile(
    r"\b(?:please|can you|could you|would you|will you|tell me|let me know|show me)\b",
    re.IGNORECASE,
)
_GESTURE_DIRECTION_CENTER_RE = re.compile(
    r"\b(?:center|centre|middle|straight ahead|back to center|return to center|come back to center)\b",
    re.IGNORECASE,
)
_GESTURE_DIRECTION_LEFT_RE = re.compile(r"\bleft\b", re.IGNORECASE)
_GESTURE_DIRECTION_RIGHT_RE = re.compile(r"\bright\b", re.IGNORECASE)
_GESTURE_DIRECTION_UP_RE = re.compile(r"\bup\b", re.IGNORECASE)
_GESTURE_DIRECTION_DOWN_RE = re.compile(r"\bdown\b", re.IGNORECASE)


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
    requires_perception: bool = False
    perception_mode: Literal["visual", "auditory", "external_unknown", ""] = ""
    report_intent: Literal["describe", "identify", "verify", "status", ""] = ""
    implicit_observation_required: bool = False


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


@dataclass(frozen=True)
class DeterministicFollowthroughStepDescriptor:
    """Authoritative descriptor for runtime-executable deterministic gesture steps."""

    request_id: str
    step_id: str
    tool_name: str
    tool_args: tuple[tuple[str, str], ...]


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
        self._compound_owner_turn_id: str | None = None
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

    def compound_owner_turn_id(self) -> str:
        """Return the active compound request owner turn, if any."""
        return self._clean_text(self._compound_owner_turn_id)

    def compound_has_open_non_report_steps(self) -> bool:
        """Return whether the active compound request has open non-report steps."""
        state = self._compound_state
        if state is None:
            return False
        return self._has_open_non_report_steps(state)

    def deterministic_followthrough_step(self) -> DeterministicFollowthroughStepDescriptor | None:
        """Return a deterministic runtime descriptor for the active gesture followthrough step.

        This remains bookkeeping-derived metadata. It is intentionally narrow and
        only emits descriptors for unambiguous low-risk gesture directions.
        """

        state = self._compound_state
        if state is None:
            return None
        active_idx = state.active_step_index
        if not isinstance(active_idx, int) or not (0 <= active_idx < len(state.steps)):
            return None
        step = state.steps[active_idx]
        if step.kind != "gesture":
            return None
        if bool(step.requires_perception):
            return None
        direction = self._gesture_direction_from_text(step.summary)
        tool_name = {
            "left": "gesture_look_left",
            "right": "gesture_look_right",
            "up": "gesture_look_up",
            "down": "gesture_look_down",
            "center": "gesture_look_center",
        }.get(direction)
        if not tool_name:
            return None
        return DeterministicFollowthroughStepDescriptor(
            request_id=state.request_id,
            step_id=step.step_id,
            tool_name=tool_name,
            tool_args=(),
        )

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
        log_method = logger.debug if reminder_due else logger.info
        log_method(
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
                f"{step.step_id}:{step.kind}:{step.status}:{int(step.requires_perception)}:{step.perception_mode}:{step.report_intent}:{self._trim_text(step.summary, 32)}"
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
        owner_turn_id = self._clean_text(payload.get("turn_id"))
        if not transcript:
            self._stance = "recovering_context"
            self._compound_state = None
            self._compound_owner_turn_id = None
            return

        is_action_request = self._is_action_request(transcript)
        followup_summary = self._extract_followup_summary(transcript, requires_action=is_action_request)
        if self._is_compound_request_candidate(transcript, is_action_request=is_action_request):
            self._compound_state = self._derive_compound_state(
                transcript,
                source=source,
                unresolved_summary=followup_summary,
            )
            self._compound_owner_turn_id = owner_turn_id or None
        else:
            self._compound_state = None
            self._compound_owner_turn_id = None

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
                    detail=self._append_owner_turn_detail("origin=user_request", owner_turn_id=owner_turn_id),
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
        owner_turn_id = self._clean_text(payload.get("turn_id"))
        allow_rebind = self._as_bool(payload.get("allow_cross_turn_rebind"))
        rebind_reason = self._clean_text(payload.get("cross_turn_rebind_reason")) or "unspecified"
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
        commitment_owner = self._owner_turn_id_for_item_id("commitment:current")
        can_update_commitment = (
            not commitment_owner
            or not owner_turn_id
            or commitment_owner == owner_turn_id
            or allow_rebind
        )
        if commitment_summary and can_update_commitment:
            self._set_item(
                ContinuityItem(
                    id="commitment:current",
                    kind="commitment",
                    summary=commitment_summary,
                    status="active",
                    priority="medium",
                    source="tool_call_started",
                    detail=self._append_owner_turn_detail(
                        f"origin=tool_followthrough tool={tool_name} call_id={call_id}",
                        owner_turn_id=owner_turn_id,
                    ),
                    expires_after_turns=4,
                )
            )
        elif commitment_summary and not can_update_commitment:
            logger.info(
                "continuity_commitment_owner_mismatch event=tool_call_started owner_turn_id=%s incoming_turn_id=%s action=preserve_existing reason=owner_turn_mismatch",
                commitment_owner or "",
                owner_turn_id or "",
            )
        self._advance_compound_state_on_tool_start(
            tool_name=tool_name,
            turn_id=owner_turn_id,
            allow_cross_turn_rebind=allow_rebind,
            rebind_reason=rebind_reason,
        )
        self._stance = "awaiting_tool"

    def _apply_tool_result_received(self, payload: dict[str, object]) -> None:
        call_id = self._clean_text(payload.get("call_id"))
        tool_name = self._clean_text(payload.get("tool_name"))
        owner_turn_id = self._clean_text(payload.get("turn_id"))
        allow_rebind = self._as_bool(payload.get("allow_cross_turn_rebind"))
        rebind_reason = self._clean_text(payload.get("cross_turn_rebind_reason")) or "unspecified"
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
        self._advance_compound_state_on_tool_result(
            tool_name=tool_name,
            turn_id=owner_turn_id,
            allow_cross_turn_rebind=allow_rebind,
            rebind_reason=rebind_reason,
        )
        self._stance = "idle"

    def _apply_response_done(self, payload: dict[str, object]) -> None:
        close_ongoing = self._as_bool(payload.get("close_ongoing"))
        close_commitment = self._as_bool(payload.get("close_commitment"))
        close_unresolved = self._as_bool(payload.get("close_unresolved"))
        complete_final_report = self._as_bool(payload.get("complete_final_report"))
        keep_ongoing = self._as_bool(payload.get("keep_ongoing"))
        owner_turn_id = self._clean_text(payload.get("turn_id"))
        allow_rebind = self._as_bool(payload.get("allow_cross_turn_rebind"))
        rebind_reason = self._clean_text(payload.get("cross_turn_rebind_reason")) or "unspecified"

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
            complete_final_report=complete_final_report,
            turn_id=owner_turn_id,
            allow_cross_turn_rebind=allow_rebind,
            rebind_reason=rebind_reason,
        )
        self._stance = "idle" if close_ongoing or close_commitment or close_unresolved or complete_final_report else "awaiting_user"

    def _advance_compound_state_on_tool_start(
        self,
        *,
        tool_name: str,
        turn_id: str,
        allow_cross_turn_rebind: bool,
        rebind_reason: str,
    ) -> None:
        state = self._compound_state
        if state is None:
            return
        if not self._ensure_compound_owner(
            turn_id=turn_id,
            allow_cross_turn_rebind=allow_cross_turn_rebind,
            reason=rebind_reason,
            event_type="tool_call_started",
        ):
            return
        active_idx = state.active_step_index
        normalized_tool = tool_name.lower()
        if active_idx is not None and active_idx < len(state.steps):
            active_step = state.steps[active_idx]
            if active_step.kind == "diagnostics" or (
                active_step.kind == "gesture" and normalized_tool.startswith(_TOOL_COMMITMENT_PREFIXES)
            ):
                return
            if active_step.kind in {"followup", "observation"}:
                return
        idx = self._first_pending_step_index(state)
        if idx is None:
            return
        step = state.steps[idx]
        if step.kind == "report":
            return
        if step.kind == "diagnostics" or normalized_tool.startswith(_TOOL_COMMITMENT_PREFIXES):
            self._compound_state = self._replace_compound_step_status(state, idx, "active")

    def _advance_compound_state_on_tool_result(
        self,
        *,
        tool_name: str,
        turn_id: str,
        allow_cross_turn_rebind: bool,
        rebind_reason: str,
    ) -> None:
        state = self._compound_state
        if state is None:
            return
        if not self._ensure_compound_owner(
            turn_id=turn_id,
            allow_cross_turn_rebind=allow_cross_turn_rebind,
            reason=rebind_reason,
            event_type="tool_result_received",
        ):
            return
        idx = state.active_step_index
        if idx is None:
            idx = self._first_pending_step_index(state)
            if idx is None:
                return
        step = state.steps[idx]
        normalized_tool = tool_name.lower()
        gesture_direction_match = self._gesture_direction_matches_step(step.summary, normalized_tool)
        clear_match = (
            step.kind == "diagnostics"
            or (
                step.kind == "gesture"
                and normalized_tool.startswith(_TOOL_COMMITMENT_PREFIXES)
                and gesture_direction_match
            )
        )
        if clear_match:
            self._compound_state = self._replace_compound_step_status(state, idx, "completed")

    def _advance_compound_state_on_response_done(
        self,
        *,
        close_commitment: bool,
        close_unresolved: bool,
        complete_final_report: bool,
        turn_id: str,
        allow_cross_turn_rebind: bool,
        rebind_reason: str,
    ) -> None:
        state = self._compound_state
        if state is None:
            return
        if not self._ensure_compound_owner(
            turn_id=turn_id,
            allow_cross_turn_rebind=allow_cross_turn_rebind,
            reason=rebind_reason,
            event_type="response_done",
        ):
            return
        updated = state
        if close_commitment and not updated.completed_step_ids:
            idx = updated.active_step_index
            if idx is None:
                idx = self._first_pending_step_index(updated, include_report=False)
            if idx is not None:
                step = updated.steps[idx]
                if step.kind != "report":
                    updated = self._replace_compound_step_status(updated, idx, "completed")
        if (
            complete_final_report
            and updated.final_followup_pending
            and not self._has_open_non_report_steps(updated)
        ):
            idx = self._first_pending_step_index(updated, kinds={"report"})
            if idx is not None:
                updated = self._replace_compound_step_status(updated, idx, "completed")
        self._compound_state = None if self._compound_state_resolved(updated) else updated
        if self._compound_state is None:
            self._compound_owner_turn_id = None

    def _ensure_compound_owner(
        self,
        *,
        turn_id: str,
        allow_cross_turn_rebind: bool,
        reason: str,
        event_type: str,
    ) -> bool:
        owner_turn_id = str(self._compound_owner_turn_id or "").strip()
        incoming_turn_id = str(turn_id or "").strip()
        if not owner_turn_id or not incoming_turn_id or owner_turn_id == incoming_turn_id:
            if allow_cross_turn_rebind and owner_turn_id and incoming_turn_id and owner_turn_id == incoming_turn_id:
                logger.info(
                    "continuity_compound_owner_rebind_accepted event=%s owner_turn_id=%s incoming_turn_id=%s action=noop reason=%s",
                    event_type,
                    owner_turn_id,
                    incoming_turn_id,
                    reason or "unspecified",
                )
            return True
        if allow_cross_turn_rebind:
            logger.info(
                "continuity_compound_owner_rebound event=%s old_owner_turn_id=%s new_owner_turn_id=%s reason=%s",
                event_type,
                owner_turn_id,
                incoming_turn_id,
                reason or "unspecified",
            )
            self._compound_owner_turn_id = incoming_turn_id
            return True
        logger.info(
            "continuity_compound_owner_mismatch event=%s owner_turn_id=%s incoming_turn_id=%s action=blocked reason=owner_turn_mismatch",
            event_type,
            owner_turn_id,
            incoming_turn_id,
        )
        return False

    def _append_owner_turn_detail(self, detail: str, *, owner_turn_id: str) -> str:
        normalized_detail = self._clean_text(detail)
        normalized_owner = self._clean_text(owner_turn_id)
        if not normalized_owner:
            return normalized_detail
        if "owner_turn_id=" in normalized_detail:
            return normalized_detail
        return f"{normalized_detail} owner_turn_id={normalized_owner}".strip()

    def _owner_turn_id_for_item_id(self, item_id: str) -> str:
        item = self._items.get(item_id)
        if item is None:
            return ""
        detail = str(getattr(item, "detail", "") or "")
        match = re.search(r"\bowner_turn_id=([^\s]+)", detail)
        if match is None:
            return ""
        return self._clean_text(match.group(1))

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

    def _has_open_non_report_steps(self, state: CompoundContinuityState) -> bool:
        return any(step.kind != "report" and step.status != "completed" for step in state.steps)

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
                    "requires_perception": step.requires_perception,
                    "perception_mode": step.perception_mode,
                    "report_intent": step.report_intent,
                    "implicit_observation_required": step.implicit_observation_required,
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

    def _is_compound_request_candidate(self, text: str, *, is_action_request: bool) -> bool:
        if is_action_request:
            return True
        if self._is_observation_request(text):
            return True
        if self._is_query_request(text):
            return True
        if self._tool_needed_by_transcript(text):
            return True
        if self._is_explicit_followthrough_request(text):
            return True
        return False

    def _is_explicit_followthrough_request(self, text: str) -> bool:
        if _FOLLOWUP_ONLY_PREFIX_RE.search(text):
            return True
        return bool(_REQUEST_INTENT_RE.search(text))

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
                prior_steps=tuple(steps),
            )
            if step is None:
                continue
            if (
                not steps
                and step.kind == "followup"
                and len(clauses) > 1
                and not self._is_explicit_followthrough_request(clause)
            ):
                continue
            followup_recorded = followup_recorded or step.kind == "report"
            normalized_index = len(steps)
            if normalized_index != idx:
                step = replace(step, step_id=f"step_{normalized_index + 1}", status="active" if normalized_index == 0 else "pending")
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
        prior_steps: tuple[CompoundContinuityStep, ...],
    ) -> CompoundContinuityStep | None:
        normalized = clause.strip()
        if not normalized:
            return None
        kind = self._classify_compound_step_kind(normalized, unresolved_summary=unresolved_summary)
        if kind == "report" and followup_recorded:
            return None
        report_traits = self._classify_report_semantics(
            clause=normalized,
            kind=kind,
            prior_steps=prior_steps,
        )
        return CompoundContinuityStep(
            step_id=f"step_{step_index + 1}",
            kind=kind,
            summary=normalized if normalized.endswith((".", "?")) else f"{normalized}.",
            status="active" if step_index == 0 else "pending",
            source_detail=f"origin={source}",
            requires_perception=bool(report_traits["requires_perception"]),
            perception_mode=report_traits["perception_mode"],
            report_intent=report_traits["report_intent"],
            implicit_observation_required=bool(report_traits["requires_perception"])
            and report_traits["perception_mode"] == "visual",
        )

    def _classify_report_semantics(
        self,
        *,
        clause: str,
        kind: CompoundStepKind,
        prior_steps: tuple[CompoundContinuityStep, ...],
    ) -> dict[str, str | bool]:
        if kind != "report":
            return {"requires_perception": False, "perception_mode": "", "report_intent": ""}

        normalized_clause = clause.lower().strip()
        normalized = f" {normalized_clause} "
        status_only = self._is_status_only_report_clause(normalized_clause)
        has_gesture_context = any(step.kind == "gesture" for step in prior_steps)
        has_structured_say = bool(_STRUCTURED_SAY_RE.search(normalized_clause))
        has_report_verb = any(token in normalized for token in ("tell me", "let me know", "report", "describe", "identify", "verify")) or has_structured_say
        has_visual_cue = any(token in normalized for token in _REPORT_VISUAL_CUE_TOKENS)
        has_auditory_cue = any(token in normalized for token in _REPORT_AUDITORY_CUE_TOKENS)

        report_intent: Literal["describe", "identify", "verify", "status", ""] = ""
        if any(token in normalized for token in _REPORT_IDENTIFY_CUES):
            report_intent = "identify"
        elif any(token in normalized for token in _REPORT_DESCRIBE_CUES):
            report_intent = "describe"
        elif "verify" in normalized or any(token in normalized for token in _REPORT_VERIFY_CUES):
            report_intent = "verify"
        elif status_only:
            report_intent = "status"

        requires_perception = False
        perception_mode: Literal["visual", "auditory", "external_unknown", ""] = ""
        if not status_only and has_report_verb and has_auditory_cue:
            requires_perception = True
            perception_mode = "auditory"
        elif not status_only and has_report_verb and has_visual_cue:
            requires_perception = True
            perception_mode = "visual"
        elif not status_only and has_report_verb and has_gesture_context and ("whether" in normalized or "what" in normalized):
            requires_perception = True
            perception_mode = "external_unknown"

        if report_intent == "" and has_report_verb and not requires_perception:
            report_intent = "status"
        return {
            "requires_perception": requires_perception,
            "perception_mode": perception_mode,
            "report_intent": report_intent,
        }

    def _is_status_only_report_clause(self, normalized_clause: str) -> bool:
        cleaned = normalized_clause.strip(" .?!")
        if not cleaned:
            return False
        normalized = f" {cleaned} "
        if any(token in normalized for token in _REPORT_IDENTIFY_CUES):
            return False
        if any(token in normalized for token in _REPORT_DESCRIBE_CUES):
            return False
        if any(token in normalized for token in _REPORT_VERIFY_CUES):
            return False
        status_terms = "|".join(re.escape(token) for token in _REPORT_STATUS_TOKENS)
        direct_status = re.fullmatch(
            rf"(?:report|say)\s+(?:(?:that|when|once)\s+)?(?:(?:it is|it's|you're|you are)\s+)?(?:{status_terms})",
            cleaned,
        )
        notify_status = re.fullmatch(
            rf"(?:tell me|let me know|report)\s+(?:when|once)\s+(?:(?:it is|it's|you're|you are)\s+)?(?:{status_terms})",
            cleaned,
        )
        return bool(direct_status or notify_status)

    def _classify_compound_step_kind(self, clause: str, *, unresolved_summary: str | None) -> CompoundStepKind:
        lowered = clause.lower().strip()
        unresolved_match = False
        if unresolved_summary:
            unresolved_match = lowered == unresolved_summary.lower().strip()
        if unresolved_match or _FOLLOWUP_ONLY_PREFIX_RE.search(clause):
            return "report"
        if _DIAGNOSTIC_STEP_RE.search(clause):
            return "diagnostics"
        if _REPORT_STEP_RE.search(clause):
            return "report"
        if _OBSERVATION_STEP_RE.search(clause):
            return "observation"
        if _GESTURE_DIRECTION_ONLY_STEP_RE.fullmatch(lowered.strip(" .?!")):
            return "gesture"
        if _GESTURE_STEP_RE.search(clause):
            return "gesture"
        return "followup"

    def _gesture_direction_matches_step(self, step_summary: str, tool_name: str) -> bool:
        expected = self._gesture_direction_from_text(step_summary)
        actual = self._gesture_direction_from_tool_name(tool_name)
        if expected == "" or actual == "":
            return True
        return expected == actual

    def _gesture_direction_from_text(self, text: str) -> str:
        normalized = text.lower().strip()
        if _GESTURE_DIRECTION_CENTER_RE.search(normalized):
            return "center"
        if _GESTURE_DIRECTION_LEFT_RE.search(normalized):
            return "left"
        if _GESTURE_DIRECTION_RIGHT_RE.search(normalized):
            return "right"
        if _GESTURE_DIRECTION_UP_RE.search(normalized):
            return "up"
        if _GESTURE_DIRECTION_DOWN_RE.search(normalized):
            return "down"
        return ""

    def _gesture_direction_from_tool_name(self, tool_name: str) -> str:
        normalized = tool_name.lower().strip()
        if "center" in normalized or "centre" in normalized:
            return "center"
        if "left" in normalized:
            return "left"
        if "right" in normalized:
            return "right"
        if "up" in normalized:
            return "up"
        if "down" in normalized:
            return "down"
        return ""
