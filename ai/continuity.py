"""Deterministic presence/continuity bookkeeping seam."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import re
from typing import Literal

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
ContinuityStance = Literal[
    "idle",
    "assisting_observation",
    "assisting_execution",
    "awaiting_user",
    "awaiting_tool",
    "awaiting_perception",
    "recovering_context",
]

_MAX_ITEMS_PER_BUCKET = 3
_TOOL_COMMITMENT_PREFIXES = ("gesture_", "move", "look", "center", "turn", "navigate")
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


@dataclass(frozen=True)
class ContinuityItem:
    id: str
    kind: ContinuityKind
    summary: str
    status: ContinuityStatus
    priority: ContinuityPriority
    source: str
    expires_after_turns: int | None = None


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
    generated_reason: str = ""


class ContinuityLedger:
    """Small in-memory continuity ledger.

    This seam only records deterministic bookkeeping. It does not arbitrate,
    schedule, or execute behavior.
    """

    def __init__(self) -> None:
        self._stance: ContinuityStance = "idle"
        self._items: dict[str, ContinuityItem] = {}

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

    def build_brief(self, run_id: str, turn_id: str, reason: str) -> ContinuityBrief:
        brief = ContinuityBrief(
            run_id=run_id,
            turn_id=turn_id,
            stance=self._stance,
            ongoing=self._bucket("ongoing", {"active", "pending"}),
            unresolved=self._bucket("unresolved", {"active", "pending", "blocked"}),
            commitments=self._bucket("commitment", {"active", "pending", "blocked"}),
            blockers=self._bucket("blocker", {"active", "pending", "blocked"}),
            constraints=self._bucket("constraint", {"active", "pending", "blocked"}),
            recently_closed=self._bucket("recently_closed", {"resolved"}),
            generated_reason=str(reason or "").strip(),
        )
        logger.info(
            "continuity_brief_built run_id=%s turn_id=%s stance=%s ongoing=%s blockers=%s commitments=%s closed=%s reason=%s",
            run_id,
            turn_id,
            brief.stance,
            len(brief.ongoing),
            len(brief.blockers),
            len(brief.commitments),
            len(brief.recently_closed),
            brief.generated_reason or "none",
        )
        return brief

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
            return

        is_action_request = self._is_action_request(transcript)
        followup_summary = self._extract_followup_summary(transcript, requires_action=is_action_request)

        self._set_item(
            ContinuityItem(
                id="request:current",
                kind="ongoing",
                summary=transcript,
                status="active",
                priority=self._priority_for_text(transcript),
                source=source,
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
                    expires_after_turns=4,
                )
            )
        self._stance = "awaiting_tool"

    def _apply_tool_result_received(self, payload: dict[str, object]) -> None:
        call_id = self._clean_text(payload.get("call_id"))
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
        self._stance = "idle" if close_ongoing or close_commitment or close_unresolved else "awaiting_user"

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
        if self._tool_needed_by_transcript(text):
            return "awaiting_tool"
        if len(text.split()) <= 1:
            return "recovering_context"
        return "idle"
