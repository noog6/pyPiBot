"""Governance layer for tool execution and approvals."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import re
import time
from typing import Any, Iterable

from core.logging import log_info


@dataclass(frozen=True)
class ToolSpec:
    tier: int
    reversible: bool
    cost_hint: str
    safety_tags: tuple[str, ...] = ()
    confirm_required: bool | None = None
    confirm_reason: str | None = None
    confirm_prompt: str | None = None
    cooldown_seconds: float = 0.0
    dry_run_supported: bool = False
    governance_tier: str = "GUARDED"
    side_effects: str = "UNKNOWN"
    sensitivity: str = "INTERNAL"
    default_confirmation: str = "ASK"


@dataclass
class ActionPacket:
    id: str
    tool_name: str
    tool_args: dict[str, Any]
    tier: int
    what: str
    why: str
    impact: str
    rollback: str
    alternatives: list[str]
    confidence: float
    cost: str
    risk_flags: list[str]
    requires_confirmation: bool
    expiry_ts: float | None = None

    def summary(self) -> str:
        return (
            f"tool={self.tool_name} tier={self.tier} cost={self.cost} "
            f"confidence={self.confidence:.2f} requires_confirmation={self.requires_confirmation}"
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tier": self.tier,
            "what": self.what,
            "why": self.why,
            "impact": self.impact,
            "rollback": self.rollback,
            "alternatives": list(self.alternatives),
            "confidence": self.confidence,
            "cost": self.cost,
            "risk_flags": list(self.risk_flags),
            "requires_confirmation": self.requires_confirmation,
            "expiry_ts": self.expiry_ts,
        }


@dataclass(frozen=True)
class GovernanceDecision:
    status: str
    reason: str
    action_summary: str | None = None
    confirm_required: bool = False
    confirm_reason: str | None = None
    confirm_prompt: str | None = None
    idempotency_key: str | None = None
    cooldown_seconds: float = 0.0
    dry_run_supported: bool = False
    decision_source: str = "tier_default"
    thresholds: dict[str, Any] | None = None

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    @property
    def needs_confirmation(self) -> bool:
        return self.status == "needs_confirmation" or self.confirm_required

    @property
    def denied(self) -> bool:
        return self.status == "denied"


class GovernanceReason(str, Enum):
    AUTONOMY_OBSERVE_ONLY = "autonomy_observe_only"
    TOOL_CALL_BUDGET_EXHAUSTED = "tool_call_budget_exhausted"
    EXPENSIVE_CALL_BUDGET_EXHAUSTED = "expensive_call_budget_exhausted"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"
    AUTONOMY_WINDOW_CLOSED = "autonomy_window_closed"
    AUTONOMY_LEVEL_REQUIRES_CONFIRMATION = "autonomy_level_requires_confirmation"
    TIER1_GUARDED_THRESHOLD_EXCEEDED = "tier1_guarded_threshold_exceeded"
    WITHIN_BOUNDS = "within_bounds"
    DRY_RUN_NOT_SUPPORTED = "dry_run_not_supported"
    TOOL_EXECUTION_COOLDOWN = "tool_execution_cooldown"


_REASON_NORMALIZATION_MAP: dict[str, GovernanceReason] = {
    "autonomy dial set to observe-only": GovernanceReason.AUTONOMY_OBSERVE_ONLY,
    "tool-call budget exhausted": GovernanceReason.TOOL_CALL_BUDGET_EXHAUSTED,
    "expensive-call budget exhausted": GovernanceReason.EXPENSIVE_CALL_BUDGET_EXHAUSTED,
    "risk threshold exceeded": GovernanceReason.RISK_THRESHOLD_EXCEEDED,
    "autonomy window closed for tiered action": GovernanceReason.AUTONOMY_WINDOW_CLOSED,
    "autonomy level requires confirmation": GovernanceReason.AUTONOMY_LEVEL_REQUIRES_CONFIRMATION,
    "tier-1 guarded runtime thresholds exceeded": GovernanceReason.TIER1_GUARDED_THRESHOLD_EXCEEDED,
    "within bounds": GovernanceReason.WITHIN_BOUNDS,
    "dry-run not supported for this tool": GovernanceReason.DRY_RUN_NOT_SUPPORTED,
}


def normalize_governance_reason(reason: str) -> str:
    lowered = str(reason or "").strip().lower()
    if lowered.startswith("tool execution paused for"):
        return GovernanceReason.TOOL_EXECUTION_COOLDOWN.value
    normalized = _REASON_NORMALIZATION_MAP.get(lowered)
    if normalized is not None:
        return normalized.value
    stable = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return stable or "unknown"


def normalized_decision_payload(
    decision: GovernanceDecision,
    *,
    action_summary_fallback: str | None = None,
) -> dict[str, Any]:
    confirm_reason = decision.confirm_reason
    if confirm_reason is None and decision.confirm_required:
        confirm_reason = decision.reason
    normalized_confirm_reason = (
        normalize_governance_reason(confirm_reason) if confirm_reason is not None else None
    )
    action_summary = " ".join(
        str(decision.action_summary or action_summary_fallback or "Action requires confirmation.").split()
    )
    if not action_summary.endswith("."):
        action_summary = f"{action_summary}."
    return {
        "action_summary": action_summary,
        "confirm_required": decision.needs_confirmation,
        "confirm_reason": normalized_confirm_reason,
        "confirm_prompt": decision.confirm_prompt,
        "idempotency_key": decision.idempotency_key,
        "cooldown_seconds": float(decision.cooldown_seconds or 0.0),
        "dry_run_supported": bool(decision.dry_run_supported),
        "decision_source": str(getattr(decision, "decision_source", "tier_default") or "tier_default"),
        "thresholds": dict(getattr(decision, "thresholds", None) or {}),
        "max_reminders": getattr(decision, "max_reminders", None),
        "reminder_schedule_seconds": getattr(decision, "reminder_schedule_seconds", None),
    }


_ARG_WHITESPACE_RE = re.compile(r"\s+")


def normalize_tool_argument_value(value: Any) -> Any:
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return ""
        return _ARG_WHITESPACE_RE.sub("", trimmed)
    if isinstance(value, dict):
        return {str(key): normalize_tool_argument_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [normalize_tool_argument_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_tool_argument_value(item) for item in value]
    return value


def normalize_tool_arguments(args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    return {str(key): normalize_tool_argument_value(value) for key, value in args.items()}


def build_normalized_idempotency_key(tool_name: str, args: dict[str, Any]) -> str:
    normalized_args = normalize_tool_arguments(args)
    payload = json.dumps(
        {"tool": str(tool_name), "args": normalized_args},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{tool_name}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class AutonomyWindowSpec:
    name: str
    start_minutes: int
    duration_s: float
    allowed_tiers: tuple[int, ...]
    enabled: bool = True


@dataclass
class AutonomyWindowState:
    name: str
    opened_at: float
    expires_at: float
    allowed_tiers: tuple[int, ...]
    source: str

    def summary(self) -> str:
        opened = datetime.fromtimestamp(self.opened_at).isoformat(timespec="seconds")
        expires = datetime.fromtimestamp(self.expires_at).isoformat(timespec="seconds")
        return (
            f"name={self.name} source={self.source} opened_at={opened} "
            f"expires_at={expires} allowed_tiers={list(self.allowed_tiers)}"
        )


class BudgetTracker:
    def __init__(self, limit: int, window_s: float) -> None:
        self.limit = limit
        self.window_s = window_s
        self.timestamps: deque[float] = deque()

    def allow(self, now: float) -> bool:
        if self.limit <= 0:
            return True
        cutoff = now - self.window_s
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        return len(self.timestamps) < self.limit

    def record(self, now: float) -> None:
        if self.limit <= 0:
            return
        self.timestamps.append(now)


class GovernanceLayer:
    def __init__(self, tool_specs: dict[str, ToolSpec], config: dict[str, Any]) -> None:
        self._tool_specs = dict(tool_specs)
        governance_cfg = config.get("governance") or {}
        self._autonomy_level = str(
            governance_cfg.get("autonomy_level", "act-with-bounds")
        ).lower()
        budgets = governance_cfg.get("budgets") or {}
        self._tool_calls_budget = BudgetTracker(
            int(budgets.get("tool_calls_per_minute", 0)),
            60.0,
        )
        self._expensive_budget = BudgetTracker(
            int(budgets.get("expensive_calls_per_day", 0)),
            60.0 * 60.0 * 24.0,
        )
        self._risk_threshold = float(governance_cfg.get("risk_threshold", 0.6))
        guarded_thresholds = governance_cfg.get("guarded_thresholds") or {}
        self._guarded_max_cost_score = float(guarded_thresholds.get("max_cost_score", 0.8))
        self._guarded_min_rate_limit_remaining = int(
            guarded_thresholds.get("min_rate_limit_remaining", 2)
        )
        self._guarded_privacy_flag_requires_confirmation = bool(
            guarded_thresholds.get("privacy_flag_requires_confirmation", True)
        )
        self._scheduled_windows = self._load_autonomy_windows(
            governance_cfg.get("autonomy_windows") or []
        )
        self._active_window: AutonomyWindowState | None = None
        self._last_window_name: str | None = None

    def _default_spec(self) -> ToolSpec:
        return ToolSpec(
            tier=2,
            reversible=False,
            cost_hint="med",
            safety_tags=("unclassified",),
            confirm_required=None,
            confirm_reason=None,
            confirm_prompt=None,
            cooldown_seconds=0.0,
            dry_run_supported=False,
        )

    def _idempotency_key(self, action: ActionPacket) -> str:
        return build_normalized_idempotency_key(action.tool_name, action.tool_args)

    def decide_tool_call(
        self,
        action: ActionPacket,
        *,
        dry_run_requested: bool = False,
        runtime_cooldown_seconds: float = 0.0,
        runtime_context: dict[str, Any] | None = None,
    ) -> GovernanceDecision:
        decision = self.review_with_runtime_context(action, runtime_context=runtime_context)
        spec = self._tool_specs.get(action.tool_name, self._default_spec())
        confirm_required = (
            bool(spec.confirm_required)
            if spec.confirm_required is not None
            else decision.needs_confirmation
        )
        confirm_reason = spec.confirm_reason or (decision.reason if confirm_required else None)
        if confirm_reason is not None:
            confirm_reason = normalize_governance_reason(confirm_reason)
        confirm_prompt = spec.confirm_prompt
        cooldown_seconds = max(0.0, float(runtime_cooldown_seconds), float(spec.cooldown_seconds))
        dry_run_supported = bool(spec.dry_run_supported)

        if dry_run_requested and not dry_run_supported:
            return GovernanceDecision(
                status="denied",
                reason=GovernanceReason.DRY_RUN_NOT_SUPPORTED.value,
                confirm_required=False,
                idempotency_key=self._idempotency_key(action),
                cooldown_seconds=cooldown_seconds,
                dry_run_supported=dry_run_supported,
                decision_source="session_override",
                thresholds=dict(getattr(decision, "thresholds", None) or {}),
            )

        if cooldown_seconds > 0:
            return GovernanceDecision(
                status="denied",
                reason=GovernanceReason.TOOL_EXECUTION_COOLDOWN.value,
                confirm_required=False,
                idempotency_key=self._idempotency_key(action),
                cooldown_seconds=cooldown_seconds,
                dry_run_supported=dry_run_supported,
                decision_source="session_override",
                thresholds=dict(getattr(decision, "thresholds", None) or {}),
            )

        if decision.status == "needs_confirmation":
            confirm_required = True

        return GovernanceDecision(
            status=decision.status,
            reason=normalize_governance_reason(decision.reason),
            action_summary=action.summary(),
            confirm_required=confirm_required,
            confirm_reason=confirm_reason,
            confirm_prompt=confirm_prompt,
            idempotency_key=self._idempotency_key(action),
            cooldown_seconds=cooldown_seconds,
            dry_run_supported=dry_run_supported,
            decision_source=str(getattr(decision, "decision_source", "tier_default") or "tier_default"),
            thresholds=dict(getattr(decision, "thresholds", None) or {}),
        )

    @staticmethod
    def coerce_decision_payload(decision: Any, action: ActionPacket | None = None) -> GovernanceDecision:
        if isinstance(decision, GovernanceDecision):
            return decision
        status = str(getattr(decision, "status", "denied"))
        reason = normalize_governance_reason(str(getattr(decision, "reason", "policy denied")))
        confirm_required = bool(
            getattr(decision, "confirm_required", False)
            or getattr(decision, "needs_confirmation", False)
            or status == "needs_confirmation"
        )
        return GovernanceDecision(
            status=status,
            reason=reason,
            action_summary=(
                " ".join(str(getattr(decision, "action_summary", "")).split())
                if getattr(decision, "action_summary", None) is not None
                else None
            ),
            confirm_required=confirm_required,
            confirm_reason=(
                normalize_governance_reason(str(getattr(decision, "confirm_reason")))
                if getattr(decision, "confirm_reason", None) is not None
                else None
            ),
            confirm_prompt=getattr(decision, "confirm_prompt", None),
            idempotency_key=getattr(decision, "idempotency_key", None)
            or (
                build_normalized_idempotency_key(action.tool_name, action.tool_args)
                if action is not None
                else None
            ),
            cooldown_seconds=float(getattr(decision, "cooldown_seconds", 0.0) or 0.0),
            dry_run_supported=bool(getattr(decision, "dry_run_supported", False)),
            decision_source=str(getattr(decision, "decision_source", "tier_default") or "tier_default"),
            thresholds=dict(getattr(decision, "thresholds", None) or {}),
        )

    def build_action_packet(
        self,
        name: str,
        call_id: str,
        args: dict[str, Any],
        *,
        reason: str | None = None,
    ) -> ActionPacket:
        spec = self._tool_specs.get(name, self._default_spec())
        estimated_cost = spec.cost_hint
        risk_score = self._estimate_risk(spec)
        risk_flags = list(spec.safety_tags)
        if not spec.reversible:
            risk_flags.append("non_reversible")
        if spec.cost_hint == "expensive":
            risk_flags.append("expensive")
        if risk_score >= self._risk_threshold:
            risk_flags.append("risk_threshold_exceeded")
        requires_confirmation = spec.tier >= 2
        impact = f"Invoke {name} with proposed arguments and return its result."
        rollback = (
            "No rollback needed (read-only)."
            if spec.reversible
            else "Manual rollback required; no automatic undo available."
        )
        alternatives = [
            "Do nothing and leave the current state unchanged.",
            "Ask for revised arguments or a different approach.",
        ]
        return ActionPacket(
            id=call_id,
            tool_name=name,
            tool_args=args,
            tier=spec.tier,
            what=f"Run {name} with proposed arguments.",
            why=reason or "Model requested tool call.",
            impact=impact,
            rollback=rollback,
            alternatives=alternatives,
            confidence=max(0.05, min(0.98, 1.0 - risk_score)),
            cost=estimated_cost,
            risk_flags=risk_flags,
            requires_confirmation=requires_confirmation,
        )

    def open_autonomy_window(
        self,
        name: str,
        *,
        duration_s: float | None = None,
        allowed_tiers: Iterable[int] | None = None,
        source: str = "manual",
    ) -> None:
        now = time.time()
        spec = next((window for window in self._scheduled_windows if window.name == name), None)
        resolved_duration = duration_s or (spec.duration_s if spec else None)
        if resolved_duration is None:
            raise ValueError("duration_s must be provided when opening an ad-hoc window")
        resolved_tiers = (
            tuple(allowed_tiers)
            if allowed_tiers is not None
            else (spec.allowed_tiers if spec else ())
        )
        if not resolved_tiers:
            raise ValueError("allowed_tiers must be provided when opening an ad-hoc window")
        state = AutonomyWindowState(
            name=name,
            opened_at=now,
            expires_at=now + float(resolved_duration),
            allowed_tiers=tuple(int(tier) for tier in resolved_tiers),
            source=source,
        )
        self._active_window = state
        self._log_window_transition(previous=self._last_window_name, current=state)

    def close_autonomy_window(self, *, name: str | None = None, reason: str = "manual") -> None:
        if self._active_window is None:
            return
        if name is not None and self._active_window.name != name:
            return
        previous = self._active_window
        self._active_window = None
        log_info(
            f"🪟 Autonomy window closed ({reason}): {previous.summary()}",
            style="bold blue",
        )
        self._log_window_transition(previous=previous.name, current=None)

    def review(self, action: ActionPacket) -> GovernanceDecision:
        return self.review_with_runtime_context(action, runtime_context=None)

    def review_with_runtime_context(
        self,
        action: ActionPacket,
        *,
        runtime_context: dict[str, Any] | None,
    ) -> GovernanceDecision:
        now = time.monotonic()
        now_wall = time.time()
        thresholds = self._build_runtime_threshold_snapshot(runtime_context)
        if self._autonomy_level in {"observe-only", "observe"}:
            return GovernanceDecision(
                status="denied",
                reason=GovernanceReason.AUTONOMY_OBSERVE_ONLY.value,
                decision_source="session_override",
                thresholds=thresholds,
            )

        if not self._tool_calls_budget.allow(now):
            return GovernanceDecision(
                status="denied",
                reason=GovernanceReason.TOOL_CALL_BUDGET_EXHAUSTED.value,
                decision_source="threshold_crossed",
                thresholds=thresholds,
            )

        spec = self._tool_specs.get(action.tool_name, self._default_spec())
        risk_score = self._estimate_risk(spec)
        thresholds["risk_score"] = risk_score
        if action.cost == "expensive" and not self._expensive_budget.allow(now):
            return GovernanceDecision(
                status="denied",
                reason=GovernanceReason.EXPENSIVE_CALL_BUDGET_EXHAUSTED.value,
                decision_source="threshold_crossed",
                thresholds=thresholds,
            )

        if risk_score >= self._risk_threshold:
            return GovernanceDecision(
                status="needs_confirmation",
                reason=GovernanceReason.RISK_THRESHOLD_EXCEEDED.value,
                decision_source="threshold_crossed",
                thresholds=thresholds,
            )

        if action.tier > 1:
            window_state = self._resolve_autonomy_window(now_wall)
            if window_state is None or action.tier not in window_state.allowed_tiers:
                return GovernanceDecision(
                    status="needs_confirmation",
                    reason=GovernanceReason.AUTONOMY_WINDOW_CLOSED.value,
                    decision_source="autonomy_window_closed",
                    thresholds=thresholds,
                )

        if self._autonomy_level in {"assist", "act-with-confirm"}:
            if action.tier > 0:
                return GovernanceDecision(
                    status="needs_confirmation",
                    reason=GovernanceReason.AUTONOMY_LEVEL_REQUIRES_CONFIRMATION.value,
                    decision_source="tier_default",
                    thresholds=thresholds,
                )

        if self._tier1_guarded_threshold_exceeded(action, runtime_context=runtime_context):
            return GovernanceDecision(
                status="needs_confirmation",
                reason=GovernanceReason.TIER1_GUARDED_THRESHOLD_EXCEEDED.value,
                decision_source="threshold_crossed",
                thresholds=thresholds,
            )

        return GovernanceDecision(
            status="approved",
            reason=GovernanceReason.WITHIN_BOUNDS.value,
            decision_source="tier_default",
            thresholds=thresholds,
        )

    def _build_runtime_threshold_snapshot(
        self,
        runtime_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        context = runtime_context or {}
        return {
            "risk_score": None,
            "risk_threshold": self._risk_threshold,
            "guarded_max_cost_score": self._guarded_max_cost_score,
            "guarded_min_rate_limit_remaining": self._guarded_min_rate_limit_remaining,
            "guarded_privacy_flag_requires_confirmation": self._guarded_privacy_flag_requires_confirmation,
            "cost_score": context.get("cost_score"),
            "rate_limit_remaining": context.get("rate_limit_remaining"),
            "privacy_flag": bool(context.get("privacy_flag")),
        }

    def _tier1_guarded_threshold_exceeded(
        self,
        action: ActionPacket,
        *,
        runtime_context: dict[str, Any] | None,
    ) -> bool:
        if action.tier != 1:
            return False
        tags = {str(tag).strip().lower() for tag in action.risk_flags if str(tag).strip()}
        is_guarded = "network" in tags or "research" in tags or "privacy" in tags
        if not is_guarded:
            return False
        context = runtime_context or {}
        cost_score = context.get("cost_score")
        if isinstance(cost_score, (int, float)) and float(cost_score) >= self._guarded_max_cost_score:
            return True
        remaining = context.get("rate_limit_remaining")
        if isinstance(remaining, (int, float)) and float(remaining) <= self._guarded_min_rate_limit_remaining:
            return True
        privacy_flag = context.get("privacy_flag")
        if self._guarded_privacy_flag_requires_confirmation and bool(privacy_flag):
            return True
        return False

    def record_execution(self, action: ActionPacket) -> None:
        now = time.monotonic()
        self._tool_calls_budget.record(now)
        if action.cost == "expensive":
            self._expensive_budget.record(now)

    def describe_tool(self, name: str) -> dict[str, Any]:
        spec = self._tool_specs.get(name, self._default_spec())
        return {
            "tier": spec.tier,
            "reversible": spec.reversible,
            "cost_hint": spec.cost_hint,
            "safety_tags": list(spec.safety_tags),
            "confirm_required": spec.confirm_required,
            "confirm_reason": spec.confirm_reason,
            "confirm_prompt": spec.confirm_prompt,
            "cooldown_seconds": spec.cooldown_seconds,
            "dry_run_supported": spec.dry_run_supported,
        }

    def _load_autonomy_windows(self, raw_windows: Iterable[dict[str, Any]]) -> list[AutonomyWindowSpec]:
        windows: list[AutonomyWindowSpec] = []
        for raw in raw_windows:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "window")
            start_time = str(raw.get("start_time") or "00:00")
            duration_minutes = float(raw.get("duration_minutes") or 0.0)
            allowed_tiers = tuple(int(tier) for tier in (raw.get("allowed_tiers") or []))
            enabled = bool(raw.get("enabled", True))
            start_minutes = self._parse_start_minutes(start_time)
            window = AutonomyWindowSpec(
                name=name,
                start_minutes=start_minutes,
                duration_s=max(0.0, duration_minutes * 60.0),
                allowed_tiers=allowed_tiers,
                enabled=enabled,
            )
            windows.append(window)
        return windows

    def _parse_start_minutes(self, start_time: str) -> int:
        try:
            parts = start_time.strip().split(":")
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            return max(0, min(23, hours)) * 60 + max(0, min(59, minutes))
        except (ValueError, IndexError):
            return 0

    def _resolve_autonomy_window(self, now_wall: float) -> AutonomyWindowState | None:
        if self._active_window is not None:
            if now_wall >= self._active_window.expires_at:
                expired = self._active_window
                self._active_window = None
                log_info(
                    f"🪟 Autonomy window expired: {expired.summary()}",
                    style="bold blue",
                )
                self._log_window_transition(previous=expired.name, current=None)
            else:
                self._log_window_transition(previous=self._last_window_name, current=self._active_window)
                return self._active_window

        scheduled = self._resolve_scheduled_window(now_wall)
        self._log_window_transition(previous=self._last_window_name, current=scheduled)
        return scheduled

    def _resolve_scheduled_window(self, now_wall: float) -> AutonomyWindowState | None:
        if not self._scheduled_windows:
            return None
        now = datetime.fromtimestamp(now_wall)
        minutes_now = now.hour * 60 + now.minute
        for window in self._scheduled_windows:
            if not window.enabled or window.duration_s <= 0 or not window.allowed_tiers:
                continue
            duration_minutes = int(window.duration_s // 60)
            if duration_minutes >= 24 * 60:
                start_dt = datetime.combine(now.date(), datetime.min.time())
                return AutonomyWindowState(
                    name=window.name,
                    opened_at=start_dt.timestamp(),
                    expires_at=start_dt.timestamp() + window.duration_s,
                    allowed_tiers=window.allowed_tiers,
                    source="scheduled",
                )
            end_minutes = (window.start_minutes + duration_minutes) % (24 * 60)
            crosses_midnight = window.start_minutes + duration_minutes >= 24 * 60
            in_window = (
                window.start_minutes <= minutes_now < end_minutes
                if not crosses_midnight
                else minutes_now >= window.start_minutes or minutes_now < end_minutes
            )
            if not in_window:
                continue
            start_date = now.date()
            if crosses_midnight and minutes_now < end_minutes:
                start_date = (now - timedelta(days=1)).date()
            start_dt = datetime.combine(
                start_date,
                datetime.min.time(),
            ) + timedelta(minutes=window.start_minutes)
            return AutonomyWindowState(
                name=window.name,
                opened_at=start_dt.timestamp(),
                expires_at=start_dt.timestamp() + window.duration_s,
                allowed_tiers=window.allowed_tiers,
                source="scheduled",
            )
        return None

    def _log_window_transition(
        self,
        *,
        previous: str | None,
        current: AutonomyWindowState | None,
    ) -> None:
        current_name = current.name if current else None
        if previous == current_name:
            return
        prev_label = previous or "none"
        current_label = current_name or "none"
        detail = f"current={current.summary()}" if current else "current=none"
        log_info(
            f"🪟 Autonomy window transition: {prev_label} -> {current_label} ({detail})",
            style="bold blue",
        )
        self._last_window_name = current_name

    def _estimate_risk(self, spec: ToolSpec) -> float:
        base = 0.2
        tier_bump = 0.2 * max(spec.tier - 1, 0)
        cost_bump = 0.2 if spec.cost_hint == "expensive" else 0.0
        reversible_bump = -0.1 if spec.reversible else 0.1
        return max(0.0, min(1.0, base + tier_bump + cost_bump + reversible_bump))


_REQUIRED_GOVERNANCE_METADATA_FIELDS = (
    "governance_tier",
    "side_effects",
    "sensitivity",
    "default_confirmation",
)

_ALLOWED_GOVERNANCE_TIERS = {"SAFE", "GUARDED", "PRIVILEGED"}
_ALLOWED_DEFAULT_CONFIRMATION = {"NEVER", "ASK", "ALWAYS"}


def _coerce_governance_metadata(tool_name: str, payload: dict[str, Any]) -> dict[str, str]:
    missing = [field for field in _REQUIRED_GOVERNANCE_METADATA_FIELDS if payload.get(field) is None]
    if missing:
        raise ValueError(
            f"Governance metadata missing for tool '{tool_name}': {', '.join(missing)}"
        )

    metadata = {
        "governance_tier": str(payload["governance_tier"]).upper(),
        "side_effects": str(payload["side_effects"]).upper(),
        "sensitivity": str(payload["sensitivity"]).upper(),
        "default_confirmation": str(payload["default_confirmation"]).upper(),
    }
    if metadata["governance_tier"] not in _ALLOWED_GOVERNANCE_TIERS:
        raise ValueError(
            f"Governance metadata for tool '{tool_name}' has invalid governance_tier "
            f"'{metadata['governance_tier']}'. Expected one of {sorted(_ALLOWED_GOVERNANCE_TIERS)}."
        )
    if metadata["default_confirmation"] not in _ALLOWED_DEFAULT_CONFIRMATION:
        raise ValueError(
            f"Governance metadata for tool '{tool_name}' has invalid default_confirmation "
            f"'{metadata['default_confirmation']}'. Expected one of "
            f"{sorted(_ALLOWED_DEFAULT_CONFIRMATION)}."
        )
    return metadata


def build_tool_specs(
    raw: dict[str, dict[str, Any]],
    *,
    registered_tool_names: Iterable[str] | None = None,
) -> dict[str, ToolSpec]:
    raw = raw or {}
    if registered_tool_names is not None:
        missing = sorted(set(registered_tool_names) - set(raw.keys()))
        if missing:
            raise ValueError(
                "Governance tool_specs missing entries for registered tools: "
                + ", ".join(missing)
            )

    specs: dict[str, ToolSpec] = {}
    for name, payload in raw.items():
        metadata = _coerce_governance_metadata(name, payload)
        specs[name] = ToolSpec(
            tier=int(payload.get("tier", 2)),
            reversible=bool(payload.get("reversible", False)),
            cost_hint=str(payload.get("cost_hint", "med")),
            safety_tags=tuple(payload.get("safety_tags", []) or []),
            confirm_required=(
                None if payload.get("confirm_required") is None else bool(payload.get("confirm_required"))
            ),
            confirm_reason=(
                str(payload.get("confirm_reason")) if payload.get("confirm_reason") is not None else None
            ),
            confirm_prompt=(
                str(payload.get("confirm_prompt")) if payload.get("confirm_prompt") is not None else None
            ),
            cooldown_seconds=float(payload.get("cooldown_seconds", 0.0) or 0.0),
            dry_run_supported=bool(payload.get("dry_run_supported", False)),
            governance_tier=metadata["governance_tier"],
            side_effects=metadata["side_effects"],
            sensitivity=metadata["sensitivity"],
            default_confirmation=metadata["default_confirmation"],
        )
    return specs


def list_tool_tags(tool_specs: dict[str, ToolSpec]) -> dict[str, list[str]]:
    return {name: list(spec.safety_tags) for name, spec in tool_specs.items()}


def unique_tags(tool_specs: dict[str, ToolSpec]) -> set[str]:
    tags: set[str] = set()
    for spec in tool_specs.values():
        tags.update(spec.safety_tags)
    return tags


def normalize_safety_tags(tags: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({tag.strip() for tag in tags if tag and tag.strip()}))
