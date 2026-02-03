"""Governance layer for tool execution and approvals."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from datetime import datetime, timedelta
import time
from typing import Any, Iterable

from core.logging import log_info


@dataclass(frozen=True)
class ToolSpec:
    tier: int
    reversible: bool
    cost_hint: str
    safety_tags: tuple[str, ...] = ()


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

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    @property
    def needs_confirmation(self) -> bool:
        return self.status == "needs_confirmation"

    @property
    def denied(self) -> bool:
        return self.status == "denied"


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
        now = time.monotonic()
        now_wall = time.time()
        if self._autonomy_level in {"observe-only", "observe"}:
            return GovernanceDecision(
                status="denied",
                reason="autonomy dial set to observe-only",
            )

        if not self._tool_calls_budget.allow(now):
            return GovernanceDecision(
                status="denied",
                reason="tool-call budget exhausted",
            )

        spec = self._tool_specs.get(action.tool_name, self._default_spec())
        risk_score = self._estimate_risk(spec)
        if action.cost == "expensive" and not self._expensive_budget.allow(now):
            return GovernanceDecision(
                status="denied",
                reason="expensive-call budget exhausted",
            )

        if risk_score >= self._risk_threshold:
            return GovernanceDecision(
                status="needs_confirmation",
                reason="risk threshold exceeded",
            )

        if action.tier > 1:
            window_state = self._resolve_autonomy_window(now_wall)
            if window_state is None or action.tier not in window_state.allowed_tiers:
                return GovernanceDecision(
                    status="needs_confirmation",
                    reason="autonomy window closed for tiered action",
                )

        if self._autonomy_level in {"assist", "act-with-confirm"}:
            if action.tier > 0:
                return GovernanceDecision(
                    status="needs_confirmation",
                    reason="autonomy level requires confirmation",
                )

        return GovernanceDecision(status="approved", reason="within bounds")

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


def build_tool_specs(raw: dict[str, dict[str, Any]]) -> dict[str, ToolSpec]:
    specs: dict[str, ToolSpec] = {}
    for name, payload in raw.items():
        specs[name] = ToolSpec(
            tier=int(payload.get("tier", 2)),
            reversible=bool(payload.get("reversible", False)),
            cost_hint=str(payload.get("cost_hint", "med")),
            safety_tags=tuple(payload.get("safety_tags", []) or []),
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
