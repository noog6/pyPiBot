"""Governance layer for tool execution and approvals."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
import time
from typing import Any, Iterable


@dataclass(frozen=True)
class ToolSpec:
    tier: int
    reversible: bool
    cost_hint: str
    safety_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionPacket:
    name: str
    call_id: str
    args: dict[str, Any]
    tool_spec: ToolSpec
    estimated_cost: str
    risk_score: float
    created_at: float = field(default_factory=time.monotonic)

    def summary(self) -> str:
        return (
            f"tool={self.name} tier={self.tool_spec.tier} cost={self.estimated_cost} "
            f"risk={self.risk_score:.2f} reversible={self.tool_spec.reversible}"
        )


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
    ) -> ActionPacket:
        spec = self._tool_specs.get(name, self._default_spec())
        estimated_cost = spec.cost_hint
        risk_score = self._estimate_risk(spec)
        return ActionPacket(
            name=name,
            call_id=call_id,
            args=args,
            tool_spec=spec,
            estimated_cost=estimated_cost,
            risk_score=risk_score,
        )

    def review(self, action: ActionPacket) -> GovernanceDecision:
        now = time.monotonic()
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

        if action.estimated_cost == "expensive" and not self._expensive_budget.allow(now):
            return GovernanceDecision(
                status="denied",
                reason="expensive-call budget exhausted",
            )

        if action.tool_spec.tier > 1 or action.risk_score >= self._risk_threshold:
            return GovernanceDecision(
                status="needs_confirmation",
                reason="tool tier requires confirmation",
            )

        if self._autonomy_level in {"assist", "act-with-confirm"}:
            if action.tool_spec.tier > 0:
                return GovernanceDecision(
                    status="needs_confirmation",
                    reason="autonomy level requires confirmation",
                )

        return GovernanceDecision(status="approved", reason="within bounds")

    def record_execution(self, action: ActionPacket) -> None:
        now = time.monotonic()
        self._tool_calls_budget.record(now)
        if action.estimated_cost == "expensive":
            self._expensive_budget.record(now)

    def describe_tool(self, name: str) -> dict[str, Any]:
        spec = self._tool_specs.get(name, self._default_spec())
        return {
            "tier": spec.tier,
            "reversible": spec.reversible,
            "cost_hint": spec.cost_hint,
            "safety_tags": list(spec.safety_tags),
        }

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
