"""Shared governance decision envelope for cross-system policy adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

GovernanceDecisionType = Literal["allow", "defer", "suppress", "clarify", "downgrade", "expire"]


_REASON_CODE_PATTERN = re.compile(r"[^a-z0-9]+")

def normalize_reason_code(value: str, *, fallback: str = "unspecified") -> str:
    """Normalize reason code values for stable logging and assertions."""

    normalized = _REASON_CODE_PATTERN.sub("_", str(value or "").strip().lower()).strip("_")
    return normalized or fallback


@dataclass(frozen=True)
class GovernanceDecision:
    """Cross-system governance decision envelope.

    Contract: ``priority`` is seam-local metadata emitted by an adapter and is
    not globally comparable across subsystems (for example battery/curiosity/
    embodiment) unless an explicit cross-system arbitration seam consumes and
    normalizes it.
    """

    decision: GovernanceDecisionType
    reason_code: str
    subsystem: str
    priority: int
    expires_at: float | None = None
    ttl_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Migration note: if/when we introduce globally comparable governance
        # priority, add an explicit comparator/helper in this module (or a
        # sibling governance arbitration module) and migrate call sites to it.
        object.__setattr__(self, "reason_code", normalize_reason_code(self.reason_code))
        object.__setattr__(self, "subsystem", normalize_reason_code(self.subsystem, fallback="unknown"))
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
