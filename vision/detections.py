"""Stable detection event schemas for vision pipelines.

Bounding boxes are normalized to the source frame dimensions and represented as
``(x, y, width, height)`` with each value expected in the inclusive range
``[0.0, 1.0]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    """Single object detection result."""

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionEvent:
    """Detection snapshot for one processed frame or sampling instant."""

    timestamp_ms: int
    detections: list[Detection]
    frame_id: int | None = None
    source: str = "imx500"
