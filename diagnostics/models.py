"""Models for diagnostics results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticStatus(str, Enum):
    """Status for diagnostics checks."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DiagnosticResult:
    """Result for a single diagnostic check."""

    name: str
    status: DiagnosticStatus
    details: str
