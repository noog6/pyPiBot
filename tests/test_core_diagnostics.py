"""Tests for core diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from core.diagnostics import probe


def test_core_probe() -> None:
    """Core probe should pass when logging is available."""

    result = probe()
    assert result.status is DiagnosticStatus.PASS
