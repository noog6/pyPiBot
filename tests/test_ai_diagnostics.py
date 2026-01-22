"""Tests for AI diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from ai.diagnostics import probe


def test_ai_probe_offline_pass() -> None:
    """AI probe should pass with a provided API key."""

    result = probe(api_key="test-key", require_websockets=False)
    assert result.status is DiagnosticStatus.PASS


def test_ai_probe_offline_missing_key() -> None:
    """AI probe should fail when no key is provided."""

    result = probe(api_key=None, require_websockets=False)
    assert result.status is DiagnosticStatus.FAIL
