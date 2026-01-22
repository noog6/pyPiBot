"""Tests for microphone diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from interaction.microphone_diagnostics import probe
from interaction.microphone_hal import FakeInputBackend


def test_microphone_probe_offline_pass() -> None:
    """Offline microphone probe should pass with a fake device."""

    backend = FakeInputBackend(devices=["mic"])
    result = probe(backend=backend)
    assert result.status is DiagnosticStatus.PASS


def test_microphone_probe_offline_fail() -> None:
    """Offline microphone probe should fail when stream open fails."""

    backend = FakeInputBackend(devices=["mic"], can_open=False)
    result = probe(backend=backend)
    assert result.status is DiagnosticStatus.FAIL
