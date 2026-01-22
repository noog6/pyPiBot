"""Tests for hardware diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from hardware.diagnostics import HardwareProbeConfig, probe


def test_hardware_probe_warns_on_missing_modules() -> None:
    """Hardware probe should warn when optional deps are missing."""

    result = probe(
        config=HardwareProbeConfig(require_all=False),
        available_modules={"smbus"},
    )
    assert result.status is DiagnosticStatus.WARN


def test_hardware_probe_fails_when_required() -> None:
    """Hardware probe should fail when required deps are missing."""

    result = probe(
        config=HardwareProbeConfig(require_all=True),
        available_modules={"numpy"},
    )
    assert result.status is DiagnosticStatus.FAIL
