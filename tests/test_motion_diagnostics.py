"""Tests for motion diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from motion.diagnostics import MotionProbeConfig, probe


def test_motion_probe_offline_pass() -> None:
    """Motion probe should pass when expected servos exist."""

    result = probe(
        servo_names=["pan", "tilt"],
        config=MotionProbeConfig(expected_servos=("pan", "tilt")),
    )
    assert result.status is DiagnosticStatus.PASS


def test_motion_probe_warns_on_missing_servo() -> None:
    """Motion probe should warn when expected servos are missing."""

    result = probe(
        servo_names=["pan"],
        config=MotionProbeConfig(expected_servos=("pan", "tilt")),
    )
    assert result.status is DiagnosticStatus.WARN
