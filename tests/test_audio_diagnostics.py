"""Tests for audio diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from interaction.audio_hal import FakeAudioBackend
from interaction.diagnostics import probe


def test_audio_probe_offline_pass() -> None:
    """Offline audio probe should pass with a fake device."""

    backend = FakeAudioBackend(devices=["speaker"])
    result = probe(backend=backend)
    assert result.status is DiagnosticStatus.PASS


def test_audio_probe_offline_fail() -> None:
    """Offline audio probe should fail when stream open fails."""

    backend = FakeAudioBackend(devices=["speaker"], can_open=False)
    result = probe(backend=backend)
    assert result.status is DiagnosticStatus.FAIL
