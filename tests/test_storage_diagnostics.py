"""Tests for storage diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from storage.diagnostics import probe


def test_storage_probe_offline(tmp_path) -> None:
    """Storage probe should pass when using a temp directory."""

    result = probe(base_dir=tmp_path)
    assert result.status is DiagnosticStatus.PASS
