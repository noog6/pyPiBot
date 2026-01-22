"""Tests for config diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from config.diagnostics import probe


def test_config_probe_offline(tmp_path) -> None:
    """Config probe should pass with a default config present."""

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("{}", encoding="utf-8")

    result = probe(base_dir=tmp_path)
    assert result.status is DiagnosticStatus.PASS
