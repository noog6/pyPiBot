"""Tests for services diagnostics."""

from __future__ import annotations

from diagnostics.models import DiagnosticStatus
from services.diagnostics import probe


def test_services_probe_offline(tmp_path) -> None:
    """Services probe should pass when a service module exists."""

    services_dir = tmp_path / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    (services_dir / "__init__.py").write_text("", encoding="utf-8")
    (services_dir / "dummy_service.py").write_text("# test", encoding="utf-8")

    result = probe(base_dir=tmp_path)
    assert result.status is DiagnosticStatus.PASS
