"""Diagnostics routines for the services subsystem."""

from __future__ import annotations

from pathlib import Path

from diagnostics.models import DiagnosticResult, DiagnosticStatus


def probe(base_dir: Path | None = None) -> DiagnosticResult:
    """Run a services probe to validate service module availability.

    Args:
        base_dir: Optional base directory for offline testing.

    Returns:
        Diagnostic result indicating service readiness.
    """

    name = "services"
    root_dir = base_dir if base_dir is not None else Path.cwd()
    services_dir = root_dir / "services"
    if not services_dir.exists():
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details=f"Services directory missing at {services_dir}",
        )

    service_modules = sorted(
        path
        for path in services_dir.glob("*.py")
        if path.name not in {"__init__.py"}
    )
    if not service_modules:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.WARN,
            details="No service modules found",
        )

    return DiagnosticResult(
        name=name,
        status=DiagnosticStatus.PASS,
        details=f"Service modules: {', '.join(path.stem for path in service_modules)}",
    )
