"""Diagnostics runner utilities."""

from __future__ import annotations

from collections.abc import Iterable

from core.logging import logger as LOGGER
from diagnostics.models import DiagnosticResult, DiagnosticStatus


def format_results(results: Iterable[DiagnosticResult]) -> str:
    """Return a human-friendly diagnostics report."""

    lines = ["Diagnostics report", "-" * 60]
    for result in results:
        status = result.status.value
        name = result.name
        details = result.details
        lines.append(f"[{status}] {name}: {details}")
    lines.append("-" * 60)
    return "\n".join(lines)


def run_diagnostics(probes: Iterable[callable]) -> list[DiagnosticResult]:
    """Run diagnostics probes and return results."""

    results: list[DiagnosticResult] = []
    for probe in probes:
        try:
            result = probe()
        except Exception as exc:  # noqa: BLE001 - diagnostics must keep running
            LOGGER.exception("Probe failed: %s", probe)
            result = DiagnosticResult(
                name=getattr(probe, "__name__", "unknown_probe"),
                status=DiagnosticStatus.FAIL,
                details=f"Probe raised exception: {exc}",
            )
        results.append(result)
    return results
