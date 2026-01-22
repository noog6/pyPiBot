"""Diagnostics routines for the core subsystem."""

from __future__ import annotations

import importlib.util

from diagnostics.models import DiagnosticResult, DiagnosticStatus


def probe() -> DiagnosticResult:
    """Run a core probe to validate logging readiness.

    Returns:
        Diagnostic result indicating core readiness.
    """

    name = "core"
    from core import logging as core_logging

    logger = core_logging.logger
    rich_available = importlib.util.find_spec("rich") is not None
    details = "Rich logging enabled" if rich_available else "Rich logging not available (fallback)"
    if logger is None:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="Core logger failed to initialize",
        )
    return DiagnosticResult(
        name=name,
        status=DiagnosticStatus.PASS,
        details=details,
    )
