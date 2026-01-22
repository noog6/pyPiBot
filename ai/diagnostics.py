"""Diagnostics routines for the AI subsystem."""

from __future__ import annotations

import importlib.util
import os

from diagnostics.models import DiagnosticResult, DiagnosticStatus


def probe(api_key: str | None = None, require_websockets: bool = True) -> DiagnosticResult:
    """Run an AI probe to validate environment requirements.

    Args:
        api_key: Optional API key override for testing.
        require_websockets: Whether to require websockets availability.

    Returns:
        Diagnostic result indicating AI readiness.
    """

    name = "ai"
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="Missing OPENAI_API_KEY",
        )

    if require_websockets and importlib.util.find_spec("websockets") is None:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="Missing websockets dependency",
        )

    return DiagnosticResult(
        name=name,
        status=DiagnosticStatus.PASS,
        details="AI configuration present",
    )
