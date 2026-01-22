"""Diagnostics routines for hardware dependencies."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from diagnostics.models import DiagnosticResult, DiagnosticStatus


@dataclass(frozen=True)
class HardwareProbeConfig:
    """Configuration for hardware dependency checks."""

    require_all: bool = False


def probe(config: HardwareProbeConfig | None = None, available_modules: set[str] | None = None) -> DiagnosticResult:
    """Run a hardware probe to validate optional dependencies.

    Args:
        config: Optional configuration for probe behavior.
        available_modules: Optional override set for offline testing.

    Returns:
        Diagnostic result indicating hardware dependency readiness.
    """

    name = "hardware"
    settings = config or HardwareProbeConfig()
    required = ["smbus"]
    optional = ["picamera2", "numpy", "PIL"]

    missing: list[str] = []
    for module_name in required + optional:
        if available_modules is not None:
            is_available = module_name in available_modules
        else:
            is_available = importlib.util.find_spec(module_name) is not None
        if not is_available:
            missing.append(module_name)

    if missing:
        status = DiagnosticStatus.FAIL if settings.require_all else DiagnosticStatus.WARN
        details = f"Missing hardware deps: {', '.join(missing)}"
        return DiagnosticResult(name=name, status=status, details=details)

    return DiagnosticResult(
        name=name,
        status=DiagnosticStatus.PASS,
        details="Hardware dependencies available",
    )
