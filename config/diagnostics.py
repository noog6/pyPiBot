"""Diagnostics routines for the configuration subsystem."""

from __future__ import annotations

from pathlib import Path

from diagnostics.models import DiagnosticResult, DiagnosticStatus


def probe(base_dir: Path | None = None) -> DiagnosticResult:
    """Run a configuration probe to validate config file availability.

    Args:
        base_dir: Optional base directory for offline testing.

    Returns:
        Diagnostic result indicating config readiness.
    """

    name = "config"
    try:
        root_dir = base_dir if base_dir is not None else Path.cwd()
        config_dir = root_dir / "config"
        default_config = config_dir / "default.yaml"
        override_config = config_dir / "override.yaml"

        if not config_dir.exists():
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.FAIL,
                details=f"Config directory missing at {config_dir}",
            )

        if not default_config.exists():
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.FAIL,
                details=f"Missing default config at {default_config}",
            )

        default_config.read_text(encoding="utf-8")
        if override_config.exists():
            override_config.read_text(encoding="utf-8")

        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.PASS,
            details=f"Config files readable at {config_dir}",
        )
    except OSError as exc:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details=f"Config access failed: {exc}",
        )
