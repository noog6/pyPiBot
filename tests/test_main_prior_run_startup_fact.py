"""Focused tests for prior-run startup fact wording."""

from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

import main
from storage.controller import SessionLedgerRecord


def _record(canonical_run_id: str) -> SessionLedgerRecord:
    return SessionLedgerRecord(
        canonical_run_id=canonical_run_id,
        run_number=123,
        started_at=1,
        ready_at=2,
        last_seen_at=3,
        shutdown_completed_at=None,
        lifecycle_state="running",
    )


def test_build_prior_run_startup_fact_clean() -> None:
    fact = main._build_prior_run_startup_fact(False, _record("run-123"))

    assert fact == "Previous runtime session ended cleanly."


def test_build_prior_run_startup_fact_unclean_includes_run_id() -> None:
    fact = main._build_prior_run_startup_fact(True, _record("run-123"))

    assert fact == (
        "Previous runtime session appears to have ended unexpectedly. "
        "Last known session: run-123."
    )


def test_build_prior_run_startup_fact_none_when_no_prior_run() -> None:
    fact = main._build_prior_run_startup_fact(False, None)

    assert fact is None
