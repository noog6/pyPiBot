"""Focused tests for get_session_ledger_status runtime payload contract."""

from __future__ import annotations

from dataclasses import dataclass

from services import tool_runtime


@dataclass(frozen=True)
class _FakeRecord:
    canonical_run_id: str
    run_number: int
    started_at: int
    ready_at: int | None
    last_seen_at: int | None
    shutdown_completed_at: int | None
    lifecycle_state: str
    shutdown_clean: bool


class _FakeStorage:
    def __init__(self, records: list[_FakeRecord]) -> None:
        self.records = records
        self.calls: list[tuple[int, bool]] = []

    def get_recent_session_records(self, *, lookback_runs: int, include_current: bool):
        self.calls.append((lookback_runs, include_current))
        return self.records[:lookback_runs]


def test_get_session_ledger_status_defaults_to_single_prior_run(monkeypatch) -> None:
    fake_storage = _FakeStorage(
        [
            _FakeRecord("run-7", 7, 700, 710, 720, 730, "shutdown_completed", True),
            _FakeRecord("run-6", 6, 600, 610, 620, None, "running", False),
        ]
    )

    monkeypatch.setattr(
        "storage.controller.StorageController.get_instance",
        lambda: fake_storage,
    )

    payload = tool_runtime.get_session_ledger_status()

    assert fake_storage.calls == [(1, False)]
    assert payload["status"] == "ok"
    assert payload["lookback_runs"] == 1
    assert payload["include_current"] is False
    assert payload["returned_runs"] == 1
    assert payload["runs"][0]["run_id"] == "run-7"


def test_get_session_ledger_status_clamps_lookback_to_five(monkeypatch) -> None:
    fake_storage = _FakeStorage([])
    monkeypatch.setattr("storage.controller.StorageController.get_instance", lambda: fake_storage)

    payload = tool_runtime.get_session_ledger_status(lookback_runs=99)

    assert fake_storage.calls == [(5, False)]
    assert payload["lookback_runs"] == 5


def test_get_session_ledger_status_supports_include_current(monkeypatch) -> None:
    fake_storage = _FakeStorage(
        [_FakeRecord("run-8", 8, 800, 810, 820, None, "running", False)]
    )
    monkeypatch.setattr("storage.controller.StorageController.get_instance", lambda: fake_storage)

    payload = tool_runtime.get_session_ledger_status(lookback_runs=1, include_current=True)

    assert fake_storage.calls == [(1, True)]
    assert payload["include_current"] is True
    assert payload["runs"][0]["lifecycle_state"] == "running"


def test_get_session_ledger_status_empty_history_is_explicit(monkeypatch) -> None:
    fake_storage = _FakeStorage([])
    monkeypatch.setattr("storage.controller.StorageController.get_instance", lambda: fake_storage)

    payload = tool_runtime.get_session_ledger_status()

    assert payload["returned_runs"] == 0
    assert payload["runs"] == []
    assert payload["summary_text"] == "no recorded prior run history available"
    assert "empty_history" in payload["interpretation_notes"]


def test_get_session_ledger_status_summary_and_notes_are_deterministic(monkeypatch) -> None:
    fake_storage = _FakeStorage(
        [
            _FakeRecord("run-9", 9, 900, 910, 920, 930, "shutdown_completed", True),
            _FakeRecord("run-8", 8, 800, 810, 820, None, "running", False),
        ]
    )
    monkeypatch.setattr("storage.controller.StorageController.get_instance", lambda: fake_storage)

    payload = tool_runtime.get_session_ledger_status(lookback_runs=2)

    assert payload["summary_text"] == (
        "run-9: clean shutdown recorded; run-8: no clean shutdown recorded"
    )
    assert "appears to have ended cleanly" in payload["interpretation_notes"]["shutdown_clean_true"]
    assert "appears to have ended unexpectedly" in payload["interpretation_notes"]["shutdown_clean_false"]
