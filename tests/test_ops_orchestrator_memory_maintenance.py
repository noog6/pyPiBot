"""Tests for periodic memory maintenance scheduling in ops orchestrator."""

from __future__ import annotations

from services.ops_orchestrator import OpsOrchestrator


def _new_orchestrator() -> OpsOrchestrator:
    OpsOrchestrator._instance = None
    orchestrator = OpsOrchestrator()
    OpsOrchestrator._instance = None
    return orchestrator


def test_memory_maintenance_runs_only_at_interval_boundary(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._memory_maintenance_enabled = True
    orchestrator._memory_maintenance_interval_s = 100.0
    orchestrator._memory_maintenance_optimize_every_runs = 2
    orchestrator._memory_maintenance_next_ts = 50.0

    calls: list[bool] = []

    class _FakeMemoryManager:
        def run_periodic_maintenance(self, *, optimize_allowed: bool) -> dict[str, int | bool]:
            calls.append(optimize_allowed)
            return {"pruned_rows": 0, "purged_rows": 0, "optimize_triggered": False}

    monkeypatch.setattr(
        "services.ops_orchestrator.MemoryManager.get_instance", lambda: _FakeMemoryManager()
    )

    orchestrator._maybe_run_memory_maintenance(now=49.9)
    assert calls == []

    orchestrator._maybe_run_memory_maintenance(now=50.0)
    assert calls == [False]
    assert orchestrator._memory_maintenance_next_ts == 150.0

    orchestrator._maybe_run_memory_maintenance(now=149.9)
    assert calls == [False]

    orchestrator._maybe_run_memory_maintenance(now=150.0)
    assert calls == [False, True]


def test_memory_maintenance_disabled_is_noop(monkeypatch) -> None:
    orchestrator = _new_orchestrator()
    orchestrator._memory_maintenance_enabled = False
    orchestrator._memory_maintenance_next_ts = 0.0

    class _FakeMemoryManager:
        def run_periodic_maintenance(self, *, optimize_allowed: bool) -> dict[str, int | bool]:
            raise AssertionError("maintenance should not execute when disabled")

    monkeypatch.setattr(
        "services.ops_orchestrator.MemoryManager.get_instance", lambda: _FakeMemoryManager()
    )

    orchestrator._maybe_run_memory_maintenance(now=10.0)
