"""Focused tests for the runtime diagnostics tool bridge."""

from __future__ import annotations

import sys
import types

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime_api import RealtimeAPI
from services import tool_runtime


def test_read_runtime_diagnostics_delegates_to_registered_provider_without_mutation() -> None:
    original_provider = tool_runtime._runtime_diagnostics_provider
    source_payload = {
        "connected": True,
        "ready": False,
        "continuity": {
            "stance": "awaiting_user",
            "stance_detail": "question_pending",
            "settlement": "unresolved_followup",
            "settlement_detail": "question_pending",
            "counts": {
                "current": 1,
                "ongoing": 0,
                "commitments": 0,
                "unresolved": 1,
                "blockers": 0,
                "constraints": 0,
                "recently_closed": 0,
            },
        },
    }

    try:
        tool_runtime.set_runtime_diagnostics_provider(lambda: source_payload)

        payload = tool_runtime.read_runtime_diagnostics()

        assert payload is source_payload
        assert payload["connected"] is True
        assert payload["ready"] is False
        assert set(payload["continuity"]) == {
            "stance",
            "stance_detail",
            "settlement",
            "settlement_detail",
            "counts",
        }
        assert set(payload["continuity"]["counts"]) == {
            "current",
            "ongoing",
            "commitments",
            "unresolved",
            "blockers",
            "constraints",
            "recently_closed",
        }
    finally:
        tool_runtime.set_runtime_diagnostics_provider(original_provider)


def test_read_runtime_diagnostics_reports_unavailable_without_provider() -> None:
    original_provider = tool_runtime._runtime_diagnostics_provider

    try:
        tool_runtime.set_runtime_diagnostics_provider(None)

        payload = tool_runtime.read_runtime_diagnostics()

        assert payload == {
            "status": "unavailable",
            "message": "Runtime diagnostics are not currently available.",
        }
    finally:
        tool_runtime.set_runtime_diagnostics_provider(original_provider)


def test_realtime_api_registers_session_health_as_runtime_diagnostics_provider(monkeypatch) -> None:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.get_session_health = lambda: {"ready": True}
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tool_runtime,
        "set_runtime_diagnostics_provider",
        lambda provider: captured.setdefault("provider", provider),
    )

    RealtimeAPI._register_runtime_diagnostics_provider(api)

    provider = captured.get("provider")
    assert callable(provider)
    assert provider() == {"ready": True}
