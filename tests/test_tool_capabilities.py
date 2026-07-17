from __future__ import annotations

import pytest
import yaml

from ai.tool_capabilities import build_tool_capability_map, validate_tool_capability_declarations
from ai.tools import tools


def _raw_tool_specs() -> dict[str, dict[str, object]]:
    with open("config/default.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["governance"]["tool_specs"]


def test_registered_query_tool_capability_matrix_and_governance_invariant() -> None:
    raw_specs = _raw_tool_specs()
    registered = {tool["name"] for tool in tools}
    caps = build_tool_capability_map(raw_specs, registered_tool_names=registered, require_declared=True)

    assert set(caps) == registered

    expected = {
        "read_battery_voltage": ("read", True, False, False, "read-only sensor query"),
        "get_output_volume": ("read", True, False, False, "read-only audio query"),
        "read_runtime_diagnostics": ("read", True, True, False, "read-only runtime diagnostics query"),
        "recall_memories": ("read", True, False, False, "read-only memory lookup"),
        "perform_research": ("action", False, False, False, "network-backed research action"),
    }
    for tool_name, (operation_kind, user_eligible, compound_eligible, runtime_only, governance) in expected.items():
        capability = caps[tool_name]
        assert capability.operation_kind == operation_kind
        assert capability.user_result_eligible is user_eligible
        assert capability.compound_intermediate_eligible is compound_eligible
        assert capability.runtime_only_background is runtime_only
        assert capability.governance_classification == governance

    for tool_name, payload in raw_specs.items():
        reason = str(payload.get("confirm_reason") or "").lower()
        if reason.startswith("read-only"):
            capability = caps[tool_name]
            assert capability.operation_kind == "read"
            assert capability.user_result_eligible or capability.compound_intermediate_eligible


def test_registered_tools_fail_validation_when_capabilities_are_missing() -> None:
    raw_specs = _raw_tool_specs()
    registered = {tool["name"] for tool in tools}
    broken_specs = dict(raw_specs)
    broken_specs["get_output_volume"] = dict(broken_specs["get_output_volume"])
    broken_specs["get_output_volume"].pop("capabilities")

    with pytest.raises(ValueError, match="Tool capabilities missing.*get_output_volume"):
        validate_tool_capability_declarations(
            broken_specs,
            registered_tool_names=registered,
        )


def test_undeclared_capabilities_are_conservative_when_not_validating() -> None:
    caps = build_tool_capability_map({"future_query": {"confirm_reason": "read-only future query"}})

    capability = caps["future_query"]
    assert capability.operation_kind == "action"
    assert capability.user_result_eligible is False
    assert capability.compound_intermediate_eligible is False
    assert capability.runtime_only_background is False
    assert capability.capability_source == "default_conservative"
