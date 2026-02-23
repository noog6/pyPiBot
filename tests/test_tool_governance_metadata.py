"""Tests for governance metadata coverage across registered tools."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def _registered_tool_names() -> list[str]:
    tools_file = Path("ai/tools.py").read_text(encoding="utf-8")
    return re.findall(r'"name":\s*"([^\"]+)"', tools_file)


def _tool_specs() -> dict[str, dict[str, object]]:
    config = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    return config["governance"]["tool_specs"]


def test_all_registered_tools_have_governance_metadata() -> None:
    tool_names = _registered_tool_names()
    tool_specs = _tool_specs()

    assert len(tool_names) == 21
    assert set(tool_names) <= set(tool_specs)

    required = {"governance_tier", "side_effects", "sensitivity", "default_confirmation"}
    for tool_name in tool_names:
        payload = tool_specs[tool_name]
        assert required <= set(payload.keys()), tool_name


def test_safe_tools_default_to_never_confirmation() -> None:
    tool_specs = _tool_specs()
    for tool_name, payload in tool_specs.items():
        if payload.get("governance_tier") == "SAFE":
            assert payload.get("default_confirmation") == "NEVER", tool_name


def test_build_tool_specs_raises_when_registered_tool_missing() -> None:
    from ai.governance import build_tool_specs

    try:
        build_tool_specs({}, registered_tool_names=["read_battery_voltage"])
    except ValueError as exc:
        assert "missing entries" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing registered tool metadata")
