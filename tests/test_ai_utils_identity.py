"""Tests for assistant identity handling in session instruction assembly."""

from __future__ import annotations

import sys
import types
from pathlib import Path

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai import utils


def test_build_session_instructions_uses_configured_assistant_name(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "SOUL.md").write_text("I am Theo.\nTheo remains Theo.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    instructions = utils.build_session_instructions(assistant_name="Nova")

    assert "I am Nova." in instructions
    assert "Nova remains Nova." in instructions
    assert "Theo" not in instructions


def test_build_session_instructions_defaults_to_soul_text_when_name_missing(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "SOUL.md").write_text("I am Theo.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    instructions = utils.build_session_instructions()

    assert instructions == "I am Theo."
