"""Utility constants for realtime API handling."""

from __future__ import annotations

from pathlib import Path

RUN_TIME_TABLE_LOG_JSON = "log/runtime_table.jsonl"
SOUL_PATH = Path("config/SOUL.md")

PREFIX_PADDING_MS = 500
SILENCE_THRESHOLD = 0.2
SILENCE_DURATION_MS = 900


def build_session_instructions(
    profile_block: str | None = None,
    lessons_block: str | None = None,
    assistant_name: str | None = None,
) -> str:
    soul_text = SOUL_PATH.read_text(encoding="utf-8").strip()
    configured_name = (assistant_name or "").strip()
    if configured_name:
        soul_text = soul_text.replace("Theo", configured_name)
    instruction_blocks = [soul_text]
    if profile_block:
        instruction_blocks.append(profile_block)
    if lessons_block:
        instruction_blocks.append(lessons_block)
    return "\n".join(instruction_blocks)
