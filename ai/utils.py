"""Utility constants for realtime API handling."""

from __future__ import annotations

RUN_TIME_TABLE_LOG_JSON = "log/runtime_table.jsonl"
BASE_SESSION_INSTRUCTIONS = """You are Theo, a friendly Raspberry Pi robot assistant.
Respond conversationally and keep responses concise unless asked for detail.
When the user directly asks to change volume (e.g. "turn it up/down", "I can’t hear you", "too loud"),
call the adjust_output_volume tool. When asked about the current volume, call the read_output_volume tool.
"""

PREFIX_PADDING_MS = 500
SILENCE_THRESHOLD = 0.2
SILENCE_DURATION_MS = 900


def build_session_instructions(
    profile_block: str | None = None,
    lessons_block: str | None = None,
) -> str:
    instruction_blocks = [BASE_SESSION_INSTRUCTIONS]
    if profile_block:
        instruction_blocks.append(profile_block)
    if lessons_block:
        instruction_blocks.append(lessons_block)
    return "\n".join(instruction_blocks)
