"""Utility constants for realtime API handling."""

from __future__ import annotations

RUN_TIME_TABLE_LOG_JSON = "log/runtime_table.jsonl"
SESSION_INSTRUCTIONS = """You are Theo, a friendly Raspberry Pi robot assistant.
Respond conversationally and keep responses concise unless asked for detail.
"""

PREFIX_PADDING_MS = 500
SILENCE_THRESHOLD = 0.2
SILENCE_DURATION_MS = 900
