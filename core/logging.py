"""Logging utilities for realtime events and session updates."""

from __future__ import annotations

import atexit
import hashlib
import importlib
import importlib.util
import json
import logging
import logging.handlers
import os
from pathlib import Path
import queue
from typing import Any, Dict, Iterable, Optional


def _rich_available() -> bool:
    return importlib.util.find_spec("rich") is not None


if _rich_available():
    rich_logging = importlib.import_module("rich.logging")
    rich_console = importlib.import_module("rich.console")
    rich_text = importlib.import_module("rich.text")
    RichHandler = rich_logging.RichHandler
    Console = rich_console.Console
    Text = rich_text.Text
    console = Console()
else:
    RichHandler = None
    Console = None
    Text = None
    console = None


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("realtime_api")
    logger.setLevel(logging.INFO)

    if RichHandler is not None:
        if not any(isinstance(h, RichHandler) for h in logger.handlers):
            handler = RichHandler(rich_tracebacks=True, console=console)
            formatter = logging.Formatter("%(message)s", datefmt="[%X]")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    else:
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    logger.propagate = False
    return logger


logger = setup_logging()

_queue_listener: logging.handlers.QueueListener | None = None
_queue_handlers: list[logging.Handler] = []
_file_log_path: Path | None = None
_atexit_registered = False


def _shutdown_file_logging() -> None:
    global _queue_listener

    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None


def _remove_queue_handlers() -> None:
    for handler in _queue_handlers:
        for target_logger in (logging.getLogger(), logger):
            if handler in target_logger.handlers:
                target_logger.removeHandler(handler)
    _queue_handlers.clear()


def enable_file_logging(log_path: Path) -> None:
    """Enable background file logging to the supplied log path."""

    global _queue_listener, _file_log_path, _atexit_registered

    log_path = log_path.expanduser()
    if _file_log_path == log_path and _queue_listener is not None:
        return

    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None

    _remove_queue_handlers()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler.setFormatter(formatter)

    log_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    logger.addHandler(queue_handler)
    _queue_handlers.append(queue_handler)

    _queue_listener = logging.handlers.QueueListener(
        log_queue,
        file_handler,
        respect_handler_level=True,
    )
    _queue_listener.start()

    if getattr(_queue_listener, "_thread", None) is not None:
        _queue_listener._thread.daemon = True

    _file_log_path = log_path

    if not _atexit_registered:
        atexit.register(_shutdown_file_logging)
        _atexit_registered = True


def _format_text(message: str, style: str) -> Any:
    if Text is None:
        return message
    return Text(message, style=style)


def log_ws_event(direction: str, event: dict[str, Any]) -> None:
    event_type = event.get("type", "Unknown")
    spammy = {
        "response.output_audio.delta",
        "response.output_audio_transcript.delta",
        "response.function_call_arguments.delta",
    }

    if event_type in spammy:
        return

    event_emojis = {
        "session.update": "🛠️",
        "session.created": "🔌",
        "session.updated": "🔄",
        "input_audio_buffer.append": "🎤",
        "input_audio_buffer.commit": "✅",
        "input_audio_buffer.speech_started": "🗣️",
        "input_audio_buffer.speech_stopped": "🤫",
        "input_audio_buffer.cleared": "🧹",
        "input_audio_buffer.committed": "📨",
        "conversation.item.create": "📝",
        "conversation.item.added": "📥",
        "conversation.item.done": "📝",
        "response.create": "➡️",
        "response.created": "📝",
        "response.content_part.added": "➕",
        "response.content_part.done": "✅",
        "response.output_item.added": "➕",
        "response.output_item.done": "✅",
        "response.output_audio_transcript.delta": "✍️",
        "response.output_audio_transcript.done": "📝",
        "response.output_audio.delta": "🔊",
        "response.output_audio.done": "🔇",
        "response.done": "✔️ ",
        "response.cancel": "⛔",
        "response.function_call_arguments.delta": "📥",
        "response.function_call_arguments.done": "📥",
        "rate_limits.updated": "⏳",
        "error": "❌",
        "conversation.item.input_audio_transcription.completed": "📝",
        "conversation.item.input_audio_transcription.failed": "⚠️",
    }
    emoji = event_emojis.get(event_type, "❓")
    icon = "⬆️ - Out" if direction == "Outgoing" else "⬇️ - In"
    style = "bold cyan" if direction == "Outgoing" else "bold green"
    logger.info(_format_text(f"{emoji} {icon} {event_type}", style=style))


def log_tool_call(function_name: str, args: Any, result: Any) -> None:
    logger.info(_format_text(f"🛠️ Calling function: {function_name} with args: {args}", "bold magenta"))
    logger.info(_format_text(f"🛠️ Function call result: {result}", "bold yellow"))


def log_error(message: str) -> None:
    logger.error(_format_text(message, style="bold red"))


def log_info(message: str, style: str = "bold white") -> None:
    logger.info(_format_text(message, style=style))


def log_warning(message: str) -> None:
    logger.warning(_format_text(message, style="bold yellow"))


MAX_STR = 38
MAX_LIST = 60
TOOL_NAME_CAP = 35
REDACT = True
REDACT_KEYS = {"event_id", "id", "session_id"}
NO_TRUNCATE_KEYS = {"instructions"}

DROP_SESSION_KEYS = (
    "tracing",
    "prompt",
    "include",
)

DEBUG_FULL_PAYLOAD = bool(int(os.getenv("THEO_LOG_SESSION_FULL", "0")))

MARK_SUMMARY = "--- SESSION_UPDATED_SUMMARY ---"
MARK_PAYLOAD = "--- SESSION_UPDATED_PAYLOAD ---"

KEEP_SESSION_KEYS = {
    "model",
    "output_modalities",
    "tool_choice",
    "max_output_tokens",
    "truncation",
    "audio",
    "tools",
    "instructions",
}


def _keep_only(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: d[k] for k in keys if k in d}


def _truncate_str(s: str, max_len: int = MAX_STR) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _first_line(s: str, max_len: int = 160) -> str:
    if not s:
        return ""
    line = s.strip().splitlines()[0]
    return _truncate_str(line, max_len)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _shorten_list(lst: list[Any], max_items: int = MAX_LIST) -> list[Any]:
    if len(lst) <= max_items:
        return lst
    return lst[:max_items] + [f"… ({len(lst) - max_items} more)"]


def _normalize_for_log(obj: Any, *, _key: Optional[str] = None) -> Any:
    if isinstance(obj, str):
        if _key in NO_TRUNCATE_KEYS:
            return obj
        return _truncate_str(obj)

    if isinstance(obj, list):
        return [_normalize_for_log(x) for x in _shorten_list(obj)]

    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k in sorted(obj.keys(), key=str):
            v = obj[k]
            if REDACT and k in REDACT_KEYS and isinstance(v, str):
                out[k] = "<redacted>"
            else:
                out[k] = _normalize_for_log(v, _key=k)
        return out

    return obj


def _compact_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []
    compact: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        params = ((tool.get("parameters") or {}).get("properties") or {})
        required = (tool.get("parameters") or {}).get("required") or []
        compact.append(
            {
                "name": tool.get("name"),
                "type": tool.get("type"),
                "required": required,
                "params": sorted(params.keys()),
            }
        )
    return compact


def _extract_summary(event: Dict[str, Any]) -> Dict[str, Any]:
    sess = (event or {}).get("session") or {}
    audio = sess.get("audio") or {}

    in_cfg = audio.get("input") or {}
    out_cfg = audio.get("output") or {}

    in_fmt = in_cfg.get("format") or {}
    out_fmt = out_cfg.get("format") or {}

    turn = in_cfg.get("turn_detection") or {}
    tools = sess.get("tools") or []

    instructions = sess.get("instructions") or ""
    instr_digest = {
        "len": len(instructions),
        "sha256": _sha256(instructions)[:12],
        "preview": _first_line(instructions),
    }

    tool_names = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("name"):
            tool_names.append(tool["name"])

    summary = {
        "type": event.get("type"),
        "event_id": event.get("event_id"),
        "session": {
            "id": sess.get("id"),
            "model": sess.get("model"),
            "output_modalities": sess.get("output_modalities"),
            "tool_choice": sess.get("tool_choice"),
            "max_output_tokens": sess.get("max_output_tokens"),
            "truncation": sess.get("truncation"),
            "expires_at": sess.get("expires_at"),
        },
        "audio": {
            "voice": out_cfg.get("voice"),
            "speed": out_cfg.get("speed"),
            "in": f"{in_fmt.get('type')}@{in_fmt.get('rate')}",
            "out": f"{out_fmt.get('type')}@{out_fmt.get('rate')}",
            "vad": {
                "type": turn.get("type"),
                "threshold": turn.get("threshold"),
                "prefix_padding_ms": turn.get("prefix_padding_ms"),
                "silence_duration_ms": turn.get("silence_duration_ms"),
                "idle_timeout_ms": turn.get("idle_timeout_ms"),
                "create_response": turn.get("create_response"),
                "interrupt_response": turn.get("interrupt_response"),
            },
        },
        "tools": {
            "count": len(tools),
            "names": tool_names[:TOOL_NAME_CAP]
            + (["…"] if len(tool_names) > TOOL_NAME_CAP else []),
        },
        "instructions_digest": instr_digest,
    }

    if REDACT:
        if summary.get("event_id"):
            summary["event_id"] = "<redacted>"
        if summary["session"].get("id"):
            summary["session"]["id"] = "<redacted>"

    return summary


def _prune_session(sess: Dict[str, Any]) -> Dict[str, Any]:
    if not DROP_SESSION_KEYS:
        return dict(sess)
    pruned = dict(sess)
    for k in DROP_SESSION_KEYS:
        pruned.pop(k, None)
    return pruned


def _prune_nones(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _prune_nones(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_prune_nones(x) for x in obj]
    return obj


def _headline(summary: Dict[str, Any]) -> str:
    sess = summary.get("session") or {}
    audio = summary.get("audio") or {}
    vad = audio.get("vad") or {}
    instr = summary.get("instructions_digest") or {}
    tools = summary.get("tools") or {}
    return (
        "SESSION_UPDATED | "
        f"model={sess.get('model')} | "
        f"voice={audio.get('voice')} | "
        f"vad={vad.get('type')}(th={vad.get('threshold')},pre={vad.get('prefix_padding_ms')},"
        f"sil={vad.get('silence_duration_ms')}) | "
        f"tools={tools.get('count')} | "
        f"instr={instr.get('sha256')} ({instr.get('len')})"
    )


def log_session_updated(event: Dict[str, Any], *, full_payload: Optional[bool] = None) -> None:
    if full_payload is None:
        full_payload = DEBUG_FULL_PAYLOAD

    print(MARK_SUMMARY)
    summary = _extract_summary(event)
    print(_headline(summary))
    print(json.dumps(_normalize_for_log(summary), indent=2, ensure_ascii=False))

    if not full_payload:
        return

    print(MARK_PAYLOAD)
    event2 = dict(event)
    event2 = _prune_nones(event2)

    sess = dict((event2.get("session") or {}))
    sess = _prune_session(sess)
    sess = _keep_only(sess, KEEP_SESSION_KEYS)

    sess["tools"] = _compact_tools(sess.get("tools"))

    event2["session"] = sess
    print(json.dumps(_normalize_for_log(event2), indent=2, ensure_ascii=False))
