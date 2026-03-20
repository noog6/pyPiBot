"""Tests for realtime model configuration normalization and wiring."""

from __future__ import annotations

import asyncio
import os
import sys
import types
from pathlib import Path

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from ai.realtime_api import RealtimeAPI
from config.controller import ConfigController


def _reset_singletons() -> None:
    ConfigController._instance = None


def test_config_controller_sets_safe_realtime_model_default(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("assistant_name: Theo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    realtime_cfg = ConfigController.get_instance().get_config()["realtime"]

    assert realtime_cfg["model"] == "gpt-realtime"


def test_config_controller_preserves_configured_realtime_model(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "assistant_name: Theo\nrealtime:\n  model: gpt-realtime-preview\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    realtime_cfg = ConfigController.get_instance().get_config()["realtime"]

    assert realtime_cfg["model"] == "gpt-realtime-preview"


def test_config_controller_blank_realtime_model_falls_back_to_default(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        "assistant_name: Theo\nrealtime:\n  model: '   '\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    realtime_cfg = ConfigController.get_instance().get_config()["realtime"]

    assert realtime_cfg["model"] == "gpt-realtime"


async def _fake_initialize_session(self, _websocket) -> None:
    return None


async def _fake_process_ws_messages(self, _websocket) -> None:
    return None


async def _fake_send_initial_prompts(self, _websocket) -> None:
    return None


async def _fake_send_audio_loop(self, _websocket) -> None:
    return None


class _ConnectContext:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _TransportStub:
    def __init__(self, seen_urls: list[str], *args, **kwargs) -> None:
        self._seen_urls = seen_urls

    def connect(self, *, url: str, headers, close_timeout: int, ping_interval: int, ping_timeout: int):
        self._seen_urls.append(url)
        return _ConnectContext()


def _make_run_api_stub(*, realtime_model: str = "gpt-realtime") -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.api_key = "test-key"
    api.prompts = None
    api.ready_event = type("Ready", (), {"clear": lambda self: None})()
    api._event_injector = type("Injector", (), {"start": lambda self: None, "stop": lambda self: None})()
    api._audio_output_device_name = None
    api.audio_player = None
    api.loop = None
    api._transport = None
    api._realtime_model = realtime_model
    api._session_connected = False
    api._note_connection_attempt = lambda: None
    api._note_connected = lambda: setattr(api, "_session_connected", True)
    api._note_disconnect = lambda _reason: setattr(api, "_session_connected", False)
    api._note_failure = lambda _reason: None
    api._note_reconnect = lambda: None
    api.initialize_session = types.MethodType(_fake_initialize_session, api)
    api.process_ws_messages = types.MethodType(_fake_process_ws_messages, api)
    api.send_initial_prompts = types.MethodType(_fake_send_initial_prompts, api)
    api.send_audio_loop = types.MethodType(_fake_send_audio_loop, api)
    api.mic = type(
        "Mic",
        (),
        {
            "is_recording": False,
            "start_recording": lambda self: None,
            "stop_recording": lambda self: None,
            "close": lambda self: None,
        },
    )()
    task_registry = type(
        "Tasks",
        (),
        {
            "cancel_all": lambda self, _why: None,
            "await_all": lambda self, timeout_s=1.0: asyncio.sleep(0),
        },
    )()
    api._runtime_task_registry = lambda: task_registry
    api.shutdown_handler = lambda: None
    return api


def test_realtime_run_uses_configured_model_in_websocket_url(monkeypatch) -> None:
    api = _make_run_api_stub(realtime_model="gpt-realtime-preview")
    seen_urls: list[str] = []

    monkeypatch.setattr("ai.realtime_api.RealtimeTransport", lambda *args, **kwargs: _TransportStub(seen_urls))
    monkeypatch.setattr("ai.realtime_api._require_websockets", lambda: type("Ws", (), {"connect": object()})())
    monkeypatch.setattr("ai.realtime_api._resolve_websocket_exceptions", lambda _websockets: (None, RuntimeError))
    monkeypatch.setattr(
        "ai.realtime_api.AudioPlayer",
        lambda **kwargs: type("Player", (), {"close": lambda self: None})(),
    )
    monkeypatch.setattr("ai.realtime_api.configure_websocket_library_logging", lambda: None)
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda *args, **kwargs: None)

    asyncio.run(api.run())

    assert seen_urls == ["wss://api.openai.com/v1/realtime?model=gpt-realtime-preview"]


def test_initialize_session_uses_configured_model_in_session_update(monkeypatch) -> None:
    sent_payloads: list[dict[str, object]] = []
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._realtime_model = "gpt-realtime-preview"
    api.profile_manager = type(
        "P",
        (),
        {
            "get_profile_context": lambda self: type(
                "Ctx", (), {"to_instruction_block": lambda self: "profile"}
            )()
        },
    )()
    api._vad_turn_detection = {
        "profile": "default",
        "threshold": 0.2,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 900,
        "create_response": True,
        "interrupt_response": True,
    }
    api._build_startup_memory_digest_note = lambda: None

    class _Transport:
        async def send_json(self, websocket, payload):
            sent_payloads.append(payload)

    api._transport = _Transport()

    monkeypatch.setattr(
        "ai.realtime_api.ReflectionManager.get_instance",
        lambda: type("R", (), {"get_recent_lessons": lambda self: []})(),
    )
    monkeypatch.setattr("ai.realtime_api.logger.info", lambda *args, **kwargs: None)

    asyncio.run(api.initialize_session(object()))

    assert sent_payloads[0]["session"]["model"] == "gpt-realtime-preview"
