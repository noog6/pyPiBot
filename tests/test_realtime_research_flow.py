"""Regression tests for realtime research intent handling."""

from __future__ import annotations

import asyncio

from ai.realtime_api import RealtimeAPI
from services.research.models import ResearchPacket, ResearchRequest


class _FakeService:
    def __init__(self, packet: ResearchPacket) -> None:
        self.packet = packet
        self.calls: list[ResearchRequest] = []

    def request_research(self, request: ResearchRequest) -> ResearchPacket:
        self.calls.append(request)
        return self.packet


def _make_api_stub() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.orchestration_state = type("S", (), {"transition": lambda *args, **kwargs: None})()
    api._research_enabled = True
    return api


def test_auto_approved_research_intent_short_circuits_normal_flow() -> None:
    api = _make_api_stub()
    api._research_permission_required = False

    calls: list[str] = []

    async def _dispatch(request: ResearchRequest, websocket: object) -> None:
        calls.append(request.prompt)

    api._dispatch_research_request = _dispatch

    handled = asyncio.run(
        api._maybe_process_research_intent(
            "find datasheet for ads1015",
            websocket=object(),
            source="text_message",
        )
    )

    assert handled is True
    assert calls == ["find datasheet for ads1015"]


def test_dispatch_research_uses_worker_thread(monkeypatch) -> None:
    api = _make_api_stub()
    sent: list[str] = []

    async def _send_assistant_message(message: str, websocket: object) -> None:
        sent.append(message)

    packet = ResearchPacket(status="ok", answer_summary="researched")
    service = _FakeService(packet)
    api._research_service = service
    api.send_assistant_message = _send_assistant_message

    thread_calls: list[tuple[object, tuple[object, ...]]] = []

    async def _fake_to_thread(func, *args, **kwargs):
        thread_calls.append((func, args))
        return func(*args, **kwargs)

    monkeypatch.setattr("ai.realtime_api.asyncio.to_thread", _fake_to_thread)

    asyncio.run(api._dispatch_research_request(ResearchRequest(prompt="find sensor pinout"), object()))

    assert len(thread_calls) == 1
    assert service.calls and service.calls[0].prompt == "find sensor pinout"
    assert sent == ["researched"]


def test_send_initial_prompt_routes_research_intent() -> None:
    api = _make_api_stub()
    api.prompts = ["Can you search the web for Waveshare Servo Driver HAT voltage requirements?"]

    calls: list[str] = []

    async def _dispatch(request: ResearchRequest, websocket: object) -> None:
        calls.append(request.prompt)

    api._dispatch_research_request = _dispatch
    api._research_permission_required = False

    class _FakeWebsocket:
        def __init__(self) -> None:
            self.events: list[str] = []

        async def send(self, payload: str) -> None:
            self.events.append(payload)

    ws = _FakeWebsocket()

    asyncio.run(api.send_initial_prompts(ws))

    assert calls == [api.prompts[0]]
    assert ws.events == []
