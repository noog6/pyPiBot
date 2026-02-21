"""Regression tests for realtime research intent handling."""

from __future__ import annotations

import asyncio

from ai.realtime_api import ConfirmationState, RealtimeAPI
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
    api._pending_action = None
    api._pending_research_request = None
    api._pending_confirmation_token = None
    api._confirmation_state = ConfirmationState.IDLE
    api._prior_research_permission_marker = None
    api._prior_research_permission_grace_s = 8.0
    api._research_permission_outcome_ttl_s = 20.0
    api._research_permission_outcomes = {}
    api._research_suppressed_fingerprints = {}
    api._presented_actions = set()
    api._tool_call_dedupe_ttl_s = 30.0
    api.send_assistant_message = lambda *args, **kwargs: None
    return api


def test_auto_approved_research_intent_short_circuits_normal_flow() -> None:
    api = _make_api_stub()
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()

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
    assert len(sent) == 1
    assert sent[0].startswith("researched\n\n")
    assert (
        "I found sources but did not fetch/parse their contents in this run"
        in sent[0]
    )


def test_send_initial_prompt_routes_research_intent() -> None:
    api = _make_api_stub()
    api.prompts = ["Can you search the web for Waveshare Servo Driver HAT voltage requirements?"]

    calls: list[str] = []

    async def _dispatch(request: ResearchRequest, websocket: object) -> None:
        calls.append(request.prompt)

    api._dispatch_research_request = _dispatch
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()

    class _FakeWebsocket:
        def __init__(self) -> None:
            self.events: list[str] = []

        async def send(self, payload: str) -> None:
            self.events.append(payload)

    ws = _FakeWebsocket()

    asyncio.run(api.send_initial_prompts(ws))

    assert calls == [api.prompts[0]]
    assert ws.events == []


def test_allowlisted_html_url_bypasses_permission_and_dispatches() -> None:
    api = _make_api_stub()
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "explicit"
    api._research_firecrawl_allowlist_domains = {"wikipedia.org"}
    api._pending_confirmation_token = None

    calls: list[str] = []

    async def _dispatch(request: ResearchRequest, websocket: object) -> None:
        calls.append(request.prompt)

    api._dispatch_research_request = _dispatch

    handled = asyncio.run(
        api._maybe_process_research_intent(
            "search the web for https://en.wikipedia.org/wiki/Python_(programming_language)",
            websocket=object(),
            source="text_message",
        )
    )

    assert handled is True
    assert calls
    assert api._pending_confirmation_token is None


def test_non_allowlisted_domain_requires_permission_token() -> None:
    api = _make_api_stub()
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "explicit"
    api._research_firecrawl_allowlist_domains = {"wikipedia.org"}

    prompts: list[str] = []

    async def _send_assistant_message(message: str, *args, **kwargs) -> None:
        prompts.append(message)

    api.send_assistant_message = _send_assistant_message

    handled = asyncio.run(
        api._maybe_process_research_intent(
            "search the web for https://example.com/deep-dive",
            websocket=object(),
            source="text_message",
        )
    )

    assert handled is True
    assert api._pending_confirmation_token is not None
    assert api._pending_confirmation_token.kind == "research_permission"
    assert prompts and "Do I have your permission" in prompts[0]


def test_pdf_url_requires_permission_token() -> None:
    api = _make_api_stub()
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()

    prompts: list[str] = []

    async def _send_assistant_message(message: str, *args, **kwargs) -> None:
        prompts.append(message)

    api.send_assistant_message = _send_assistant_message

    handled = asyncio.run(
        api._maybe_process_research_intent(
            "search the web for https://arxiv.org/pdf/1706.03762.pdf",
            websocket=object(),
            source="text_message",
        )
    )

    assert handled is True
    assert api._pending_confirmation_token is not None
    assert api._pending_confirmation_token.kind == "research_permission"
    assert prompts and "Do I have your permission" in prompts[0]
