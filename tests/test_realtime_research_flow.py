"""Regression tests for realtime research intent handling."""

from __future__ import annotations

import asyncio

from ai.realtime_api import ConfirmationState, RealtimeAPI
from services.research.models import ResearchPacket, ResearchRequest
from storage.trusted_domains import TrustedDomainStore


class _FakeService:
    def __init__(self, packet: ResearchPacket, *, discovered_domains: list[str] | None = None) -> None:
        self.packet = packet
        self.calls: list[ResearchRequest] = []
        self.discovered_domains = discovered_domains or []

    def request_research(self, request: ResearchRequest) -> ResearchPacket:
        self.calls.append(request)
        return self.packet

    def discover_domains(self, request: ResearchRequest) -> list[str]:
        return list(self.discovered_domains)

class _InMemoryTrustedDomainStore:
    def __init__(self) -> None:
        self.domains: set[str] = set()

    def add_domain(self, domain_or_url: str, *, added_by: str = "user") -> str | None:
        domain = str(domain_or_url).strip().lower()
        if not domain:
            return None
        self.domains.add(domain)
        return domain

    def get_domain_set(self) -> set[str]:
        return set(self.domains)



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
    api._trusted_domain_store = _InMemoryTrustedDomainStore()
    api._trusted_research_domains = set()
    api._research_service = _FakeService(ResearchPacket(status="ok", answer_summary="ok"))
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
    assert api._pending_confirmation_token.metadata.get("domains") == ["example.com"]
    assert api._pending_confirmation_token.metadata.get("domains_known") is True
    assert prompts and "example.com" in prompts[0]


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
    assert prompts and "arxiv.org" in prompts[0]


def test_search_preflight_domains_are_included_in_permission_prompt() -> None:
    api = _make_api_stub()
    api._research_mode = "ask"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()
    api._research_service = _FakeService(
        ResearchPacket(status="ok", answer_summary="ok"),
        discovered_domains=["https://nxp.com/products", "slashdot.org/articles/1"],
    )

    prompts: list[str] = []

    async def _send_assistant_message(message: str, *args, **kwargs) -> None:
        prompts.append(message)

    api.send_assistant_message = _send_assistant_message

    handled = asyncio.run(
        api._maybe_process_research_intent(
            "search the web for RP2040 errata",
            websocket=object(),
            source="text_message",
        )
    )

    assert handled is True
    assert prompts
    assert "nxp.com" in prompts[0]
    assert "slashdot.org" in prompts[0]
    assert api._pending_confirmation_token is not None
    assert api._pending_confirmation_token.metadata.get("domains_known") is True


def test_permission_approval_adds_domain_and_dispatches() -> None:
    api = _make_api_stub()
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "explicit"
    api._research_firecrawl_allowlist_domains = {"wikipedia.org"}

    calls: list[str] = []

    async def _dispatch(request: ResearchRequest, websocket: object) -> None:
        calls.append(request.prompt)

    api._dispatch_research_request = _dispatch

    async def _send_assistant_message(message: str, *args, **kwargs) -> None:
        return None

    api.send_assistant_message = _send_assistant_message

    asyncio.run(
        api._maybe_process_research_intent(
            "search the web for https://example.com/deep-dive",
            websocket=object(),
            source="text_message",
        )
    )

    handled = asyncio.run(api._maybe_handle_research_permission_response("yes", object()))

    assert handled is True
    assert calls == ["search the web for https://example.com/deep-dive"]
    assert "example.com" in api._trusted_research_domains
    assert "example.com" in api._research_firecrawl_allowlist_domains


def test_permission_decline_does_not_add_domain() -> None:
    api = _make_api_stub()
    api._research_mode = "auto"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "explicit"
    api._research_firecrawl_allowlist_domains = {"wikipedia.org"}

    async def _send_assistant_message(message: str, *args, **kwargs) -> None:
        return None

    api.send_assistant_message = _send_assistant_message

    asyncio.run(
        api._maybe_process_research_intent(
            "search the web for https://example.com/deep-dive",
            websocket=object(),
            source="text_message",
        )
    )

    handled = asyncio.run(api._maybe_handle_research_permission_response("no", object()))

    assert handled is True
    assert "example.com" not in api._trusted_research_domains


def test_unknown_domains_prompt_offers_preview_first() -> None:
    api = _make_api_stub()
    api._research_mode = "ask"
    api._research_provider = "openai"
    api._research_firecrawl_enabled = False
    api._research_firecrawl_allowlist_mode = "public"
    api._research_firecrawl_allowlist_domains = set()
    api._research_service = _FakeService(ResearchPacket(status="ok", answer_summary="ok"), discovered_domains=[])

    prompts: list[str] = []

    async def _send_assistant_message(message: str, *args, **kwargs) -> None:
        prompts.append(message)

    api.send_assistant_message = _send_assistant_message

    handled = asyncio.run(
        api._maybe_process_research_intent(
            "please look up the latest ai chip news",
            websocket=object(),
            source="text_message",
        )
    )

    assert handled is True
    assert prompts
    assert "don't know the source domains yet" in prompts[0]
    assert "preview domains first" in prompts[0]
    assert api._pending_confirmation_token is not None
    assert api._pending_confirmation_token.metadata.get("domains_known") is False


def test_trusted_domain_store_persists_across_instances(tmp_path) -> None:
    db_path = tmp_path / "trusted.db"

    first = TrustedDomainStore(db_path=db_path)
    added = first.add_domain("https://Example.COM/deep-dive?foo=1")

    second = TrustedDomainStore(db_path=db_path)
    domains = second.get_domain_set()

    assert added == "example.com"
    assert "example.com" in domains
