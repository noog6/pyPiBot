from __future__ import annotations

import asyncio
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import ai.tools as ai_tools
from ai.realtime_api import RealtimeAPI
from services.memory_manager import MemoryManager, MemoryScope
from storage.memories import MemoryStore


def _base_api() -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api._stop_words = {"halt"}
    api._preference_recall_cache = {}
    api._preference_recall_cooldown_s = 10.0
    api._pending_preference_recall_trace = None
    api._tool_call_records = []
    api._turn_diagnostic_timestamps = {}
    api._preference_recall_max_attempts = 1
    api._preference_recall_min_semantic_score = 0.15
    api._preference_recall_min_lexical_score = 0.1
    api._preference_recall_allow_low_score_debug = False
    api._memory_retrieval_scope = "user_global"
    api._current_run_id = lambda: "run-test"
    api._current_turn_id_or_unknown = lambda: "turn_1"
    api._extract_preference_keywords = RealtimeAPI._extract_preference_keywords.__get__(api, RealtimeAPI)
    api._preference_recall_memories_from_payload = RealtimeAPI._preference_recall_memories_from_payload.__get__(api, RealtimeAPI)
    api._sanitize_memory_cards_text_for_user = RealtimeAPI._sanitize_memory_cards_text_for_user.__get__(api, RealtimeAPI)
    api._build_preference_query_fingerprint = RealtimeAPI._build_preference_query_fingerprint.__get__(api, RealtimeAPI)
    return api


def _make_memory_manager(store: MemoryStore) -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._active_user_id = "default"
    manager._active_session_id = None
    manager._default_scope = MemoryScope.USER_GLOBAL
    manager._store = store
    manager._embedding_worker = None
    manager._semantic_config = SimpleNamespace(enabled=False, rerank_enabled=False)
    manager._last_turn_retrieval_at = {}
    manager._auto_pin_min_importance = 5
    manager._auto_pin_requires_review = True
    manager._auto_reflection_semantic_dedupe_enabled = False
    manager._auto_reflection_dedupe_recent_limit = 24
    manager._auto_reflection_dedupe_high_risk_cosine = 0.9
    manager._auto_reflection_dedupe_policy = "skip_write"
    manager._auto_reflection_dedupe_importance = 2
    manager._auto_reflection_dedupe_clear_pin = True
    manager._auto_reflection_dedupe_needs_review = True
    manager._auto_reflection_dedupe_apply_to_manual_tool = False
    manager._recall_trace_enabled = True
    manager._recall_trace_level = "info"
    return manager


def test_find_stop_word_delegates_to_runtime() -> None:
    api = _base_api()
    with patch("ai.realtime_api.preference_recall_runtime._find_stop_word", return_value="halt") as mocked:
        result = api._find_stop_word("please halt now")
    mocked.assert_called_once_with(api, "please halt now")
    assert result == "halt"


def test_preference_recall_intent_delegates_to_runtime() -> None:
    api = _base_api()
    websocket = object()
    with patch(
        "ai.realtime_api.preference_recall_runtime._maybe_handle_preference_recall_intent",
        new=AsyncMock(return_value=True),
    ) as mocked:
        result = asyncio.run(api._maybe_handle_preference_recall_intent("text", websocket, source="test"))
    mocked.assert_awaited_once_with(api, "text", websocket, source="test")
    assert result is True


def test_preference_recall_filters_irrelevant_low_score() -> None:
    api = _base_api()
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)
    payload = {
        "memories": [
            {"content": "User's preferred editor is Vim."},
            {"content": "User's pants are grey in color."},
        ],
        "memory_cards": [
            {
                "memory": "User's preferred editor is Vim.",
                "why_relevant": "It matches your query about 'editor'. Evidence: lexical exact match on 'editor'.",
                "confidence": "High",
            },
            {
                "memory": "User's pants are grey in color.",
                "why_relevant": "It matches your query about 'preference'. Evidence: semantic similarity=0.00.",
                "confidence": "Low",
            },
        ],
        "memory_cards_text": "placeholder",
    }

    filtered = api._filter_preference_recall_payload_for_user(payload, query="favorite editor")

    assert len(filtered["memory_cards"]) == 1
    assert "pants" not in filtered["memory_cards_text"].lower()


def test_preference_recall_no_double_attempt_by_default() -> None:
    api = _base_api()
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)
    recall_fn = AsyncMock(return_value={"memories": [], "memory_cards": [], "memory_cards_text": ""})

    asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=recall_fn,
            source="test",
            resolved_turn_id="turn_1",
            query="favorite editor",
        )
    )

    assert recall_fn.await_count == 1


def test_preference_query_builder_keeps_preference_cues_and_domain() -> None:
    api = _base_api()

    query = RealtimeAPI._build_preference_recall_query(
        api,
        "hey theo do you remember what my favorite editor is",
        keywords=["hey", "theo", "remember", "editor"],
    )

    assert "editor" in query
    assert "favorite" in query
    assert "preferred" in query


def test_preference_query_builder_keeps_keyword_lookup_token() -> None:
    api = _base_api()

    query = RealtimeAPI._build_preference_recall_query(
        api,
        "can you check your memories for anything related to vim",
        keywords=["can", "check", "your", "memories", "anything", "related", "vim"],
    )

    assert "vim" in query


def test_preference_recall_run441_query_variants_return_vim() -> None:
    api = _base_api()
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)

    async def _fake_recall(*, query: str, limit: int, scope: str):
        _ = (limit, scope)
        if "editor" in query or "vim" in query:
            return {
                "memories": [
                    {"content": "User's preferred editor is Vim."},
                    {"content": "User's favorite editor is Vim."},
                ],
                "memory_cards": [
                    {
                        "memory": "User's preferred editor is Vim.",
                        "why_relevant": "It matches your query about 'Vim'. Evidence: lexical exact match on 'vim'.",
                        "confidence": "High",
                    },
                    {
                        "memory": "User's favorite editor is Vim.",
                        "why_relevant": "It matches your query about 'editor'. Evidence: lexical exact match on 'editor'.",
                        "confidence": "High",
                    },
                ],
                "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
                "trace": {
                    "retrieval_mode": "lexical",
                    "candidate_counts": {
                        "lexical_candidates": 2,
                        "semantic_candidates": 0,
                        "combined_candidates": 2,
                    },
                },
            }
        return {"memories": [], "memory_cards": [], "memory_cards_text": "", "trace": {"retrieval_mode": "lexical", "candidate_counts": {"lexical_candidates": 0, "semantic_candidates": 0, "combined_candidates": 0}}}

    query_1 = RealtimeAPI._build_preference_recall_query(
        api,
        "hey theo do you remember what my favorite editor is",
        keywords=["hey", "theo", "remember", "editor"],
    )
    payload_1, _ = asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=_fake_recall,
            source="input_audio_transcription",
            resolved_turn_id="turn_2",
            query=query_1,
        )
    )
    assert any("Vim" in memory["content"] for memory in payload_1["memories"])

    query_2 = RealtimeAPI._build_preference_recall_query(
        api,
        "can you check your memories for anything related to vim",
        keywords=["can", "check", "your", "memories", "anything", "related", "vim"],
    )
    payload_2, _ = asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=_fake_recall,
            source="input_audio_transcription",
            resolved_turn_id="turn_3",
            query=query_2,
        )
    )
    assert any("Vim" in memory["content"] for memory in payload_2["memories"])


def test_preference_recall_empty_logs_empty_reason() -> None:
    api = _base_api()
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)
    recall_fn = AsyncMock(
        return_value={
            "memories": [],
            "memory_cards": [],
            "memory_cards_text": "",
            "trace": {
                "retrieval_mode": "lexical",
                "candidate_counts": {
                    "lexical_candidates": 0,
                    "semantic_candidates": 0,
                    "combined_candidates": 0,
                },
            },
        }
    )

    with patch("ai.realtime_api.logger.info") as mocked_info:
        asyncio.run(
            RealtimeAPI._run_preference_recall_with_fallbacks(
                api,
                recall_fn=recall_fn,
                source="input_audio_transcription",
                resolved_turn_id="turn_2",
                query="favorite editor",
            )
        )

    rendered = "\n".join(str(call.args[0]) % call.args[1:] for call in mocked_info.call_args_list if call.args)
    assert "empty_reason=no_candidates" in rendered
    assert "query_lineage=0:editor|domain_only" in rendered


def test_preference_recall_empty_without_trace_logs_trace_unavailable_reason() -> None:
    api = _base_api()
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)
    recall_fn = AsyncMock(return_value={"memories": [], "memory_cards": [], "memory_cards_text": ""})

    with patch("ai.realtime_api.logger.info") as mocked_info:
        asyncio.run(
            RealtimeAPI._run_preference_recall_with_fallbacks(
                api,
                recall_fn=recall_fn,
                source="input_audio_transcription",
                resolved_turn_id="turn_2",
                query="favorite editor",
            )
        )

    rendered = "\n".join(str(call.args[0]) % call.args[1:] for call in mocked_info.call_args_list if call.args)
    assert "empty_reason=no_hit_trace_unavailable" in rendered
    assert "query_lineage=0:editor|domain_only" in rendered


def test_preference_recall_hit_contract_sets_none_reason_and_synthesizes_text() -> None:
    api = _base_api()
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)
    recall_fn = AsyncMock(
        return_value={
            "memories": [{"content": "User's preferred editor is Vim."}],
            "memory_cards": [
                {
                    "memory": "User's preferred editor is Vim.",
                    "why_relevant": "It matches your query about 'editor'. Evidence: lexical exact match on 'editor'.",
                    "confidence": "High",
                }
            ],
            "memory_cards_text": "",
            "trace": {
                "retrieval_mode": "lexical",
                "candidate_counts": {
                    "lexical_candidates": 2,
                    "semantic_candidates": 0,
                    "combined_candidates": 2,
                },
            },
        }
    )

    payload, handled = asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=recall_fn,
            source="input_audio_transcription",
            resolved_turn_id="turn_2",
            query="favorite editor",
        )
    )

    assert handled is True
    assert payload["hit"] is True
    assert payload["empty_reason"] == "none"
    assert payload["returned_count"] >= 1
    assert "Relevant memory" in payload["memory_cards_text"]


def test_preference_recall_miss_contract_sets_non_none_empty_reason() -> None:
    api = _base_api()
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)
    recall_fn = AsyncMock(
        return_value={
            "memories": [],
            "memory_cards": [],
            "memory_cards_text": "",
            "trace": {
                "retrieval_mode": "lexical",
                "candidate_counts": {
                    "lexical_candidates": 0,
                    "semantic_candidates": 0,
                    "combined_candidates": 0,
                },
            },
        }
    )

    payload, handled = asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=recall_fn,
            source="input_audio_transcription",
            resolved_turn_id="turn_2",
            query="favorite editor",
        )
    )

    assert handled is True
    assert payload["hit"] is False
    assert payload["empty_reason"] != "none"


def test_preference_recall_scope_forwarded_to_recall_tool() -> None:
    api = _base_api()
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)
    observed_scopes: list[str] = []

    async def _fake_recall(*, query: str, limit: int, scope: str):
        _ = (query, limit)
        observed_scopes.append(scope)
        return {"memories": [], "memory_cards": [], "memory_cards_text": ""}

    api._memory_retrieval_scope = "user_global"
    asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=_fake_recall,
            source="input_audio_transcription",
            resolved_turn_id="turn_1",
            query="favorite editor",
        )
    )
    api._memory_retrieval_scope = "user_profile"
    asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=_fake_recall,
            source="input_audio_transcription",
            resolved_turn_id="turn_1",
            query="favorite editor",
        )
    )

    assert observed_scopes == ["user_global", "user_profile"]



def test_preference_query_variants_keep_domain_and_value_tokens() -> None:
    api = _base_api()
    api._build_preference_recall_query_variants = RealtimeAPI._build_preference_recall_query_variants.__get__(api, RealtimeAPI)

    variants = RealtimeAPI._build_preference_recall_query_variants(api, "editor favorite vim user preference")

    assert variants[0] == ("editor", "domain_only")
    assert any("editor" in variant and "vim" in variant for variant, _ in variants)
    assert all(" user " not in f" {variant} " for variant, _ in variants)


def test_preference_recall_uses_variant_retry_for_run441_style_miss() -> None:
    api = _base_api()
    api._preference_recall_max_attempts = 2
    api._build_preference_recall_query_variants = RealtimeAPI._build_preference_recall_query_variants.__get__(api, RealtimeAPI)
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)

    calls: list[str] = []

    async def _fake_recall(*, query: str, limit: int, scope: str):
        _ = (limit, scope)
        calls.append(query)
        if "preferred editor vim" in query or "favorite editor vim" in query:
            return {
                "memories": [
                    {"content": "User's preferred editor is Vim."},
                    {"content": "User's favorite editor is Vim."},
                ],
                "memory_cards": [
                    {
                        "memory": "User's preferred editor is Vim.",
                        "why_relevant": "It matches your query about 'Vim'. Evidence: lexical exact match on 'vim'.",
                        "confidence": "High",
                    },
                    {
                        "memory": "User's favorite editor is Vim.",
                        "why_relevant": "It matches your query about 'editor'. Evidence: lexical exact match on 'editor'.",
                        "confidence": "High",
                    },
                ],
                "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
                "trace": {"retrieval_mode": "hybrid", "candidate_counts": {"lexical_candidates": 2, "semantic_candidates": 0, "combined_candidates": 2}},
            }
        return {"memories": [], "memory_cards": [], "memory_cards_text": "", "trace": {"retrieval_mode": "hybrid", "candidate_counts": {"lexical_candidates": 0, "semantic_candidates": 0, "combined_candidates": 0}}}

    payload, _ = asyncio.run(
        RealtimeAPI._run_preference_recall_with_fallbacks(
            api,
            recall_fn=_fake_recall,
            source="input_audio_transcription",
            resolved_turn_id="turn_2",
            query="editor favorite vim user preference",
        )
    )

    assert len(calls) == 2
    assert calls[0] == "editor"
    assert any("preferred editor vim" in call or "favorite editor vim" in call for call in calls)
    assert any("Vim" in memory["content"] for memory in payload["memories"])


def test_preference_recall_query_lineage_tracks_variants_and_empty_reason() -> None:
    api = _base_api()
    api._preference_recall_max_attempts = 2
    api._build_preference_recall_query_variants = RealtimeAPI._build_preference_recall_query_variants.__get__(api, RealtimeAPI)
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)

    async def _fake_recall(*, query: str, limit: int, scope: str):
        _ = (query, limit, scope)
        return {
            "memories": [],
            "memory_cards": [],
            "memory_cards_text": "",
            "trace": {
                "retrieval_mode": "hybrid",
                "candidate_counts": {
                    "lexical_candidates": 0,
                    "semantic_candidates": 0,
                    "combined_candidates": 0,
                },
            },
        }

    with patch("ai.realtime_api.logger.info") as mocked_info:
        asyncio.run(
            RealtimeAPI._run_preference_recall_with_fallbacks(
                api,
                recall_fn=_fake_recall,
                source="input_audio_transcription",
                resolved_turn_id="turn_2",
                query="editor favorite vim user preference",
            )
        )

    rendered = "\n".join(str(call.args[0]) % call.args[1:] for call in mocked_info.call_args_list if call.args)
    assert "empty_reason=no_candidates" in rendered
    assert "query_lineage=0:editor|domain_only" in rendered
    assert "query_lineage=0:editor|domain_only;1:preferred editor vim|canonical" in rendered

def test_preference_recall_hit_attaches_memory_context_for_primary_response() -> None:
    api = _base_api()
    api._current_input_event_key = "evt_1"
    api._is_preference_recall_intent = Mock(return_value=(True, ["editor"]))
    api._build_preference_recall_query = Mock(return_value="editor")
    api._mark_preference_recall_candidate = Mock()
    api._clear_preference_recall_candidate = Mock()
    api._run_preference_recall_with_fallbacks = AsyncMock(
        return_value=(
            {
                "memories": [{"content": "User's favorite editor is Vim."}],
                "memory_cards": [],
                "memory_cards_text": "Relevant memory:\n- \"User's favorite editor is Vim.\"",
            },
            True,
        )
    )
    api._preference_recall_locked_input_event_keys = set()
    captured_contexts: list[dict[str, object]] = []

    def _capture_context(**kwargs):
        captured_contexts.append(kwargs)

    api._set_pending_preference_memory_context = _capture_context
    api.send_assistant_message = AsyncMock()

    with patch("ai.realtime.preference_recall_runtime.function_map", {"recall_memories": AsyncMock()}):
        handled = asyncio.run(api._maybe_handle_preference_recall_intent("remember my editor", object(), source="input_audio_transcription"))

    assert handled is False
    assert captured_contexts
    memory_context = captured_contexts[0]["memory_context"]
    assert memory_context["hit"] is True
    assert "Vim" in str(memory_context["prompt_note"])
    api.send_assistant_message.assert_not_awaited()
    assert "evt_1" not in api._preference_recall_locked_input_event_keys


def test_preference_recall_miss_context_has_no_memory_claim_language() -> None:
    api = _base_api()
    api._current_input_event_key = "evt_2"
    api._is_preference_recall_intent = Mock(return_value=(True, ["pants"]))
    api._build_preference_recall_query = Mock(return_value="pants color")
    api._mark_preference_recall_candidate = Mock()
    api._clear_preference_recall_candidate = Mock()
    api._run_preference_recall_with_fallbacks = AsyncMock(return_value=({"memories": [], "memory_cards": [], "memory_cards_text": ""}, True))
    api._preference_recall_locked_input_event_keys = set()
    captured_contexts: list[dict[str, object]] = []

    def _capture_context(**kwargs):
        captured_contexts.append(kwargs)

    api._set_pending_preference_memory_context = _capture_context
    api.send_assistant_message = AsyncMock()

    with patch("ai.realtime.preference_recall_runtime.function_map", {"recall_memories": AsyncMock()}):
        handled = asyncio.run(api._maybe_handle_preference_recall_intent("what color are my pants", object(), source="input_audio_transcription"))

    assert handled is False
    assert captured_contexts
    prompt_note = str(captured_contexts[0]["memory_context"]["prompt_note"]).lower()
    assert "i remember" not in prompt_note
    assert "you mentioned" not in prompt_note
    assert "do not claim memory" in prompt_note
    api.send_assistant_message.assert_not_awaited()


def test_deliverable_seen_true_for_tool_output_turn() -> None:
    api = _base_api()
    api._utterance_info_summary = {"deliverable_seen": False}
    api._mark_utterance_info_summary = RealtimeAPI._mark_utterance_info_summary.__get__(api, RealtimeAPI)
    api._extract_assistant_text_from_content = RealtimeAPI._extract_assistant_text_from_content.__get__(api, RealtimeAPI)
    api._is_active_response_guarded = lambda: False
    api._cancel_micro_ack = lambda **_kwargs: None
    api._mark_first_assistant_utterance_observed_if_needed = lambda _text: None
    api.state_manager = SimpleNamespace(update_state=lambda *_args, **_kwargs: None)
    api._set_response_delivery_state = lambda **_kwargs: None
    api._current_turn_id_or_unknown = lambda: "turn_4"
    api._active_response_input_event_key = "evt_4"
    api._current_input_event_key = "evt_4"
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    RealtimeAPI._on_assistant_output_item_added(
        api,
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Found Vim memories."}],
        },
    )

    assert bool(api._utterance_info_summary.get("deliverable_seen")) is True


def test_preference_recall_run453_divergence_parity_with_direct_domain_recall(monkeypatch, tmp_path) -> None:
    api = _base_api()
    api._preference_recall_max_attempts = 2
    api._build_preference_recall_query_variants = lambda _query: [("favorite preference", "canonical")]
    api._preference_recall_fallback_query = RealtimeAPI._preference_recall_fallback_query.__get__(api, RealtimeAPI)
    api._filter_preference_recall_payload_for_user = RealtimeAPI._filter_preference_recall_payload_for_user.__get__(api, RealtimeAPI)

    store = MemoryStore(db_path=tmp_path / "memories.db")
    manager = _make_memory_manager(store)
    manager.remember_memory(
        content="User's preferred editor is Vim.",
        tags=["preference", "editor"],
        importance=4,
        scope=MemoryScope.USER_GLOBAL,
    )
    manager.remember_memory(
        content="User's favorite editor is Vim.",
        tags=["preference", "editor"],
        importance=4,
        scope=MemoryScope.USER_GLOBAL,
    )
    monkeypatch.setattr(ai_tools.MemoryManager, "get_instance", lambda: manager)

    direct_payload = asyncio.run(ai_tools.recall_memories(query="editor", scope="user_global"))
    assert direct_payload["memories"]
    assert isinstance(direct_payload.get("trace"), dict)

    attempted_calls: list[dict[str, str]] = []
    direct_recall_fn = ai_tools.recall_memories

    async def _recall_via_function_map(*, query: str, limit: int, scope: str):
        attempted_calls.append({"query": query, "scope": scope})
        return await direct_recall_fn(query=query, limit=limit, scope=scope)

    monkeypatch.setitem(ai_tools.function_map, "recall_memories", _recall_via_function_map)

    query = RealtimeAPI._build_preference_recall_query(
        api,
        "hey theo do you remember what my favorite editor is",
        keywords=["hey", "theo", "remember", "favorite", "editor"],
    )

    with patch("ai.realtime_api.logger.info") as mocked_info:
        payload, handled = asyncio.run(
            RealtimeAPI._run_preference_recall_with_fallbacks(
                api,
                recall_fn=ai_tools.function_map["recall_memories"],
                source="text_message",
                resolved_turn_id="turn_3",
                query=query,
            )
        )

    assert handled is True
    assert [call["query"] for call in attempted_calls] == ["favorite preference", "editor"]
    assert {call["scope"] for call in attempted_calls} == {"user_global"}

    rendered_lines = [
        str(call.args[0]) % call.args[1:]
        for call in mocked_info.call_args_list
        if call.args and str(call.args[0]).startswith("preference_recall_tool_result")
    ]
    assert len(rendered_lines) == 2
    assert "query=favorite preference" in rendered_lines[0]
    assert "empty_reason=no_candidates" in rendered_lines[0]
    assert "scope=user_global" in rendered_lines[0]
    assert "retrieval_backend=" in rendered_lines[0]
    assert "query=editor" in rendered_lines[1]
    assert "empty_reason=none" in rendered_lines[1]
    assert "scope=user_global" in rendered_lines[1]
    assert "retrieval_backend=" in rendered_lines[1]

    domain_hit_succeeded = any("query=editor" in line and "empty_reason=none" in line for line in rendered_lines)
    assert direct_payload["memories"]
    assert payload["memories"] or payload["memory_cards"] or domain_hit_succeeded
