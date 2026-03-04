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

from ai.realtime_api import RealtimeAPI


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

def test_preference_recall_does_not_block_response_when_audio_started() -> None:
    api = _base_api()
    api._is_preference_recall_intent = Mock(return_value=(True, ["editor"]))
    api._build_preference_recall_query = Mock(return_value="editor")
    api._mark_preference_recall_candidate = Mock()
    api._clear_preference_recall_candidate = Mock()
    api._run_preference_recall_with_fallbacks = AsyncMock(return_value=({"memories": [], "memory_cards": [], "memory_cards_text": ""}, True))

    with patch("ai.realtime.preference_recall_runtime.function_map", {"recall_memories": AsyncMock()}):
        handled = asyncio.run(api._maybe_handle_preference_recall_intent("remember my editor", object(), source="input_audio_transcription"))

    assert handled is False


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
