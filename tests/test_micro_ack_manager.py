from __future__ import annotations

import asyncio
import random

from ai.micro_ack_manager import MicroAckCategory, MicroAckConfig, MicroAckContext, MicroAckManager


def _context(*, turn_id: str, category: str | MicroAckCategory = "speech", channel: str = "voice", intent: str | None = None) -> MicroAckContext:
    return MicroAckContext(
        category=category,
        channel=channel,
        run_id="run-1",
        session_id="session-1",
        turn_id=turn_id,
        intent=intent,
    )


def test_micro_ack_schedules_and_emits() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []
    logs: list[tuple[str, str, str, int | None, str | None, str | None, str | None, str | None, str | None]] = []
    now = 0.0

    def _now() -> float:
        return now

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=20, global_cooldown_ms=1, per_turn_max=1),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id: logs.append((event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id)),
        suppression_reason=lambda: None,
        now_fn=_now,
        rng=random.Random(7),
    )

    manager.on_user_speech_ended()
    manager.maybe_schedule(context=_context(turn_id="turn-1"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.03))

    assert emits and emits[0][0].turn_id == "turn-1"
    assert any(event == "scheduled" for event, *_ in logs)
    assert any(
        event == "emitted" and category == "speech" and channel == "voice"
        for event, _turn_id, _reason, _delay, category, channel, _intent, _action, _tool_call_id in logs
    )
    loop.close()


def test_micro_ack_cancelled_before_emit() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=30),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda *_args, **_kwargs: None,
        suppression_reason=lambda: None,
    )

    manager.maybe_schedule(context=_context(turn_id="turn-1"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    manager.cancel(turn_id="turn-1", reason="response_started")
    loop.run_until_complete(asyncio.sleep(0.04))

    assert emits == []
    loop.close()


def test_micro_ack_suppresses_for_recent_talk_over() -> None:
    loop = asyncio.new_event_loop()
    logs: list[tuple[str, str, str, int | None, str | None, str | None, str | None, str | None, str | None]] = []

    manager_holder: dict[str, MicroAckManager] = {}
    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=20, talk_over_risk_window_ms=10000),
        on_emit=lambda *_args: None,
        on_log=lambda event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id: logs.append((event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id)),
        suppression_reason=lambda: manager_holder["manager"].suppression_baseline_reason(),
    )
    manager_holder["manager"] = manager
    manager.mark_talk_over_incident()
    manager.maybe_schedule(context=_context(turn_id="turn-2"), reason="speech_stopped", loop=loop, expected_delay_ms=900)

    assert any(event == "suppressed" and turn_id == "turn-2" and reason == "talk_over_risk" and category == "speech" and channel == "voice" for event, turn_id, reason, _delay, category, channel, _intent, _action, _tool_call_id in logs)
    loop.close()


def test_channel_disabled_logs_single_suppression_without_scheduling_state_change() -> None:
    loop = asyncio.new_event_loop()
    logs: list[tuple[str, str, str, int | None, str | None, str | None, str | None, str | None, str | None]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=20, channel_enabled={"voice": False}),
        on_emit=lambda *_args: None,
        on_log=lambda event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id: logs.append((event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id)),
        suppression_reason=lambda: None,
    )

    manager.maybe_schedule(context=_context(turn_id="turn-channel-disabled", channel="voice"), reason="speech_stopped", loop=loop, expected_delay_ms=900)

    assert logs == [("suppressed", "turn-channel-disabled", "channel_disabled", None, "speech", "voice", None, None, None)]
    assert manager._scheduled == {}
    assert manager._scheduled_reason == {}
    loop.close()


def test_same_turn_different_category_allowed_per_policy() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=1, per_turn_max=2, dedupe_ttl_ms=5000, long_wait_second_ack_ms=0),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda *_args: None,
        suppression_reason=lambda: None,
        rng=random.Random(7),
    )

    manager.maybe_schedule(context=_context(turn_id="turn-1", category="speech", intent="search"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(context=_context(turn_id="turn-1", category="watchdog", intent="search"), reason="watchdog", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))

    assert [context.category for context, *_ in emits] == ["speech", "watchdog"]
    loop.close()


def test_same_category_intent_within_ttl_is_suppressed() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []
    logs: list[tuple[str, str, str, int | None, str | None, str | None, str | None, str | None, str | None]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=1, per_turn_max=3, dedupe_ttl_ms=5000, long_wait_second_ack_ms=0),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id: logs.append((event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id)),
        suppression_reason=lambda: None,
        rng=random.Random(7),
    )

    context = _context(turn_id="turn-1", category="speech", intent="weather")
    manager.maybe_schedule(context=context, reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(context=context, reason="speech_stopped", loop=loop, expected_delay_ms=900)

    assert len(emits) == 1
    assert any(event == "suppressed" and turn_id == "turn-1" and reason == "duplicate_within_ttl" and category == "speech" and channel == "voice" for event, turn_id, reason, _delay, category, channel, _intent, _action, _tool_call_id in logs)
    loop.close()


def test_different_channels_have_separate_limits() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(
            delay_ms=10,
            global_cooldown_ms=1,
            global_cooldown_scope="channel",
            per_turn_max=1,
            dedupe_ttl_ms=5000,
            long_wait_second_ack_ms=0,
        ),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda *_args: None,
        suppression_reason=lambda: None,
        rng=random.Random(7),
    )

    manager.maybe_schedule(context=_context(turn_id="turn-1", channel="voice", intent="help"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(context=_context(turn_id="turn-1", channel="text", intent="help"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))

    assert [context.channel for context, *_ in emits] == ["voice", "text"]
    loop.close()


def test_global_cooldown_scope_channel_allows_voice_then_text() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(
            delay_ms=10,
            global_cooldown_ms=1000,
            global_cooldown_scope="channel",
            per_turn_max=3,
            dedupe_ttl_ms=0,
            long_wait_second_ack_ms=0,
            channel_cooldown_ms={"voice": 1000, "text": 1000},
        ),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda *_args: None,
        suppression_reason=lambda: None,
        rng=random.Random(7),
    )

    manager.maybe_schedule(context=_context(turn_id="turn-1", channel="voice", intent="help"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(context=_context(turn_id="turn-2", channel="text", intent="help"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))

    assert [context.channel for context, *_ in emits] == ["voice", "text"]
    loop.close()


def test_global_cooldown_scope_all_blocks_voice_then_text() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []
    logs: list[tuple[str, str, str, int | None, str | None, str | None, str | None, str | None, str | None]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(
            delay_ms=10,
            global_cooldown_ms=1000,
            global_cooldown_scope="all",
            per_turn_max=3,
            dedupe_ttl_ms=0,
            long_wait_second_ack_ms=0,
            channel_cooldown_ms={"voice": 0, "text": 0},
        ),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id: logs.append((event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id)),
        suppression_reason=lambda: None,
        rng=random.Random(7),
    )

    manager.maybe_schedule(context=_context(turn_id="turn-1", channel="voice", intent="help"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(context=_context(turn_id="turn-2", channel="text", intent="help"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))

    assert [context.channel for context, *_ in emits] == ["voice"]
    assert any(event == "suppressed" and reason == "cooldown" and channel == "text" for event, _turn, reason, _delay, _category, channel, _intent, _action, _tool_call_id in logs)
    loop.close()


def test_per_channel_cooldown_blocks_same_channel_only() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []
    logs: list[tuple[str, str, str, int | None, str | None, str | None, str | None, str | None, str | None]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(
            delay_ms=10,
            global_cooldown_ms=0,
            per_turn_max=3,
            dedupe_ttl_ms=0,
            long_wait_second_ack_ms=0,
            channel_cooldown_ms={"voice": 1000, "text": 0},
        ),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id: logs.append((event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id)),
        suppression_reason=lambda: None,
        rng=random.Random(7),
    )

    manager.maybe_schedule(
        context=_context(turn_id="turn-1", channel="voice", category="speech", intent="weather"),
        reason="speech_stopped",
        loop=loop,
        expected_delay_ms=900,
    )
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(
        context=_context(turn_id="turn-2", channel="voice", category="watchdog", intent="stocks"),
        reason="watchdog",
        loop=loop,
        expected_delay_ms=900,
    )
    manager.maybe_schedule(
        context=_context(turn_id="turn-3", channel="text", category="watchdog", intent="stocks"),
        reason="watchdog",
        loop=loop,
        expected_delay_ms=900,
    )
    loop.run_until_complete(asyncio.sleep(0.02))

    assert [context.channel for context, *_ in emits] == ["voice", "text"]
    assert any(event == "suppressed" and reason == "channel_cooldown" and category == "watchdog" and channel == "voice" for event, _turn, reason, _delay, category, channel, _intent, _action, _tool_call_id in logs)
    loop.close()


def test_category_drives_phrase_family() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=0, per_turn_max=1, long_wait_second_ack_ms=0),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda *_args: None,
        suppression_reason=lambda: None,
    )

    manager.maybe_schedule(
        context=_context(turn_id="turn-1", category=MicroAckCategory.START_OF_WORK),
        reason="speech_stopped",
        loop=loop,
        expected_delay_ms=900,
    )
    loop.run_until_complete(asyncio.sleep(0.02))

    assert emits
    assert emits[0][1].startswith("start_of_work_")
    loop.close()


def test_per_category_cooldown_blocks_same_category_only() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []
    logs: list[tuple[str, str, str, int | None, str | None, str | None, str | None, str | None, str | None]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(
            delay_ms=10,
            global_cooldown_ms=0,
            per_turn_max=3,
            dedupe_ttl_ms=0,
            long_wait_second_ack_ms=0,
            category_cooldown_ms={
                MicroAckCategory.LATENCY_MASK.value: 1000,
                MicroAckCategory.START_OF_WORK.value: 0,
            },
        ),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id: logs.append((event, turn_id, reason, delay_ms, category, channel, intent, action, tool_call_id)),
        suppression_reason=lambda: None,
    )

    manager.maybe_schedule(
        context=_context(
            turn_id="turn-1",
            channel="voice",
            category=MicroAckCategory.LATENCY_MASK,
            intent="weather",
        ),
        reason="transcript_finalized",
        loop=loop,
        expected_delay_ms=900,
    )
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(
        context=_context(
            turn_id="turn-2",
            channel="voice",
            category=MicroAckCategory.LATENCY_MASK,
            intent="stocks",
        ),
        reason="transcript_finalized",
        loop=loop,
        expected_delay_ms=900,
    )
    manager.maybe_schedule(
        context=_context(
            turn_id="turn-3",
            channel="voice",
            category=MicroAckCategory.START_OF_WORK,
            intent="stocks",
        ),
        reason="speech_stopped",
        loop=loop,
        expected_delay_ms=900,
    )
    loop.run_until_complete(asyncio.sleep(0.02))

    assert [context.category for context, *_ in emits] == [MicroAckCategory.LATENCY_MASK, MicroAckCategory.START_OF_WORK]
    assert any(
        event == "suppressed" and reason == "category_cooldown" and category == MicroAckCategory.LATENCY_MASK.value
        for event, _turn, reason, _delay, category, _channel, _intent, _action, _tool_call_id in logs
    )
    loop.close()


def test_phrase_selection_is_deterministic_for_same_dedupe_key() -> None:
    loop = asyncio.new_event_loop()
    emits_a: list[tuple[MicroAckContext, str, str]] = []
    emits_b: list[tuple[MicroAckContext, str, str]] = []

    manager_a = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=0, per_turn_max=1, long_wait_second_ack_ms=0),
        on_emit=lambda context, phrase_id, phrase: emits_a.append((context, phrase_id, phrase)),
        on_log=lambda *_args: None,
        suppression_reason=lambda: None,
    )
    manager_b = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=0, per_turn_max=1, long_wait_second_ack_ms=0),
        on_emit=lambda context, phrase_id, phrase: emits_b.append((context, phrase_id, phrase)),
        on_log=lambda *_args: None,
        suppression_reason=lambda: None,
    )

    context = _context(turn_id="turn-1", category=MicroAckCategory.LATENCY_MASK, intent="weather")
    manager_a.maybe_schedule(context=context, reason="transcript_finalized", loop=loop, expected_delay_ms=900)
    manager_b.maybe_schedule(context=context, reason="transcript_finalized", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))

    assert emits_a and emits_b
    assert emits_a[0][1] == emits_b[0][1]
    assert emits_a[0][2] == emits_b[0][2]
    loop.close()


def test_non_safety_categories_never_emit_safety_phrase_ids() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=0, per_turn_max=3, dedupe_ttl_ms=0, long_wait_second_ack_ms=0),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda *_args: None,
        suppression_reason=lambda: None,
    )

    for idx, category in enumerate((MicroAckCategory.START_OF_WORK, MicroAckCategory.LATENCY_MASK, MicroAckCategory.FAILURE_FALLBACK), start=1):
        manager.maybe_schedule(
            context=_context(turn_id=f"turn-{idx}", category=category, intent=f"intent-{idx}"),
            reason="speech_stopped",
            loop=loop,
            expected_delay_ms=900,
        )
    loop.run_until_complete(asyncio.sleep(0.03))

    assert emits
    assert all(not phrase_id.startswith("safety_gate_") for _context_value, phrase_id, _phrase in emits)
    loop.close()
