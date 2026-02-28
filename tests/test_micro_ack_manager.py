from __future__ import annotations

import asyncio
import random

from ai.micro_ack_manager import MicroAckConfig, MicroAckContext, MicroAckManager


def _context(*, turn_id: str, category: str = "speech", channel: str = "voice", intent: str | None = None) -> MicroAckContext:
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
    logs: list[tuple[str, str, str, int | None]] = []
    now = 0.0

    def _now() -> float:
        return now

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=20, global_cooldown_ms=1, per_turn_max=1),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda event, turn_id, reason, delay_ms: logs.append((event, turn_id, reason, delay_ms)),
        suppression_reason=lambda: None,
        now_fn=_now,
        rng=random.Random(7),
    )

    manager.on_user_speech_ended()
    manager.maybe_schedule(context=_context(turn_id="turn-1"), reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.03))

    assert emits and emits[0][0].turn_id == "turn-1"
    assert any(event == "scheduled" for event, *_ in logs)
    assert any(event == "emitted" for event, *_ in logs)
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
    logs: list[tuple[str, str, str, int | None]] = []

    manager_holder: dict[str, MicroAckManager] = {}
    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=20, talk_over_risk_window_ms=10000),
        on_emit=lambda *_args: None,
        on_log=lambda event, turn_id, reason, delay_ms: logs.append((event, turn_id, reason, delay_ms)),
        suppression_reason=lambda: manager_holder["manager"].suppression_baseline_reason(),
    )
    manager_holder["manager"] = manager
    manager.mark_talk_over_incident()
    manager.maybe_schedule(context=_context(turn_id="turn-2"), reason="speech_stopped", loop=loop, expected_delay_ms=900)

    assert ("suppressed", "turn-2", "talk_over_risk", None) in logs
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
    logs: list[tuple[str, str, str, int | None]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=1, per_turn_max=3, dedupe_ttl_ms=5000, long_wait_second_ack_ms=0),
        on_emit=lambda context, phrase_id, phrase: emits.append((context, phrase_id, phrase)),
        on_log=lambda event, turn_id, reason, delay_ms: logs.append((event, turn_id, reason, delay_ms)),
        suppression_reason=lambda: None,
        rng=random.Random(7),
    )

    context = _context(turn_id="turn-1", category="speech", intent="weather")
    manager.maybe_schedule(context=context, reason="speech_stopped", loop=loop, expected_delay_ms=900)
    loop.run_until_complete(asyncio.sleep(0.02))
    manager.maybe_schedule(context=context, reason="speech_stopped", loop=loop, expected_delay_ms=900)

    assert len(emits) == 1
    assert ("suppressed", "turn-1", "duplicate_within_ttl", None) in logs
    loop.close()


def test_different_channels_have_separate_limits() -> None:
    loop = asyncio.new_event_loop()
    emits: list[tuple[MicroAckContext, str, str]] = []

    manager = MicroAckManager(
        config=MicroAckConfig(delay_ms=10, global_cooldown_ms=1, per_turn_max=1, dedupe_ttl_ms=5000, long_wait_second_ack_ms=0),
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
