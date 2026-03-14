from __future__ import annotations

import asyncio
import sys
import time
import types

import pytest

if "audioop" not in sys.modules:
    sys.modules["audioop"] = types.ModuleType("audioop")

from ai.realtime.asr_trust import is_current_visual_question
from ai.tools import function_map
from ai.realtime_api import RealtimeAPI


class _CameraStub:
    def __init__(self, *, pending_images: list[object] | None = None, alive: bool = True) -> None:
        self._pending_images = pending_images or []
        self._alive = alive

    def is_vision_loop_alive(self) -> bool:
        return self._alive

    def claim_one_pending_image_for_active_fresh_look(self) -> object | None:
        if not self._pending_images:
            return None
        return self._pending_images.pop(0)


def _build_api(*, camera: object | None = None) -> RealtimeAPI:
    api = RealtimeAPI.__new__(RealtimeAPI)
    api.camera_controller = camera
    api._fresh_look_enabled = True
    api._fresh_look_cooldown_s = 10.0
    api._fresh_look_wait_timeout_s = 0.15
    api._fresh_look_motion_settle_extension_s = 0.12
    api._last_fresh_look_request_at_monotonic = None
    api._last_fresh_look_completed_at_monotonic = None
    api._fresh_look_by_turn_id = {}
    api._latest_final_transcript_by_turn_id = {}
    api._current_run_id = lambda: "run-test"
    api._last_vision_frame_sent_at_monotonic = None
    return api


def test_current_visual_question_detection() -> None:
    assert is_current_visual_question("what do you see right now")
    assert is_current_visual_question("can you check what's on the shelf now")
    assert not is_current_visual_question("what did you see earlier")




def test_current_visual_question_detection_covers_natural_phrasing() -> None:
    assert is_current_visual_question("tell me what you see in front of you")
    assert is_current_visual_question("let me know what you see in front of you")
    assert is_current_visual_question("what are you looking at right now")
    assert is_current_visual_question("can you look and tell me what's there")
    assert is_current_visual_question("look at what's in front of you and describe it")
    assert is_current_visual_question("go back to center and then tell me what I have in my hands")
    assert is_current_visual_question("what am I holding")
    assert is_current_visual_question("can you see it now")
    assert is_current_visual_question("look at this")
    assert is_current_visual_question("look at my hand")


def test_mixed_motion_and_visual_utterance_triggers_fresh_look() -> None:
    assert is_current_visual_question("go back to center and tell me what you see in front of you")


def test_current_visual_question_detection_covers_conditional_see_phrasing() -> None:
    assert is_current_visual_question("look back at center and then tell me if you see a can of pop in your field of view")
    assert is_current_visual_question("tell me if you see a can of pop")
    assert is_current_visual_question("do you see a can of pop in your field of view")
    assert is_current_visual_question("can you check whether there is a can of pop in front of you")


def test_non_current_visual_question_detection_stays_false() -> None:
    assert not is_current_visual_question("what did you see earlier")
    assert not is_current_visual_question("what do you remember seeing last night")
    assert not is_current_visual_question("tell me if you saw a can of pop earlier")

def test_non_current_visual_question_does_not_trigger_fresh_look(monkeypatch) -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    called = []

    async def _capture(**_kwargs):
        called.append(True)
        return {}

    api._attempt_fresh_look_for_turn = _capture
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    async def _send_assistant_message(*_args, **_kwargs):
        return None

    api.send_assistant_message = _send_assistant_message
    api._prefer_explicit_inspect_owner_for_turn(
        turn_id="turn-7",
        transcript="go back to center and tell me what you see in front of you",
        seam="transcript_final",
    )

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what did you see before",
            websocket=object(),
            turn_id="turn-1",
            input_event_key="evt-1",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert called == []


def test_verify_on_risk_defers_clarify_while_explicit_inspect_pending() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    call_order: list[str] = []

    capture_kwargs: list[dict[str, object]] = []

    async def _capture(**kwargs):
        capture_kwargs.append(kwargs)
        call_order.append("fresh_look")
        return {"requested": True, "completed": False, "blocked_reason": "timeout"}

    async def _send_assistant_message(*_args, **_kwargs):
        call_order.append("clarify")
        return None

    api._attempt_fresh_look_for_turn = _capture
    api.send_assistant_message = _send_assistant_message
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""
    api._prefer_explicit_inspect_owner_for_turn(
        turn_id="turn-7",
        transcript="go back to center and tell me what you see in front of you",
        seam="transcript_final",
    )

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="go back to center and tell me what you see in front of you",
            websocket=object(),
            turn_id="turn-7",
            input_event_key="evt-7",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert call_order == []
    assert capture_kwargs == []
    state = api._fresh_look_state_for_turn(turn_id="turn-7")
    assert state["visual_actuator"] == "explicit_inspect"




def test_verify_on_risk_clarifies_when_explicit_inspect_failed() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    call_order: list[str] = []

    async def _capture(**_kwargs):
        call_order.append("fresh_look")
        return {}

    async def _send_assistant_message(*_args, **_kwargs):
        call_order.append("clarify")
        return None

    api._attempt_fresh_look_for_turn = _capture
    api.send_assistant_message = _send_assistant_message
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    state = api._fresh_look_state_for_turn(turn_id="turn-failed")
    state["visual_actuator"] = "explicit_inspect"
    state["visual_intent_class"] = "explicit_inspect"
    state["explicit_inspect_status"] = "timeout"

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="Can you see it now?",
            websocket=object(),
            turn_id="turn-failed",
            input_event_key="evt-failed",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert call_order == ["clarify"]

def test_verify_on_risk_skips_heuristic_when_explicit_inspect_already_selected() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    called = []

    async def _capture(**_kwargs):
        called.append(True)
        return {}

    api._attempt_fresh_look_for_turn = _capture
    state = api._fresh_look_state_for_turn(turn_id="turn-explicit")
    state["visual_actuator"] = "explicit_inspect"
    state["visual_intent_class"] = "explicit_inspect"
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    async def _send_assistant_message(*_args, **_kwargs):
        return None

    api.send_assistant_message = _send_assistant_message

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="tell me what you see in front of you",
            websocket=object(),
            turn_id="turn-explicit",
            input_event_key="evt-explicit",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert called == []


def test_verify_on_risk_seeds_explicit_owner_before_fallback() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    called = []

    async def _capture(**_kwargs):
        called.append(True)
        return {}

    api._attempt_fresh_look_for_turn = _capture
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    async def _send_assistant_message(*_args, **_kwargs):
        return None

    api.send_assistant_message = _send_assistant_message

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="Can you see it now?",
            websocket=object(),
            turn_id="turn-seeded",
            input_event_key="evt-seeded",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    state = api._fresh_look_state_for_turn(turn_id="turn-seeded")
    assert state["visual_actuator"] == "explicit_inspect"
    assert called == []


def test_fresh_look_respects_cooldown() -> None:
    api = _build_api(camera=_CameraStub())
    api._last_fresh_look_request_at_monotonic = time.monotonic()
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-2")

    assert not allowed
    assert reason == "cooldown"


def test_fresh_look_respects_busy_state() -> None:
    api = _build_api(camera=_CameraStub())
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-2")

    assert not allowed
    assert reason == "injection_not_ready"




def test_get_vision_state_uses_camera_instance_fallback_when_controller_unset() -> None:
    api = _build_api(camera=None)
    api.camera_instance = _CameraStub(pending_images=[], alive=True)
    api._last_vision_frame_sent_at_monotonic = time.monotonic()

    state = api.get_vision_state()

    assert state["can_capture"] is True
    assert state["camera_active"] is True
    assert state["available"] is True


def test_fresh_look_gating_uses_camera_instance_fallback_when_controller_unset() -> None:
    api = _build_api(camera=None)
    api.camera_instance = _CameraStub(pending_images=[])
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-fallback")

    assert not allowed
    assert reason == "injection_not_ready"

def test_fresh_look_alive_camera_not_marked_camera_unavailable() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    allowed, reason = api._fresh_look_gating_decision(turn_id="turn-camera-alive")

    assert not allowed
    assert reason != "camera_missing"
    assert reason != "camera_loop_inactive"
    assert reason == "injection_not_ready"


def test_successful_fresh_look_marks_current_provenance() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-3"))

    assert state["requested"] is True
    assert state["completed"] is True
    assert state["visual_answer_mode"] == "fresh_current"
    assert state["last_fresh_capture_age_ms"] == 0




def test_verify_on_risk_does_not_reclaim_after_explicit_inspect_when_metadata_missing() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    called: list[dict[str, object]] = []

    async def _capture(**kwargs):
        called.append(kwargs)
        return {"requested": True, "completed": False, "blocked_reason": "timeout"}

    async def _send_assistant_message(*_args, **_kwargs):
        return None

    state = api._fresh_look_state_for_turn(turn_id="turn-1")
    state["visual_actuator"] = "explicit_inspect"
    state["visual_intent_class"] = "explicit_inspect"

    api._attempt_fresh_look_for_turn = _capture
    api.send_assistant_message = _send_assistant_message
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="what do you see right now",
            websocket=object(),
            turn_id="turn-1",
            input_event_key="evt-1",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert called == []


def test_explicit_inspect_marks_visual_actuator_metadata() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    async def _center(**_kwargs):
        return {"queued": True}

    original = function_map.get("gesture_look_center")
    function_map["gesture_look_center"] = _center
    try:
        result = asyncio.run(api._inspect_current_view_tool(recenter=True, delay_ms=5))
    finally:
        if original is None:
            function_map.pop("gesture_look_center", None)
        else:
            function_map["gesture_look_center"] = original

    assert result["visual_actuator"] == "explicit_inspect"
    assert result["visual_intent_class"] == "explicit_inspect"


def test_visual_answer_provenance_ambient_recent_when_queue_present() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    state = api.get_vision_state()

    mode = api._classify_visual_answer_provenance(turn_id="turn-ambient", vision_state=state)

    assert mode == "ambient_recent"




def test_get_vision_state_reports_camera_alive_and_available_when_recent_frame_exists() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._last_vision_frame_sent_at_monotonic = time.monotonic()

    state = api.get_vision_state()

    assert state["can_capture"] is True
    assert state["camera_active"] is True
    assert state["available"] is True

def test_camera_active_clarify_never_uses_camera_unavailable_message() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-clarify"))
    message = api._visual_unavailable_clarify_text_for_turn(
        turn_id="turn-clarify",
        vision_state=api.get_vision_state(),
    )

    assert state["blocked_reason"] == "injection_not_ready"
    assert "camera isn’t available" not in message


def test_mixed_motion_visual_prompt_preserves_truthful_ambient_provenance() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    api._tool_call_records = [{"turn_id": "turn-mixed", "name": "gesture_look_center"}]
    api._fresh_look_state_for_turn(turn_id="turn-mixed")["completed"] = False

    provenance = api._classify_visual_answer_provenance(
        turn_id="turn-mixed",
        vision_state=api.get_vision_state(),
    )

    assert provenance == "ambient_recent"


def test_bounded_visual_clarify_never_uses_camera_unavailable_when_injection_not_ready() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (False, "interaction_state=speaking") if with_reason else False
    asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-bounded"))

    final_text = api._normalize_verify_clarify_message(
        message="I can’t take a fresh look right now because the camera isn’t available.",
        metadata={
            "trigger": "asr_verify_on_risk",
            "reason": "visual_unavailable",
            "clarify_mode": "bounded",
            "turn_id": "turn-bounded",
        },
    )

    assert "camera isn’t available" not in final_text
    assert "vision is busy" in final_text


def test_failed_fresh_look_returns_truthful_timeout_wording() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-4"))
    message = api._visual_unavailable_clarify_text_for_turn(
        turn_id="turn-4",
        vision_state=api.get_vision_state(),
    )

    assert state["blocked_reason"] == "timeout"
    assert "I tried to take a fresh look" in message


def test_repeated_look_now_requests_do_not_spam_captures() -> None:
    camera = _CameraStub(pending_images=[object()])
    api = _build_api(camera=camera)
    api._fresh_look_cooldown_s = 30.0
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    first = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-5"))
    camera._pending_images = []
    second = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-6"))

    assert first["completed"] is True
    assert second["completed"] is False
    assert second["blocked_reason"] == "cooldown"


def test_fresh_look_times_out_without_eligible_motion_extension() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-no-motion"))

    assert state["blocked_reason"] == "timeout"
    assert state["motion_settle_extension_used"] is False
    assert state["extended_deadline_until_monotonic"] is None


def test_fresh_look_deadline_extends_once_for_eligible_same_turn_motion() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._fresh_look_wait_timeout_s = 0.08
    api._fresh_look_motion_settle_extension_s = 0.25
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    async def _trigger_motion() -> None:
        await asyncio.sleep(0.02)
        api._mark_fresh_look_motion_request_for_turn(turn_id="turn-extend-once", tool_name="gesture_look_center", source="intent_commit")
        first_motion = api._fresh_look_state_for_turn(turn_id="turn-extend-once")["motion_requested_at_monotonic"]
        await asyncio.sleep(0.04)
        api._mark_fresh_look_motion_request_for_turn(turn_id="turn-extend-once", tool_name="gesture_look_center", source="intent_commit")
        second_motion = api._fresh_look_state_for_turn(turn_id="turn-extend-once")["motion_requested_at_monotonic"]
        assert isinstance(first_motion, float)
        assert isinstance(second_motion, float)

    async def _run() -> dict[str, object]:
        fresh_task = asyncio.create_task(api._attempt_fresh_look_for_turn(turn_id="turn-extend-once"))
        motion_task = asyncio.create_task(_trigger_motion())
        await asyncio.gather(motion_task, fresh_task)
        return fresh_task.result()

    state = asyncio.run(_run())

    assert state["blocked_reason"] == "timeout"
    assert state["motion_settle_extension_used"] is True
    assert isinstance(state["extended_deadline_until_monotonic"], float)


def test_fresh_look_succeeds_when_frame_arrives_during_extended_settle_window() -> None:
    camera = _CameraStub(pending_images=[])
    api = _build_api(camera=camera)
    api._fresh_look_wait_timeout_s = 0.08
    api._fresh_look_motion_settle_extension_s = 0.30
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    async def _trigger_motion_and_frame() -> None:
        await asyncio.sleep(0.03)
        api._mark_fresh_look_motion_request_for_turn(
            turn_id="turn-extended-success",
            tool_name="gesture_look_center",
            source="intent_commit",
        )
        await asyncio.sleep(0.10)
        api._mark_fresh_look_motion_request_for_turn(
            turn_id="turn-extended-success",
            tool_name="gesture_look_center",
            source="tool_result_late",
        )
        camera._pending_images.append(object())

    async def _run() -> dict[str, object]:
        fresh_task = asyncio.create_task(api._attempt_fresh_look_for_turn(turn_id="turn-extended-success"))
        producer_task = asyncio.create_task(_trigger_motion_and_frame())
        await asyncio.gather(producer_task, fresh_task)
        return fresh_task.result()

    state = asyncio.run(_run())

    assert state["completed"] is True
    assert state["blocked_reason"] == "none"
    assert state["motion_settle_extension_used"] is True


def test_non_visual_turn_motion_mark_does_not_trigger_extension() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    state = api._fresh_look_state_for_turn(turn_id="turn-non-visual")
    state["requested"] = True
    state["is_current_visual_turn"] = False

    api._mark_fresh_look_motion_request_for_turn(turn_id="turn-non-visual", tool_name="gesture_look_center", source="intent_commit")

    assert state["motion_requested_at_monotonic"] is None


def test_ineligible_tool_does_not_trigger_fresh_look_extension() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    state = api._fresh_look_state_for_turn(turn_id="turn-ineligible")
    state["requested"] = True
    state["is_current_visual_turn"] = True

    api._mark_fresh_look_motion_request_for_turn(turn_id="turn-ineligible", tool_name="gesture_nod", source="intent_commit")

    assert state["motion_requested_at_monotonic"] is None


def test_repeated_eligible_motion_requests_do_not_stack_extensions() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    state = api._fresh_look_state_for_turn(turn_id="turn-repeat-motion")
    state["requested"] = True
    state["completed"] = False
    state["is_current_visual_turn"] = True

    api._mark_fresh_look_motion_request_for_turn(turn_id="turn-repeat-motion", tool_name="gesture_look_center", source="intent_commit")
    first_mark = state["motion_requested_at_monotonic"]
    state["motion_settle_extension_used"] = True
    api._mark_fresh_look_motion_request_for_turn(turn_id="turn-repeat-motion", tool_name="gesture_look_center", source="tool_result")

    assert isinstance(first_mark, float)
    assert isinstance(state["motion_requested_at_monotonic"], float)
    assert state["motion_requested_at_monotonic"] == first_mark


def test_motion_intent_before_tool_result_extends_fresh_look() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._fresh_look_wait_timeout_s = 0.08
    api._fresh_look_motion_settle_extension_s = 0.22
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    async def _run() -> dict[str, object]:
        fresh_task = asyncio.create_task(api._attempt_fresh_look_for_turn(turn_id="turn-intent-early"))
        await asyncio.sleep(0.03)
        api._mark_fresh_look_motion_request_for_turn(
            turn_id="turn-intent-early",
            tool_name="gesture_look_center",
            source="intent_commit",
        )
        await asyncio.sleep(0.12)
        api._mark_fresh_look_motion_request_for_turn(
            turn_id="turn-intent-early",
            tool_name="gesture_look_center",
            source="tool_result_late",
        )
        await fresh_task
        return fresh_task.result()

    state = asyncio.run(_run())

    assert state["blocked_reason"] == "timeout"
    assert state["motion_settle_extension_used"] is True
    assert state["eligible_motion_tool_name"] == "gesture_look_center"


def test_extension_holds_when_tool_result_arrives_after_base_timeout() -> None:
    camera = _CameraStub(pending_images=[])
    api = _build_api(camera=camera)
    api._fresh_look_wait_timeout_s = 0.07
    api._fresh_look_motion_settle_extension_s = 0.28
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    async def _run() -> dict[str, object]:
        fresh_task = asyncio.create_task(api._attempt_fresh_look_for_turn(turn_id="turn-late-result"))
        await asyncio.sleep(0.02)
        api._mark_fresh_look_motion_request_for_turn(
            turn_id="turn-late-result",
            tool_name="gesture_look_center",
            source="intent_commit",
        )
        await asyncio.sleep(0.08)
        api._mark_fresh_look_motion_request_for_turn(
            turn_id="turn-late-result",
            tool_name="gesture_look_center",
            source="tool_result_late",
        )
        await asyncio.sleep(0.08)
        camera._pending_images.append(object())
        await fresh_task
        return fresh_task.result()

    state = asyncio.run(_run())

    assert state["completed"] is True
    assert state["motion_settle_extension_used"] is True


def test_multi_seam_motion_marking_preserves_first_timestamp() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    state = api._fresh_look_state_for_turn(turn_id="turn-first-ts")
    state["requested"] = True
    state["completed"] = False
    state["is_current_visual_turn"] = True

    api._mark_fresh_look_motion_request_for_turn(
        turn_id="turn-first-ts",
        tool_name="gesture_look_center",
        source="execute_function_call_commit",
    )
    first_ts = state["motion_requested_at_monotonic"]
    time.sleep(0.01)
    api._mark_fresh_look_motion_request_for_turn(
        turn_id="turn-first-ts",
        tool_name="gesture_look_center",
        source="tool_result_queued",
    )

    assert isinstance(first_ts, float)
    assert state["motion_requested_at_monotonic"] == first_ts


def test_extension_applies_when_motion_before_base_deadline_even_if_loop_observes_late() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._fresh_look_wait_timeout_s = 0.02
    api._fresh_look_motion_settle_extension_s = 0.22
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    async def _run() -> dict[str, object]:
        fresh_task = asyncio.create_task(api._attempt_fresh_look_for_turn(turn_id="turn-loop-jitter"))
        await asyncio.sleep(0.005)
        api._mark_fresh_look_motion_request_for_turn(
            turn_id="turn-loop-jitter",
            tool_name="gesture_look_center",
            source="execute_function_call_commit",
        )
        await fresh_task
        return fresh_task.result()

    state = asyncio.run(_run())

    assert state["blocked_reason"] == "timeout"
    assert state["motion_settle_extension_used"] is True
    assert isinstance(state["extended_deadline_until_monotonic"], float)


def test_fresh_look_claims_single_pending_frame_during_response_in_progress() -> None:
    frame_one = object()
    frame_two = object()
    camera = _CameraStub(pending_images=[frame_one, frame_two])
    api = _build_api(camera=camera)
    api.is_ready_for_injections = lambda with_reason=False: (False, "response_in_progress") if with_reason else False

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-claim-single"))

    assert state["completed"] is True
    assert state["claimed_pending_frame"] is True
    assert len(camera._pending_images) == 1


def test_fresh_look_claimed_frame_is_not_reused_by_later_turn() -> None:
    camera = _CameraStub(pending_images=[object()])
    api = _build_api(camera=camera)
    api._fresh_look_wait_timeout_s = 0.03
    api._fresh_look_cooldown_s = 0.0
    api.is_ready_for_injections = lambda with_reason=False: (False, "response_in_progress") if with_reason else False

    first = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-first"))
    second = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-second"))

    assert first["completed"] is True
    assert first["claimed_pending_frame"] is True
    assert second["completed"] is False
    assert second["blocked_reason"] == "timeout"


def test_fresh_look_timeout_when_pending_frame_never_arrives_during_response_in_progress() -> None:
    camera = _CameraStub(pending_images=[])
    api = _build_api(camera=camera)
    api._fresh_look_wait_timeout_s = 0.03
    api.is_ready_for_injections = lambda with_reason=False: (False, "response_in_progress") if with_reason else False

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-no-pending"))

    assert state["completed"] is False
    assert state["blocked_reason"] == "timeout"
    assert state["claimed_pending_frame"] is False


def test_fresh_look_does_not_claim_pending_frame_when_injection_lane_ready() -> None:
    camera = _CameraStub(pending_images=[object()])
    api = _build_api(camera=camera)
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True

    state = asyncio.run(api._attempt_fresh_look_for_turn(turn_id="turn-no-claim-ready-lane"))

    assert state["completed"] is True
    assert state["claimed_pending_frame"] is False
    assert len(camera._pending_images) == 1

def test_explicit_inspect_tool_registered() -> None:
    from ai.tools import tools

    inspect_spec = next((tool for tool in tools if tool.get("name") == "inspect_current_view"), None)

    assert inspect_spec is not None
    assert inspect_spec["parameters"]["properties"]["recenter"]["type"] == "boolean"


def test_inspect_current_view_returns_structured_success_contract() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True
    api._current_turn_id_or_unknown = lambda: "turn-inspect-ok"

    result = asyncio.run(api._inspect_current_view_tool(recenter=False))

    assert result["status"] == "ok"
    assert result["visual_answer_mode"] == "fresh_current"
    assert result["claimed_pending_frame"] is False
    assert result["recenter_applied"] is False
    assert result["evidence_source"] == "pending_queue_visible"


def test_inspect_current_view_returns_timeout_status_without_false_success() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._fresh_look_wait_timeout_s = 0.03
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True
    api._current_turn_id_or_unknown = lambda: "turn-inspect-timeout"

    result = asyncio.run(api._inspect_current_view_tool(recenter=False))

    assert result["status"] == "timeout"
    assert result["blocked_reason"] == "timeout"
    assert result["visual_answer_mode"] != "fresh_current"


def test_inspect_current_view_recenter_uses_existing_motion_tool() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    api.is_ready_for_injections = lambda with_reason=False: (True, "ready") if with_reason else True
    api._current_turn_id_or_unknown = lambda: "turn-inspect-recenter"

    from ai import realtime_api as realtime_module

    calls: list[tuple[str, int]] = []
    original = realtime_module.function_map["gesture_look_center"]

    async def _fake_center(delay_ms: int = 0) -> dict[str, object]:
        calls.append(("gesture_look_center", delay_ms))
        return {"queued": True, "gesture": "look_center", "delay_ms": delay_ms}

    realtime_module.function_map["gesture_look_center"] = _fake_center
    try:
        result = asyncio.run(api._inspect_current_view_tool(recenter=True, delay_ms=120))
    finally:
        realtime_module.function_map["gesture_look_center"] = original

    assert calls == [("gesture_look_center", 120)]
    assert result["recenter_applied"] is True
    state = api._fresh_look_state_for_turn(turn_id="turn-inspect-recenter")
    assert state["eligible_motion_tool_name"] == "gesture_look_center"


@pytest.mark.parametrize("explicit_status", ["blocked", "timeout", "unavailable"])
def test_verify_on_risk_forces_visual_unavailable_clarify_when_explicit_inspect_non_ok(explicit_status: str) -> None:
    api = _build_api(camera=_CameraStub(pending_images=[object()]))
    state = api._fresh_look_state_for_turn(turn_id="turn-explicit-nonok")
    state["visual_actuator"] = "explicit_inspect"
    state["visual_intent_class"] = "explicit_inspect"
    state["explicit_inspect_status"] = explicit_status

    captured_metadata: list[dict[str, object]] = []

    async def _send_assistant_message(message: str, _websocket: object, *, response_metadata: dict[str, object] | None = None, **_kwargs) -> None:
        assert message
        captured_metadata.append(dict(response_metadata or {}))

    api.send_assistant_message = _send_assistant_message
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="tell me what object is in front of you",
            websocket=object(),
            turn_id="turn-explicit-nonok",
            input_event_key=f"evt-explicit-{explicit_status}",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert clarified is True
    assert captured_metadata
    assert captured_metadata[0]["reason"] == "visual_unavailable"
    verdict = api._get_response_gating_verdict(
        turn_id="turn-explicit-nonok",
        input_event_key=f"evt-explicit-{explicit_status}",
    )
    assert verdict is not None
    assert verdict.action == "CLARIFY"
    assert verdict.reason == "visual_unavailable"


@pytest.mark.parametrize("inspect_status", ["blocked", "timeout", "unavailable"])
def test_execute_function_call_inspect_non_ok_forces_bounded_clarify_followup(inspect_status: str) -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._tool_call_records = []
    api._last_tool_call_results = []
    api._current_turn_id_or_unknown = lambda: "turn-exec-inspect"
    api._normalize_realtime_call_id = lambda call_id: call_id
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._mark_or_suppress_research_spoken_response = lambda _rid: False
    api._tool_followup_input_event_key = lambda *, call_id: f"tool:{call_id}"
    api._active_input_event_key_by_turn_id = {"turn-exec-inspect": "item_parent"}

    sent_events: list[dict[str, object]] = []
    followup_events: list[dict[str, object]] = []

    class _Transport:
        async def send_json(self, _ws: object, payload: dict[str, object]) -> None:
            sent_events.append(payload)

    async def _fake_send_response_create(_ws: object, response_create_event: dict[str, object], **_kwargs) -> bool:
        followup_events.append(response_create_event)
        return True

    async def _fake_inspect_current_view_tool(**_kwargs):
        return {
            "status": inspect_status,
            "blocked_reason": "injection_not_ready",
            "visual_answer_mode": "ambient_stale",
        }

    api._get_or_create_transport = lambda: _Transport()
    api._send_response_create = _fake_send_response_create
    api._inspect_current_view_tool = _fake_inspect_current_view_tool

    asyncio.run(
        api.execute_function_call(
            "inspect_current_view",
            "call-inspect-nonok",
            {"recenter": False},
            websocket=object(),
        )
    )

    assert sent_events
    assert followup_events
    response_payload = followup_events[0].get("response") or {}
    metadata = response_payload.get("metadata") or {}
    assert metadata.get("explicit_inspect_outcome") == inspect_status
    assert metadata.get("reason") == "visual_unavailable"
    assert metadata.get("trigger") == "asr_verify_on_risk"
    assert metadata.get("clarify_mode") == "bounded"
    assert response_payload.get("tool_choice") == "none"


def test_visual_tool_descriptions_bias_semantic_requests_to_inspect() -> None:
    from ai.tools import tools

    inspect_spec = next(tool for tool in tools if tool.get("name") == "inspect_current_view")
    center_spec = next(tool for tool in tools if tool.get("name") == "gesture_look_center")

    inspect_description = str(inspect_spec.get("description") or "").lower()
    center_description = str(center_spec.get("description") or "").lower()

    assert "primary visual-semantic tool" in inspect_description
    assert "what am i holding" in inspect_description
    assert "motion-only setup" in center_description
    assert "semantic visual questions use inspect_current_view" in center_description


def test_transcript_final_visual_turn_prefers_explicit_inspect_owner() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    async def _send_assistant_message(*_args, **_kwargs):
        return None

    api.send_assistant_message = _send_assistant_message

    owned = api._prefer_explicit_inspect_owner_for_turn(
        turn_id="turn-pref",
        transcript="Hey Theo, look at this.",
        seam="transcript_final",
    )

    assert owned is True
    state = api._fresh_look_state_for_turn(turn_id="turn-pref")
    assert state["visual_actuator"] == "explicit_inspect"
    assert state["visual_intent_class"] == "explicit_inspect"


def test_verify_on_risk_uses_heuristic_only_when_explicit_inspect_unavailable() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    state = api._fresh_look_state_for_turn(turn_id="turn-fallback")
    state["explicit_inspect_status"] = "timeout"
    called: list[dict[str, object]] = []

    async def _capture(**kwargs):
        called.append(kwargs)
        return {"requested": True, "completed": False, "blocked_reason": "timeout"}

    async def _send_assistant_message(*_args, **_kwargs):
        return None

    api._attempt_fresh_look_for_turn = _capture
    api.send_assistant_message = _send_assistant_message
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="Can you see it now?",
            websocket=object(),
            turn_id="turn-fallback",
            input_event_key="evt-fallback",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert called
    assert called[0]["visual_actuator"] == "heuristic_fresh_look"


def test_verify_on_risk_successful_fresh_capture_not_overridden_by_clarify() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._fresh_look_state_for_turn(turn_id="turn-success")["explicit_inspect_status"] = "unavailable"
    sent = []

    async def _capture(**_kwargs):
        state = api._fresh_look_state_for_turn(turn_id="turn-success")
        state["requested"] = True
        state["completed"] = True
        state["blocked_reason"] = "none"
        state["visual_actuator"] = "heuristic_fresh_look"
        state["visual_intent_class"] = "fallback_visual_question"
        return state

    async def _send_assistant_message(*_args, **_kwargs):
        sent.append(True)
        return None

    api._attempt_fresh_look_for_turn = _capture
    api.send_assistant_message = _send_assistant_message
    api._asr_verify_on_risk_enabled = True
    api._asr_clarify_asked_input_event_keys = set()
    api._asr_clarify_count_by_turn = {}
    api._asr_verify_max_clarify_per_turn = 1
    api._asr_verify_short_utterance_ms = 300
    api._asr_verify_min_confidence = 0.6
    api._response_gating_verdict_by_input_event_key = {}
    api._pending_server_auto_response_by_turn_id = {}
    api._canonical_utterance_key = lambda *, turn_id, input_event_key: f"run-test:{turn_id}:{input_event_key}"
    api._record_cancel_issued_timing = lambda *_args, **_kwargs: None
    api._stale_response_ids_set = set()
    api._mark_pending_server_auto_response_cancelled = lambda **_kwargs: None
    api._suppress_cancelled_response_audio = lambda *_args, **_kwargs: None
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api.assistant_reply = ""
    api._assistant_reply_accum = ""

    clarified = asyncio.run(
        api._maybe_verify_on_risk_clarify(
            transcript="Can you see it now?",
            websocket=object(),
            turn_id="turn-success",
            input_event_key="evt-success",
            snapshot={"run_id": "run-test", "asr_confidence": 0.95},
        )
    )

    assert clarified is False
    assert sent == []


def test_execute_function_call_inspect_suppresses_spurious_recenter() -> None:
    api = _build_api(camera=_CameraStub(pending_images=[]))
    api._tool_call_records = []
    api._last_tool_call_results = []
    api._current_turn_id_or_unknown = lambda: "turn-recenter"
    api._normalize_realtime_call_id = lambda call_id: call_id
    api._track_outgoing_event = lambda *_args, **_kwargs: None
    api._mark_utterance_info_summary = lambda **_kwargs: None
    api._mark_or_suppress_research_spoken_response = lambda _rid: False
    api._tool_followup_input_event_key = lambda *, call_id: f"tool:{call_id}"
    api._active_input_event_key_by_turn_id = {"turn-recenter": "item_parent"}
    api._latest_final_transcript_by_turn_id = {"turn-recenter": "Can you see it now?"}
    api._send_response_create = lambda *_args, **_kwargs: asyncio.sleep(0, result=True)
    api._get_or_create_transport = lambda: type("T", (), {"send_json": staticmethod(lambda *_a, **_k: asyncio.sleep(0))})()
    api._last_user_input_text = "Can you see it now?"

    seen_args: list[dict[str, object]] = []

    async def _fake_inspect_current_view_tool(**kwargs):
        seen_args.append(dict(kwargs))
        return {
            "status": "ok",
            "blocked_reason": "none",
            "visual_answer_mode": "fresh_current",
        }

    api._inspect_current_view_tool = _fake_inspect_current_view_tool

    asyncio.run(
        api.execute_function_call(
            "inspect_current_view",
            "call-recenter",
            {"recenter": True, "delay_ms": 0},
            websocket=object(),
        )
    )

    assert seen_args
    assert seen_args[0]["recenter"] is False
