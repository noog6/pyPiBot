from ai.realtime.metadata_contract import (
    PROVIDER_METADATA_MAX_PROPERTIES,
    PROVIDER_METADATA_MAX_VALUE_LENGTH,
    normalize_provider_metadata,
)


def test_followthrough_catchup_payload_externalized_and_values_bounded() -> None:
    stored = []

    def store(payload):
        stored.append(payload)
        return "rctx_000001"

    metadata = {
        "turn_id": "turn_1077",
        "input_event_key": "item_1077:required_deliverable_followthrough:0",
        "parent_input_event_key": "item_1077",
        "tool_followup": "true",
        "tool_followup_release": "true",
        "followthrough_step_output_policy": "required_deliverable",
        "followthrough_post_completion_reason": "required_deliverable_owed",
        "followthrough_required_tool_name": "read_runtime_diagnostics",
        "followthrough_required_tool_already_executed": "true",
        "followthrough_catchup_payload": "x" * 694,
    }

    result = normalize_provider_metadata(metadata, store_local_context=store)

    assert "followthrough_catchup_payload" not in result.metadata
    assert result.metadata["followthrough_context_id"] == "rctx_000001"
    assert result.metadata["metadata_schema_version"] == "provider_metadata.v1"
    assert stored and stored[0]["followthrough_catchup_payload"] == "x" * 694
    assert len(result.metadata) <= PROVIDER_METADATA_MAX_PROPERTIES
    assert all(isinstance(value, str) for value in result.metadata.values())
    assert all(len(value) <= PROVIDER_METADATA_MAX_VALUE_LENGTH for value in result.metadata.values())


def test_metadata_priority_capping_preserves_required_deliverable_routing() -> None:
    metadata = {
        "turn_id": "turn_1",
        "input_event_key": "item_1:required_deliverable_followthrough:0",
        "parent_turn_id": "turn_1",
        "parent_input_event_key": "item_1",
        "tool_call_id": "call_diag",
        "tool_followup": "true",
        "tool_followup_release": "true",
        "blocked_by_response_id": "resp_parent",
        "consumes_canonical_slot": "false",
        "explicit_multipart": "true",
        "local_runtime_followthrough": "true",
        "followthrough_step_output_policy": "required_deliverable",
        "followthrough_post_completion_reason": "required_deliverable_owed",
        "followthrough_required_tool_name": "read_runtime_diagnostics",
        "followthrough_required_tool_already_executed": "true",
        "tool_followup_suppress_if_parent_covered": "true",
        "tool_name": "read_runtime_diagnostics",
        "gesture_motion_status": "completed",
        "followthrough_runtime_tool_args": "{}",
        "tool_followup_tool_choice_reason": "diagnostic only",
    }

    result = normalize_provider_metadata(metadata)

    assert len(result.metadata) == PROVIDER_METADATA_MAX_PROPERTIES
    for key in (
        "turn_id",
        "input_event_key",
        "parent_input_event_key",
        "tool_followup",
        "tool_followup_release",
        "followthrough_step_output_policy",
        "followthrough_post_completion_reason",
        "followthrough_required_tool_name",
        "followthrough_required_tool_already_executed",
    ):
        assert key in result.metadata
    assert "gesture_motion_status" not in result.metadata



def test_future_action_phrase_alone_does_not_demote_substantive_final(monkeypatch) -> None:
    import sys
    import types

    monkeypatch.setitem(sys.modules, "audioop", types.SimpleNamespace())
    from ai.realtime_api import RealtimeAPI

    api = RealtimeAPI.__new__(RealtimeAPI)
    text = "Everything is healthy, and I’ll keep monitoring the battery."
    assert api._classify_deliverable_text(text) == "final"


def test_response_create_runtime_normalizes_oversized_followthrough_metadata() -> None:
    from ai.realtime.response_create_runtime import ResponseCreateRuntime

    class API:
        def __init__(self):
            self.stored = []

        def _extract_response_create_metadata(self, event):
            return event.setdefault("response", {}).setdefault("metadata", {})

        def _store_response_create_local_context(self, payload):
            self.stored.append(payload)
            return "rctx_000777"

    api = API()
    runtime = ResponseCreateRuntime(api)
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_1077",
                "input_event_key": "item_1077:required_deliverable_followthrough:0",
                "parent_input_event_key": "item_1077",
                "tool_followup": "true",
                "tool_followup_release": "true",
                "followthrough_step_output_policy": "required_deliverable",
                "followthrough_post_completion_reason": "required_deliverable_owed",
                "followthrough_catchup_payload": "x" * 694,
            }
        },
    }

    runtime._enforce_tool_followup_metadata_limit(response_create_event=event, canonical_key="ck")
    metadata = event["response"]["metadata"]

    assert "followthrough_catchup_payload" not in metadata
    assert metadata["followthrough_context_id"] == "rctx_000777"
    assert api.stored[0]["followthrough_catchup_payload"] == "x" * 694
    assert len(metadata) <= PROVIDER_METADATA_MAX_PROPERTIES
    assert max(len(value) for value in metadata.values()) <= PROVIDER_METADATA_MAX_VALUE_LENGTH


def test_required_deliverable_envelope_survives_all_mandatory_candidates() -> None:
    stored = []

    def store(payload):
        stored.append(payload)
        return "rctx_required"

    metadata = {
        "turn_id": "turn_1077",
        "input_event_key": "item_1077:required_deliverable_followthrough:0",
        "parent_input_event_key": "item_1077",
        "parent_turn_id": "turn_1077",
        "tool_call_id": "call_diag",
        "tool_followup": "true",
        "tool_followup_release": "true",
        "blocked_by_response_id": "resp_parent",
        "consumes_canonical_slot": "false",
        "explicit_multipart": "true",
        "local_runtime_followthrough": "true",
        "followthrough_step_output_policy": "required_deliverable",
        "tool_followup_step_output_policy": "required_deliverable",
        "followthrough_post_completion_reason": "required_deliverable_owed",
        "tool_followup_post_completion_reason": "required_deliverable_owed",
        "followthrough_required_tool_name": "read_runtime_diagnostics",
        "tool_followup_required_tool_name": "read_runtime_diagnostics",
        "followthrough_required_tool_already_executed": "true",
        "tool_followup_suppress_if_parent_covered": "true",
        "followthrough_catchup_payload": "x" * 694,
        "gesture_motion_status": "completed",
        "tool_name": "read_runtime_diagnostics",
    }

    result = normalize_provider_metadata(metadata, store_local_context=store)

    assert len(result.metadata) <= PROVIDER_METADATA_MAX_PROPERTIES
    assert result.metadata["followthrough_context_id"] == "rctx_required"
    assert result.metadata["followthrough_required_tool_name"] == "read_runtime_diagnostics"
    assert result.metadata["followthrough_required_tool_already_executed"] == "true"
    assert result.metadata["turn_id"] == "turn_1077"
    assert result.metadata["input_event_key"] == "item_1077:required_deliverable_followthrough:0"
    assert "tool_followup_step_output_policy" not in result.metadata
    assert "tool_followup_post_completion_reason" not in result.metadata
    assert "tool_followup_required_tool_name" not in result.metadata
    assert "gesture_motion_status" not in result.metadata
    assert stored and stored[0]["followthrough_catchup_payload"] == "x" * 694


def test_metadata_validation_error_retries_required_deliverable_with_reduced_envelope(monkeypatch) -> None:
    import asyncio
    import sys
    import types

    monkeypatch.setitem(sys.modules, "audioop", types.SimpleNamespace())
    from ai.realtime_api import RealtimeAPI

    api = RealtimeAPI.__new__(RealtimeAPI)
    api._current_run_id = lambda: "run-test"
    api._current_turn_id_or_unknown = lambda: "turn_1077"
    api._response_create_queue_drain_source = ""
    api.response_in_progress = True
    api._response_in_flight = True
    api._metadata_validation_retry_keys = set()
    api._last_response_create_ts = None
    api._trace_context_marks_required_deliverable_followthrough = lambda metadata: True
    api._drain_response_create_queue = lambda source_trigger=None: None
    sent = []

    class Transport:
        async def send_json(self, _websocket, payload):
            sent.append(payload)

    api._get_or_create_transport = lambda: Transport()
    api._last_sent_response_create_event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_1077",
                "input_event_key": "item_1077:required_deliverable_followthrough:0",
                "parent_input_event_key": "item_1077",
                "tool_followup": "true",
                "tool_followup_release": "true",
                "consumes_canonical_slot": "false",
                "explicit_multipart": "true",
                "local_runtime_followthrough": "true",
                "followthrough_step_output_policy": "required_deliverable",
                "followthrough_post_completion_reason": "required_deliverable_owed",
                "followthrough_required_tool_name": "read_runtime_diagnostics",
                "followthrough_required_tool_already_executed": "true",
                "followthrough_context_id": "rctx_000001",
                "metadata_schema_version": "provider_metadata.v1",
                "followthrough_catchup_payload": "x" * 694,
                "gesture_motion_status": "completed",
            },
            "instructions": "Deliver the report now.",
        },
    }
    event = {
        "error": {
            "message": "Invalid 'response.metadata.followthrough_catchup_payload': string too long. Expected a string with maximum length 512, but got a string with length 694 instead."
        }
    }

    asyncio.run(api.handle_error(event, object()))

    assert len(sent) == 1
    retry_metadata = sent[0]["response"]["metadata"]
    assert retry_metadata["metadata_retry"] == "validation_reduced_envelope"
    assert retry_metadata["followthrough_context_id"] == "rctx_000001"
    assert "followthrough_catchup_payload" not in retry_metadata
    assert "gesture_motion_status" not in retry_metadata
    assert len(retry_metadata) <= 16


def test_run1077_multistep_followthrough_metadata_contract_and_cleanup(monkeypatch) -> None:
    import asyncio
    import sys
    import types

    monkeypatch.setitem(sys.modules, "audioop", types.SimpleNamespace())
    from ai.realtime.response_create_runtime import ResponseCreateRuntime
    from ai.realtime.response_terminal_handlers import ResponseTerminalHandlers

    class API:
        def __init__(self):
            self._response_create_local_context_by_id = {}
            self._response_create_local_context_counter = 0
            self.sent = []
            self.steps = []

        def _extract_response_create_metadata(self, event):
            return event.setdefault("response", {}).setdefault("metadata", {})

        def _store_response_create_local_context(self, payload):
            self._response_create_local_context_counter += 1
            context_id = f"rctx_{self._response_create_local_context_counter:06d}"
            self._response_create_local_context_by_id[context_id] = {"payload": payload}
            return context_id

        def _cleanup_response_create_local_context_by_metadata(self, metadata):
            self._response_create_local_context_by_id.pop(metadata.get("followthrough_context_id"), None)

    api = API()
    runtime = ResponseCreateRuntime(api)
    api.steps.extend(["gesture_look_right:completed", "read_runtime_diagnostics:completed"])
    event = {
        "type": "response.create",
        "response": {
            "metadata": {
                "turn_id": "turn_1077",
                "input_event_key": "item_1077:required_deliverable_followthrough:0",
                "parent_input_event_key": "item_1077",
                "tool_call_id": "compgest_diag_1077",
                "tool_followup": "true",
                "tool_followup_release": "true",
                "blocked_by_response_id": "resp_parent_1077",
                "consumes_canonical_slot": "false",
                "explicit_multipart": "true",
                "local_runtime_followthrough": "true",
                "followthrough_step_output_policy": "required_deliverable",
                "tool_followup_step_output_policy": "required_deliverable",
                "followthrough_post_completion_reason": "required_deliverable_owed",
                "tool_followup_post_completion_reason": "required_deliverable_owed",
                "followthrough_required_tool_name": "read_runtime_diagnostics",
                "tool_followup_required_tool_name": "read_runtime_diagnostics",
                "followthrough_required_tool_already_executed": "true",
                "followthrough_catchup_payload": "x" * 694,
                "gesture_motion_status": "completed",
                "followthrough_dispatch_source": "deterministic_followthrough_motion_gate",
                "followthrough_required_tool_execution": "false",
            },
            "tool_choice": "none",
            "instructions": "Completed gesture_look_right and read_runtime_diagnostics. Deliver the report now; do not call tools again.",
        },
    }

    runtime._enforce_tool_followup_metadata_limit(response_create_event=event, canonical_key="ck_1077")
    metadata = event["response"]["metadata"]
    api.sent.append(event)

    assert api.steps == ["gesture_look_right:completed", "read_runtime_diagnostics:completed"]
    assert event["response"]["tool_choice"] == "none"
    assert metadata["followthrough_required_tool_already_executed"] == "true"
    assert "followthrough_catchup_payload" not in metadata
    assert "tool_followup_step_output_policy" not in metadata
    assert "gesture_motion_status" not in metadata
    assert len(metadata) <= PROVIDER_METADATA_MAX_PROPERTIES
    assert all(len(value) <= PROVIDER_METADATA_MAX_VALUE_LENGTH for value in metadata.values())
    assert len(api.sent) == 1
    context_id = metadata["followthrough_context_id"]
    assert context_id in api._response_create_local_context_by_id
    api._cleanup_response_create_local_context_by_metadata(metadata)
    assert context_id not in api._response_create_local_context_by_id
