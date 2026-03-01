# Realtime API Migration Map

This map decomposes `ai/realtime_api.py` into explicit ownership regions, documents what has already been extracted, and lays out a prioritized slice plan for the remaining work.

## 1) Responsibility map of `ai/realtime_api.py` (functional regions + function pointers)

> Current monolith size: **11,316 LOC**.

### A. Bootstrapping, dependency wiring, and runtime guards
- Core constructor and dependency initialization:
  - `RealtimeAPI.__init__`
  - `_require_websockets`, `_resolve_websocket_exceptions`, `_validate_outbound_endpoint`
- Session-level setup and readiness state:
  - `_configure_event_router`, `is_ready_for_injections`, `get_session_health`

### B. Event routing + websocket lifecycle handling
- Router hookup and fallback exception policy:
  - `_configure_event_router`, `_on_event_handler_exception`
- Connection attempt/accounting:
  - `_note_connection_attempt`, `_note_connected`, `_note_disconnect`, `_note_failure`, `_note_reconnect`
- Transport usage sites (still called from monolith):
  - `_get_or_create_transport`

### C. Turn/utterance correlation and response lifecycle state
- Correlation and canonical keys:
  - `_build_utterance_context`, `_utterance_context_scope`, `_canonical_utterance_key`
  - `_bind_active_input_event_key_for_turn`, `_rebind_active_response_correlation_key`
- Response obligations and lifecycle traces:
  - `_response_obligation_key`, `_emit_response_lifecycle_trace`, `_record_response_trace_context`
- Queue + release mechanics:
  - `_schedule_pending_response_create`, `_sync_pending_response_create_queue`, `_can_release_queued_response_create`

### D. Audio input/output and transcript handling
- Microphone/turn-detection configuration:
  - `_create_microphone`, `_initialize_microphone`, `_resolve_vad_turn_detection`
- User transcript processing:
  - `_extract_transcript`, `_log_user_transcript`, `_should_log_partial_user_transcript`
- Playback transitions:
  - `_on_playback_complete`

### E. Injection and stimulus gating
- External stimulus policy and startup gate checks:
  - `_can_accept_external_stimulus`, `_is_allowed_awaiting_confirmation_stimulus`
  - `_maybe_defer_startup_injection`, `_release_startup_injection_gate`
- Injection entrypoints:
  - `inject_event`, `inject_system_context`, `_emit_injected_event`, `_emit_system_context_payload`

### F. Confirmation + governance + intent dedupe
- Action staging and normalization:
  - `_stage_action`, `_normalize_confirmation_decision`, `_describe_staged_action`
- Confirmation prompting and parse/detect:
  - `_build_approval_prompt`, `_parse_confirmation_decision`, `_detect_alternate_intent_while_confirmation_pending`
- Idempotency and duplicate suppression:
  - `_idempotency_key_for_action`, `_evaluate_intent_guard`, `_is_duplicate_tool_call`

### G. Tool execution pipeline and structured no-op logging
- Tool context + normalization helpers:
  - `_build_tool_runtime_context`, `_normalize_tool_intent`, `_extract_dry_run_flag`
- Execution controls:
  - `_tool_execution_cooldown_remaining`, `_record_executed_tool_call`
- Audit + no-op trace semantics:
  - `_log_structured_noop_event`, `_format_action_packet`

### H. Memory retrieval / preference recall / auto-memory reflection
- Turn memory retrieval and startup digest:
  - `_prepare_turn_memory_brief`, `_build_startup_memory_digest_note`
- Preference recall detection and cleanup:
  - `_is_preference_recall_intent`, `_normalize_memory_recall_answer`, `_memory_pin_followup_needed`
- Reflection and memory write gate:
  - `_enqueue_response_done_reflection`, `_should_store_auto_memory`, `_parse_response_done_payload`

### I. Research permissioning + domain allowlist handling
- Research gating:
  - `should_request_research_permission`, `_research_can_run_now`
- Request/domain fingerprinting:
  - `_build_research_request_fingerprint`, `_extract_primary_research_domain`
- Allowlist policy:
  - `_research_request_domains_allowlisted`, `_is_research_domain_allowlisted`

### J. Sensor aggregation and shutdown
- Sensor aggregation utilities:
  - `_sensor_aggregation_key`, `_is_critical_sensor_trigger`, `_get_injection_priority`
- Shutdown and signal handling:
  - `shutdown_handler`, `_request_shutdown`

## 2) Current extracted modules and ownership boundaries

These modules already exist and should be treated as owned seams with narrow responsibilities.

- `ai/realtime/event_router.py`
  - Ownership: event-type registration, dispatch, fallback, exception callback wrapping.
  - Boundary: no lifecycle state mutation; routes only.
- `ai/realtime/transport.py`
  - Ownership: websocket connect/send/recv wrappers and outbound endpoint validation hook.
  - Boundary: no policy decisions or retries.
- `ai/realtime/injections.py`
  - Ownership: startup injection gate, queueing, timeout-based release.
  - Boundary: no stimulus eligibility policy.
- `ai/realtime/input_audio_events.py`
  - Ownership: input-audio speech start/stop/transcript event handlers.
  - Boundary: delegates state writes to `RealtimeAPI` facade methods.
- `ai/realtime/response_lifecycle.py`
  - Ownership: deterministic empty-response retry policy + lifecycle helper utilities.
  - Boundary: no websocket loop ownership.
- `ai/realtime/response_terminal_handlers.py`
  - Ownership: terminal response event handlers (done/failed/cancelled) and cleanup choreography.
  - Boundary: calls back into API; should not absorb unrelated policy.
- `ai/realtime/shutdown.py`
  - Ownership: one-time shutdown request guard, cancellation ordering, websocket close guard.
  - Boundary: no event routing or injection policy.
- `ai/realtime/types.py`
  - Ownership: shared state dataclasses (`UtteranceContext`, `CanonicalResponseState`, `PendingResponseCreate`).
  - Boundary: types only, no behavior.

## 3) Target module tree

```text
ai/realtime/
  adapter.py                    # façade around websocket + event loop glue currently in RealtimeAPI.run
  confirmation_coordinator.py   # pending action state machine, prompt/parse, dedupe + timeout handling
  injection_bus.py              # external stimulus policy + startup gate orchestration + queueing policies
  tool_pipeline.py              # tool arg normalization, runtime context, execution/no-op accounting
  lifecycle_state.py            # canonical key correlation, obligation store, response create queue state
  event_router.py               # (existing)
  transport.py                  # (existing)
  injections.py                 # (existing startup gate primitive)
  input_audio_events.py         # (existing)
  response_lifecycle.py         # (existing)
  response_terminal_handlers.py # (existing)
  shutdown.py                   # (existing)
  types.py                      # (existing)
```

Ownership intent:
- `RealtimeAPI` remains an orchestration shell.
- Policy/state transitions move to coordinators (`confirmation_coordinator`, `injection_bus`, `lifecycle_state`).
- IO adapters (`adapter`, `transport`) remain thin and deterministic.

## 4) Prioritized slice plan (3–6 slices)

## Slice 1 (P0): `lifecycle_state.py` extraction
**Moved code/functions**
- Correlation and canonical key management:
  - `_build_utterance_context`, `_utterance_context_scope`, `_canonical_utterance_key`
  - `_bind_active_input_event_key_for_turn`, `_active_input_event_key_for_turn`
- Response-create queue coordination:
  - `_schedule_pending_response_create`, `_sync_pending_response_create_queue`, `_can_release_queued_response_create`
- Obligation state helpers:
  - `_response_obligation_key`, `_canonical_response_state_store`

**Interface signatures**
- `class LifecycleStateCoordinator:`
  - `def canonical_key(self, *, turn_id: str, input_event_key: str | None) -> str`
  - `def bind_input_event(self, *, turn_id: str, input_event_key: str) -> None`
  - `def enqueue_response_create(self, item: PendingResponseCreate) -> None`
  - `def pop_releasable_response_creates(self, *, trigger: str, metadata: dict[str, Any]) -> list[PendingResponseCreate]`
  - `def obligation_for(self, *, turn_id: str, input_event_key: str | None) -> dict[str, Any] | None`

**Tests to add/update**
- Add: `tests/test_realtime_lifecycle_state_coordinator.py`
  - canonical key lineage/normalization
  - queue release gating across confirmation hold transitions
- Update: `tests/test_realtime_lifecycle_invariants.py`, `tests/test_realtime_response_origin_correlation.py`

**Risk + rollback strategy**
- Risk: mis-correlated turn IDs causing dropped/duplicate response creates.
- Rollback: keep adapter shim methods in `RealtimeAPI` forwarding to old code path behind a config flag (`realtime.lifecycle_state_v2=false`).

---

## Slice 2 (P0): `confirmation_coordinator.py` extraction
**Moved code/functions**
- Action staging + normalization:
  - `_stage_action`, `_normalize_confirmation_decision`, `_describe_staged_action`
- Approval prompt/parse + alternate-intent detection:
  - `_build_approval_prompt`, `_parse_confirmation_decision`, `_detect_alternate_intent_while_confirmation_pending`
- Confirmation timeout/dedupe bookkeeping:
  - `_record_confirmation_timeout`, `_suppressed_confirmation_outcome`, `_is_suppressed_after_confirmation_timeout`

**Interface signatures**
- `class ConfirmationCoordinator:`
  - `def stage(self, action: ActionPacket) -> dict[str, Any]`
  - `def build_prompt(self, action: ActionPacket, staging: dict[str, Any]) -> str`
  - `def parse_decision(self, transcript: str) -> NormalizedConfirmationDecision | None`
  - `def on_timeout(self, *, idempotency_key: str | None, cause: str) -> None`
  - `def should_suppress(self, *, idempotency_key: str | None, now: float | None = None) -> bool`

**Tests to add/update**
- Add: `tests/test_realtime_confirmation_coordinator.py`
  - deterministic parse matrix (approve/deny/unclear)
  - timeout suppression and idempotency-key windows
- Update: `tests/test_realtime_confirmation_flow.py`, `tests/test_tool_governance_metadata.py`

**Risk + rollback strategy**
- Risk: confirmation loops or false approvals.
- Rollback: dual-run mode that logs both legacy + coordinator decisions and executes legacy decision until parity reaches threshold.

---

## Slice 3 (P1): `injection_bus.py` extraction
**Moved code/functions**
- Stimulus policy + startup gate checks:
  - `_can_accept_external_stimulus`, `_is_allowed_awaiting_confirmation_stimulus`
  - `_maybe_defer_startup_injection`, `_startup_gate_is_critical_allowed`
- Injection queueing/emit helpers:
  - `_emit_injected_event`, `_emit_system_context_payload`, `_format_event_for_injection`

**Interface signatures**
- `class InjectionBus:`
  - `def accept(self, *, source: str, kind: str, priority: str, metadata: dict[str, Any]) -> tuple[bool, str]`
  - `def defer_or_emit_event(self, payload: dict[str, Any]) -> None`
  - `def defer_or_emit_system_context(self, payload: dict[str, Any]) -> None`
  - `def release_startup_gate(self, *, reason: str) -> int`

**Tests to add/update**
- Add: `tests/test_realtime_injection_bus.py`
  - awaiting-confirmation source allow/deny matrix
  - startup gate critical-priority bypass behavior
- Update: `tests/test_realtime_injections.py`, `tests/test_realtime_stimulus_gating.py`, `tests/test_system_context_injection.py`

**Risk + rollback strategy**
- Risk: over-suppression (missed critical stimuli) or under-suppression (chatter during confirmation).
- Rollback: fallback to direct `RealtimeAPI` path via one feature flag and retain queue serialization format unchanged.

---

## Slice 4 (P1): `tool_pipeline.py` extraction
**Moved code/functions**
- Argument normalization/runtime context:
  - `_extract_dry_run_flag`, `_build_tool_runtime_context`, `_normalize_tool_intent`
- Duplicate/idempotency + call fingerprints:
  - `_build_tool_call_fingerprint`, `_is_duplicate_tool_call`, `_record_executed_tool_call`
- Structured noop and action formatting:
  - `_log_structured_noop_event`, `_format_action_packet`

**Interface signatures**
- `class ToolPipeline:`
  - `def normalize_call(self, *, tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]`
  - `def runtime_context(self, *, action: ActionPacket) -> dict[str, Any]`
  - `def should_execute(self, *, tool_name: str, args: dict[str, Any]) -> tuple[bool, str]`
  - `def record_result(self, *, tool_name: str, args: dict[str, Any], outcome: str) -> None`

**Tests to add/update**
- Add: `tests/test_realtime_tool_pipeline.py`
  - duplicate suppression and fingerprint stability
  - dry-run flag coercion behavior
- Update: `tests/test_realtime_research_flow.py`, `tests/test_tool_governance_metadata.py`

**Risk + rollback strategy**
- Risk: accidental behavior change in governance/no-op telemetry.
- Rollback: preserve existing `ActionPacket` fields and logger keys; route through legacy execution when pipeline returns unknown state.

---

## Slice 5 (P2): `adapter.py` extraction (outer orchestration shell)
**Moved code/functions**
- Websocket loop orchestration currently in `run` + event dispatch glue.
- Connection/reconnect accountings:
  - `_note_connection_attempt`, `_note_connected`, `_note_disconnect`, `_note_failure`, `_note_reconnect`
- Transport dependency setup:
  - `_get_or_create_transport`

**Interface signatures**
- `class RealtimeAdapter:`
  - `async def run(self) -> None`
  - `async def connect_once(self) -> None`
  - `async def dispatch_event(self, event: dict[str, Any], websocket: Any) -> None`
  - `def session_health(self) -> dict[str, Any]`

**Tests to add/update**
- Add: `tests/test_realtime_adapter_reconnect.py`
  - reconnect bookkeeping and backoff hooks
  - route-to-handler fallback behavior parity
- Update: `tests/test_realtime_event_routing_regression.py`, `tests/test_realtime_session_health.py`, `tests/test_realtime_websocket_close_guard.py`

**Risk + rollback strategy**
- Risk: reconnect regressions or shutdown ordering drift.
- Rollback: keep `RealtimeAPI.run()` as a thin switch with `adapter_enabled` guard, defaulting off until parity tests pass.

## 5) Completed slices: before/after LOC tracking

The following slices are already complete (extracted today in-tree). LOC figures are direct file line counts.

| Completed slice | New module(s) LOC | Estimated monolith reduction (net) | Notes |
|---|---:|---:|---|
| Event router extraction | 44 | -44 | `event_router.py` now owns type->handler dispatch. |
| Transport wrapper extraction | 47 | -47 | `transport.py` isolates connect/send/recv. |
| Startup injection coordinator extraction | 77 | -77 | `injections.py` holds startup gate queue + timeout release. |
| Input audio event handler extraction | 144 | -144 | `input_audio_events.py` keeps speech/transcript callbacks. |
| Response lifecycle helper extraction | 231 | -231 | `response_lifecycle.py` owns deterministic empty-response retries. |
| Response terminal handler extraction | 457 | -457 | `response_terminal_handlers.py` owns terminal cleanup choreography. |
| Shutdown coordinator extraction | 80 | -80 | `shutdown.py` owns idempotent shutdown and websocket close guard. |
| Shared realtime types extraction | 44 | -44 | `types.py` centralizes dataclasses used across collaborators. |

### Aggregate tracking snapshot
- `ai/realtime_api.py` current size: **11,316 LOC**.
- Extracted helper modules total: **1,124 LOC**.
- Approximate pre-extraction monolith equivalent: **12,440 LOC** (current monolith + extracted LOC).

> Tracking rule for future completed slices: capture `wc -l ai/realtime_api.py ai/realtime/<new_module>.py` before and after merge, then append a delta row in this table.
