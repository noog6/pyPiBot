# Realtime API Surgery Plan

## Phase 0 — Baseline + Cut List

- Baseline LOC (`ai/realtime_api.py`): **9964** lines.
- Top 5 largest regions by method span (rough responsibilities):
  1. `handle_function_call` (tool-governance/business logic path) — ~484 lines.
  2. `__init__` (runtime wiring + state initialization) — ~449 lines.
  3. `_handle_response_created_event` (response lifecycle orchestration) — ~412 lines.
  4. `_send_response_create` (response.create business policy + queue gate decisions) — ~239 lines.
  5. `_drain_response_create_queue` (queue arbitration + release policy) — ~202 lines.

### CUT LIST (non-core adapter candidates)

Candidate regions outside strict vendor adapter responsibilities:
- Response-create policy + queue arbitration logic (`_send_response_create`, `_drain_response_create_queue`).
- Confirmation reminder dedupe/release gates.
- Tool-governance decision pipeline (`handle_function_call`).

### Selected slices for this pass

1. **Response-create send policy**: move `_send_response_create` logic into `ai/realtime/response_create_runtime.py`.
2. **Response-create queue drain arbitration**: move `_drain_response_create_queue` logic into `ai/realtime/response_create_runtime.py`.

### Functions/classes removed from `realtime_api.py` (planned)

- `RealtimeAPI._send_response_create`
- `RealtimeAPI._drain_response_create_queue`

(Original implementations deleted and replaced with thin delegating glue.)

### New home modules + interface surface

- Modified module: `ai/realtime/response_create_runtime.py`
  - `ResponseCreateRuntime.send_response_create(...)`
  - `ResponseCreateRuntime.drain_response_create_queue(...)`

### Tests to protect behavior

- Add/extend contract tests asserting `RealtimeAPI` wrappers delegate to `ResponseCreateRuntime` with unchanged signatures.
- Run targeted realtime response/confirmation tests that exercise response.create send/queue behavior.

### Rollback strategy

- Revert commit to restore prior in-class implementations.
- No storage format changes or config schema changes in this slice.

---

## Final Results

- After LOC (`ai/realtime_api.py`): **9545** lines.
- Net change: **-419 lines** in `ai/realtime_api.py`.
- Removed sections/functions/classes:
  - Deleted in-file implementation of `RealtimeAPI._send_response_create`.
  - Deleted in-file implementation of `RealtimeAPI._drain_response_create_queue`.
- New modules created/modified:
  - Modified `ai/realtime/response_create_runtime.py` with moved implementations and required imports.
  - Added `tests/test_realtime_response_create_runtime_contract.py` for delegation contracts.
- Behavior changes: **None intended** (delegation target changed, not semantics).
- Negative confirmation list:
  - No vendor websocket session lifecycle flow changes.
  - No event schema or payload format changes.
  - No governance threshold/policy changes.
  - No persistence format/config key changes.
