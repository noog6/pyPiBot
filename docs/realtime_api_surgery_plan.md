# Realtime API Surgery Plan (Theo)

## Phase 0 — Baseline + Cut List

- Baseline LOC (`ai/realtime_api.py`): **9545**.
- Top large regions (rough spans by method size):
  1. `handle_function_call` (~484 lines)
  2. `__init__` (~449 lines)
  3. `_handle_response_created_event` (~412 lines)
  4. `_maybe_handle_preference_recall_intent` (~204 lines)
  5. `_handle_event_legacy` / `_handle_input_audio_transcription_completed_event` (~150 lines each)

### CUT LIST (non-core adapter responsibilities)
- Preference recall workflow (query, cache, fallback, replacement response emission)
- Turn-level preference-recall suppression / trace bookkeeping
- Misc memory-intent stop-word helper logic
- Tool governance execution path (deferred to future pass due blast radius)

### Selected slice for this pass
1. **Preference recall workflow extraction** (low-risk boundary: method-level extraction, same call sites).

### Planned removals from `ai/realtime_api.py`
- `_suppress_preference_recall_server_auto_response`
- `_emit_preference_recall_skip_trace_if_needed`
- `_maybe_handle_preference_recall_intent`
- `_find_stop_word`

### New home modules + interfaces
- `ai/realtime/preference_recall_runtime.py`
  - `_suppress_preference_recall_server_auto_response(api, websocket)`
  - `_emit_preference_recall_skip_trace_if_needed(api, turn_id=...)`
  - `_maybe_handle_preference_recall_intent(api, text, websocket, source=...)`
  - `_find_stop_word(api, text)`

### Tests to protect behavior
- Add contract tests proving `RealtimeAPI` methods delegate to extracted runtime functions unchanged.

### Rollback strategy
- Revert single commit to restore in-file implementations and remove new module.

---

## Final Results

- Before LOC: **9545**
- After LOC: **9230**
- Net change: **-315 lines**

### Removed sections/functions/classes (from `ai/realtime_api.py`)
- Full method bodies removed and replaced by thin delegation wrappers:
  - `_suppress_preference_recall_server_auto_response`
  - `_emit_preference_recall_skip_trace_if_needed`
  - `_maybe_handle_preference_recall_intent`
  - `_find_stop_word`

### New modules created/modified
- Added `ai/realtime/preference_recall_runtime.py`
- Updated `ai/realtime_api.py` imports + delegating wrappers

### Tests added/updated
- Added `tests/test_realtime_preference_recall_runtime_contract.py`

### Behavior changes
- Intended behavior change: **none** (logic moved; wrappers preserve call contract).

### Negative confirmation list
- No changes to websocket transport lifecycle.
- No changes to response scheduling algorithms.
- No changes to tool governance decisions.
- No persistence format changes.
