# EXECUTIVE SUMMARY (1–2 pages)

## What’s broken or risky today

- **Double audible responses can still happen for one utterance** when a speculative `server_auto` response reaches audio start, while a queued `assistant_message` replacement survives under a synthetic key and speaks later. This is visible in run 413 (`responses=3` after one user utterance in that window). 
- **Turn/correlation ownership is fragmented** across `turn_id`, `input_event_key`, canonical key, and response-id-derived non-consuming keys, which can hide duplicates from single-flight guards.
- **Policy ownership is split** between `_send_response_create`, `handle_event(response.created)`, watchdog fallback, and `send_assistant_message` key rewriting.
- **Cancellation is race-prone**: the pre-audio cancellation guard exists, but once audio starts, follow-up scheduling paths can still emit if keyed differently.

## Why it happens (top causal chain)

1. Server emits `response.created` with no client metadata; Theo classifies as `server_auto` and binds it to a synthetic or pending key.
2. Transcript final rebinds to item key and triggers preference-recall path.
3. Preference recall schedules assistant follow-up while server response is active.
4. `send_assistant_message` rewrites to a synthetic key during in-flight suppression context, detaching follow-up from canonical utterance guards.
5. Server-auto reaches first audio delta (audible response already happened).
6. Later queue drain emits assistant response under synthetic non-consuming canonical key, so “already delivered” checks don’t stop it.

## Shortest safe path to “no double responses”

- Stop rewriting preference-recall assistant follow-ups to synthetic input keys while a response is in flight.
- Keep parent `input_event_key`, so canonical delivery-state checks block late duplicate creates after server-auto already delivered/done.
- Add regression test asserting parent key preservation in this path.

---

# LIFECYCLE MAPS (authoritative)

## 2.1 turn_id lifecycle

```mermaid
flowchart TD
    A[_next_response_turn_id()] --> B[_resolve_response_create_turn_id()]
    B -->|metadata.turn_id| C[Use provided turn_id]
    B -->|assistant_message + micro_ack_turn_id| D[Use micro_ack turn]
    B -->|current response turn exists| E[Reuse _current_response_turn_id]
    B -->|fallback| F[Generate turn_N]

    C --> G[_utterance_context_scope]
    D --> G
    E --> G
    F --> G

    G --> H[_ensure_response_create_correlation]
    H --> I[_active_input_event_key_by_turn_id[turn_id] = input_event_key]

    J[response.created] --> K[_current_turn_id_or_unknown]
    K --> I

    L[input_audio_transcription.completed] --> M[item_id => input_event_key]
    M --> I
    M --> N[_rebind_active_response_correlation_key]
```

### turn_id Sources & Sinks

| Source (event/path) | Generator | Stored In | Consumers | Risks | Evidence |
|---|---|---|---|---|---|
| `response.create` scheduling | `_resolve_response_create_turn_id` | `_current_response_turn_id`, utterance context | `_send_response_create`, lifecycle logs | Reuse of active turn in mixed-origin windows | `ai/realtime_api.py::_resolve_response_create_turn_id` |
| Assistant injected message | metadata `turn_id` default from context | response metadata + context | response origin queue, binding | wrong turn reuse if context stale | `send_assistant_message` |
| Transcript final | current turn + `item_id` | `_active_input_event_key_by_turn_id` | server_auto rebinding and obligations | stale map entry if not replaced | `conversation.item.input_audio_transcription.completed` handler |

## 2.2 server_auto lifecycle

```mermaid
flowchart TD
    A[speech_stopped] --> B[server emits response.created origin=server_auto]
    B --> C[handle_event(response.created)]
    C --> D[bind to pending/current input_event_key]
    D --> E{suppression/obligation/preference guard?}
    E -->|yes + pre-audio| F[send response.cancel]
    E -->|no| G[allow response]
    G --> H[first audio delta]
    H --> I[canonical first_audio_started=true]

    J[transcript final] --> K[rebind synthetic->item key]
    K --> L[preference recall may queue assistant_message]

    I --> M[response.done]
    M --> N[drain queued response.create]
    N --> O{same canonical key?}
    O -->|yes| P[block duplicate]
    O -->|synthetic key| Q[duplicate slips through]
```

### Response Origins

| origin | who initiates | typical triggers | intended policy | actual policy | evidence |
|---|---|---|---|---|---|
| `server_auto` | server realtime VAD/config | speech stop before transcript final | allow only if not replaced/suppressed; cancel before audio when replaced | works pre-audio, but may already start audio before cancellation path completes | `handle_event(response.created)` guard + run 413 excerpt |
| `prompt` | client startup prompt | initial synthetic prompt | one startup response | behaves as expected in log | run 413 excerpt |
| `assistant_message` | Theo (`send_assistant_message`) | tool/memory follow-up | single replacement response for utterance | synthetic key rewrite can decouple and allow extra audible response | `send_assistant_message` + run 413 excerpt |

---

# CURRENT STATE ARCHITECTURE REPORT

## Components and responsibilities

- **`RealtimeAPI`**: owns response scheduling, origin queueing, key correlation, suppression, cancellation, and transcript flow.
- **`InteractionLifecycleController`**: canonical state transitions and cancellation decisions.
- **Micro-ack manager/watchdog**: side-channel conversational latency mitigation.

## Policy ownership (current)

- **Allow/cancel/suppress/defer** decisions are split across:
  - `_send_response_create` (single-flight / delivered / created checks)
  - `handle_event` response.created branch (server_auto guards)
  - `send_assistant_message` (key mutation policy)
  - watchdog outcome path (`response_not_scheduled`, micro-ack)

## Violations / fragilities

- Duplicated responsibility around key canonicalization and response replacement.
- Ambiguous truth between `_response_delivery_ledger`, `_response_created_canonical_keys`, and non-consuming lifecycle keys.
- Inconsistent keying path (`turn_id + item_id`) vs synthetic keys allows bypass of canonical dedupe.

## Interaction handling features inventory

| Feature | Purpose | Where | Failure modes | Recommendation |
|---|---|---|---|---|
| micro-ack scheduler | bridge latency | micro-ack manager + realtime hooks | can coexist with active response if guards drift | Keep, but ensure strict cancel on first audio |
| playback-busy watchdog | avoid dropped turns | transcript watchdog + queue retry | retries can enqueue stale creates if keying diverges | Keep, tie strictly to canonical key state |
| response obligation tracking | ensure one response owed per utterance | `_response_obligations` APIs | split ownership with suppression flags | Keep, but centralize in one state machine in Path B |
| suppression logic | avoid redundant chatter | preference recall guard paths | late cancel no-op after audio start | Keep, add pre-audio invariant tests |
| tool-driven assistant scheduling | explicit follow-up response.create | `send_assistant_message` | synthetic key rewrite created duplicate window | **Change now (done)** |

---

# FIX PLAN (sequenced)

## Path A (minimal fix, fastest)

1. **Remove synthetic key rewrite for preference recall follow-ups.**
   - Edit: `ai/realtime_api.py::send_assistant_message`.
   - Change: keep original `input_event_key` even when response is in flight.
   - Safety: preserves canonical guard behavior; no protocol changes.
   - Tests: add unit regression ensuring key preservation while in-flight.
   - Expected logs: no `synthetic_preference_recall_*` for this path; more `response_not_scheduled ... already_handled` instead.

2. **Validate duplicate prevention using existing duplicate-response repro tests.**
   - Use targeted pytest for `tests/test_realtime_duplicate_response_repro.py`.

## Path B (architectural tidy)

1. Introduce a unified **utterance lifecycle state machine** keyed by canonical key (`run_id:turn_id:input_event_key`).
2. Move all allow/cancel/defer decisions behind one policy adapter.
3. Deprecate non-consuming synthetic response keying except explicit multipart responses.
4. Consolidate obligations + delivery ledger + created-set into one state object with invariants.
5. Add deterministic fixture-driven integration stream tests for response.created/delta/done races.

Trade-off: bigger refactor risk, but removes multi-source-of-truth drift.

---

# TESTS

## Added/extended tests

- Added: `test_preference_recall_preserves_parent_input_event_key_during_active_response`.
  - Asserts in-flight preference recall keeps parent `input_event_key` and does not synthesize a detached key.

## Required invariant coverage status

- **No >1 audible response per utterance**: covered indirectly by existing duplicate-response repro tests plus canonical guard tests; should be strengthened further in Path B with stream fixture assertions.
- **server_auto deterministically cancelled pre-audio or allowed once**: existing lifecycle invariant tests exercise pre-audio cancellation behavior.
- **turn_id uniqueness and non-reuse from synthetic prompts**: existing lifecycle invariant tests cover startup/user turn separation.
- **audio_playback_busy rescheduler no duplicate create**: existing turn-response guard tests cover retry/drop behavior.

---

# Open Questions / Uncertainty

- **Confidence: high** that synthetic key rewriting was a key enabler of duplicates in the run-413 pattern.
- **Confidence: medium** on whether any other path still intentionally requires non-consuming synthetic key follow-ups for preference recall UX.
- **Confidence: medium** that watchdog + queued replacement interactions could still produce edge duplicates if external ordering changes under high latency.
