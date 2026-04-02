# Theo Continuity / Presence (Current-State, Code-Aligned)

Purpose: formalize what continuity/presence **actually does today** so future work
stays seam-local and does not accidentally expand continuity into hidden
arbitration, governance, or planning authority.

## Owning layer

**Owning layer:** Layer 4 (short-horizon presence/continuity bookkeeping).

This is the correct layer because continuity is implemented as a deterministic,
in-memory ledger (`ContinuityLedger`) that projects read-only briefs,
settlement classifications, and inspection summaries. It is integrated in runtime
as context/diagnostic state and a narrow followthrough guard input.

Continuity does **not** own:
- response-create arbitration
- terminal-deliverable selection
- semantic-owner selection
- tool-followup release/hold/suppress arbitration
- governance/confirmation policy
- memory retrieval policy
- embodiment cue execution
- planner-like decision making

## What continuity/presence is today

Continuity is currently **ledger + weak presence substrate**:

- **Ledger:** deterministic item bookkeeping across transcript/tool/response
  events (`transcript_final`, `tool_call_started`, `tool_result_received`,
  `response_done`).
- **Presence substrate:** emits a bounded continuity stance plus stance detail,
  tracks short-horizon open/closed items, and models compound request
  followthrough state.
- **Read surfaces:** exposes read-only brief/settlement/diagnostics/debug summary
  APIs used by runtime logs, session health, and selective consultative inputs.

It is not purely passive logging anymore, but it is also not an execution
controller.

## Implemented data model

### Bounded enums and states

- Item kinds: `ongoing`, `unresolved`, `commitment`, `blocker`, `constraint`,
  `recently_closed`.
- Item statuses: `active`, `blocked`, `pending`, `resolved`, `expired`.
- Item priorities: `low`, `medium`, `high`.
- Stance labels: `idle`, `assisting_observation`, `assisting_execution`,
  `assisting_query`, `awaiting_user`, `awaiting_tool`,
  `awaiting_perception`, `recovering_context`.
- `awaiting_perception` is part of the bounded stance enum, but current default
  transcript/tool/response heuristics do not actively emit it.
- Turn settlement states: `settled`, `awaiting_tool`,
  `followthrough_remaining`, `unresolved_followup`, `recently_closed_only`,
  `active_items_only`.

### Core records

- `ContinuityItem`: id, kind, summary, status, priority, source, detail,
  `expires_after_turns`.
- `CompoundContinuityStep`: bounded step kind/status plus perception/report
  traits.
- `CompoundContinuityState`: request-level chain with active/completed/pending
  step pointers and a `final_followup_pending` flag.
- `ContinuityBrief`: read projection with bucketed open/closed items, `current`
  projection, stance + detail, and optional compound state.
- `ContinuityTurnSettlement`: read-only settlement classification flags.

### Short-horizon / TTL behavior

- Items are expiring turn-by-turn via `expires_after_turns` decrement on every
  applied continuity event.
- Typical horizons in current implementation:
  - transcript-derived ongoing/commitment/unresolved: 4 turns
  - tool blockers: 2 turns
  - recently closed snapshots: 1 turn

### Bucket semantics

- `blockers`: active/pending/blocked `blocker` items.
- `current`: prioritized projection of blockers → commitments → unresolved →
  ongoing → constraints (capped).
- `recently_closed`: resolved `recently_closed` items only (capped).
- Each bucket is bounded (`_MAX_ITEMS_PER_BUCKET = 3`) and priority-sorted.

### Fingerprint + cooldown semantics

Continuity brief logging dedupes by a material fingerprint containing stance,
stance_detail, settlement, counts, projected current/recently_closed signatures,
and compound signature. If unchanged, a reminder log is emitted only after
cooldown (`_CONTINUITY_BRIEF_LOG_COOLDOWN_S`, default 10s).

## Implemented behaviors

### Event application (producer side)

Continuity updates on four runtime events:
- `transcript_final`: sets request ongoing item, optional commitment/unresolved,
  stance derivation, optional compound-state derivation.
- `tool_call_started`: adds blocker, may promote commitment detail, may advance
  compound active step, sets `awaiting_tool`.
- `tool_result_received`: closes blocker to `recently_closed`, keeps commitment
  active, may complete compound step, sets `idle`.
- `response_done`: conditionally closes/removes ongoing/commitment/unresolved,
  closes blockers, advances/resolves compound state, sets `idle` or
  `awaiting_user`.

### Diagnostic-only behavior

- `continuity_event_applied` INFO log on every applied continuity event.
- `continuity_inspection_summary` INFO with bounded projected items + settlement.
- `continuity_brief_built` INFO with deduped fingerprint/reminder semantics.
- `CONTINUITY SUMMARY` INFO turn-close summary (runtime-config gated).
- Read-only inspection APIs (`get_continuity_diagnostics`,
  `get_continuity_debug_summary`, `get_continuity_turn_settlement`).

### Consultative/context behavior

- Quiet Intent input path reads continuity stance (`get_continuity_brief(...).stance`),
  normalizes it, and uses it as one signal in consultative posture-bias
  selection.
- Session health includes compact continuity diagnostics (stance, settlement,
  counts).

### Narrow behavior-shaping behavior

Continuity provides one narrow behavior-shaping input: followthrough-chain guard
for terminal deliverable arbitration bridging.

- Runtime checks continuity settlement/compound followup pending for the turn
  when handling `response.done`.
- If followthrough remains, terminal arbitration can preserve tool-followup
  precedence for empty tool-output done events.
- This is scoped to turn-close followthrough protection and does not create new
  arbitration authority inside continuity.

## Response terminal + followthrough bridge runtime contract

This section documents runtime contracts exercised at `response.done` time.

### Non-deliverable intermediate tool-followup

- A tool-output `response.done` can be intentionally marked non-deliverable when
  terminal arbitration returns `selected=false` with
  `selection_reason=tool_followup_precedence`.
- In that state, runtime treats the event as an intermediate followthrough step,
  not the turn’s final user-facing answer.
- The runtime still applies terminal-selection bookkeeping for auditability, but
  continuity close is not committed from that event alone.

### When followthrough is still open

- Followthrough openness is evaluated by
  `_turn_followthrough_chain_remaining(..., include_report_followup=...)`, which
  inspects continuity settlement + compound state.
- With `include_report_followup=False` (the response-done guard path), report-only
  pending steps do not by themselves force reopening; unresolved non-report steps
  do.

### Intentional `selected=false` and turn progression behavior

- `selected=false` is expected for intermediate/non-deliverable tool-followup
  terminals and should not be interpreted as a runtime fault by itself.
- When followthrough remains, continuity handoff sets `keep_ongoing=true` and
  withholds close commitment; when followthrough is resolved and a normal
  substantive terminal is selected, close commitment can proceed.
- This is why turn progression may stay open after one `response.done` and close
  only after a later followthrough completion.

## Empty-response retry contract (bridge-aware)

### Empty-response candidate criteria

Retry is considered only when the canonical response is empty of user-facing
evidence (no assistant text, no terminal response text, no audio started, no
deliverable observed, and no progress/final deliverable class), websocket is
available, and delivery state is not terminal (`cancelled`/`failed`/`errored`).

### Origin gating and narrow bridge exception

- Default retryable origins are `prompt`, `server_auto`, and `user_transcript`.
- `tool_output` is normally origin-gated out.
- Narrow exception: when an empty `tool_output` terminal is a followthrough bridge
  case (`tool_followup_precedence` or `empty_tool_followup_non_deliverable` plus
  followthrough-chain remaining), retry origin is override-promoted for dispatch
  as `server_auto`.

### What promotion does and does not mean

- Promotion is dispatch materialization for the retry create event
  (`empty_retry_materialization=report_followup`), not a global ownership rewrite.
- This keeps retry transport semantics compatible with server-auto handling while
  preserving separate ownership/provisional rules.

## Contract breach detection (observability contract)

- Contract breach detection is an INFO-level diagnostic aid for suspicious seam
  combinations; it is not a hard runtime exception path.
- Current breach types:
  - `EMPTY_TOOL_FOLLOWUP_DONE`
  - `FOLLOWTHROUGH_REMAINING_WITH_NON_DELIVERABLE_OUTPUT`
  - `TOOL_FOLLOWUP_BLOCKED_BY_SAME_TURN_OWNER`
- `recommended_action` is operational guidance for investigation/repair intent,
  not an automatically enforced recovery command.
- Breach logs should be treated as investigate-first signals. Do not suppress the
  log family without proving the seam facts are benign and covered by tests.

## Log anchor interpretation (high-signal only)

- `response_done_followthrough_guard_decision`
  - Emitted by followthrough guard evaluation.
  - `decision=true` means followthrough is still open for the evaluated scope.
  - `include_report_followup=false` indicates report-only followup should be
    ignored for close-blocking.

- `terminal_deliverable_selection_applied`
  - Authoritative terminal-selection write point for a response.
  - `canonical_key` is execution ownership; `semantic_owner_canonical_key` can
    differ when semantic parent promotion occurs.
  - `selected=false` with reasons like `tool_followup_precedence` or
    `provisional_server_auto_awaiting_transcript_final` is an intentional
    non-terminal selection, not necessarily a fault.

- `semantic_answer_owner_resolved`
  - Emitted when semantic owner is reassigned away from execution key (typically
    tool-output parent promotion).
  - If absent, semantic owner remained execution-scoped.

- `continuity_response_done_handoff`
  - Summarizes close intent:
    - `close_commitment=true` means runtime is trying to settle commitment
      continuity.
    - `complete_final_report=true` means selected+substantive final report path
      completed.
    - `followthrough_chain_remaining=true` means progression remains open.
    - `allow_cross_turn_rebind=true` with
      `rebind_reason=semantic_owner_parent_promoted` indicates parent-turn
      continuity handoff.

- `empty_response_detected`
  - Confirms empty-response evidence for a canonical done path before retry
    arbitration.

- `empty_response_retry_origin_override_applied`
  - Narrow exception path only: bridged tool-output followthrough empty-done is
    being retried through `server_auto` dispatch semantics.
  - Should appear with bridge metadata on the resulting create event
    (`empty_retry_materialization=report_followup`).

- `contract_breach_detected`
  - Advisory diagnostic artifact, not hard runtime failure.
  - `recommended_action` means “investigate this seam posture”; do not treat as
    auto-remediation.

## Runtime integration map

### Producers (write path)

- Transcript finalization path applies `transcript_final` events.
- Tool execution path applies `tool_call_started` and `tool_result_received`.
- `response.done` handler applies `response_done` with close flags derived from
  terminal close conditions.

### Consumers (read path)

- Quiet Intent state-transition refresh: reads continuity stance (**consultative**).
- Session health / diagnostics / debug summaries (**diagnostic**).
- Response-done followthrough guard (`_turn_followthrough_chain_remaining`):
  reads settlement + compound followup flag (**narrow behavior-shaping**).

### Explicitly non-consuming authority seams

Current continuity outputs are **not** consumed as direct authority in:
- response-create gating
- semantic-owner reassignment policy
- governance confirmation policy
- tool-followup release/hold/suppress policy
- memory retrieval policy selection
- embodiment cue execution policy

## Authority boundary contract (must hold)

### Authority continuity HAS today

- Deterministic short-horizon bookkeeping of continuity items/stance.
- Read-only settlement classification and diagnostics.
- Narrow followthrough-chain signal contribution at response-done time.

### Authority continuity DOES NOT have today

- No direct send/suppress/defer of `response.create`.
- No terminal deliverable winner selection logic.
- No semantic-owner chooser logic.
- No governance approval/confirmation decisions.
- No planner or multi-turn scheduling control.
- No memory retrieval strategy control.
- No embodiment execution control.

### Guardrail statement

Treat continuity as a bounded bookkeeping/context seam. If future work requires
new authority, it must land in the owning arbitration/governance seam rather
than being added implicitly to continuity helpers.


## Drift / seam-local follow-up opportunities

- Add one explicit negative-boundary regression test proving continuity updates
  are observational/context-only with respect to response-create, governance,
  and tool-followup policy seams (for example: continuity state changes alone do
  not alter `response.create` gating outcomes, confirmation requirements, or
  tool-followup policy decisions without their seam owners changing).

## Evidence anchors

Primary implementation anchors:
- `ai/continuity.py`
- `ai/realtime_api.py`
- `ai/realtime/response_terminal_handlers.py`
- `docs/architecture/quiet_intent.md`

Primary test anchors:
- `tests/test_continuity.py`
- `tests/test_realtime_session_health.py`
- `tests/test_terminal_deliverable_arbitration.py`
- `tests/test_quiet_intent_runtime_calibration.py`
