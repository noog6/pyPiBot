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
