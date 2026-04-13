# Seam Family: Continuity and Commitment State

Owning layer: Layer 4 continuity/presence bookkeeping.

Status: **Stable seam** (narrow authority), but with **fragile handoffs** at terminal close boundaries.

## A) Purpose

Maintain short-horizon state of ongoing/unresolved/commitment/blocker items and compound followthrough steps, then provide settlement context without becoming an authority seam.

## B) Owning modules/files

- `ai/continuity.py` (ledger, state model, settlement classification).
- Runtime integrations in `ai/realtime_api.py` and `ai/realtime/response_terminal_handlers.py`.
- Adjacent influence: `ai/quiet_intent.py` consultative stance use.

## C) Entry points

- Continuity event applications (`transcript_final`, tool start/result, `response_done`).
- Turn settlement reads used by terminal followthrough guards.

## D) Exit points / outcomes

- `ContinuityBrief` and settlement projections.
- Close/update flags for ongoing/commitment/unresolved/required-deliverable.
- Diagnostic summaries for runtime and incident triage.

## E) Upstream dependencies

- Event ordering from runtime.
- Accurate semantic owner/turn binding at response done.
- Correct step output policy metadata for compound chains.

## F) Downstream impacts

- Followthrough chain remaining guards.
- Required-deliverable pending detection.
- Consultative posture/context signals.

## G) Core invariants

- Continuity is bookkeeping/context, not arbitration authority.
- Required-deliverable pending should track compound-step reality.
- Settlement states must remain bounded and deterministic.

## H) Failure signatures

- `close_commitment` says complete while required deliverable remains pending.
- Followthrough chain marked closed too early.
- Continuity stance contradicts terminal settlement logs.

## I) Known tricky handoffs

- Semantic owner parent promotion with cross-turn rebind.
- Non-deliverable intermediate tool-output done events.
- Report-step pending vs non-report followthrough guard scopes.
- Settlement classification when continuity buckets are empty but `compound_request.final_followup_pending=true` during required-deliverable redrive windows.

## J) Related seams

- [Terminal selection and settlement](terminal_selection_and_settlement.md)
- [Required deliverable contracts](required_deliverable_contracts.md)
- [Observability and tracing surfaces](observability_and_tracing_surfaces.md)

## K) First places to inspect

1. `ai/continuity.py` compound step transition helpers for required deliverable.
2. `ai/continuity.py::classify_turn_settlement(...)` handling of `compound_request.final_followup_pending` and `next_pending_step_id`.
3. Runtime continuity handoff payload in `response_terminal_handlers`.
4. `get_continuity_turn_settlement` call sites.
5. Logs: `continuity_event_applied`, `continuity_inspection_summary`, `CONTINUITY SUMMARY`.
