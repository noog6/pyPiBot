# Theo Curiosity Engine (Tiny, Governed, Low-Rate)

## Purpose

The curiosity engine is a narrow **relevance filter seam** that records a small
number of potentially interesting signals (currently conversation repeated-topic
anchors) and may mark one as surface-eligible when runtime/arbitration context
is safe.

This is intentionally not a planner, autonomous behavior loop, or tool caller.

## Owning layer

**Owning layer: Decision Arbitration Layer (emerging), with signal intake from Perception/Memory interfaces.**

Runtime integration is limited to feeding candidates and honoring existing
response/create + confirmation + obligation gates.

## What curiosity is

- Tiny scoring over explicit candidates.
- Bounded, TTL-based short-term candidate memory.
- Dedupe and cooldown to keep output low-rate.
- Structured reason codes for inspectable behavior.
- Suggestion generation only; no direct execution.

## What curiosity is not

- Not a second orchestration framework.
- Not a bypass for governance/confirmations.
- Not a background chatter loop.
- Not hidden policy in transport plumbing.

## Current signal scope (v1)

- **Conversation repeated topic anchors** extracted from transcript trust snapshot.
- Candidate fields include `source`, `reason_code`, `score`, `dedupe_key`,
  `created_at`, `expires_at`, and optional `suggested_followup`.

## Output semantics

- `ignore`: below threshold or deduped.
- `record`: candidate kept in bounded recent window but not surfaced.
- `surface`: candidate marked surface-eligible for later turn use.

## Guardrails implemented

- Low-rate dedupe window.
- Surface cooldown.
- Bounded/decaying repeated-topic anchor stats (`anchor_max_entries`, `anchor_decay_window_s`).
- Suppress surfacing with explicit reasons: `suppressed_listening`, `confirmation_pending`, `obligation_open`, `busy_turn`.
- No direct tool execution path.
- Bounded in-memory candidate window.
- Structured logs:
  - `curiosity_candidate_detected`
  - `curiosity_candidate_suppressed`
  - `curiosity_candidate_recorded`
  - `curiosity_suggestion_surface_eligible`
  - `curiosity_suggestion_surface_suppressed`

## Ownership boundaries

- `ai/curiosity_engine.py`: scoring, dedupe/cooldown, bounded state, reasoned decision.
- `ai/realtime_api.py`: candidate ingestion from transcript trust snapshots,
  arbitration-aware surface suppression, bounded/decaying anchor bookkeeping,
  and bounded surfaced-candidate metadata cleanup.

## Anti-patterns to avoid

- Letting curiosity call tools directly.
- Adding recursive curiosity triggers from curiosity outputs.
- Storing unbounded curiosity history.
- Moving curiosity policy into websocket transport internals.
