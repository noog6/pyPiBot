# Seam Family: Response Lifecycle and Identity

Owning layer: Layer 1 runtime / nervous system.

Status: **Fragile seam** and **historic seam with regressions** around sparse metadata and parent linkage.

Confidence: medium (core ownership is clear; sparse metadata and stale-context ordering remain edge-sensitive).

## A) Purpose

Maintain canonical identity from response creation through terminal completion, including retries, provisional upgrade/replacement, and semantic linkage inputs.

## B) Owning modules/files

- `ai/realtime/lifecycle_state.py` (canonical key construction + obligation transitions).
- `ai/realtime/response_lifecycle.py` (empty-response retry decisions and retry lineage).
- `ai/realtime_api.py` canonical response state stores, trace mapping, selection store.
- Adjacent influence: `ai/semantic_owner_arbitration.py`, `ai/realtime/response_terminal_handlers.py`.

## C) Entry points

- Canonical key builders and turn/input binding helpers.
- Response created/done event ingesters in runtime API/terminal handlers.
- Empty response retry evaluator (`decide_empty_response_done_action`).

## D) Exit points / outcomes

- Canonical state transitions (`created`, `done`, deliverable markers).
- Retry dispatch with idempotency key and lineage suffixing.
- Response-to-turn mapping used by terminal and continuity seams.

## E) Upstream dependencies

- Run id / turn id / input event key quality.
- Provider metadata completeness.
- Transport event ordering and stale context handling.

## F) Downstream impacts

- Semantic-owner promotion eligibility.
- Terminal deliverable write correctness.
- Required-deliverable completion attribution.

## G) Core invariants

- Canonical key should be stable for the same logical utterance lineage.
- Retry lineage must be explicit and idempotent.
- Response done should reconcile against correct canonical and trace context.

## H) Failure signatures

- `response.done` appears but binds to wrong turn/canonical key.
- Duplicate settlement or missing settlement due to stale response context.
- Empty retry loops or retries skipped despite empty done evidence.

## I) Known tricky handoffs

- Provisional `server_auto` pre-transcript identity to transcript-final upgraded identity.
- Empty retry materialization using server-auto dispatch semantics without claiming provisional ownership.
- Parent linkage when metadata is sparse or tool-prefixed.

## J) Related seams

- [Response create arbitration](response_create_arbitration.md)
- [Terminal selection and settlement](terminal_selection_and_settlement.md)
- [Required deliverable contracts](required_deliverable_contracts.md)

## K) First places to inspect

1. `LifecycleStateCoordinator` key/obligation transitions.
2. `ResponseLifecycleTracker.maybe_schedule_empty_response_retry(...)`.
3. `RealtimeAPI` canonical response state mutation helpers.
4. Logs: `empty_response_*`, `terminal_deliverable_selection_applied`, canonical marker/class logs.
