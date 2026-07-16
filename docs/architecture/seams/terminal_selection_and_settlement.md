# Seam Family: Terminal Selection and Settlement

Owning layer: Layer 3 arbitration + Layer 4 turn-settlement bookkeeping.

Status: **Active seam** with fragile cross-seam authority handoffs.

Confidence: medium (selection authority is clear; settlement handoff ordering remains integration-sensitive).

## A) Purpose

Decide whether a terminal response event is the turn deliverable and apply settlement effects (selection store updates, continuity close progression, tool followup release timing).

## B) Owning modules/files

- `ai/terminal_deliverable_arbitration.py` (selection policy).
- `ai/realtime/response_terminal_handlers.py` (terminal event execution and handoff).
- `ai/realtime_api.py` (`_apply_terminal_deliverable_selection`, settlement helpers).
- `ai/semantic_owner_arbitration.py` (semantic owner resolution).

## C) Entry points

- `handle_response_done(...)` and related terminal handlers.
- `arbitrate_terminal_deliverable_selection(...)`.
- `decide_semantic_owner(...)`.

## D) Exit points / outcomes

- Selection decision persisted per response id with reason.
- Canonical/semantic owner assignment for continuity and deliverable reconciliation.
- Continuity close/update dispatch (`close_commitment`, `complete_required_deliverable`, etc.).

## E) Upstream dependencies

- Response origin and provisional state.
- Empty/non-empty evidence, transcript-final status.
- Followthrough and required-deliverable open state.

## F) Downstream impacts

- Whether turn is considered complete.
- Whether followup work remains open.
- Whether parent turn receives semantic completion credit.

## G) Core invariants

- `selected=false` is valid for intermediate non-deliverable states; not automatically a fault.
- `selected=false reason=empty_tool_followup_non_deliverable` can still be an expected terminal close when tool motion/followthrough is fully complete and no report is owed.
- Semantic owner reassignment is narrow and reason-coded.
- Cancelled/non-deliverable reasons should be explicitly recorded.

## H) Failure signatures

- Turn closes on non-deliverable output.
- Turn fails to close after normal selected terminal.
- Parent semantic promotion misattributes completion.

## I) Known tricky handoffs

- Tool-output `response.done` with followthrough still open.
- Tool-output empty non-deliverable terminal: suppress silent-incident + allow continuity close only when no followthrough/report remains.
- Provisional server-auto done before transcript-final.
- Selection reason to continuity close-flag translation.

## J) Related seams

- [Response lifecycle and identity](response_lifecycle_and_identity.md)
- [Required deliverable contracts](required_deliverable_contracts.md)
- [Continuity and commitment state](continuity_and_commitment_state.md)

## K) First places to inspect

1. `ai/terminal_deliverable_arbitration.py` reason path chosen.
2. `ai/realtime/response_terminal_handlers.py` around `selection_decision` and continuity handoff.
3. `ai/realtime_api.py` selection store writer and final-deliverable checks.
4. Logs: `terminal_deliverable_selection_applied`, `semantic_answer_owner_resolved`, `continuity_response_done_handoff`.

## L) Progress parents with open read-result obligations

Owning layer: Layer 3 terminal selection and canonical lifecycle.

When a user-requested read-result obligation is open, terminal selection cannot let a progress-only parent response consume the terminal deliverable slot. A response classified as `progress` with `covered=false` remains a progress acknowledgement across `response.done`, playback completion, queue drain, and canonical projection unless new substantive response content is attached. The read-only-helper precedence relaxation is only valid when the parent text is not progress-only and there is genuine result coverage.

Canonical projection and semantic coverage must not share a field in a way that silently upgrades progress to final. Any classification mutation requires new evidence and structured tracing with old/new class and coverage values. The defensive rejection log for this seam is `read_result_parent_terminal_rejected reason=open_read_result_obligation_progress_parent`.

### Positive read-result coverage gate

Terminal selection separates three facts: the response was selected, the response has user-facing text, and the response covers the requested read result. For an open user-requested read follow-up, selected text may relax pending read-helper precedence only when correlated result coverage is `result_covered_true`. `coverage_unknown` and `result_covered_false` both preserve follow-up precedence. This prevents substantive but incomplete parents such as “The battery sensor is working normally” from answering “What is the voltage?”.
