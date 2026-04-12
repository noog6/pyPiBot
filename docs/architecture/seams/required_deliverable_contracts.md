# Seam Family: Required-Deliverable Contracts

Owning layer: Layer 3 arbitration + Layer 4 continuity handoff.

Status: **Historic seam with past regressions** and **fragile handoff seam**.

Confidence: medium (contract intent is clear; completeness evidence and redrive eligibility remain edge-sensitive).

## A) Purpose

Ensure user-visible required deliverables (especially report/followthrough outputs) are not marked complete until substantive content and required tool execution evidence are present.

## B) Owning modules/files

- `ai/realtime/response_terminal_handlers.py` (required-deliverable completion checks + redrive).
- `ai/realtime_api.py` required-deliverable trace markers and active-step checks.
- `ai/continuity.py` compound step state with `required_deliverable` output policy.
- Adjacent influence: `ai/realtime/response_create_runtime.py` followthrough overrides.

## C) Entry points

- `response.done` handler required-deliverable gating path.
- Trace metadata interpretation (`followthrough_step_output_policy`, post-completion reason).
- Continuity active required-step inspection.

## D) Exit points / outcomes

- Accept completion and allow settlement progression.
- Reject completion and defer settlement (`missing_substantive_content` or `missing_tool_execution`).
- Optional materialization redrive dispatch with retry budget.

## E) Upstream dependencies

- Tool followup metadata quality.
- Continuity compound-step state correctness.
- Terminal selection and semantic owner context.

## F) Downstream impacts

- Turn close commitment accuracy.
- Final report completion semantics.
- User-visible completeness vs progress mismatch incidents.

## G) Core invariants

- Required-deliverable paths must not settle on bridge/progress-only output.
- Missing tool execution should block completion even if text exists.
- Redrive budget must be bounded and explicit.

## H) Failure signatures

- Report obligation appears complete but user never received substantive answer.
- Repeated redrive attempts without materialization.
- Continuity `required_deliverable_pending` drift from terminal logs.

## I) Known tricky handoffs

- Tool output marked non-deliverable while chain remains open.
- Redrive eligibility depending on both trace markers and continuity-open state.
- Completion attribution when semantic owner is parent-promoted.

## J) Related seams

- [Tool followup and followthrough](tool_followup_and_followthrough.md)
- [Terminal selection and settlement](terminal_selection_and_settlement.md)
- [Continuity and commitment state](continuity_and_commitment_state.md)

## K) First places to inspect

1. Required-deliverable checks inside `handle_response_done` terminal path.
2. `RealtimeAPI._response_done_marks_required_deliverable_followthrough(...)`.
3. `ContinuityLedger` required step pending/completion transitions.
4. Logs: `required_deliverable_completion_rejected`, `required_deliverable_followthrough_materialization_redrive*`.
