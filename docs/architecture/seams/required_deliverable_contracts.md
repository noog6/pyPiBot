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
- Missing tool execution should block completion even if text exists; diagnostics/report required steps infer `read_runtime_diagnostics` as the required tool so completed diagnostics evidence can satisfy the report contract.
- Tool execution evidence may include both the runtime-observed `turn_id` and the continuity-resolved `owner_turn_id`; required-tool matching must consider both while still requiring the specific inferred tool name.
- Required-tool evidence remains exact-name matched; unrelated non-gesture tool calls must not satisfy an unknown required-tool contract by default.
- Redrive budget must be bounded and explicit.
- Required-deliverable followthrough retry keys must remain parent-scoped
  (`parent:required_deliverable_followthrough:<n>`) and must not append retry
  suffixes onto a prior followthrough key.
- When a report follows a completed deterministic diagnostics step, first-attempt
  materialization should mark the diagnostics result as already available, include
  a compact completed-result summary, and instruct the model to deliver the report
  now rather than narrating progress. Local companion diagnostics at the report
  boundary must use the dedicated required-deliverable dispatcher rather than a
  generic tool-followup create.
- Metadata capping must preserve both required-tool aliases
  (`followthrough_required_tool_name` and `tool_followup_required_tool_name`) plus
  `followthrough_required_tool_already_executed` for required-deliverable creates.
- Required-deliverable `tool_output` response.done candidates must not be
  demoted by report-step self-precedence; only remaining **non-report**
  followthrough may block terminal selection.

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
4. Followthrough create lineage: `input_event_key`, `parent_input_event_key`, retry count, required-tool aliases, and already-executed markers (`followthrough_required_tool_already_executed`).
5. First-attempt instructions for completed diagnostics: verify the dedicated dispatcher included a completed diagnostic result summary and did not route through generic tool-followup materialization.
6. Logs: `required_deliverable_completion_rejected`, `required_deliverable_followthrough_materialization_redrive*`.
