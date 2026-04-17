# Seam Family: Tool Followup and Followthrough

Owning layer: Layer 3 turn arbitration with runtime execution in Layer 1.

Status: **Likely seam-factory zone** and **active seam**.

## A) Purpose

Control whether tool-followup outputs are released, held, or suppressed; preserve followthrough chains until user-facing obligations are truly satisfied.

## B) Owning modules/files

- `ai/tool_followup_arbitration.py` (pure release/hold/suppress policy seam).
- `ai/realtime_api.py` parent coverage evaluation + queue release/suppression integration.
- `ai/realtime/response_terminal_handlers.py` followthrough handoff at `response.done`.
- Adjacent influence: `ai/terminal_deliverable_arbitration.py`, `ai/continuity.py`.

## C) Entry points

- `decide_tool_followup_arbitration(...)`.
- Runtime queue release paths (`queue_release_parent_eval` style call sites).
- Terminal event handlers where followthrough timing/markers are recorded.

## D) Exit points / outcomes

- Followup dispatched immediately, held pending classification, or suppressed due to parent coverage.
- Timing markers and diagnostics for followup lifecycle.
- Contract-breach hints when followup is blocked by same-turn ownership.

## E) Upstream dependencies

- Tool result distinctiveness.
- Parent response coverage and classification status.
- Followthrough chain remaining state from continuity/trace context.

## F) Downstream impacts

- Whether required-deliverable completion can ever occur.
- Whether terminal selection treats done event as intermediate vs final.
- Whether users see redundant gesture-only followup output.

## G) Core invariants

- Suppress only when parent coverage is actually qualified.
- Hold is transitional and should resolve into release/suppress deterministically.
- Non-distinct or gesture-only outputs should not consume final deliverable ownership.
- `gesture_intermediate_inject_only` suppression is valid only when runtime can deterministically dispatch the next non-report step; if the active next step is non-gesture and no runtime descriptor exists, keep `response.create` enabled so model dispatch can continue the chain.

## H) Failure signatures

- Tool executes but no visible followup output appears.
- Redundant followups after already-final parent response.
- Followup stuck in hold due to parent classification never resolving.

## I) Known tricky handoffs

- Parent coverage source differences (canonical vs terminal selection).
- Classification pending state timing relative to terminal events.
- Interaction with same-turn owner drop in response-create arbitration.
- Provider metadata capping can remove classifier-critical keys (for example `tool_name`), which can flip followup arbitration from suppress -> release even after parent coverage is established.

## J) Related seams

- [Required deliverable contracts](required_deliverable_contracts.md)
- [Terminal selection and settlement](terminal_selection_and_settlement.md)
- [Continuity and commitment state](continuity_and_commitment_state.md)

## K) First places to inspect

1. `ai/tool_followup_arbitration.py` decision reason path.
2. `ai/realtime_api.py` parent coverage evaluation + queue release logs.
3. `ai/realtime/response_create_runtime.py` metadata capping path (`_enforce_tool_followup_metadata_limit`) when arbitration reasons unexpectedly shift (for example `non_gesture_tool` after queue drain).
4. `ai/realtime/response_terminal_handlers.py` followthrough guard + selection reason.
5. Logs: `queue_release_parent_eval`, `tool_followup_*`, `response_done_followthrough_guard_decision`, `tool_followup_metadata_capped`.
