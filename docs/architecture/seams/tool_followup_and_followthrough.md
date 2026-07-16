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

- Suppress only when parent coverage is actually qualified; streaming text classification must use the accumulated response text, not an isolated delta, so progress acknowledgements such as “checking…” cannot become final coverage when later delta fragments omit progress words.
- If completed parent text is available, defensive parent-coverage checks must reclassify that text and reject cached `final` state when the text is progress-only.
- Active required-deliverable followthrough keeps the necessary post-tool follow-up eligible even after a parent acknowledgement response has completed.
- Hold is transitional and should resolve into release/suppress deterministically.
- Non-distinct or gesture-only outputs should not consume final deliverable ownership.
- Local runtime tool results that deterministically advance continuity to an active low-risk gesture step should dispatch that gesture before generic tool-output `response.create` scheduling; if no safe descriptor exists, leave the generic fallback path intact.
- `gesture_intermediate_inject_only` suppression is valid only when runtime can deterministically dispatch the next non-report step; if the active next step is non-gesture and no runtime descriptor exists, keep `response.create` enabled so model dispatch can continue the chain.
- `followthrough_remaining` settlement alone is insufficient to keep a followup open: guard decisions must also confirm a still-open non-report step, a pending required report step, or motion that is physically still in `queued/started`.

## H) Failure signatures

- Tool executes but no visible followup output appears.
- Redundant followups after already-final parent response.
- Followup stuck in hold due to parent classification never resolving.
- Local runtime diagnostics completes between gestures, then generic response-create is blocked by followthrough exclusivity before the next deterministic gesture dispatches.

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
2. `ai/realtime_api.py` parent coverage evaluation, local runtime post-tool deterministic dispatch, and queue release logs.
3. `ai/realtime/response_create_runtime.py` metadata capping path (`_enforce_tool_followup_metadata_limit`) when arbitration reasons unexpectedly shift (for example `non_gesture_tool` after queue drain).
4. `ai/realtime/response_terminal_handlers.py` followthrough guard + selection reason.
5. Logs: `queue_release_parent_eval`, `tool_followup_*`, `response_done_followthrough_guard_decision`, `tool_followup_metadata_capped`.

## L) Multi-step catch-up context ownership

Completed-step catch-up material for required-deliverable followthrough is local orchestration context, not provider-visible metadata. The model-facing report instruction may include the completed-step facts needed to answer (for example that `gesture_look_right` and `read_runtime_diagnostics` already completed), but `response.metadata` carries only compact routing/contract fields and, when needed, a local `followthrough_context_id`.

Context lifecycle:

1. Tool-result and deterministic followthrough code buffers completed-step facts by turn.
2. Required-deliverable dispatch consumes those facts into response instructions and response-create local context.
3. Metadata normalization externalizes oversized `followthrough_catchup_payload` and sends the compact context id.
4. Response events can inspect the id through echoed metadata/trace context, but current event-time behavior does not require rehydrating the context; the model-facing catch-up facts are still supplied through the response instructions.
5. `response.done` cleanup removes the process-local context when the trace metadata contains the id; stale entries are bounded by store size.

This is an incremental migration boundary: the local context store currently provides retention/correlation and future migration scaffolding, not authoritative event-time rehydration. Additional verbose arbitration evidence, diagnostic summaries, and canonical lineage details should move behind local context ids rather than adding more provider metadata fields. Missing local context should fail visibly as a controlled lifecycle/context incident rather than silently settling a required deliverable.

## M) Asynchronous gesture step completion

Owning layer: Layer 1 runtime execution + Layer 2 continuity/compound-request state + Layer 3 deterministic followthrough dispatch.

Ordered compound gestures distinguish **tool invocation acceptance** from **physical execution completion**. A gesture tool result with `queued=true` and `motion_request_key=<key>` means only that the motion substrate accepted the request; it does not complete the compound step. Continuity records that step as `executing`; runtime owns a bounded structured correlation registry keyed by `motion_request_key` with owner turn, compound request id, step id, tool call id, tool name, registration state, and last observed motion status.

Step advancement rules:

1. `tool_result_received` for a compound-owned gesture captures the active compound descriptor before the step transitions away from `active`, registers structured runtime correlation, then leaves the continuity step in `executing`.
2. Motion-controller lifecycle callbacks must not mutate continuity directly. They capture immutable snapshots and schedule the runtime motion handler onto Theo's owning asyncio event loop.
3. The event-loop handler validates `motion_request_key`, owner turn, compound request id, step id, tool call id, and current continuity status before applying a motion event.
4. `gesture_motion_registered state=started` is observational; it keeps the step executing and must not dispatch the next ordered step.
5. `gesture_motion_registered state=completed` for the matching structured correlation is the terminal-success signal. Only then does continuity mark the step `completed`, activate the next step, and allow deterministic followthrough dispatch.
6. `failed`, `cancelled`, `timed_out`, and `superseded` are terminal non-success states. They block the chain with an observable continuity blocker and must not be treated as success.

Race reconciliation: after a gesture tool result binds its `tool_call_id` to a `motion_request_key`, runtime rereads current motion state and routes any already-observed `started`/terminal status through the same event-loop-owned handler used for live observer events. This protects fast motions whose completion callback arrives before normal tool-result processing finishes.

Correlation cleanup: runtime removes structured correlation after successful completion, terminal failure/cancellation/timeout/supersession, stale request/step/turn detection, actual required-deliverable/settled-turn cleanup, runtime shutdown, TTL expiry, or bounded-capacity pruning. Intermediate `response.done` events must preserve correlations whose same compound request still has the correlated gesture in `executing`. Observer registration returns an unsubscribe callback; runtime shutdown unregisters it so old `RealtimeAPI` instances are not retained by the process-global motion observer list. The observer is runtime-lifetime, not per-WebSocket-connection, so reconnect cleanup must not unregister it.

Scope boundary: only gestures structurally owned by the active ordered compound request gate this chain. Runtime-only gestures such as thinking cues, speaking posture, attention holds, idle presence, startup, and shutdown gestures remain independent unless they are explicitly dispatched as the active compound step.

Primary logs: `compound_async_step_registered`, `compound_async_step_motion_event_scheduled`, `compound_async_step_completion_observed`, `compound_async_step_advanced`, `compound_async_step_completion_ignored`, `compound_async_step_correlation_cleaned`, and `compound_async_step_failed`.

## N) User-initiated read-result delivery

Owning layer: Layer 2 continuity/obligation state + Layer 3 tool-followup arbitration.

A user-initiated read tool whose result is intended to answer the user's request creates a result-delivery obligation for the owning turn. The obligation is structural: it follows from the user request selecting a read tool whose output is needed for the answer, not from optional phrases such as “tell me” or “report back”. Internal runtime context reads, background observations, system-health polls, model-context-only reads, and compound-intermediate reads do not automatically create a spoken-result obligation.

Progress acknowledgements are valid audible responses, but they are non-terminal and non-substantive for this obligation. Parent coverage may suppress a later tool followup only when the parent contains correlated tool-result evidence (for example, the actual voltage value for a voltage read). Text presence, audio start, canonical bookkeeping, playback completion, or queue draining are not result evidence.

The queue release path must preserve one substantive followup while the read-result obligation remains open. Duplicate result events should be suppressed by stable turn/tool/canonical identities, and failures should produce one controlled failure response rather than fabricating a value or settling silently.

Implemented read-result logs in this seam currently include `read_result_parent_coverage_evaluated` and `read_result_parent_terminal_rejected`. Additional obligation lifecycle logs remain future instrumentation, not current behavior.

### Incremental obligation marker and positive coverage

Pending user-facing read follow-up is currently the structural obligation marker at the tool-followup and terminal-selection seam. The observable states are: result owed (pending user-requested read follow-up exists), result available (tool result has produced a follow-up create), follow-up scheduled/released (tool-followup state is pending/release), and result delivered (the follow-up response becomes the terminal deliverable).

For user-requested reads, parent suppression requires positive correlated coverage for the same parent canonical key and tool call. Negative evidence such as “the parent is not progress” is insufficient. If coverage is unknown, or known false, the follow-up remains authoritative and should be released once the response boundary permits. Compound-intermediate reads remain governed by the compound followthrough chain and do not independently create a standalone spoken-result obligation.
