# Seam Family: Response Create Arbitration

Owning layer: Layer 1 runtime execution + Layer 3 turn arbitration.

Status: **High-churn seam** / **likely seam-factory zone**.

## A) Purpose

Decide whether a `response.create` attempt should `SEND`, `SCHEDULE`, `BLOCK`, or `DROP`, and do so deterministically with reason codes.

## B) Owning modules/files

- `ai/interaction_lifecycle_policy.py` (lifecycle candidate generation).
- `ai/response_create_arbitration.py` (final contender arbitration).
- `ai/realtime/response_create_runtime.py` (authoritative execution/orchestration).
- Adjacent influence: `ai/realtime_api.py` state/metadata helpers, `ai/contract_breach.py` observational breach signals.

## C) Entry points

- `InteractionLifecyclePolicy.decide_response_create(...)`
- `decide_response_create_arbitration(...)`
- `ResponseCreateRuntime.evaluate_response_create_attempt(...)`

## D) Exit points / outcomes

- Immediate send over transport.
- Scheduled pending create queue entry.
- Explicit blocked/drop outcome with reason code.
- Optional contract-breach diagnostic emission when handoff looks unsafe.

## E) Upstream dependencies

- Active response/audio state.
- Canonical slot ownership and single-flight guards.
- Origin normalization (`server_auto`, `tool_output`, `micro_ack`, etc.).
- Transcript-final readiness and suppression flags.

## F) Downstream impacts

- Which response becomes authoritative execution candidate.
- Whether terminal settlement will later have a valid response lineage to reconcile.
- Whether tool-followup and required-deliverable flows can progress.

## G) Core invariants

- Exactly one winning create action per evaluated attempt.
- Higher-priority hard guards (lineage/terminal) beat lifecycle send/defer candidates.
- Same-turn owner suppression should be explicit (`DROP`) rather than silent loss.
- Transcript-final attention admission remains fail-closed at create time: a verdict reason of
  `attention_gate_closed` must force `BLOCK` even if a create attempt reaches runtime execution.

## H) Failure signatures

- Repeated create deferrals with no eventual send.
- Tool followup attempts dropped by same-turn owner unexpectedly.
- Canonical terminal blocks firing while downstream expects open followthrough.

## I) Known tricky handoffs

- Lifecycle `DEFER` vs arbitration `DROP` precedence.
- Provisional `server_auto` ownership vs materialization bridge dispatch.
- Transition from queued create to active response id binding.

## J) Related seams

- [Response lifecycle and identity](response_lifecycle_and_identity.md)
- [Tool followup and followthrough](tool_followup_and_followthrough.md)
- [Terminal selection and settlement](terminal_selection_and_settlement.md)

## K) First places to inspect

1. `ai/realtime/response_create_runtime.py` snapshot + execution decision assembly.
2. `ai/response_create_arbitration.py` candidate priorities.
3. `ai/interaction_lifecycle_policy.py` reason-code candidate ordering.
4. `response_create_*` + `create_seam_parent_coverage_eval` log families.
