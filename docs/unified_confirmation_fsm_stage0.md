# Unified Confirmation FSM (Stage 0 Design)

## Current state map (before refactor)
- Tool governance confirmation:
  - State fields: `RealtimeAPI._pending_action`, `RealtimeAPI._awaiting_confirmation_completion`, orchestration phase `AWAITING_CONFIRMATION`.
  - Entry: `handle_function_call()` -> `_request_tool_confirmation()`.
  - Decision intake: `_maybe_handle_approval_response()`.
- Research intent permission:
  - State field: `RealtimeAPI._pending_research_request`.
  - Entry: `_maybe_process_research_intent()`.
  - Decision intake: `_maybe_handle_research_permission_response()`.
- Response queue and guard interactions:
  - Queue stale-drop: `_is_stale_queued_response_create()`.
  - Confirmation guard: `_should_guard_confirmation_response()`, `handle_transcribe_response_done()`, `handle_response_done()`.

## Proposed FSM contract
- `ConfirmationState` enum:
  - `IDLE` -> no active confirmation token
  - `PENDING_PROMPT` -> token created, prompt may be queued/deferred
  - `AWAITING_DECISION` -> prompt sent, waiting for user yes/no
  - `RESOLVING` -> decision accepted/rejected and cleanup/execution in progress
  - `COMPLETED` -> terminal transition before returning to `IDLE`
- Single active token: `PendingConfirmationToken | None`.
- Token kinds:
  - `research_permission`
  - `tool_governance`

## PendingConfirmationToken schema
- `id: str`
- `kind: str`
- `tool_name: str | None`
- `request: ResearchRequest | None`
- `pending_action: PendingAction | None`
- `created_at: float`
- `expiry_ts: float | None`
- `retry_count: int`
- `max_retries: int`
- `prompt_sent: bool`
- `reminder_sent: bool`
- `metadata: dict[str, Any]` (queue/stale behavior hints)

## Transition invariants
- At most one active token.
- `state == IDLE` iff active token is `None`.
- Confirmation prompts (token metadata `approval_flow=true`) are never stale-dropped while token exists.
- Decision parser is unified (`_parse_confirmation_decision`) for both kinds.
- If transcript callback is missing, `response.done` path still emits one reminder for guarded non-decision.

## Failure-mode mapping
- INC-321-001 (prompt dropped stale): token-aware stale-drop exemption for active confirmation prompts.
- INC-321-002 (missing transcript callback): `handle_response_done()` fallback checks guarded response + active token and injects reminder/cleanup.
- INC-321-003 (strict parser): research permission handler uses `_parse_confirmation_decision`.

## Migration stages
1. Stage 1 scaffolding: add enum/token dataclass and adapter methods while keeping old fields as compatibility mirrors.
2. Stage 2 dual-path: both research permission and governance confirmation create/own token.
3. Stage 3 cutover: decision intake, stale-drop, queue release, and reminder fallback become token-driven; old split checks reduced to compatibility shims.
4. Stage 4 hardening: structured FSM logs, invariants, and race-order tests.

## Rollback plan
- Revert to previous behavior by disabling token entry points and restoring legacy field checks:
  - Keep `_pending_action` and `_pending_research_request` as canonical sources.
  - Remove token-aware stale-drop exemption.
  - Disable `response.done` reminder fallback.
- Safe rollback granularity: revert commits stage-by-stage (4 -> 0) to isolate regressions.
