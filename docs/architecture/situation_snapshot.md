# SituationSnapshot (read-only projection)

`SituationSnapshot` is a read-only projection over existing Theo runtime and service seams.

## Contract
- It **does not own authority** for lifecycle, arbitration, continuity, tool followup, motion, vision, battery, or startup decisions.
- It **must not mutate** source state.
- It composes best-effort summaries from already-authoritative owners.
- It does not make policy decisions and does not route model behavior.

## Intended use
- diagnostics/status inspection
- compact context injection inputs
- future harness-owned situated-core work (observability first)

Authoritative seams remain unchanged (interaction lifecycle, response-create arbitration/runtime, continuity, tool governance/followup, and service owners).

## Phase 2 observability exposure
- `SituationSnapshot.compact_summary()` provides a concise deterministic summary string for humans, logs, and future context plumbing.
- `RealtimeAPI.get_runtime_diagnostics()` now includes:
  - `situation_summary` (compact string)
  - `situation_snapshot` (serialized `to_dict()` payload)
- `RealtimeAPI.get_situation_diagnostics()` provides a dedicated situation bundle with both summary and full snapshot.

### Recursion guardrail
- Snapshot building accepts optional precomputed health:
  - `build_situation_snapshot(runtime, *, health: dict[str, Any] | None = None)`
- Diagnostics paths compute `health` once and pass it into snapshot construction, avoiding `get_session_health()` recursion.

### Side-effect + noise guardrails
- Snapshot reads from existing read-only surfaces only (cached battery telemetry and motion status introspection).
- No new runtime behavior is triggered by diagnostics reads (no response creation, model output, speech, tool calls, camera capture, or motion execution).
- No per-turn INFO log emission is added; snapshot visibility is via explicit diagnostics reads and compact summary surfaces.

### Field-fidelity notes (Phase 2.01)
- `interaction.state` resolves from `state_manager.state` first (authoritative lifecycle owner), then falls back to legacy runtime state fields.
- `run_id` is best-effort and read-only: `_run_id`/`run_id`/`session_id`, then `_current_run_id()` if available.
- `model.voice` resolves from session output voice configuration first, then legacy voice fields.
- `compact_summary()` treats `injection_ready_reason=response_in_progress` as `startup=busy` (transient), while preserving the underlying startup snapshot fields unchanged.
