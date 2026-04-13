# Theo Seam Inventory (v1)

Owning layer: Cross-layer architecture index.

This catalog is the index for seam families and first-pass contracts.

| Seam name | Owning files/modules | Upstream inputs | Downstream outputs | Primary invariants | Common failure symptoms | Doc pointer |
| --- | --- | --- | --- | --- | --- | --- |
| Response create arbitration | `ai/interaction_lifecycle_policy.py`, `ai/response_create_arbitration.py`, `ai/realtime/response_create_runtime.py` | Origin, in-flight state, canonical slot, suppression/transcript-final flags | `SEND`/`SCHEDULE`/`BLOCK`/`DROP` execution decision | Single deterministic winner per attempt; precedence is explicit | Defers forever, same-turn ownership drops tool followup, terminal block surprises | `docs/architecture/seams/response_create_arbitration.md` |
| Response lifecycle and identity | `ai/realtime/lifecycle_state.py`, `ai/realtime/response_lifecycle.py`, `ai/realtime_api.py` | run/turn/input keys, provider metadata, response events | Canonical state bindings, retry lineage, done reconciliation | Canonical lineage stability; idempotent retries | Wrong turn binding, duplicate settlement, empty retry loops | `docs/architecture/seams/response_lifecycle_and_identity.md` |
| Tool followup and followthrough | `ai/tool_followup_arbitration.py`, `ai/realtime_api.py`, `ai/realtime/response_terminal_handlers.py` | Tool result quality, parent coverage, chain remaining state | Release/hold/suppress followup decisions, followup dispatch/suppression | Suppress only with qualified parent coverage; holds are transitional | Tool result but no answer, duplicate followups, stalled hold state | `docs/architecture/seams/tool_followup_and_followthrough.md` |
| Required-deliverable contracts | `ai/realtime/response_terminal_handlers.py`, `ai/realtime_api.py`, `ai/continuity.py` | Followthrough policy metadata, continuity required-step state, tool evidence | Complete/defer required deliverable, optional materialization redrive | Do not settle without substantive + required tool-backed evidence | Report marked complete early, repeated redrive, pending drift | `docs/architecture/seams/required_deliverable_contracts.md` |
| Terminal selection and settlement | `ai/terminal_deliverable_arbitration.py`, `ai/realtime/response_terminal_handlers.py`, `ai/realtime_api.py`, `ai/semantic_owner_arbitration.py` | Done event evidence, origin/provisional state, followthrough open/closed | Selection write, semantic owner resolution, continuity close flags | `selected=false` can be intentional; semantic promotion is narrow | Never-settles or early-settles turns; wrong semantic owner | `docs/architecture/seams/terminal_selection_and_settlement.md` |
| Continuity and commitment state | `ai/continuity.py` (+ runtime integrations) | Transcript/tool/response events, semantic owner handoff | Continuity brief, settlement class, required-deliverable pending state | Bookkeeping-only authority; bounded deterministic state machine | close_commitment drift, stale pending state, stance mismatch | `docs/architecture/seams/continuity_and_commitment_state.md` |
| Startup and system context injection | `ai/realtime/injections.py`, `ai/realtime/injection_bus.py`, `ai/realtime_api.py` | Startup gate state, first utterance observed, timeout | Deferred context queue flush, startup context visibility | Flush once; avoid hidden arbitration authority | Missing startup context, double injection, timing race | `docs/architecture/seams/startup_and_system_context.md` |
| Memory and preference recall paths | `ai/realtime/memory_runtime.py`, `ai/realtime/preference_recall_runtime.py`, `services/memory_manager.py` | User text cues, memory backend readiness, provisional response state | Memory brief injection, recall hints, optional takeover/suppression flags | Explicit intent subtypes; graceful retrieval fallback | Recalled info ignored, weak memory grounding, failed takeover | `docs/architecture/seams/memory_and_preference_recall_paths.md` |
| Observability and tracing surfaces | `ai/decision_arbitration_adapter.py`, `ai/contract_breach.py` | Seam decision payloads, IDs/reason codes | Turn traces, diagnostics summaries, breach artifacts | Observational-only, deterministic reason metadata | Missing diagnostics, non-deterministic reason trails | `docs/architecture/seams/observability_and_tracing_surfaces.md` |

## Symptom-to-seam shortcuts

| Symptom | Likely seam | First inspect file | Adjacent seam |
| --- | --- | --- | --- |
| Runtime aborts before first turn (e.g., `ModuleNotFoundError` during realtime bootstrap import) | Startup and system context injection | `ai/realtime_api.py` top-level imports and `ai/__init__.py` runtime surface export | Observability and tracing surfaces (absence of seam logs is itself a signal) |
| Required-deliverable report never becomes terminal | Required-deliverable contracts | `ai/realtime/response_terminal_handlers.py` | Tool followup and continuity |
| Settlement shows `settled` while required-deliverable redrive is still pending | Continuity and commitment state | `ai/continuity.py::classify_turn_settlement(...)` (`compound_request.final_followup_pending`) | Required-deliverable contracts |
| Tool result exists but no final answer appears | Tool followup/followthrough | `ai/tool_followup_arbitration.py`, `ai/realtime/response_create_runtime.py` | Terminal selection |
| `close_commitment` drifts from actual completion | Continuity and terminal settlement | `ai/continuity.py` | Semantic owner and required-deliverable |
| Sparse metadata corrupts lineage | Response lifecycle and identity | `ai/realtime/lifecycle_state.py` | Semantic owner |
| Provisional server_auto stuck/odd | Response create arbitration | `ai/interaction_lifecycle_policy.py` | Memory preference takeover |


## Seam-factory hotspots

- `ai/realtime/response_terminal_handlers.py`: high-risk due to dense terminal, deliverable, and continuity authority handoffs in one handler path.
- `ai/realtime/response_create_runtime.py`: high-risk due to lifecycle policy, arbitration overlays, queueing, and metadata guardrails converging in one runtime seam.
- `ai/realtime_api.py`: high-risk integration surface where many seam boundaries meet, increasing boundary-blur regressions.
