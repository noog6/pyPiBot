# Theo Runtime Seam Map — Overview (Cartography v1)

Owning layer: Cross-layer cartography (Layers 1–5 with consultative seams).

## Purpose

This is the top-level map of Theo's runtime seam landscape. It documents **where authority lives today**, how major runtime flows connect, and where seam regressions most often originate.

This document is intentionally code-grounded and expandable. It is not a redesign proposal.

## How to use this map

1. Start with the symptom table below.
2. Jump to the seam-family document.
3. Confirm invariants and handoff contracts in code.
4. Use `docs/architecture/seam_inventory.md` for file-level ownership and adjacent seams.

## Runtime seam families (v1)

| Seam family | Primary layer | Current status | Why it matters |
| --- | --- | --- | --- |
| Response create arbitration | Layer 1 + 3 | **High-churn seam** | Decides `SEND`/`SCHEDULE`/`BLOCK`/`DROP` for response attempts. |
| Response lifecycle and identity | Layer 1 | **Fragile seam** | Binds canonical key identity across create/created/done and retry/redrive behavior. |
| Tool followup and followthrough | Layer 3 | **Likely seam-factory zone** | Converts tool completion into followup dispatch/release/suppress decisions. |
| Required-deliverable contracts | Layer 3 + 4 | **Historic seam with past regressions** | Prevents premature terminal settlement before owed user-facing output exists. |
| Terminal selection and settlement | Layer 3 + 4 | **Active seam** | Chooses whether `response.done` becomes the turn deliverable and closes continuity correctly. |
| Continuity and commitment state | Layer 4 | **Stable seam (narrow authority)** | Tracks open/closed obligations and followthrough chain state. |
| Startup and system context injection | Layer 1/2 boundary | **Fragile seam** | Controls first-turn grounding timing without stealing runtime authority. |
| Memory and preference recall paths | Layer 2 | **Active seam** | Shapes retrieval, recall takeover, and response grounding metadata. |
| Observability and tracing surfaces | Cross-cutting | **Stable seam** | Provides reason-coded diagnostics for arbitration and contract breaches. |

## Stability taxonomy (concise)

- **Stable seam**: contract shape is steady; changes are mostly additive and low-risk.
- **Active seam**: current behavior is intentional but still being extended regularly.
- **Fragile seam**: correctness is sensitive to event ordering, sparse metadata, or edge branch precedence.
- **Historic seam with regressions**: seam has a known history of repeat incidents.
- **High-churn seam**: frequent edits or policy toggles; regression risk remains elevated.
- **Likely seam-factory zone**: broad integration seam where authority handoffs blur and new seam bugs are most likely.

## Seam-factory hotspots (first triage note)

- `ai/realtime/response_terminal_handlers.py`: dense terminal/continuity/required-deliverable handoffs in one execution path.
- `ai/realtime/response_create_runtime.py`: create arbitration + queueing + metadata overlays converge here.
- `ai/realtime_api.py`: broad integration surface where many seam boundaries meet and can blur.

## Major runtime flows (cross-seam)

### 1) User input -> response create decision

1. Turn/key context enters `RealtimeAPI` + lifecycle coordinator.
2. `ai/interaction_lifecycle_policy.py` produces lifecycle candidate.
3. `ai/response_create_arbitration.py` merges lifecycle candidate with runtime overlays (lineage/terminal/same-turn owner).
4. `ai/realtime/response_create_runtime.py` executes result (send/schedule/drop/block).

### 2) `response.done` -> terminal settlement

1. Terminal event is ingested by `ai/realtime/response_terminal_handlers.py`.
2. `ai/terminal_deliverable_arbitration.py` decides whether terminal is deliverable.
3. `ai/semantic_owner_arbitration.py` may promote semantic owner to parent canonical key.
4. Continuity handoff runs through `ai/continuity.py` integration points in runtime.

### 3) Tool output -> followthrough completion

1. Tool metadata/trace context marks followup + step output policy.
2. `ai/tool_followup_arbitration.py` decides release/hold/suppress.
3. Required-deliverable guards in runtime prevent settlement unless substantive/tool-evidenced completion exists.
4. If output is empty/non-substantive, redrive/materialization paths may dispatch a retry create.

## Cross-seam dependency highlights (high-value handoffs)

- **Create identity -> done settlement:** canonical key chosen before/at create influences deliverable selection and final continuity close.
- **Tool followup arbitration -> required-deliverable completion:** suppressing or releasing followups directly affects whether the required report can be marked complete.
- **Semantic owner promotion -> continuity close target:** semantic owner reassignment can rebind close semantics to parent turn lineage.
- **Startup context injection -> early arbitration quality (not authority):** startup context influences grounding but should not override response-create authority seams.
- **Memory recall takeover -> server-auto provisional behavior:** preference recall paths can suppress/replace provisional server-auto output, affecting transcript-final upgrade/cancel-and-replace behavior.
- **Empty-response retry -> terminal materialization:** empty done detection in lifecycle can create redrive attempts that later appear as normal terminal settlement activity.

## Debugging by symptom (quick index)

| Symptom | Likely seam family | First file to inspect | Common adjacent seam |
| --- | --- | --- | --- |
| Tool result exists but no final user answer | Tool followup and required-deliverable contracts | `ai/realtime/response_terminal_handlers.py` | Terminal selection and continuity |
| `response.done` arrives but turn never settles | Terminal selection and continuity | `ai/terminal_deliverable_arbitration.py` and `ai/continuity.py` | Semantic owner arbitration |
| Duplicate or conflicting response ownership | Response lifecycle and identity | `ai/realtime/lifecycle_state.py` | Response create arbitration |
| Sparse metadata causes odd parent linkage | Response lifecycle and identity | `ai/realtime_api.py` identity/trace helpers | Semantic owner and tool followup |
| Required deliverable marked complete too early | Required-deliverable contracts | `ai/realtime/response_terminal_handlers.py` | Continuity required-step state |
| Provisional server_auto behaves inconsistently | Response create and startup/memory | `ai/interaction_lifecycle_policy.py` and `ai/realtime/preference_recall_runtime.py` | Transcript-final upgrade path |

## Map boundaries (v1)

Included deeply:
- Response lifecycle/identity
- Tool followup/followthrough
- Required-deliverable contracts
- Terminal selection/settlement
- Continuity commitment state

Included at secondary depth:
- Startup/system context injection
- Memory/preference recall takeover
- Observability/tracing surfaces

Not yet exhaustively mapped:
- Every low-level helper in `ai/realtime_api.py`
- Motion/hardware substrate internals
- Non-critical auxiliary tools and scripts
