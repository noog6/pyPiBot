# Theo Cognitive Stack (Code-Aligned)

Purpose: map Theo's **current** cognitive/behavioral stack to real code seams so future work lands in the right layer.

## You are here (March 2026)

Theo is no longer at a "continuity missing" stage.

- Runtime/lifecycle determinism is the stabilized base.
- Response-create, terminal-deliverable, semantic-owner, and tool-followup arbitration seams are now implemented.
- Continuity exists as a deterministic bookkeeping layer (not a planner).
- Perception/memory is partial: useful retrieval and intent shaping are present, but still heuristic-heavy.
- Higher-order cognition (global planner, long-horizon policy, cost/energy optimizer) is still backlog.

## Current stack reconstruction

### Layer 0 — Physical + motion substrate (**implemented / stable enough**)

**Owns:** hardware I/O, gesture execution primitives.

**Anchors:** `motion/*`, `services/tool_runtime.py`, gesture tools in `ai/tools.py`.

**Notes:** Motion/gesture behavior is exposed through deterministic tool calls; this is an execution substrate, not cognition.

---

### Layer 1 — Runtime / Nervous System (**implemented / plateau**)

**Owns:** realtime transport, response lifecycle, queue/single-flight behavior, cancellation/replace, shutdown/task hygiene.

**Anchors:** `ai/realtime_api.py`, `ai/realtime/response_create_runtime.py`, `ai/realtime/lifecycle_state.py`, `ai/realtime/shutdown.py`.

**Now true in code:**
- `response.create` attempts are normalized into explicit outcome actions (`SEND`, `SCHEDULE`, `BLOCK`, `DROP`).
- Single-flight and canonical-key guards are enforced before create/send.
- Server-auto/provisional paths are handled as first-class lifecycle cases.

**Boundary:** do not add "what should Theo do" policy here unless it is lifecycle safety/determinism.

---

### Layer 2 — Perception + memory interface (**implemented, partial quality frontier**)

**Owns:** memory-intent classification, retrieval query shaping, per-turn memory brief construction, preference/topic recall hooks.

**Anchors:** `ai/realtime/memory_runtime.py`, `services/memory_manager.py`, `ai/realtime/preference_recall_runtime.py`, memory-facing helpers in `ai/tools.py`.

**Now true in code:**
- Memory intent is explicitly typed (`preference_recall`, `topic_recall`, `general_memory`).
- Retrieval includes lexical/semantic hybrid diagnostics and readiness/fallback paths.
- Preference recall can suppress server-auto responses and force same-turn context injection.

**Partial/in-progress:** quality and signal discipline are still heuristic-driven (pattern sets, thresholds, readiness canary behavior).

---

### Layer 3 — Turn arbitration seams (**implemented, still expanding**)

**Owns:** per-turn decision surfaces that decide which candidate path wins.

**Anchors:**
- `ai/interaction_lifecycle_policy.py` (response-create + server-auto-created gating)
- `ai/realtime/response_create_runtime.py` (candidate construction + execution decision)
- `ai/terminal_deliverable_arbitration.py` (terminal deliverable selection)
- `ai/semantic_owner_arbitration.py` (semantic owner reassignment)
- `ai/tool_followup_arbitration.py` (release/hold/suppress followups)
- `ai/decision_arbitration_adapter.py` (normalized observational diagnostics)

**Now true in code:**
- Arbitration is not a single monolith; it is seam-local and deterministic by policy domain.
- Terminal reconciliation handles provisional/server-auto and transcript-final upgrade behavior.
- Tool follow-up release is policy-driven (`RELEASE`, `HOLD`, `SUPPRESS`) with parent coverage semantics.
- Semantic owner can be reassigned from tool-output lineage to parent canonical turn when conditions match.
- `decision_arbitration_adapter` remains observational-only normalization/diagnostics and is not a runtime authority seam.

**Partial/in-progress:** this is still mostly turn-local arbitration; cross-turn/global goal arbitration is not implemented.

---

### Layer 4 — Continuity + stance bookkeeping (**implemented, intentionally narrow**)

**Owns:** short-horizon continuity ledger state and classification for diagnostics/context.

**Anchors:** `ai/continuity.py`, use-sites in `ai/realtime_api.py`, `ai/realtime/response_terminal_handlers.py`.

**Reference:** `docs/architecture/continuity_presence.md` is the current-state contract for continuity ownership, data model, runtime integration, and explicit non-authority boundaries.

**Now true in code:**
- Continuity has explicit item kinds/status/priority and stance labels.
- Compound-step requests and turn-settlement classification are modeled.
- Module contract explicitly states bookkeeping-only: not an authority for arbitration/scheduling.

**Boundary:** continuity must not silently become a hidden planner or execution controller.

---

### Layer 5 — Governance spine (**implemented, partial formalization**)

**Owns:** tool risk tiers, confirmation requirements, idempotency key normalization, cooldown/budget/autonomy-window policy.

**Anchors:** `ai/governance.py`, shared decision envelope in `ai/governance_spine.py`, confirmation runtime in `ai/realtime/confirmation_runtime.py`.

**Now true in code:**
- Governance decisions are normalized (`approved`, `needs_confirmation`, `denied` semantics).
- Tool argument normalization/idempotency keying is explicit.
- Confirmation reminders/guard rails are runtime-integrated.

**Partial/in-progress:** cross-subsystem governance priority remains seam-local metadata (per `governance_spine`) unless a dedicated global arbitration seam is added.

---

### Layer 6 — Conversational pacing + embodiment cues (**implemented, bounded behavior layer**)

**Owns:** micro-ack pacing, deterministic embodiment cue policy, and consultative quiet-intent posture-bias diagnostics.

**Anchors:** `ai/micro_ack_manager.py`, `ai/embodiment_policy.py`, `ai/attention_continuity.py`, `ai/quiet_intent.py`, integration in `ai/realtime_api.py`.

**Now true in code:**
- Micro-acks are rate-limited, deduped, category/channel gated, and suppressible when followup/near-ready conditions apply.
- Embodiment policy maps interaction state + attention snapshot to governed cue actions.
- Attention continuity provides a short ASR churn hold window; it is not a second runtime machine.
- Quiet Intent emits deterministic consultative posture biases plus a diagnostic snapshot, refreshed on interaction-state transitions; runtime currently logs/stores this output but does not route it into response-create arbitration, governance, tool followup, or terminal selection authority seams.

**Boundary:** these cues are expression/pacing aids and consultative diagnostics, not substitutes for terminal deliverables or policy authority.

---

### Layers not yet implemented as full architecture

- Global multi-turn planner / explicit long-horizon goals.
- Cost/energy-aware action optimizer spanning model/tool/hardware budgets.
- Persona/character control as a first-class module with drift metrics.
- Multi-agent ecosystem coordination.

## Current status map

| Area | Status | What to assume |
| --- | --- | --- |
| Runtime lifecycle determinism | Stable plateau | Preserve semantics; avoid opportunistic rewrites. |
| Response/terminal/followup arbitration | Implemented, expanding | Extend seam-locally with explicit reason codes. |
| Perception/memory | Partial | Improve relevance/discipline before adding broad new policy. |
| Continuity ledger | Implemented narrow | Use as context/diagnostic signal, not control authority. |
| Quiet Intent posture biasing | Implemented consultative | Treat as posture/diagnostic signal; not an authority seam. |
| Governance | Implemented partial | Keep fail-closed and confirmation semantics explicit. |
| Higher cognition (planner/cost/persona/ecosystem) | Aspirational/backlog | Do not imply these are already present. |

## Anti-patterns and boundaries

- Do not collapse arbitration, governance, and runtime queueing into one module.
- Do not treat continuity settlement as execution authority.
- Do not bypass terminal-deliverable arbitration with "helpful" direct response writes.
- Do not let micro-acks or gesture-only tool outputs count as final user deliverables.
- Do not hide policy changes in transport/lifecycle plumbing without explicit arbitration/governance seams.

## Guidance for future Codex passes

1. Classify change by owning layer first.
2. If touching response paths, state which seam is authoritative (`response_create`, terminal selection, semantic owner, tool-followup, governance).
3. Preserve deterministic reason codes and observability payloads when evolving behavior.
4. Mark docs and commit/PR notes with: `Owning layer: <layer name>`.
5. When uncertain, treat code + logs as source of truth and downgrade stale roadmap language.
