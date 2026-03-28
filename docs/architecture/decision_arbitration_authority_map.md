# Decision Arbitration Authority Map (Implemented Runtime)

Purpose: document **what authority exists today in code** for Theo's decision arbitration, where each seam lives, and what each seam explicitly does **not** own.

This is a code-aligned map, not a redesign proposal. It describes distributed arbitration as implemented across seam-local modules.

## Scope and framing

- Arbitration is **distributed, not monolithic**.
- Runtime authority is exercised at multiple explicit seams.
- `decision_arbitration_adapter` is observational-only normalization and diagnostics.
- Governance/confirmation is a separate authority domain for tool risk/approval flow, not response-deliverable arbitration.

## Implemented seam inventory

### 1) Response-create lifecycle policy seam (authoritative, narrow)

- Module: `ai/interaction_lifecycle_policy.py`
- Entry points:
  - `decide_response_create(...)`
  - `decide_server_auto_created(...)`
  - `decide_watchdog_timeout(...)`

**Owns**
- Deterministic lifecycle-gate candidate selection and precedence for `response.create` (in-flight defer, audio busy defer, single-flight refusal, suppression, transcript-final wait, etc.).
- Deterministic server-auto-created pre-audio allow/defer/cancel decisions.

**Does not own**
- Final response-create contender arbitration across runtime overlays (same-turn owner, lineage, terminal-state overlays).
- Tool governance/confirmation policy.
- Terminal deliverable or semantic-owner selection.

---

### 2) Response-create arbitration seam (authoritative)

- Module: `ai/response_create_arbitration.py`
- Entry point: `decide_response_create_arbitration(...)`

**Owns**
- Final contender selection for response-create execution action (`SEND`/`SCHEDULE`/`BLOCK`/`DROP`) after combining lifecycle decision with runtime overlays.
- Priority between lineage block, terminal-state block, same-turn owner drop, and lifecycle candidate.

**Does not own**
- Transport send/queue mutation itself.
- Tool-followup release/suppress logic.
- Terminal deliverable selection.

---

### 3) Response-create runtime seam (authoritative execution/orchestration seam)

- Module: `ai/realtime/response_create_runtime.py`
- Core methods:
  - `prepare_response_create_snapshot(...)`
  - `_decide_response_create_action_with_lifecycle(...)`
  - `evaluate_response_create_attempt(...)`
  - `_finalize_response_create_execution_decision(...)`
  - `send_response_create(...)`
  - `schedule_pending_response_create(...)`
  - queue drain helper(s)

**Owns**
- Assembly of decision inputs from runtime state.
- Calling lifecycle policy + response-create arbitration seam.
- Enforcing execution-path postchecks (e.g., tool-followup final-deliverable suppression).
- Queue/send side effects.

**Explicit seam type**
- This is an **authoritative execution/orchestration seam**, not a pure policy/arbitration rule seam.

**Does not own**
- Terminal deliverable and semantic-owner arbitration rules (delegates to other seams).
- Governance risk policy (separate authority domain for tool risk/approval flow).

---

### 4) Terminal deliverable arbitration seam (authoritative)

- Module: `ai/terminal_deliverable_arbitration.py`
- Entry point: `arbitrate_terminal_deliverable_selection(...)`

**Owns**
- Whether a terminal response becomes the turn deliverable (`selected` true/false with reason).
- Non-deliverable and deferral conditions (cancelled, micro_ack, tool-followup precedence, provisional-empty, provisional-server-auto awaiting transcript-final, exact-phrase obligation, etc.).

**Does not own**
- Semantic owner reassignment.
- Tool-followup release/suppress decisions.
- Governance confirmations.

---

### 5) Semantic owner arbitration seam (authoritative, narrow)

- Module: `ai/semantic_owner_arbitration.py`
- Entry point: `decide_semantic_owner(...)`

**Owns**
- Canonical semantic owner assignment decision (`retain_execution` vs `reassign_parent`) after terminal selection context is known.

**Does not own**
- Whether terminal was selected in the first place.
- Queueing/scheduling new response work.
- Governance policy.

---

### 6) Tool-followup arbitration seam (authoritative, narrow)

- Module: `ai/tool_followup_arbitration.py`
- Entry point: `decide_tool_followup_arbitration(...)`

**Owns**
- Tool-followup release/hold/suppress decision and reason codes.
- Parent-coverage classification effect on gesture/status-only followups.

**Does not own**
- Terminal deliverable selection itself.
- Response-create lifecycle gating.
- Governance confirmations.

---

### 7) Tool governance + confirmation seams (separate authority domain for tool risk/approval flow)

- Modules:
  - `ai/governance.py`
  - `ai/realtime/confirmation_runtime.py`
  - shared envelope: `ai/governance_spine.py`

**Owns**
- Tool-call risk/budget/autonomy/confirmation decisioning.
- Idempotency key normalization.
- Confirmation token lifecycle, reminders, and timeout handling.

**Does not own**
- Response-create/terminal/semantic-owner/tool-followup arbitration outcomes.

---

### 8) Decision arbitration adapter seam (observational only)

- Module: `ai/decision_arbitration_adapter.py`

**Owns**
- Normalized observation payloads, trace merge, diagnostics/review summaries.

**Does not own**
- Any runtime authority. The module contract explicitly states it must not drive authority.

---

### 9) Consultative/context seams (consultative only)

- Examples: continuity (`ai/continuity.py` integration), quiet intent (`ai/quiet_intent.py`), attention continuity (`ai/attention_continuity.py`).

**Owns**
- Context enrichment, bookkeeping, and posture/diagnostic hints.

**Does not own**
- Arbitration winner selection, governance approvals, or lifecycle hard gates.

## Runtime composition (single-turn path)

This is the implemented composition order in runtime code.

1. **Transcript final arrives** and turn/key ownership is rebound to transcript-linked canonical key.
2. If a provisional `server_auto` response exists, upgrade logic evaluates **cancel-and-replace** eligibility (`should_cancel_and_replace(...)`); if allowed, runtime issues cancel then replacement create flow.
3. `response.create` attempts run through response-create runtime:
   - snapshot prep,
   - lifecycle policy decision,
   - response-create arbitration decision,
   - execution-path finalization,
   - send/schedule/drop.
4. On terminal events (`response.done` / `response.completed` admitted by shared ingress gate):
   - terminal deliverable arbitration runs,
   - semantic owner arbitration runs,
   - terminal selection + semantic owner reconciliations are applied,
   - blocked tool followups are released/held/suppressed as appropriate.
5. Response terminal close then checks confirmation-hold transition behavior before phase progression/mic recovery.

## Authoritative vs observational map

- **Authoritative seams**
  - `ai/interaction_lifecycle_policy.py`
  - `ai/response_create_arbitration.py`
  - `ai/realtime/response_create_runtime.py`
  - `ai/terminal_deliverable_arbitration.py`
  - `ai/semantic_owner_arbitration.py`
  - `ai/tool_followup_arbitration.py`
  - `ai/governance.py` + `ai/realtime/confirmation_runtime.py` (tool governance only)

- **Observational/diagnostic only**
  - `ai/decision_arbitration_adapter.py`

- **Consultative only**
  - continuity / quiet intent / attention continuity seams

## Boundary guardrails (code-aligned)

- Do not treat `decision_arbitration_adapter` outputs as authority signals.
- Do not move tool governance decisions into response arbitration seams.
- Do not move response arbitration decisions into governance/confirmation seams.
- Keep continuity/quiet-intent inputs consultative unless a seam explicitly promotes them to authority with tests.
- Keep arbitration changes seam-local with explicit reason codes.

## Most important correction captured by this doc

The critical correction is that Theo does **not** have one global arbiter: implemented authority is distributed across explicit seams, with `decision_arbitration_adapter` remaining observational-only.
