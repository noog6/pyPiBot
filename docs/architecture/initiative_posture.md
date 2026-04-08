# Theo Initiative Posture (Current-State, Code-Aligned)

Purpose: define the bounded initiative-posture seam as a consultative, reason-coded recommendation layer that complements realtime turn generation without creating a second hidden planner.

## Owning layer

**Owning layer:** Layer 6.5 (bounded initiative posture recommendation, consultative).

This seam is adjacent to Layer 6 posture cues (quiet intent, attention continuity) and above Layer 4 continuity bookkeeping inputs. It remains consultative and does not own arbitration authority.

## What this seam does

- Produces a single bounded recommendation:
  - `answer_directly`
  - `clarify_first`
  - `continue_followthrough`
  - `await_user`
  - `observe_only`
- Emits `confidence`, `confidence_band`, and deterministic `reason_codes`.
- Refreshes on interaction-state transitions (low-rate runtime seam, not continuous prompt chatter).
- Attaches compact consultative metadata to `response.create` events for observability.

## Inputs (initial scope)

- Interaction state.
- Conversation-active short-horizon signal.
- Continuity stance and followthrough-open signal.
- Recent-utterance lexical flags (`direct_question`, `ambiguous_request`).
- Busy-turn/response-in-flight suppression signal.
- Confirmation-pending signal.
- Latest quiet-intent mode (consultative context only).

## Explicit non-authority boundaries

Initiative posture does **not**:

- execute tools,
- approve or deny governance decisions,
- select terminal deliverables,
- reassign semantic owner,
- release/suppress tool-followups,
- override response-create arbitration winners,
- create long-horizon plans or scheduling loops.

Any future authority promotion must be explicit, seam-local, and regression-tested.

## Runtime integration (first pass)

- Runtime computes and logs `initiative_posture_decision` with dedupe fingerprinting.
- Response-create runtime attaches consultative metadata keys:
  - `initiative_posture`
  - `initiative_confidence_band`
  - `initiative_reason_codes`

This is metadata-only influence; no arbitration logic reads these values yet.

## Relationship to neighboring seams

- **Quiet Intent:** remains posture-bias consultative. Initiative posture may read quiet-intent mode, but does not expand quiet intent authority.
- **Continuity:** remains bookkeeping and followthrough context. Initiative posture reads stance/settlement; continuity still does not arbitrate execution.
- **Curiosity:** remains low-rate suggestion only.
- **Distributed arbitration seams:** remain authoritative decision points for runtime behavior.
