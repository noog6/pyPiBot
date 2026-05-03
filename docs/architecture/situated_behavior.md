# Situated behavior posture policy (Minimal Phase 3A)

Minimal Phase 3A formalizes existing state-driven posture behavior into a first-class policy contract.

## Scope
- Policy implementation: `ai/embodiment_policy.py`
- Primary contract: `SituatedPostureDecision`
- Integration seam: `EmbodimentPolicy.decide_state_cue(...)`

## What the policy controls
- Outward posture/gesture-cue eligibility only (e.g., listening, thinking, speaking, followthrough suppression).
- Deterministic, read-only decisions from existing runtime facts.
- Stable reason-code outputs for observability/tests.

## What the policy does not control
- Lifecycle transitions
- Response-create arbitration
- Terminal selection/settlement
- Tool-followup execution/arbitration
- Continuity authority
- Autonomy/model behavior

These seams remain authoritative in their existing owners.

## Followthrough safety boundary
- During followthrough/turn-contract gesture blocks, nonessential runtime state cues are suppressed.
- Task-required gesture/tool behavior remains owned by existing followthrough/governance/tool seams.

## Compatibility goal
Phase 3A is contract extraction/clarification over existing runtime behavior, not new gesture autonomy.


## Phase 3B: lifecycle startup/shutdown cue migration

Phase 3B extends situated behavior to lifecycle-owned startup/shutdown situations without changing lifecycle authority.

- Runtime/lifecycle remains authoritative for **when** startup and shutdown happen.
- Situated behavior/policy now owns only the embodied cue choice via `EmbodimentPolicy.decide_lifecycle_posture(...)`.
- Startup maps deterministically to `gesture_startup_presence` with reason code `lifecycle_startup_presence`.
- Shutdown maps deterministically to `gesture_shutdown_rest` with reason code `lifecycle_shutdown_settle`.
- Unmapped lifecycle events produce no cue with `lifecycle_event_no_cue`.
- Existing canned startup/shutdown frame selection is migrated behind named gesture definitions, so runtime paths dispatch named lifecycle gestures rather than constructing ad-hoc frames inline. For these lifecycle gestures, authored targets preserve legacy absolute pan/tilt startup and shutdown poses.

Future situated lifecycle policies should follow the same seam contract: explicit input event, deterministic output decision, stable reason codes, no lifecycle authority migration, and no background autonomy loops.

## Phase 3C: situational speech/direct-address posture cues

Phase 3C adds first-class situational cue policy for conversational turn-boundary events, while preserving runtime authority boundaries.

- Runtime remains authoritative for detecting `speech_stopped` and transcript-final direct-address admission.
- Situated behavior now owns cue selection via `EmbodimentPolicy.decide_situational_cue(...)`.
- `speech_stopped_ack` maps deterministically to `gesture_nod`.
- `direct_address_ack` maps deterministically to `gesture_attention_snap`.
- Suppression reason-codes are explicit and stable:
  - `situational_cue_suppressed_response_in_flight`
  - `situational_cue_suppressed_motion_busy`
  - `situational_cue_suppressed_turn_contract`
  - `situational_cue_suppressed_followthrough`
  - `situational_event_no_cue`
- Execution still routes through existing runtime motion/gating seams (`_enqueue_gesture_cue` + cooldown/motion guards); situational policy does not bypass those guards.
- Camera/image “blink” cues are intentionally deferred in this phase.

Future situational extensions must follow this rule: explicit event → deterministic policy decision (stable reason codes) → existing execution/gating seam.
