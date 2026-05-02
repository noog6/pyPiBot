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
