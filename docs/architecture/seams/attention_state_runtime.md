# Attention state runtime seam (transcript-final)

Owning layer: Layer 1 — Runtime / Nervous System.

This seam formalizes transcript-final attention admission into an explicit runtime state model in `ai/realtime_api.py`.

## State vocabulary

- `closed`
- `direct_address_active`
- `hold_active`
- `bypass_active`
- `blocked_recovery_cooldown`

Each state carries:
- `reason` (transition reason/source),
- `entered_at_monotonic`,
- optional `hold_until_monotonic`,
- optional `suppress_until_monotonic`,
- source `turn_id` and `input_event_key`.

## Transition rules (current behavior, now explicit)

- Direct configured-name address enters `direct_address_active` and opens/refreshes the bounded hold window.
- Follow-up during active hold enters `hold_active` without extending hold indefinitely.
- Confirmation/stop-abort exception admission enters `bypass_active`.
- Attention-gate block enters `blocked_recovery_cooldown`; runtime still re-arms quiet LISTENING as before.
- Hold expiry/cooldown expiry returns to `closed`.

## Contract boundary

- This seam is a source of truth for attention state and transition reasons.
- It does **not** broaden ambient admission or bypass policy.
- Response-create fail-closed behavior (`attention_gate_closed` backstop) remains authoritative downstream.
- Visual verify remains post-admission/post-bypass only.

## Out of scope

- Wake-word fuzzy matching, phonetics, diarization, speaker-ID, probabilistic scoring.
- Cross-runtime policy engines or authority migration into continuity/embodiment seams.
