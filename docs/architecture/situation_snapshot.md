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
