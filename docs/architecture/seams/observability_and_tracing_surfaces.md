# Seam Family: Observability and Tracing Surfaces

Owning layer: Cross-cutting observational layer.

Status: **Stable seam** (diagnostic authority only).

## A) Purpose

Expose reason-coded, seam-local diagnostics that make arbitration/settlement decisions auditable without granting decision authority to observational adapters.

## B) Owning modules/files

- `ai/decision_arbitration_adapter.py` (normalized observation payloads + summaries).
- `ai/contract_breach.py` (breach artifact detection and recommended-action hints).
- Structured runtime logs in `ai/realtime/response_create_runtime.py` and `ai/realtime/response_terminal_handlers.py`.

## C) Entry points

- Observation builders for create/terminal/semantic-owner decisions.
- Contract breach snapshot detection calls in create/done paths.

## D) Exit points / outcomes

- Turn-level arbitration traces and review summaries.
- Contract breach artifacts with fingerprints and recommended action kind.
- High-signal log anchors used in incident triage.
- End-of-run lifecycle report banner emitted from `main.py` shutdown finalization with machine-friendly status and battery/power summary fields.

## E) Upstream dependencies

- Accurate seam-local decision payloads.
- Canonical key / turn id / response id availability.
- Consistent reason-code semantics.

## F) Downstream impacts

- Faster incident triage and patch review confidence.
- Verification evidence for seam contract claims.
- Regression detection for known seam-factory zones.

## G) Core invariants

- Observability modules are non-authoritative.
- Breach detection should be investigative, not silent auto-repair.
- Reason codes should stay deterministic and comparable across runs.

## H) Failure signatures

- Missing or contradictory reason codes for same turn.
- Breach artifacts absent where expected dual-evidence conditions are present.
- Adapter output used as policy authority (architecture violation).

## I) Known tricky handoffs

- Merging multi-seam observations for one turn without losing precedence context.
- Fingerprint stability when optional fields are sparse.

## J) Related seams

- [Response create arbitration](response_create_arbitration.md)
- [Terminal selection and settlement](terminal_selection_and_settlement.md)
- [Continuity and commitment state](continuity_and_commitment_state.md)

## K) First places to inspect

1. `decision_arbitration_adapter` build/merge/summarize calls.
2. `detect_contract_breach(...)` call sites and emitted evidence fields.
3. Logs: `turn_arbitration_*`, `contract_breach_detected` style anchors.
