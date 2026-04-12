# Seam Mapping Method (Theo)

Owning layer: Architecture documentation method (cross-layer).

Use this method when adding or revising seam docs so the map remains consistent.

## 1) Pick seam scope first

Define one behavioral contract boundary (not a folder):
- what decisions happen here,
- what this seam must not own.

## 2) Record ownership explicitly

For each seam document include:
- Purpose
- Owning files/modules
- Entry points
- Exit points/outcomes
- Upstream dependencies
- Downstream impacts
- Core invariants
- Failure signatures
- Known tricky handoffs
- Related seams
- First places to inspect

## 3) Classify seam health

Label seam status with one or more:
- Stable seam
- Active seam
- Fragile seam
- Historic seam with past regressions
- High-churn seam
- Likely seam-factory zone

## 3.5) Add lightweight confidence signaling

For seams with notable uncertainty (heuristics, sparse metadata, edge-ordering sensitivity), add one line near the top:

- `Confidence: high` — mapping is strongly code-confirmed with stable behavior.
- `Confidence: medium` — mapping is code-grounded but edge ordering or integration variability remains.
- `Confidence: low` — ownership/flow exists but evidence is partial or currently volatile.

Use this selectively; do not add confidence markers where they do not improve triage.

## 4) Show cross-seam dependencies directly

Every seam doc should include at least 2 explicit cross-seam handoffs (who hands authority to whom and when).

## 5) Keep claims evidence-ready

When you claim behavior, ensure:
- reason-coded logs or deterministic branch points exist,
- owning file functions can be pointed to quickly.

## 6) Update index + overview every time

When adding/editing seam docs:
- update `seam_inventory.md` row(s),
- update symptom shortcuts when new recurring incident class appears,
- keep `seam_map_overview.md` runtime-flow list current.

## 7) Prefer current reality over ideal state

If ownership is distributed/messy, document the observed ownership and mark uncertainty clearly.
