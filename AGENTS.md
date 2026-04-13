# AGENTS.md

Purpose: This file gives AI coding agents repo-specific instructions so they can set up quickly, follow team conventions, and avoid low-value mistakes.

## Environment setup
Run this command as part of setup for each environment:
`python -m pip install pyyaml -q`

## Two high-impact defaults (recommended)
1. **Always run a fast verification pass before finishing work**
   - Minimum: run targeted tests or lint for changed files.
   - Include the exact commands and results in the final summary.

2. **Keep changes small and explain intent, not just edits**
   - Prefer focused, single-purpose commits.
   - In the final report, include: what changed, why, and any risks/follow-ups.

## Operating principles (mandatory)
- Facts are law, not lore: logs, tests, and concrete code paths outrank intuition.
- Negative confirmation matters: explicitly say what was checked and appears to be working.
- Do not infer broad architectural failure from one local contradiction unless multiple seams show the same break.
- Prefer deterministic logic over model cleverness for memory, tool gating, and fallbacks.
- Do not introduce new persistence formats outside the storage controller.
- Favor backward-compatible changes first; only introduce breaking behavior with explicit rationale and migration guidance.

## Cognitive stack roadmap (mandatory)
- Read `docs/architecture/theo_cognitive_stack.md` before major architecture work.
- Treat the following as “major architecture work”:
  - Changes to lifecycle gating/response create semantics.
  - Changes to memory retrieval/recall flow.
  - Changes to tool governance/confirmation behavior.
  - Any new queueing/arbitration/policy module.
- When any major-architecture trigger applies, include a one-line `Owning layer:` declaration in the commit and PR summary.
- Current plateau: runtime/nervous-system behavior is meaningfully stabilized; perception/memory is partial; decision arbitration and higher cognition are the active frontier.
- Classify each task by owning layer before editing code.
- Do not conflate arbitration, idempotency/single-flight, scheduling/queueing, governance/policy gating, and perception/context enrichment.
- Preserve deterministic runtime behavior while evolving higher-order cognition in upper layers.

## Seam triage defaults (mandatory)
- For runtime bugs, identify the **single best owning seam** before proposing changes.
- Distinguish explicitly:
  - **Actually broken**
  - **Merely suspicious**
  - **Working / leave alone**
- Prefer the **smallest correct fix**, but verify the **largest affected contract**.
- Do not redesign adjacent seams unless the evidence shows the owning seam cannot contain the fix.
- When proposing a fix, include:
  - exact divergence point
  - upstream impact review
  - downstream impact review
  - focused regression tests for the owning seam
  - documentation impact

## Architecture map usage
- For runtime or lifecycle bugs, consult:
  - `docs/architecture/seam_map_overview.md`
  - `docs/architecture/seam_inventory.md`
- If the issue maps cleanly to an existing seam family, use that seam language in the summary.
- If the issue does not map cleanly, say so explicitly instead of forcing it into the wrong seam.
- If a code change materially improves or invalidates a seam-map routing shortcut, update the shortcut in the same change when practical.

## Verification baseline (minimum commands)
- Run a targeted pytest pass for the area you touched (or full suite if unsure).
- If you changed memory retrieval, governance, or logging semantics: add/extend tests that prove the new behavior.
- If verification is partial, blocked, collection-fails, or no tests match, state that explicitly in the final summary.

## Documentation touchpoint (mandatory)
- For any non-trivial code change, explicitly decide one of:
  - `Docs updated`
  - `Docs intentionally not updated`
- If docs were not updated, include a one-line reason in the final summary.
- At minimum, consider whether changes affect:
  - `docs/architecture/`
  - `docs/reports/`
  - `config/default.yaml`
  - repro steps or runbook-style notes

## Documentation requirements for seam changes
- If a fix changes seam behavior, seam boundaries, required metadata, settlement rules, or first-inspect guidance, update the relevant file(s) under `docs/architecture/`.
- If a fix changes incident understanding, add or update a short report under `docs/reports/` when useful.
- If a bug hunt revealed a new recurring symptom or routing shortcut, update the seam inventory or overview.

## Doc drift rule
- If code and seam docs disagree during investigation, call it out explicitly.
- Do not silently follow the docs over the code; treat the code and logs as source of truth, then update docs if needed.

## Logging conventions
- INFO: operator-actionable events and state transitions.
- DEBUG: high-volume trace (candidate scoring, verbose audits).
- Use structured logs (key=value), avoid large payload dumps unless behind DEBUG.

## Tool governance + confirmations
- Any tool change must document: risk tier, idempotency key strategy, and whether confirmation is required.
- Avoid repeated/looping confirmations; escalate once, then fail closed with a clear reason.

## Run log / repro discipline
- Include a minimal repro command and the expected log “anchor strings”.
- If a run_id is present (e.g., run-386), use it in incident/report identifiers.

## Config + dependency hygiene
- New knobs must have safe defaults and be documented in config/default.yaml.
- Avoid heavy dependencies; prefer stdlib unless there is a strong justification.

## Regression philosophy
- Prefer one focused reproducer for the dominant seam over many broad tests with weak relevance.
- Add or update tests that prove:
  - the bug before the fix
  - the seam contract after the fix
  - one adjacent non-regression when risk is non-local
- If a command returns “no tests matched” or fails during collection, report that plainly; do not overclaim verification.

## Do-not-do list
- No ad-hoc persistence under var/ (use storage controller).
- No tight-loop INFO logging.
- No infinite background loops without stop_event + clean shutdown hooks.
- Do not broaden tool permissions as a shortcut.

## Model-vs-runtime boundary
- Do not patch Theo runtime to compensate for model-native wording quirks unless explicitly requested.
- If a defect appears to be model output quality rather than runtime contract failure, label it as such and avoid seam churn.

## Final summary format (mandatory)
Include:
- Owning seam
- What changed
- Why it changed
- Verification performed
- Upstream impact
- Downstream impact
- Documentation impact
- Remaining risks / follow-ups
