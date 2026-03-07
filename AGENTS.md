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
- Facts are law, not lore: if you claim behavior, cite the run log lines and the code locations.
- Prefer deterministic logic over model cleverness for memory, tool gating, and fallbacks.
- Do not introduce new persistence formats outside the storage controller.
- Favor backward-compatible changes first; only introduce breaking behavior with explicit rationale and migration guidance.

## Cognitive stack roadmap (mandatory)
- Read `docs/architecture/theo_cognitive_stack.md` before major architecture work.
- Current plateau: runtime/nervous-system behavior is meaningfully stabilized; perception/memory is partial; decision arbitration and higher cognition are the active frontier.
- Classify each task by owning layer before editing code.
- Do not conflate arbitration, idempotency/single-flight, scheduling/queueing, governance/policy gating, and perception/context enrichment.
- Preserve deterministic runtime behavior while evolving higher-order cognition in upper layers.

## Verification baseline (minimum commands)
- Run a targeted pytest pass for the area you touched (or full suite if unsure).
- If you changed memory retrieval, governance, or logging semantics: add/extend tests that prove the new behavior.

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

## Do-not-do list
- No ad-hoc persistence under var/ (use storage controller).
- No tight-loop INFO logging.
- No infinite background loops without stop_event + clean shutdown hooks.
- Do not broaden tool permissions as a shortcut.
