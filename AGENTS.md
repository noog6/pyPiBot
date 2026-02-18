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
