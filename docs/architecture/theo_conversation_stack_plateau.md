# Theo Conversation Stack Plateau (Good Enough Bar)

Owning layer: Runtime / Nervous System.

This is the current stabilization bar for Theo's conversation/runtime stack. It is a
plateau checklist, not a redesign roadmap.

## Stable enough for now when healthy runs consistently show

- One user request yields at most one user-visible terminal answer per canonical turn.
- No dead turns: queued or deferred follow-ups either release, stay intentionally held,
  or are explicitly suppressed with a reason.
- Transcript-final handoff and provisional/server-auto replacement paths converge to one
  coherent terminal deliverable without reopening duplicate-response regressions.
- Tool follow-up chains resolve parent ownership coherently enough that queue release
  drains and shutdown stays clean.
- Fresh-turn repeat gestures may bypass same-turn cooldowns when the repeat is a new,
  low-risk reversible action; same-turn duplicate execution protection still holds.

## Acceptable quirks that should not trigger more whack-a-mole work

- Benign mechanistic churn inside a turn when the final selected deliverable is still
  singular and coherent.
- Temporary provisional or deferred states that later reconcile cleanly.
- Healthy-path stimulus suppression (especially camera/startup/confirmation gating)
  when it is expected and non-actionable.

## Reopen conversation-stack work if any of these regress

- Duplicate user-visible answers, especially from transcript-final or replacement seams.
- Follow-ups stuck in queue/blocked state without a clear terminal release or suppression path.
- Parent/semantic ownership drift that changes which response becomes the final user-facing answer.
