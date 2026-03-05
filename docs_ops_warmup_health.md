# Ops warmup health design (run_id=483)

## Warmup entry and exit criteria

- Warmup starts when `OpsOrchestrator.start_loop()` begins a new loop.
- Warmup exits when either:
  - realtime reports `connected=1` and `ready=1`, and all required startup probes are healthy (except tolerated startup transients), or
  - a bounded grace timeout expires (`health.warmup.grace_period_s`, default `20s`).

## Startup transients tolerated during warmup

The warmup state intentionally tolerates:

- `realtime=degraded` (for pre-connection/offline startup), and
- `audio=degraded` (for partial/late device enumeration).

During warmup, snapshots are emitted as `status=warmup` with a human-friendly summary while retaining full per-probe details in snapshot metadata.

## Post-warmup semantics

After warmup exits, health returns to existing status derivation:

- `ok` when all probes are healthy,
- `degraded` when non-fatal issues persist,
- `failing` when critical issues persist.

A single transition log is emitted:

`[Ops] Warmup transition: warmup -> active reason=<criteria_met|timeout> elapsed_s=<...>`
