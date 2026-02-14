# Operations Monitoring & Orchestration

This document describes the operational monitoring layer that runs alongside the
realtime agent. It focuses on the operational models, orchestrator loop, health
probes, session counters, alert routing, and rolling budget windows.

## Overview

The operations layer is centered on the `OpsOrchestrator`, a singleton loop that
runs in the background once the realtime agent is initialized. It coordinates
health probes, emits health snapshots and heartbeats, applies budget windows,
and routes alerts onto the shared event bus for downstream injection.

## Operational Models (ops_models)

Operational state is captured by a set of lightweight data models:

- **HealthStatus**: Enum describing runtime status (`ok`, `degraded`, `failing`).
- **HealthSnapshot**: A timestamped summary of the current health state plus
  structured details for each probe.
- **BudgetCounters**: Mutable counters for ticks, heartbeats, and errors.
- **ModeState**: Operational modes (`startup`, `idle`, `active`, `shutdown`).
- **OpsEvent**: Logged events emitted by the orchestrator (e.g., heartbeat,
  health snapshot).
- **DebouncedState**: Tracks per-probe debounced health status to smooth noisy
  probe results.

These models give the orchestrator a consistent shape for health reporting,
budget metrics, and event emission.【F:core/ops_models.py†L1-L65】

## Ops Orchestrator Loop

The orchestrator runs a periodic tick loop that:

1. Executes health probes (audio, battery, motion, realtime session, and
   optional network checks).
2. Debounces probe results and derives an overall health classification.
3. Updates tick counters and emits a `HealthSnapshot` when the health state
   changes or when no snapshot exists yet.
4. Emits a heartbeat event on a longer cadence to track session liveness.
5. Applies budget windows for sensor reads, logging, and micro-presence motion
   gestures.

The loop maintains session counters (`ticks`, `heartbeats`, and `errors`) in the
`BudgetCounters` struct and exposes them via `get_counters()` for diagnostics or
UI visibility.【F:services/ops_orchestrator.py†L34-L231】

## Health Probes & Debounce Logic

Health probes return `HealthProbeResult` records with a name, status, summary,
and structured details. Current probes include:

- **Audio**: Checks realtime API audio input/output availability and records
  device metadata.
- **Battery**: Reads the battery monitor state and reports battery percent and
  severity.
- **Motion**: Confirms the motion control loop is alive.
- **Realtime Session**: Reports connection readiness and session counters such
  as failures, reconnects, and connection attempts.
- **Network (optional)**: Attempts a TCP connection to a configurable host.

Probe results are debounced for a configurable window so that transient probe
failures do not immediately flip the overall health state. The orchestrator
tracks both stable and pending statuses and only commits changes once the
pending status exceeds the debounce interval.【F:services/health_probes.py†L1-L199】【F:services/ops_orchestrator.py†L116-L345】

## Session Counters

Two levels of counters are tracked:

- **Orchestrator counters** (`ticks`, `heartbeats`, `errors`) in
  `BudgetCounters`, incremented by the ops loop and emitted in heartbeat events.
- **Realtime session counters** from the realtime API health payload, including
  `failures`, `reconnects`, `connections`, and `connection_attempts` as captured
  by the realtime session health probe.

These counters provide a running view of runtime stability and connectivity.
【F:core/ops_models.py†L24-L32】【F:services/ops_orchestrator.py†L122-L200】【F:services/health_probes.py†L149-L199】

## Alerts & Alert Policy

The ops layer emits alerts for two main cases:

- **Health alerts**: When health degrades or fails, the orchestrator publishes
  an alert with `warning` or `critical` severity and a short summary.
- **Budget alerts**: When a rolling budget is exhausted (sensor reads, logs, or
  micro-presence), a warning alert is published.

Alerts are routed through `AlertPolicy`, which enforces per-alert cooldowns,
assigns severity-based priorities, and sets TTLs before publishing to the event
bus. Default cooldowns and TTLs can be configured in `config/default.yaml`.
【F:services/ops_orchestrator.py†L342-L392】【F:core/alert_policy.py†L1-L79】【F:config/default.yaml†L54-L61】

## Budget Windows

Rolling budgets are backed by `RollingWindowBudget`, which tracks timestamps of
recent events and enforces configurable limits over a sliding time window. The
ops orchestrator applies these budgets to:

- **Sensor reads** per minute (health probes)
- **Micro-presence gestures** per hour
- **Log events** per minute

Budgets are configured under `ops.budgets` in the default config and can be
raised or set to `0` to disable limits entirely. The log budget additionally
influences whether the ops loop emits heartbeat and health snapshot logs.
【F:core/budgeting.py†L1-L62】【F:services/ops_orchestrator.py†L76-L431】【F:config/default.yaml†L37-L54】

## Configuration Summary

Relevant configuration keys include:

- `health.debounce_s` and `health.network` for probe debounce and optional
  network checks.
- `ops.budgets` for sensor read, micro-presence, and log budgets.
- `ops.micro_presence` for enabling gestures and defining intervals/battery
  thresholds.
- `alerts.cooldown_s` and `alerts.ttl_s` for alert throttling.

See `config/default.yaml` for defaults and in-line comments on each field.
【F:config/default.yaml†L37-L61】【F:config/default.yaml†L200-L219】


## Battery Event Lifecycle (Operator View)

The battery path has two layers:

1. **Telemetry/severity layer** in `BatteryMonitor` (sampling, thresholds,
   hysteresis, transition metadata).
2. **Conversation-response layer** in realtime injection policy (whether a
   battery event should create a response).

```text
ADS1015 voltage read
   ↓
BatteryMonitor._build_event
  - convert to percent_of_range
  - apply warning/critical thresholds
  - apply hysteresis to avoid flips
  - derive transition + delta_percent + rapid_drop
   ↓
BatteryMonitor.create_event_bus_handler
  - publish Event(source=battery, kind=status, metadata=...)
  - set request_response based on battery.response policy
   ↓
RealtimeAPI.inject_event
  - format battery text
  - re-evaluate request_response from metadata + config
  - optional query-context bypass for explicit user battery questions
   ↓
log line: source=battery kind=status create_response=true|false
   ↓
If true → response.create (assistant speaks)
If false → passive telemetry only (no speaking)
```

## Battery Config Keys and Meanings

All battery tuning lives under `battery:` in config.

### Severity/telemetry keys

- `battery.voltage_min` / `battery.voltage_max`
  - Voltage range used to normalize battery percentage (`percent_of_range`).
- `battery.warning_percent`
  - Warning boundary (normalized percent, 0-100).
- `battery.critical_percent`
  - Critical boundary (normalized percent, 0-100).
- `battery.hysteresis_percent`
  - Stabilization margin to prevent warning/info boundary chatter.

### Response policy keys

- `battery.response.enabled`
  - Global on/off for battery-triggered assistant responses.
- `battery.response.cooldown_s`
  - Minimum spacing between battery-triggered response requests.
- `battery.response.allow_warning`
  - Whether warning-level events are eligible to request a response.
- `battery.response.allow_critical`
  - Whether critical-level events are eligible to request a response.
- `battery.response.require_transition`
  - If true, steady-state repeats are suppressed and only transitions qualify.
- `battery.response.query_context_window_s` (optional)
  - Window for explicit user battery query context (e.g., “how’s battery?”),
    allowing response bypass of normal suppression checks.

## Tuning Examples

### Quiet mode (minimal battery chatter)

Use this when you want passive monitoring and very few spoken battery updates.

```yaml
battery:
  warning_percent: 40
  critical_percent: 20
  hysteresis_percent: 8
  response:
    enabled: true
    cooldown_s: 300
    allow_warning: false
    allow_critical: true
    require_transition: true
```

Expected behavior:

- No speaking for routine warning telemetry.
- Speaking only on critical (or explicit user battery query context).
- Fewer warning/info flips due to larger hysteresis.

### Verbose mode (more proactive battery voice updates)

Use this when you want conversational awareness during battery decline.

```yaml
battery:
  warning_percent: 55
  critical_percent: 25
  hysteresis_percent: 3
  response:
    enabled: true
    cooldown_s: 45
    allow_warning: true
    allow_critical: true
    require_transition: false
```

Expected behavior:

- Warning and critical updates can request responses more often.
- Faster user awareness but more potential chatter.

## Troubleshooting: Unexpected Battery Chatter

Use this checklist in order:

1. **Confirm policy log output**
   - Check for lines with `source=battery kind=status create_response=...`.
   - If `create_response=true` appears unexpectedly, inspect transition and
     severity metadata on that event.
2. **Check warning policy settings**
   - If chatter is warning-heavy, set `allow_warning: false` or increase
     `cooldown_s`.
3. **Enable transition-only speaking**
   - Set `require_transition: true` to suppress steady-state repeats.
4. **Increase hysteresis**
   - Raise `hysteresis_percent` if warning/info boundary jitter causes flips.
5. **Validate query-context behavior**
   - Recent explicit prompts like “how’s battery?” intentionally allow response
     bypass for a short window (`query_context_window_s`).
6. **Review thresholds for your battery chemistry**
   - Mismatched `voltage_min/max` can exaggerate normalized swings.

## Why `battery.warning_percent` May Differ from `ops.micro_presence.battery_min_percent`

These values control different subsystems and should not be forced to match:

- `battery.warning_percent` (battery monitor)
  - Controls **telemetry severity** and possible conversational signaling.
- `ops.micro_presence.battery_min_percent` (ops orchestrator)
  - Controls whether **micro-presence gestures** are allowed.

A common pattern is:

- Keep micro-presence conservative (higher minimum battery reserve) to preserve
  energy for core interaction.
- Keep warning threshold lower (or differently tuned) so spoken warnings are
  aligned with user expectations rather than gesture energy budget.

This separation lets operators tune battery chatter and motion behavior
independently, without code changes.
