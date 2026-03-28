# Theo Quiet Intent (Current-State, Code-Aligned)

Purpose: capture what Quiet Intent **actually does today** so future sessions do
not have to reverse-engineer `ai/quiet_intent.py` + `ai/realtime_api.py`.

## Owning layer

**Owning layer:** Layer 6 (Conversational pacing + embodiment-adjacent consultative posture biasing).

Quiet Intent is a deterministic selector that emits a posture-bias profile and a
diagnostic snapshot. It is refreshed in `RealtimeAPI` during interaction-state
transitions, beside attention continuity and embodiment cue policy, which makes
Layer 6 the correct ownership boundary.

Quiet Intent does **not** own:
- response-create arbitration
- terminal-deliverable selection
- semantic-owner reassignment
- tool-followup arbitration
- governance/confirmation enforcement
- tool execution

## Concrete implementation

### Modules and runtime hooks
- Selector + enums + decision payloads: `ai/quiet_intent.py`
- Runtime refresh + logging/fingerprinting: `RealtimeAPI._refresh_quiet_intent`
  and helpers in `ai/realtime_api.py`
- Regression coverage: `tests/test_quiet_intent.py`,
  `tests/test_quiet_intent_runtime_calibration.py`

### Inputs consumed today

`QuietIntentInputs` currently includes:
- `interaction_state` (`InteractionState`)
- `conversation_active` (derived from recent input and response-in-flight flags)
- `continuity_stance` (bounded enum from normalization)
- `continuity_stance_raw` (source-preserved string)
- `ops_severity` (bounded enum from normalization)
- `ops_severity_raw` (source-preserved string)
- `recent_utterance_flags` (ordered lexical flags from most recent user text)
- `attention_active` (from attention continuity snapshot)

#### Lexical flags (current classifier)

From most recent user text:
- `calm_context`: token match on `tea`, `chill`, `calm`, `rest`, `quiet`
- `observation_context`: token match on `observe`, `notice`, `watch`, `look`
- `curiosity_signal`: **regex-only** on `why`, `how`, `curious`, `wonder`
- `anomaly_signal`: token match on `alert`, `alarm`, `warning`, `critical`, `anomaly`

Ordered output tuple (if present):
`calm_context`, `observation_context`, `curiosity_signal`, `anomaly_signal`.

Normalization helpers bound noisy runtime strings:
- `normalize_continuity_stance` → `idle`, `awaiting_user`, `assisting_query`,
  `recovering_context`, or `other`
- `normalize_ops_severity` → `unknown`, `info`, `warning`, `critical`

### Modes implemented today

- `sentinel`
- `companion_presence`
- `curious_witness`
- `resting_ritual`
- `observer`

### Output shape (consultative vs diagnostic)

Quiet Intent emits three related payload surfaces:

1. **Consultative posture-bias output** (`to_consultative_bias_output`)
   - `mode`, `confidence_band`, `reason_codes`
   - `initiative_level`, `verbosity_bias`, `gesture_bias`,
     `interruption_tolerance`, `observation_threshold`
   - intentionally excludes runtime-state snapshot fields

2. **Diagnostic snapshot** (`to_diagnostic_snapshot`)
   - consultative output fields plus
   - `confidence` and all bounded/raw input fields

3. **Transitional logging alias** (`to_log_payload`)
   - explicitly logging-only compatibility surface
   - returns the same payload as `to_diagnostic_snapshot`

### Selection/scoring behavior

Selector behavior is deterministic and intentionally conservative:
- Scores start at zero for all modes.
- Signals add weighted increments per mode.
- Highest score wins; tie-break order is fixed and explicit:
  `sentinel` > `companion_presence` > `curious_witness` >
  `resting_ritual` > `observer`.
- Confidence is clamped to `[0.35, 1.00]` and mapped to `low|medium|high`.

Notable calibrations in current code/tests:
- Curiosity classification is narrower: generic “do you know …” text does not
  auto-raise `curiosity_signal`; explicit curiosity language still does.
- `assisting_query` stance no longer promotes `curious_witness` by itself; it
  only contributes when curiosity context is present.

## Runtime integration and authority boundary

### Refresh trigger
Quiet Intent is refreshed from `RealtimeAPI._handle_state_gesture` whenever
interaction state transitions are processed.

### Logging + dedupe
- Runtime builds a fingerprint from consultative fields + reason codes + recent
  flags.
- If unchanged, runtime emits DEBUG `quiet_intent_decision_unchanged`.
- If changed, runtime emits INFO `quiet_intent_decision ...` with the full
  diagnostic snapshot fields.

### What consumes the output now
Current runtime behavior stores `self._latest_quiet_intent_decision` and logs
the snapshot. No authoritative policy seam currently consumes it for execution
control.

## What Quiet Intent does **not** do today

Quiet Intent does not currently:
- gate or schedule `response.create`
- approve/deny tool actions
- alter governance tiers/confirmation requirements
- release/hold/suppress tool followups
- assign semantic owner
- pick terminal deliverables
- execute gestures/tools directly

## Current-state vs future aspirations

Current state is consultative posture-bias interpretation and diagnostics.
Future architecture may choose to consume these outputs more directly, but that
authority does not exist in today’s implementation.
