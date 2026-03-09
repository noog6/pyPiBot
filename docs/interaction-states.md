# Interaction States & UX Cues

This document describes the interaction state model used by the realtime API and how to
hook state-driven UX cues (gestures/earcons).

## State Model

The system tracks a simple four-state model:

- **IDLE**: No user speech detected and no assistant response in progress.
- **LISTENING**: User speech detected by server-side VAD.
- **THINKING**: User speech ended or a response is being prepared.
- **SPEAKING**: Assistant response is streaming (audio or text).

## Transition Map

The realtime API emits state transitions based on websocket events:

| Event | Transition |
| --- | --- |
| `input_audio_buffer.speech_started` | `IDLE` → `LISTENING` |
| `input_audio_buffer.speech_stopped` | `LISTENING` → `THINKING` |
| `response.created` | `LISTENING/IDLE` → `THINKING` |
| `response.output_audio.delta` / `response.text.delta` | `THINKING` → `SPEAKING` |
| `response.output_audio.done` / `response.output_audio_transcript.done` | `SPEAKING` → `IDLE` |

## Cue Hooks

`InteractionStateManager` exposes handlers for gesture and earcon cues. Register handlers to
invoke your preferred behavior (motion controller, audio player, etc.). For example:

```python
state_manager.set_gesture_handler(my_gesture_handler)
state_manager.set_earcon_handler(my_earcon_handler)
```

The handlers receive the new `InteractionState`, so they can dispatch different cues per
state.

## Ambient vs Requested Gestures (Turn-Contract Precedence)

Theo may emit **ambient gestures** (also called innate/background/expressive gestures)
from state cues while a turn is active. These cues are allowed by default unless a caller
explicitly forbids gesture output for that turn.

Definitions used in runtime docs:

- **Ambient gesture**: state-driven embodiment motion emitted for presence or expressiveness
  (for example, thinking/listening tilt or idle nod).
- **User-requested action/gesture**: an explicit action the user asked Theo to perform.
- **Speech obligation / turn contract**: required speech or phrase obligations attached to
  the turn outcome (including exact/inclusion phrase repairs when required).

Precedence rule:

> Ambient gestures may continue unless explicitly forbidden by the active turn contract
> or runtime policy, but they must not cause loss, suppression, delay-based failure, or
> semantic corruption of the user-requested action/speech contract.

In this context, **interference** includes:

- Replacing the requested action with ambient motion.
- Preventing or dropping required speech output.
- Collapsing “do action + say phrase” into only one half.
- Suppressing a required repair/follow-up response that is needed to satisfy the turn.
- Materially corrupting timing/ordering so the requested contract is not fulfilled.

Allowed coexistence example:

- Theo performs a subtle thinking tilt while still completing: “Do one attention snap, then
  say ‘Sentinel Theo online.’”

Not allowed examples:

- Ambient motion replaces the requested attention snap.
- Ambient behavior causes the required phrase obligation to be dropped.
- Theo closes the turn after a micro-ack without delivering the requested phrase.

Current limitation:

- Ambient gesture emitters can operate somewhat independently from upper intent/arbitration
  layers. The current guarantee is therefore **outcome-based** (requested contract must still
  be fulfilled), not “total gesture subsystem silence” during a turn.


## Embodiment Layer (Phase A + B)

The embodiment layer is a narrow seam between interaction state signals and physical cue execution.

- **Phase A (complete):** `EmbodimentPolicy` owns deterministic state-cue selection/suppression reasons.
- **Phase B (current):** `AttentionContinuity` tracks a tiny embodied-attention hold across speech-stop/transcript-final/defer churn so Theo does not appear to disengage too early.
- **Phase C (future):** richer continuity and arbitration-aware presence behaviors (without moving lifecycle transport logic into policy).

Ownership boundaries:

- Runtime (`ai/realtime_api.py`) observes speech/state/lifecycle events and updates continuity state.
- Policy (`ai/embodiment_policy.py`) reads continuity facts and decides whether to emit/suppress a state cue.
- Runtime executes motion actions after policy decisions.

Anti-patterns to avoid:

- Rebuilding a second lifecycle engine inside embodiment helpers.
- Moving servo/camera target planning into attention continuity.
- Burying embodiment policy branches back into raw websocket handlers.
- Letting continuity helpers create/own response lifecycle decisions.

## Configuration

State cues and timing thresholds are configured in `config/default.yaml` under
`interaction_states`:

- `cues_enabled`: Master toggle for cue emission.
- `gesture_enabled`: Enable/disable gesture callbacks.
- `earcon_enabled`: Enable/disable earcon callbacks.
- `min_state_duration_ms`: Debounce window to prevent rapid cue spam.
- `cue_delays_ms`: Per-state delay before emitting cues (in milliseconds).

Use `config/override.yaml` to customize these settings per deployment.

## Micro-Acks During Slow Thinking Windows

Theo can emit a very short "micro-ack" (for example, "One sec—checking.") when a user
utterance has ended and the real answer is likely to take longer than usual.

Guardrails:

- Emit at most once per turn by default, with a global cooldown.
- Suppress while user speech is active, in `LISTENING`, or when talk-over risk is elevated.
- Schedule on a short timer and cancel if the real answer starts quickly.
- Treat micro-acks as non-substantive acknowledgements; they must not trigger tool calls,
  confirmation loops, or memory persistence.

Relevant realtime config keys:

- `realtime.micro_ack_enabled`
- `realtime.micro_ack_delay_ms`
- `realtime.micro_ack_expected_wait_threshold_ms`
- `realtime.micro_ack_long_wait_second_ack_ms`
- `realtime.micro_ack_global_cooldown_ms`
- `realtime.micro_ack_per_turn_max`

## Approval Prompts & Stop Words

When a tool call requires confirmation, the realtime agent presents a structured
action packet (what/why/impact/rollback/cost/confidence/alternatives) and waits
for an explicit approval response before executing the action.【F:ai/realtime_api.py†L720-L840】

Stop words (configured in `config/default.yaml`) immediately cancel pending
actions and place tool execution into a cooldown period so users can safely
interrupt automated actions.【F:ai/realtime_api.py†L231-L540】【F:config/default.yaml†L12-L16】

## Injected Event Responses

When the runtime injects events (for example, image or text messages pushed into the
conversation outside of live audio), the realtime client can request a response by sending a
`response.create` event. These injected responses are gated to avoid interrupting live
interaction.

The injected response request is skipped when any of the following are true:

- The interaction state is not `IDLE` or `LISTENING`.
- A response is already in progress.
- A cooldown window has not elapsed since the last injected response.
- The per-minute injected response limit is exhausted.
- The session rate limits report zero remaining requests/responses.

Configure the thresholds in `config/default.yaml`:

- `injection_response_cooldown_s`: Minimum seconds between injected responses.
- `max_injection_responses_per_minute`: Maximum injected responses per rolling minute.
- `image_response_mode`: Controls whether image injections can request responses.
  - `respond`: Enqueue image messages as injected stimuli (default).
  - `catalog_only`: Still send images to the model, but skip the injected response request.
- `injection_response_triggers`: Per-trigger overrides for injected responses (keyed by trigger name).
  - `cooldown_s`: Minimum seconds between injected responses for the trigger.
  - `max_per_minute`: Maximum injected responses per rolling minute for the trigger.
  - `priority`: Priority value; positive values bypass the global cooldown/rate limits while still
    honoring the trigger-specific limits.
