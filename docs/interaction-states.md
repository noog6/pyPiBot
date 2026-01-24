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

## Configuration

State cues and timing thresholds are configured in `config/default.yaml` under
`interaction_states`:

- `cues_enabled`: Master toggle for cue emission.
- `gesture_enabled`: Enable/disable gesture callbacks.
- `earcon_enabled`: Enable/disable earcon callbacks.
- `min_state_duration_ms`: Debounce window to prevent rapid cue spam.
- `cue_delays_ms`: Per-state delay before emitting cues (in milliseconds).

Use `config/override.yaml` to customize these settings per deployment.
