# Seam Family: Startup and System Context Injection

Owning layer: Layer 1 runtime gating with Layer 2 context enrichment effects.

Status: **Fragile seam** (timing-sensitive), currently narrower and relatively stable.

## A) Purpose

Gate startup/system-context payload injection during early session phases so first-turn behavior is grounded without racing unsettled response lifecycle authority.

## B) Owning modules/files

- `ai/realtime/injections.py` (`InjectionCoordinator` gate/defer/release/timeout).
- `ai/realtime/injection_bus.py` and runtime task wiring.
- `ai/realtime_api.py` integration points that emit queued injected events.

## C) Entry points

- Startup coordinator initialization and timeout scheduling.
- `should_defer(...)` checks for first-turn unsettled state.
- `release(...)` flushing deferred injection queue.

## D) Exit points / outcomes

- Deferred event/system context payload queue flush.
- Timeout-based forced release.
- First-assistant-utterance-triggered release boundary.

## E) Upstream dependencies

- Event loop readiness.
- First-assistant utterance detection.
- Startup gate timeout config.

## F) Downstream impacts

- Early-turn grounding quality.
- Initial context available to response generation.
- Potential confusion if startup context arrives too late.

## G) Core invariants

- Startup gate should not become hidden response arbitration authority.
- Deferred payloads should flush exactly once when released.
- Timeout release should be safe and idempotent.

## H) Failure signatures

- Missing expected startup/system context in early turns.
- Double injection after both utterance and timeout triggers.
- Early-turn behavior ignoring known system context.

## I) Known tricky handoffs

- Release timing relative to first `response.create` send.
- Context injection interacting with preference recall takeover paths.

## J) Related seams

- [Response create arbitration](response_create_arbitration.md)
- [Memory and preference recall paths](memory_and_preference_recall_paths.md)

## K) First places to inspect

1. `InjectionCoordinator.gate_active/should_defer/release`.
2. Runtime wiring for startup timeout task scheduling.
3. Logs: `startup_injection_deferred`, `startup_injection_flush`.
