# Seam Family: Memory and Preference Recall Paths

Owning layer: Layer 2 perception + memory interface.

Status: **Active seam** and **high-churn seam** (heuristic thresholds and takeover paths).

Confidence: medium (heuristic intent classification + takeover edge ordering).

## A) Purpose

Classify memory intent, retrieve scoped memory evidence, and optionally alter same-turn response behavior (including preference-recall takeover/suppression paths).

## B) Owning modules/files

- `ai/realtime/memory_runtime.py` (intent classification + turn memory retrieval).
- `ai/realtime/preference_recall_runtime.py` (direct preference recall and server-auto takeover checks).
- `services/memory_manager.py` (retrieval backend and debug metadata surfaces).
- Adjacent influence: `ai/realtime/response_create_runtime.py` memory-intent instruction guardrails.

## C) Entry points

- Memory intent classification for user text.
- Per-turn memory brief preparation.
- Preference recall runtime handlers that can request same-turn takeover.

## D) Exit points / outcomes

- Memory brief/context payload attached to response path.
- Preference recall prompt notes / direct-answer hints.
- Optional suppression/override of provisional server-auto outputs.

## E) Upstream dependencies

- User utterance text quality and intent cues.
- Memory manager readiness (semantic + lexical retrieval health).
- Active response/provisional state when takeover is considered.

## F) Downstream impacts

- Response-create policy inputs (suppression/lock flags).
- Quality and directness of memory-backed answers.
- Early-turn context decisions and transcript-final replacement behavior.

## G) Core invariants

- Memory intent subtype should be explicit (`preference_recall`, `topic_recall`, `general_memory`, `none`).
- Retrieval failures should degrade gracefully and observably.
- Takeover request must stay bounded to eligible provisional server-auto cases.

## H) Failure signatures

- Preference hit exists but response ignores it or prefaces incorrectly.
- Memory retrieval unavailable causes repeated empty/weak grounding.
- Provisional server-auto answer persists despite high-confidence recall takeover condition.

## I) Known tricky handoffs

- Memory recall lock/suppression flags crossing into response-create arbitration.
- Topic recall vs preference recall disambiguation.
- Metadata propagation from retrieval verdict into response instructions.

## J) Related seams

- [Response create arbitration](response_create_arbitration.md)
- [Startup and system context](startup_and_system_context.md)
- [Observability and tracing surfaces](observability_and_tracing_surfaces.md)

## K) First places to inspect

1. `MemoryRuntime.classify_memory_intent(...)` and retrieval query shaping.
2. Preference recall takeover evaluation helpers.
3. Memory manager debug metadata and readiness signals.
4. Logs: `turn_memory_retrieval_*`, preference recall takeover diagnostics.
