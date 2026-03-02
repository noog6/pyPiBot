# Preference Recall Runtime Audit

## Trigger conditions
- `memory_intent=true` is set from `_is_memory_intent(...)` during transcript-final handling in `RealtimeAPI`; this then routes into `_maybe_handle_preference_recall_intent(...)`. 
- Preference recall is entered when `_is_preference_recall_intent(...)` matches either:
  - broad memory intent, or
  - marker+domain detection.

## Query generation
- Primary query: `_build_preference_recall_query(...)` now builds a compact, intent-specific query from extracted keywords/entities plus domain canonicals.
- Query normalization now drops conversational noise (`hey`, `remember`, `check`, `memories`) while preserving preference/domain cues and subject terms (for example, `favorite`, `preferred`, `editor`, `vim`).
- Attempt=0 uses the normalized primary query built from domain + value tokens first (for example, `editor favorite vim user preference`).
- When `preference_recall.max_attempts > 1`, attempt=1 uses deterministic query variants (domain-first and marker-first forms such as `preferred editor vim`) before optional narrow fallback.
- Fallback remains conditional:
  - only if a strict fallback marker is present,
  - only if a narrow fallback query can be derived.

## Retrieval backend + thresholds
- Recall execution uses `recall_memories` (tool), which uses `MemoryManager.recall_memories_with_trace(...)` and emits `memory_cards`.
- Runtime now filters user-facing cards/payload with minimum thresholds:
  - semantic score floor,
  - lexical score floor,
  - or token overlap / lexical exact match.
- Low-score entries are excluded from user-facing output unless debug override is enabled.
- Preference-recall result logging now emits a structured retrieval summary (`retrieval_backend`, `filters_applied`, `candidate_count`, `returned_count`, `empty_reason`) so empty responses can be diagnosed from a single INFO line.

## Result injection path
- Tool payload formatting comes from `render_memory_cards_for_assistant(...)`.
- Preference recall now gathers context only and does **not** directly send replacement assistant messages.
- Normal server/model response flow remains authoritative for final response delivery.

## Scheduling + suppression behavior
- Preference recall no longer suppresses or cancels active `server_auto` responses in the intent handler.
- `handled_preference_recall` now means context collection completed, not lifecycle ownership.
- This prevents preference recall from competing with in-flight canonical audio and avoids delivery suppression side effects.

## Run-441 corrective summary
- The run-441 misses were caused by query compaction that could drop salient preference/value tokens before retrieval.
- Preference recall now retries deterministic query variants while staying on the same `recall_memories` backend (same scope, same manager, same lexical/semantic behavior).
- INFO logs include the retrieval backend summary and `empty_reason` to explain why any empty result was produced.
