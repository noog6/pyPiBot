# Preference Recall Runtime Audit

## Trigger conditions
- `memory_intent=true` is set from `_is_memory_intent(...)` during transcript-final handling in `RealtimeAPI`; this then routes into `_maybe_handle_preference_recall_intent(...)`. 
- Preference recall is entered when `_is_preference_recall_intent(...)` matches either:
  - broad memory intent, or
  - marker+domain detection.

## Query generation
- Primary query: `_build_preference_recall_query(...)` now builds a compact, intent-specific query from extracted keywords/entities plus domain canonicals.
- Attempt=0 uses only the normalized primary query.
- Fallback is now conditional:
  - only if a strict fallback marker is present,
  - only if a narrow fallback query can be derived,
  - only when `preference_recall.max_attempts > 1`.

## Retrieval backend + thresholds
- Recall execution uses `recall_memories` (tool), which uses `MemoryManager.recall_memories_with_trace(...)` and emits `memory_cards`.
- Runtime now filters user-facing cards/payload with minimum thresholds:
  - semantic score floor,
  - lexical score floor,
  - or token overlap / lexical exact match.
- Low-score entries are excluded from user-facing output unless debug override is enabled.

## Result injection path
- Tool payload formatting comes from `render_memory_cards_for_assistant(...)`.
- Preference recall now gathers context only and does **not** directly send replacement assistant messages.
- Normal server/model response flow remains authoritative for final response delivery.

## Scheduling + suppression behavior
- Preference recall no longer suppresses or cancels active `server_auto` responses in the intent handler.
- `handled_preference_recall` now means context collection completed, not lifecycle ownership.
- This prevents preference recall from competing with in-flight canonical audio and avoids delivery suppression side effects.
