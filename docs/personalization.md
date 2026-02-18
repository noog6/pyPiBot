# Personalization & User Profiles

## Overview

Theo’s personalization is layered from:

1. Base assistant/session instructions.
2. Active user profile context.
3. Memory hydration (tiny startup digest + turn-time retrieval).

## Default population

- Assistant name defaults to `Theo`.
- Active user defaults to `default`.
- Empty profiles are created lazily with unknown/missing fields.

## User-controlled profile fields

Via `update_user_profile`:
- `name`
- `preferences`
- `favorites`

Stored in `var/user_profiles.db`.

## Memory system

Durable memory is stored in `var/memories.db`.

### Ingestion paths

Theo ingests memories via two explicit paths:

1. **Manual tool call** (`remember_memory`)
   - `source=manual_tool`
   - optional manual pinning (`pinned=true`)
2. **Response-done auto-memory** (reflection)
   - `source=auto_reflection`
   - config-gated by:
     - `memory.auto_memory_enabled`
     - `memory.require_confirmation_for_auto_memory`
     - `memory.auto_memory_min_confidence`

### Scope contract

Memory scope is explicit:

- `user_global`: available across runs for active user.
- `session_local`: isolated to active runtime `session_id`.

Theo assigns runtime `session_id=run-<run_id>` at startup, so session-local behavior is stable.

### Two-tier hydration policy (balanced context)

Theo uses a two-tier hydration model to avoid empty context *and* avoid full preload:

#### Tier 1: startup digest (tiny preload)

At session initialization, Theo injects a small pinned digest:
- sourced from pinned, review-approved user-global memories,
- capped by `memory_hydration.startup_digest_max_items` and `startup_digest_max_chars`.

Typical preload examples:
- core identity/favorite name pronunciation,
- persistent preference like "prefers concise answers",
- stable safety preference.

#### Tier 2: per-turn retrieval (query-aware)

On each user turn, Theo retrieves only highly relevant memories:
- lexical/tag relevance + importance + recency ranking,
- stale suppression and near-duplicate suppression,
- strict caps (`memory_retrieval.max_memories`, `memory_retrieval.max_chars`),
- anti-bloat skip for short/noisy user inputs,
- cooldown between retrieval injections.

Typical turn retrieval examples:
- user asks about coffee setup → recall brewing preference,
- user asks project status → recall project-specific memory,
- short/noisy input (“ok”, “thanks”) → skip retrieval.

### Pinning and review

Memories support pinning metadata:
- `pinned` marks items eligible for startup digest.
- `needs_review` keeps auto-pinned items out of startup preload until reviewed.

Auto-reflection memories can be auto-pinned at high importance using:
- `memory.auto_pin_min_importance`
- `memory.auto_pin_requires_review`

### Auditing metadata

Each memory row stores:
- scope identifiers (`user_id`, optional `session_id`),
- `source` (`manual_tool` / `auto_reflection`),
- pin/review flags (`pinned`, `needs_review`).

### Forgetting

`forget_memory(memory_id=...)` deletes a memory row by id.

## Semantic memory disable runbook

The semantic memory rollout can be disabled explicitly with config only:

```yaml
memory_semantic:
  enabled: false
  rerank_enabled: false
```

- Set both keys together so semantic retrieval and reranking are unambiguously off.
- **No DB rollback is required** when disabling this feature.
- Existing `memories` and `memory_embeddings` rows can remain in place; the runtime
  will continue using lexical retrieval paths.

## Semantic embedding write mode when background worker is disabled

By default, `memory_semantic.background_embedding_enabled: false` keeps writes purely lexical and does not attempt embedding generation.

To opt into synchronous write-time embeddings in this mode, set:

```yaml
memory_semantic:
  enabled: true
  background_embedding_enabled: false
  inline_embedding_on_write_when_background_disabled: true
```

This path is **fail-open**: memory rows are still written even if embedding calls time out or fail, and embedding status is recorded as pending/error for observability.

Tradeoff: inline embedding increases per-write CPU and can add write latency on constrained hardware, so the flag remains opt-in.
