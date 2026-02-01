# Personalization & User Profiles

## Overview

Theo’s “personality” is driven by two layers:

1. **Base session instructions** that define Theo’s default voice and response style.
2. **User profile context** that is appended to the session instructions to personalize replies.

When a session starts, the realtime API builds the session instructions by concatenating
the base personality text with the active user’s profile context (if available). This
means personalization is always layered on top of the default personality rather than
replacing it.

## What is populated by default

These values are present even before any user data is stored:

- **Assistant name**: `assistant_name` defaults to `"Theo"`.
- **Active user id**: `active_user_id` defaults to `"default"`.
- **Profile fields**: If there is no stored profile, a new profile is created with:
  - `name`: `None`
  - `preferences`: `{}` (empty object)
  - `favorites`: `[]` (empty list)
  - `last_seen`: `None`

The system will still include a profile context block with “Unknown” or “None” values
and will prompt politely for missing info.

## What users can change

Users (or integrators) can change these values via configuration or tools:

- **Active user id**: Configure `active_user_id` in `config/default.yaml` or override files
  to switch the active profile.
- **Profile data**: The `update_user_profile` tool can update:
  - `name` (string)
  - `preferences` (object)
  - `favorites` (array of strings)

These updates are persisted to the SQLite profile database in `var/user_profiles.db`.

## What changes over time through interactions

Some fields are updated automatically as the assistant is used:

- **`last_seen`**: Updated whenever the active profile is loaded and whenever
  `update_user_profile` is called.

Over time, this enables more natural continuity because the assistant can keep track of
recent activity and use stored preferences or favorites when responding.

## How memories work

Theo has a separate, durable memory system that stores long-lived facts independently
from the profile fields above. Memories are only added or queried when Theo calls the
memory tools (there is no automatic background capture), and the base session
instructions explicitly guide Theo to use them only for stable, reusable facts the
user confirms.

### Capturing memories

Theo captures memories through the `remember_memory` tool. Each memory stores:

- **Content**: A short normalized text snippet. Content is trimmed and capped at
  400 characters to keep prompts concise.
- **Tags**: Optional, normalized tags (lowercased, deduped, max 6 tags, 24 characters
  each).
- **Importance**: A 1–5 score that gets clamped to safe bounds.
- **Scope**: The active `user_id` and `session_id` are stored with each entry so that
  recall can filter to the current user/session when needed.

Memories are persisted to SQLite in `var/memories.db` using the memory store.

### Recalling memories

Theo recalls memories via the `recall_memories` tool. The query is a lightweight
SQL `LIKE` match over content and tags, and results are filtered by the active
`user_id`/`session_id`. Results are ordered by **importance** (descending) and then
by **recency**, and the recall limit is capped at 10.

### Forgetting memories

When a user asks Theo to delete a memory, the `forget_memory` tool removes the row by
its `memory_id`.
