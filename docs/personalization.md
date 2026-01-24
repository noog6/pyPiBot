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
