# Persistence Access Contract

This document defines the canonical persistence access patterns for runtime services.

## 1) When to use `StorageController`

Use `StorageController` when data must be tied to the shared runtime/logging database lifecycle (run-scoped DB file, shared lock, shared connection).

Typical use:
- Run logs and generic runtime data (`logs`, `data` tables).
- Research budget state + usage audit rows, which are expected to share the runtime connection/lock.

## 2) When direct store classes are allowed

Production service wiring must use helper constructors from `storage.factories` for default store creation, including `MemoryEmbeddingWorker` (which now defaults through `create_memory_store()`).

Direct constructor injection with concrete store classes (`MemoryStore`, `UserProfileStore`, `ResearchBudgetStorage`) remains allowed only for explicit tests and advanced composition/injection scenarios where the caller intentionally owns persistence setup.

Canonical helpers:

- `create_memory_store()`
- `create_user_profile_store()`
- `create_research_budget_store()`

These helpers centralize path/config/connection resolution so services do not duplicate persistence setup logic.

## 3) Lock + connection ownership expectations

- `StorageController` owns its SQLite connection and lock for run-scoped storage.
- `ResearchBudgetStorage` should be created from `create_research_budget_store()` so it shares the `StorageController` connection and lock.
- `MemoryStore` and `UserProfileStore` own their own SQLite connections/locks (file-backed stores under canonical var-dir resolution).
- Services should not replace, close, or mutate store internals (`_conn`, `_lock`) outside explicit constructor injection in tests/composition.

## 4) Configuration/path rules

All service-level store creation should resolve filesystem paths via storage helpers so `storage.var_dir` (or fallback `var_dir`) behavior stays consistent across stores.
