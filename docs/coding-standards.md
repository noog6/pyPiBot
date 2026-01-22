# Python Coding Standards

This document defines the baseline standards for all Python code in this project.

## Core Principles

- **Clarity over cleverness.** Optimize for readability and maintainability.
- **Small, testable units.** Prefer functions and classes with single responsibility.
- **Explicitness wins.** Avoid hidden side effects and implicit state.

## Formatting & Style

- Follow **PEP 8**.
- Use **Black** for formatting (no manual formatting exceptions).
- Line length: **88 characters** (Black default).
- Use **isort** (compatible with Black) for import ordering.

## Type Hints

- All new code **must include type hints**.
- Prefer concrete types over `Any`.
- Use `from __future__ import annotations` for forward references in new modules.

## Docstrings

- Required for all public modules, classes, and functions.
- Use **Google-style docstrings**.
- Docstrings must describe:
  - Purpose/behavior
  - Arguments and types
  - Return value and type
  - Exceptions raised (if any)

## Linting & Static Analysis

- Use **ruff** for linting and formatting checks.
- Use **mypy** for type checking (strict where possible).
- Linting and type checks should run in CI.

## Imports

- Group imports in this order:
  1. Standard library
  2. Third-party
  3. Local application
- Do **not** use try/except around imports to hide missing dependencies.

## Error Handling

- Raise specific exceptions.
- Avoid bare `except`.
- Log errors with context; do not suppress errors silently.

## Logging

- Use the project logging utilities (once defined).
- Avoid `print()` in production code.
- Log at appropriate levels (DEBUG/INFO/WARNING/ERROR/CRITICAL).

## Testing

- Use **pytest**.
- Tests should be deterministic and isolated.
- Hardware interactions must be mockable.
- Expose a lightweight `probe()` function in each subsystem to support
  diagnostics and offline tests.
- Keep probes side-effect safe by default (no destructive actions without an
  explicit opt-in).

## Diagnostics

- Store high-level diagnostics orchestration in `diagnostics/`.
- Probes should return a structured result (status + details) for consistent,
  human-readable output.
- Runtime diagnostics should be callable from the CLI without requiring the full
  robot runtime to start.

## Project Structure

- Keep modules focused and cohesive.
- Avoid circular dependencies.
- Use clear, descriptive names.

## Dependencies

- Keep dependencies minimal.
- Prefer standard library when practical.
- Pin versions for runtime and development dependencies.

## Security & Safety

- Validate external input.
- Avoid executing shell commands with user input.
- Treat hardware operations as potentially destructive—include safeguards.

## Optional Dependencies

- Optional dependencies (e.g., audio or hardware libraries) should be imported
  lazily at runtime.
- Missing optional dependencies must fail **gracefully** with a clear error
  message and without crashing the entire runtime.

## Configuration Conventions

- Use intuitive, stable configuration keys.
- If nested sections are used, group related settings under a shared prefix
  (for example, `storage` for persistence settings).
- Configuration updates should avoid breaking existing keys where possible.

## Resource Lifecycle

- Any resource that opens hardware, sockets, files, or databases must provide a
  deterministic shutdown path (for example, a `close()` method).
- Runtime code should explicitly clean up resources on shutdown.
