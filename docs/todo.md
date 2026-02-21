# Todo

## Diagnostics follow-ups

- Keep probes side-effect light (small temp files, short-lived checks).
- Standardize probe result details for operator-friendly output.
- Continue lazy imports for optional dependencies so diagnostics can run anywhere.

## Next steps

- Expand hardware abstractions and mock drivers.
- Build user interaction pipelines (speech/text -> intent -> action).
- Add more external integrations in `services/`.

## Security investigations

- Review subprocess usage that shells out to system tools (e.g., `amixer`) and confirm inputs are fully controlled/validated to prevent command injection or unexpected device access.【F:services/output_volume.py†L14-L84】
- Validate outbound network endpoints and TLS usage for realtime/HTTP API calls to ensure no plaintext or untrusted endpoints are reachable in production builds.【F:ai/realtime_api.py†L160-L260】【F:ai/realtime_api.py†L840-L900】

## Test-suite maintainability follow-ups

- Create shared `tests/helpers/semantic_config.py` fixture/builders so semantic retrieval tests do not repeat partial `SimpleNamespace` configs that can drift from `SemanticMemoryConfig` defaults.
- Add focused retrieval debug contract tests that assert required keys/types rather than exact `mode` string values for every branch, to reduce brittleness when fallback categorization evolves.
- Introduce reusable `RealtimeAPI` confirmation-gating test stubs/fixtures that track both legacy (`_pending_action`) and tokenized (`_pending_confirmation_token`) flows while the migration remains in progress.
