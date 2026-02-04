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

- Audit API key handling/log redaction coverage (e.g., bearer tokens and OpenAI keys) to ensure secrets never leak to logs or persisted diagnostics files.【F:core/logging.py†L220-L280】【F:core/logging.py†L340-L420】
- Review subprocess usage that shells out to system tools (e.g., `amixer`) and confirm inputs are fully controlled/validated to prevent command injection or unexpected device access.【F:services/output_volume.py†L14-L84】
- Validate outbound network endpoints and TLS usage for realtime/HTTP API calls to ensure no plaintext or untrusted endpoints are reachable in production builds.【F:ai/realtime_api.py†L160-L260】【F:ai/realtime_api.py†L840-L900】
