# pyPiBot

A Raspberry Pi–based AI control system for a small robotic observer, integrating
perception, interaction, and hardware control.

## Current Focus

- Target hardware: Raspberry Pi Zero W and Raspberry Pi Zero 2.
- Initial AI provider: OpenAI realtime models.
- Deployment target: systemd-managed service with git-based updates.

## What this is / What this isn’t

**What this is**
- A Raspberry Pi–hosted control stack for a small robotic observer (“Theo”).
- A Python codebase that ties together perception, interaction, motion, and GPIO.
- A foundation for running the runtime as a managed service on Raspberry Pi OS.

**What this isn’t**
- A fully autonomous or safety-certified robotics platform.
- A general-purpose robotics framework or drop-in SDK for arbitrary hardware.
- A project that self-modifies or claims unattended decision authority.

## Project Layout

```
pyPiBot/
├── main.py        # Runtime entrypoint
├── ai/            # AI provider integrations (realtime API + tools)
├── config/        # YAML configuration controller + defaults
├── core/          # Shared runtime support (policies, models, logging, diagnostics)
├── hardware/      # GPIO, sensors, and actuator drivers
├── interaction/   # Audio input/output and user interaction helpers
├── motion/        # Motion controller and keyframe sequencing
├── services/      # External services/integrations (ops, memory, research)
├── storage/       # Persistent storage (SQLite) controller
├── systemd/       # systemd unit templates
└── docs/          # Documentation (coding standards, requirements, setup)
```

## Getting Started

### Prerequisites

- Python 3.10+
- Audio dependencies for Raspberry Pi: `pyaudio`, `numpy`
- Realtime API dependencies: `websockets`
- Camera/vision dependencies (Raspberry Pi): `picamera2`, `Pillow`

### Installation

See [docs/installation.md](docs/installation.md) for Raspberry Pi OS setup
steps, git clone details, and the environment setup script. Hardware assembly
notes live in [docs/hardware-setup.md](docs/hardware-setup.md).

If you enable optional Firecrawl-backed research
(`research.firecrawl.enabled: true`), set `FIRECRAWL_API_KEY` with:

```bash
./scripts/update-firecrawl-key.sh
```

### Run the Runtime

```bash
python main.py --prompts "Say Hello World!"
```

The runtime will:
- Load configuration from `config/default.yaml`.
- Initialize the storage layer and log run metadata.
- Write per-run logs using incrementing numeric run IDs (for example `log/314/run_314.log`, with the current value tracked in `var/current_run`).
- Start the Realtime API runtime (required); startup exits with a non-zero status if this dependency cannot initialize.
- Attempt to start audio input/output (gracefully degrades if unavailable).
- Attempt to start optional hardware peripherals when present, including motion control, camera vision, IMU monitoring, and battery monitoring.

## Configuration

Configuration is stored in YAML under `config/`:
- `config/default.yaml` for baseline settings
- `config/override.yaml` for runtime updates (auto-archived on updates)

Default keys include:
- `log_dir` and `var_dir` for storage
- `assistant_name` and `startup_prompts`
- `logging_level`
- `stop_words` and `stop_word_cooldown_s` to pause tool execution on emergency phrases
- `health`, `ops`, and `alerts` for operational health probes, budgets, and alert policy
- `governance` settings (autonomy level, autonomy windows, budgets, and tool tier specs)
- `research` settings for web-lookup behavior, user permission gating, provider wiring, budget limits, and cache controls

Micro-ack tracing tip (Realtime): when micro-ack logging is enabled, each `micro_ack_scheduled`,
`micro_ack_emitted`, and `micro_ack_suppressed` line includes `dedupe_fp=<short_fingerprint>` so operators can
trace one micro-ack decision across schedule/emit/suppress transitions. Suppressions also include
`suppression_source` (for example `baseline`, `confirmation`, or `cooldown`) to quickly identify the gating layer.

### Web Research Capability

The runtime includes a web-research subsystem that:

- detects explicit web-research intent in user text,
- runs the OpenAI-backed research path in production when research is enabled,
- controls user confirmation gating via `research.permission_required`,
- keeps Firecrawl scraping optional (`research.firecrawl.enabled`) and disabled by default, and
- returns a structured `research_packet_v1` summary with extracted facts/sources while persisting per-request transcripts under the current run directory.

See [docs/web-research.md](docs/web-research.md) and [`config/default.yaml`](config/default.yaml) for configuration, defaults, behavior, and operations notes.

## Systemd Deployment

See [docs/systemd-setup.md](docs/systemd-setup.md) and the template unit in
[systemd/pyPiBot.service](systemd/pyPiBot.service) for Raspberry Pi deployment
instructions, including best-effort pre-start git sync logging and the
logrotate template at `ops/logrotate/pypibot`.

For systemd deployments that use Firecrawl, operators can manage
`FIRECRAWL_API_KEY` with `./scripts/update-firecrawl-key.sh` before restarting
the service.

## Diagnostics & Tests

Run the diagnostics suite (offline uses fake hardware backends where available):

```bash
python -m diagnostics.run --offline
```

To run diagnostics against live hardware and configured services:

```bash
python -m diagnostics.run
```

Unit tests are executed with pytest:

```bash
pytest -q
```

## Documentation

- [docs/architecture/README.md](docs/architecture/README.md)
- [docs/agent-map.md](docs/agent-map.md)
- [docs/architecture/theo_cognitive_stack.md](docs/architecture/theo_cognitive_stack.md)
- [docs/coding-standards.md](docs/coding-standards.md)
- [docs/hardware-setup.md](docs/hardware-setup.md)
- [docs/installation.md](docs/installation.md)
- [docs/interaction-states.md](docs/interaction-states.md)
- [docs/operations.md](docs/operations.md)
- [docs/personalization.md](docs/personalization.md)
- [docs/requirements.md](docs/requirements.md)
- [docs/systemd-setup.md](docs/systemd-setup.md)
- [docs/todo.md](docs/todo.md)
- [docs/web-research.md](docs/web-research.md)
