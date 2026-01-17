# pyPiBot

Theo is a Raspberry Pi robot assistant that combines AI interaction, hardware
control, and user-facing I/O in a clean, extensible Python codebase.

## Current Focus

- Target hardware: Raspberry Pi Zero W and Raspberry Pi Zero 2.
- Initial AI provider: OpenAI realtime models.
- Deployment target: systemd-managed service with git-based updates.

## Project Layout

```
pyPiBot/
├── ai/            # AI provider integrations (realtime API + tools)
├── config/        # YAML configuration controller + defaults
├── core/          # Runtime orchestration and app lifecycle
├── hardware/      # GPIO, sensors, and actuator drivers
├── interaction/   # Audio input/output and user interaction helpers
├── motion/        # Motion controller and keyframe sequencing
├── services/      # External services/integrations (placeholder)
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

See `docs/installation.md` for recommended `apt-get` packages and optional
virtualenv setup.

### Run the Runtime

```bash
python main.py --prompts "Say Hello World!"
```

The runtime will:
- Load configuration from `config/default.yaml`.
- Initialize the storage layer and log run metadata.
- Attempt to start audio input/output (gracefully degrades if unavailable).
- Start the motion controller and camera vision loop when hardware is present.

## Configuration

Configuration is stored in YAML under `config/`:
- `config/default.yaml` for baseline settings
- `config/override.yaml` for runtime updates (auto-archived on updates)

Default keys include:
- `log_dir` and `var_dir` for storage
- `assistant_name` and `startup_prompts`
- `logging_level`

## Systemd Deployment

See `docs/systemd-setup.md` and the template unit in `systemd/pyPiBot.service`
for Raspberry Pi deployment instructions.

## Documentation

- `docs/coding-standards.md`
- `docs/requirements.md`
- `docs/systemd-setup.md`
- `docs/installation.md`

## Next Steps

- Expand hardware abstractions and mock drivers.
- Build user interaction pipelines (speech/text -> intent -> action).
- Add more external integrations in `services/`.
