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

See [docs/installation.md](docs/installation.md) for Raspberry Pi OS setup
steps, git clone details, and the environment setup script. Hardware assembly
notes live in [docs/hardware-setup.md](docs/hardware-setup.md).

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
- `stop_words` and `stop_word_cooldown_s` to pause tool execution on emergency phrases
- `health`, `ops`, and `alerts` for operational health probes, budgets, and alert policy
- `governance` settings (autonomy level, autonomy windows, budgets, and tool tier specs)
- `imx500_enabled`, `imx500_model`, `imx500_fps_cap`, `imx500_min_confidence`, `imx500_event_buffer_size`, and `imx500_interesting_classes` for optional IMX500 object-detection settings

IMX500 controller skeleton lives at `hardware/imx500_controller.py`. It is disabled by default and will no-op with a single warning when dependencies are unavailable. Detection payload types are defined in `vision/detections.py`: `Detection` and `DetectionEvent` with normalized `(x, y, w, h)` bounding boxes in `[0..1]`. The controller keeps a ring buffer of time-ordered events and exposes `get_latest_event()` / `get_recent_events()` for non-blocking reads.

When `save_camera_images: true`, camera captures are still written asynchronously; if an IMX500 event is available, a sibling JSON sidecar is also written (for example `image_*.jpg` with `image_*.detections.json`).

## Systemd Deployment

See [docs/systemd-setup.md](docs/systemd-setup.md) and the template unit in
[systemd/pyPiBot.service](systemd/pyPiBot.service) for Raspberry Pi deployment
instructions.

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

- [docs/agent-map.md](docs/agent-map.md)
- [docs/coding-standards.md](docs/coding-standards.md)
- [docs/hardware-setup.md](docs/hardware-setup.md)
- [docs/installation.md](docs/installation.md)
- [docs/interaction-states.md](docs/interaction-states.md)
- [docs/operations.md](docs/operations.md)
- [docs/personalization.md](docs/personalization.md)
- [docs/requirements.md](docs/requirements.md)
- [docs/systemd-setup.md](docs/systemd-setup.md)
- [docs/todo.md](docs/todo.md)
