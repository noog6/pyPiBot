# Project Requirements (Initial)

This document captures an initial, high-level set of requirements for the project.
These will evolve as we refine scope and integrate existing code.

## Functional Requirements

1. **AI Interaction**
   - Support a pluggable AI interface for local and/or remote inference.
   - Provide a consistent prompt/response pipeline for higher-level behaviors.
   - Allow configuration of AI providers and models via configuration files.

2. **Hardware Control (Raspberry Pi)**
   - Provide a hardware abstraction layer for GPIO, sensors, and actuators.
   - Support mock drivers for development and testing without hardware.
   - Expose safe operations for hardware actions (failsafes for critical actions).

3. **User Interaction**
   - Support multiple input modalities (text, speech, sensor triggers).
   - Support multiple output modalities (speech, LEDs, displays).
   - Provide a clear command/intent interface to map AI outputs to actions.

4. **System Integration**
   - Central configuration system for environment, hardware, and AI settings.
   - Standard logging and event tracing for troubleshooting.
   - Health checks and status reporting.

5. **Extensibility**
   - Allow adding new hardware drivers without core changes.
   - Allow adding new AI providers without core changes.
   - Provide a plugin or adapter pattern for new capabilities.

## Non-Functional Requirements

1. **Reliability**
   - Deterministic behavior for critical hardware actions.
   - Graceful degradation when AI providers or peripherals are unavailable.

2. **Security & Safety**
   - Validate external input.
   - Avoid executing shell commands with untrusted input.
   - Implement safeguards for destructive hardware actions.

3. **Performance**
   - Low-latency responses for local interactions.
   - Efficient use of Raspberry Pi resources.

4. **Maintainability**
   - Strong typing and linting enforced in CI.
   - Clear module boundaries and documentation.
   - Test coverage for core logic.

5. **Portability**
   - Linux-first support (Raspberry Pi OS).
   - Minimal external dependencies where practical.

## Current Decisions

- **Target hardware:** Raspberry Pi Zero W and Raspberry Pi Zero 2.
- **Initial AI provider:** OpenAI realtime models.
- **Offline behavior:** The system should continue operating in a degraded mode if
  network connectivity is lost (no hard failures).
- **Deployment approach (target):** systemd-managed service.
- **Deployment approach (current):** Git-based updates on device.

## Open Questions

- Should we provide a helper script/installer for systemd unit setup to simplify
  deployments?
