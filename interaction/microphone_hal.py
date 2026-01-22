"""Thin microphone HAL for offline diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class AudioInputBackend(Protocol):
    """Minimal audio input backend interface for offline probes."""

    def list_input_devices(self) -> list[str]:
        """Return available input device names."""

    def open_input_stream(self) -> None:
        """Open and close a simple input stream."""


@dataclass
class FakeInputBackend:
    """Fake audio input backend for offline diagnostics."""

    devices: list[str] = field(default_factory=lambda: ["offline-mic"])
    can_open: bool = True

    def list_input_devices(self) -> list[str]:
        """Return configured fake input devices."""

        return list(self.devices)

    def open_input_stream(self) -> None:
        """Simulate opening an input stream."""

        if not self.can_open:
            raise RuntimeError("Failed to open fake input stream")
