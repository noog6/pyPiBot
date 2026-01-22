"""Thin audio HAL for offline diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class AudioOutputBackend(Protocol):
    """Minimal audio output backend interface for offline probes."""

    def list_output_devices(self) -> list[str]:
        """Return available output device names."""

    def open_output_stream(self) -> None:
        """Open and close a simple output stream."""


@dataclass
class FakeAudioBackend:
    """Fake audio backend for offline diagnostics."""

    devices: list[str] = field(default_factory=lambda: ["offline-speaker"])
    can_open: bool = True

    def list_output_devices(self) -> list[str]:
        """Return configured fake devices."""

        return list(self.devices)

    def open_output_stream(self) -> None:
        """Simulate opening an output stream."""

        if not self.can_open:
            raise RuntimeError("Failed to open fake audio stream")
