"""Audio utility constants."""

from __future__ import annotations

import importlib
import importlib.util


CHANNELS = 1
RATE = 24000
CHUNK = 2048


def resolve_format() -> int:
    """Resolve the PyAudio format constant."""

    if importlib.util.find_spec("pyaudio") is None:
        raise RuntimeError("PyAudio is required for audio IO")

    pyaudio = importlib.import_module("pyaudio")
    return pyaudio.paInt16
