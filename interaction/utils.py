"""Audio utility constants."""

from __future__ import annotations

import importlib
import importlib.util


CHANNELS = 1
RATE = 24000
CHUNK = 2048


def resolve_device_index(
    audio: object,
    device_name: str | None,
    *,
    require_input: bool = False,
    require_output: bool = False,
) -> int:
    """Resolve an audio device index by exact device name."""

    if not device_name:
        raise RuntimeError("Audio device name is required")

    get_count = getattr(audio, "get_device_count")
    get_info = getattr(audio, "get_device_info_by_index")
    for i in range(get_count()):
        info = get_info(i)
        if require_input and info.get("maxInputChannels", 0) <= 0:
            continue
        if require_output and info.get("maxOutputChannels", 0) <= 0:
            continue
        if info.get("name") == device_name:
            return int(info.get("index", i))

    raise RuntimeError(f"Audio device named '{device_name}' not found")


def resolve_format() -> int:
    """Resolve the PyAudio format constant."""

    if importlib.util.find_spec("pyaudio") is None:
        raise RuntimeError("PyAudio is required for audio IO")

    pyaudio = importlib.import_module("pyaudio")
    return pyaudio.paInt16
