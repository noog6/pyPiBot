"""Diagnostics routines for audio input."""

from __future__ import annotations

import importlib
import importlib.util

from diagnostics.models import DiagnosticResult, DiagnosticStatus
from interaction.microphone_hal import AudioInputBackend
from interaction.utils import CHANNELS, CHUNK, RATE, resolve_format


def probe(backend: AudioInputBackend | None = None) -> DiagnosticResult:
    """Run an audio input probe to validate microphone availability.

    Args:
        backend: Optional offline audio input backend for testing.

    Returns:
        Diagnostic result indicating audio input readiness.
    """

    name = "audio_input"

    if backend is not None:
        try:
            devices = backend.list_input_devices()
            if not devices:
                return DiagnosticResult(
                    name=name,
                    status=DiagnosticStatus.WARN,
                    details="No offline input devices configured",
                )

            backend.open_input_stream()
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.PASS,
                details=f"Offline input devices: {', '.join(devices)}",
            )
        except Exception as exc:  # noqa: BLE001 - probe should not raise
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.FAIL,
                details=f"Offline audio input probe failed: {exc}",
            )

    if importlib.util.find_spec("pyaudio") is None:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="PyAudio is not installed",
        )

    if importlib.util.find_spec("numpy") is None:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="NumPy is not installed",
        )

    try:
        pyaudio = importlib.import_module("pyaudio")
        audio = pyaudio.PyAudio()
        try:
            device_info = audio.get_default_input_device_info()
            stream = audio.open(
                format=resolve_format(),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_info["index"],
                frames_per_buffer=CHUNK,
                start=False,
            )
            stream.close()
        finally:
            audio.terminate()

        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.PASS,
            details=f"Default input device: {device_info.get('name')}",
        )
    except Exception as exc:  # noqa: BLE001 - probe should not raise
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details=f"Audio input probe failed: {exc}",
        )
