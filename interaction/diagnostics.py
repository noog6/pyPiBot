"""Diagnostics routines for the audio subsystem."""

from __future__ import annotations

import importlib
import importlib.util

from diagnostics.models import DiagnosticResult, DiagnosticStatus
from interaction.audio import FRAMES_PER_BUFFER, OUTPUT_RATE
from interaction.audio_hal import AudioOutputBackend
from interaction.utils import CHANNELS, resolve_format


def probe(backend: AudioOutputBackend | None = None) -> DiagnosticResult:
    """Run an audio probe to validate output availability.

    Args:
        backend: Optional offline audio backend for testing.

    Returns:
        Diagnostic result indicating audio output readiness.
    """

    name = "audio_output"

    if backend is not None:
        try:
            devices = backend.list_output_devices()
            if not devices:
                return DiagnosticResult(
                    name=name,
                    status=DiagnosticStatus.WARN,
                    details="No offline audio devices configured",
                )

            backend.open_output_stream()
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.PASS,
                details=f"Offline audio devices: {', '.join(devices)}",
            )
        except Exception as exc:  # noqa: BLE001 - probe should not raise
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.FAIL,
                details=f"Offline audio probe failed: {exc}",
            )

    if importlib.util.find_spec("pyaudio") is None:
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details="PyAudio is not installed",
        )

    try:
        pyaudio = importlib.import_module("pyaudio")
        audio = pyaudio.PyAudio()
        try:
            device_info = audio.get_default_output_device_info()
            stream = audio.open(
                format=resolve_format(),
                channels=CHANNELS,
                rate=OUTPUT_RATE,
                output=True,
                output_device_index=device_info["index"],
                frames_per_buffer=FRAMES_PER_BUFFER,
                start=False,
            )
            stream.close()
        finally:
            audio.terminate()

        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.PASS,
            details=f"Default output device: {device_info.get('name')}",
        )
    except Exception as exc:  # noqa: BLE001 - probe should not raise
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details=f"Audio output probe failed: {exc}",
        )
