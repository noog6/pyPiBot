"""Diagnostics routines for the audio subsystem."""

from __future__ import annotations

import importlib
import importlib.util

from config import ConfigController
from core.logging import logger
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
        config = ConfigController.get_instance().get_config()
        audio_cfg = config.get("audio") or {}
        output_cfg = audio_cfg.get("output") or {}
        output_device_index = output_cfg.get("device_index")
        if output_device_index is None:
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.FAIL,
                details="Audio output device index not configured",
            )
        pyaudio = importlib.import_module("pyaudio")
        audio = pyaudio.PyAudio()
        try:
            device_info = audio.get_device_info_by_index(int(output_device_index))
            stream = audio.open(
                format=resolve_format(),
                channels=CHANNELS,
                rate=OUTPUT_RATE,
                output=True,
                output_device_index=int(output_device_index),
                frames_per_buffer=FRAMES_PER_BUFFER,
                start=False,
            )
            stream.close()
        finally:
            audio.terminate()

        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.PASS,
            details=f"Output device: {device_info.get('name')}",
        )
    except Exception as exc:  # noqa: BLE001 - probe should not raise
        try:
            _log_devices(require_output=True)
        except Exception:
            logger.exception("Failed to list output devices after audio probe error")
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details=f"Audio output probe failed: {exc}",
        )


def _log_devices(*, require_output: bool = False) -> None:
    if importlib.util.find_spec("pyaudio") is None:
        return
    pyaudio = importlib.import_module("pyaudio")
    audio = pyaudio.PyAudio()
    try:
        logger.info(
            "[AUDIO DIAG] Listing devices (output=%s)",
            require_output,
        )
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if require_output and info.get("maxOutputChannels", 0) <= 0:
                continue
            logger.info(
                "[AUDIO DIAG] Device %s: %s | Output Channels: %s",
                i,
                info.get("name"),
                info.get("maxOutputChannels"),
            )
    finally:
        audio.terminate()
