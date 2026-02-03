"""Diagnostics routines for audio input."""

from __future__ import annotations

import importlib
import importlib.util

from config import ConfigController
from core.logging import logger
from diagnostics.models import DiagnosticResult, DiagnosticStatus
from interaction.microphone_hal import AudioInputBackend
from interaction.utils import CHANNELS, CHUNK, RATE, resolve_device_index, resolve_format


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
        config = ConfigController.get_instance().get_config()
        audio_cfg = config.get("audio") or {}
        input_cfg = audio_cfg.get("input") or {}
        input_device_name = input_cfg.get("device_name")
        if not input_device_name:
            return DiagnosticResult(
                name=name,
                status=DiagnosticStatus.FAIL,
                details="Audio input device name not configured",
            )
        pyaudio = importlib.import_module("pyaudio")
        audio = pyaudio.PyAudio()
        try:
            input_device_index = resolve_device_index(
                audio,
                input_device_name,
                require_input=True,
            )
            device_info = audio.get_device_info_by_index(input_device_index)
            stream = audio.open(
                format=resolve_format(),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK,
                start=False,
            )
            stream.close()
        finally:
            audio.terminate()

        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.PASS,
            details=f"Input device: {device_info.get('name')}",
        )
    except Exception as exc:  # noqa: BLE001 - probe should not raise
        try:
            _log_devices(require_input=True)
        except Exception:
            logger.exception("Failed to list input devices after audio probe error")
        return DiagnosticResult(
            name=name,
            status=DiagnosticStatus.FAIL,
            details=f"Audio input probe failed: {exc}",
        )


def _log_devices(*, require_input: bool = False) -> None:
    if importlib.util.find_spec("pyaudio") is None:
        return
    pyaudio = importlib.import_module("pyaudio")
    audio = pyaudio.PyAudio()
    try:
        logger.info(
            "[MIC DIAG] Listing devices (input=%s)",
            require_input,
        )
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if require_input and info.get("maxInputChannels", 0) <= 0:
                continue
            logger.info(
                "[MIC DIAG] Device %s: %s | Input Channels: %s",
                i,
                info.get("name"),
                info.get("maxInputChannels"),
            )
    finally:
        audio.terminate()
