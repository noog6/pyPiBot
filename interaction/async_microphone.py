"""Async microphone input helper."""

from __future__ import annotations

import importlib
import importlib.util
import queue
from typing import Any

from core.logging import logger
from interaction.utils import CHANNELS, CHUNK, RATE, resolve_device_index, resolve_format


class AsyncMicrophone:
    """Asynchronous microphone capture with internal buffering."""

    def __init__(
        self,
        input_device_name: str | None = None,
        debug_list_devices: bool = False,
    ) -> None:
        if importlib.util.find_spec("pyaudio") is None:
            raise RuntimeError("PyAudio is required for AsyncMicrophone")

        if importlib.util.find_spec("numpy") is None:
            raise RuntimeError("NumPy is required for AsyncMicrophone")

        pyaudio = importlib.import_module("pyaudio")
        self._np = importlib.import_module("numpy")
        self._pa_continue = pyaudio.paContinue
        audio_format = resolve_format()

        self.p = pyaudio.PyAudio()

        if debug_list_devices:
            self._log_devices(require_input=True)

        if not input_device_name:
            raise RuntimeError(
                "Audio input device name is required. Set audio.input.device_name"
            )

        input_device_index = resolve_device_index(
            self.p,
            input_device_name,
            require_input=True,
        )
        try:
            info = self.p.get_device_info_by_index(input_device_index)
            logger.info(
                "[ASYNC MIC] Input device (selected): %s idx=%s defaultRate=%s",
                info.get("name"),
                info.get("index"),
                info.get("defaultSampleRate"),
            )
        except Exception:
            logger.info(
                "[ASYNC MIC] Input device selected: %s (idx=%s)",
                input_device_name,
                input_device_index,
            )

        try:
            self.stream = self.p.open(
                format=audio_format,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=CHUNK,
                stream_callback=self.callback,
            )
        except Exception as exc:
            self._log_devices(require_input=True)
            raise RuntimeError(
                "Failed to open input audio device "
                f"'{input_device_name}' (idx={input_device_index})"
            ) from exc
        self.queue: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self.is_recording = False
        self.is_receiving = False
        logger.info("AsyncMicrophone initialized")

    def callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: dict[str, Any],
        status: int,
    ) -> tuple[None, int]:
        """Callback for the PyAudio stream."""

        if self.is_recording and not self.is_receiving:
            try:
                self.queue.put_nowait(in_data)
            except queue.Full:
                pass
        return (None, self._pa_continue)

    def _log_devices(
        self,
        *,
        require_input: bool = False,
        require_output: bool = False,
    ) -> None:
        logger.info(
            "[ASYNC MIC] Listing audio devices (input=%s output=%s)",
            require_input,
            require_output,
        )
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if require_input and info.get("maxInputChannels", 0) <= 0:
                continue
            if require_output and info.get("maxOutputChannels", 0) <= 0:
                continue
            logger.info(
                "[ASYNC MIC] Device %s: %s | Input Channels: %s | Output Channels: %s",
                i,
                info.get("name"),
                info.get("maxInputChannels"),
                info.get("maxOutputChannels"),
            )

    def rms_numpy(self, audio_bytes: bytes, sample_width: int = 2) -> float:
        """Compute RMS (Root Mean Square) volume level with NumPy."""

        dtype_map = {1: self._np.int8, 2: self._np.int16, 4: self._np.int32}
        dtype = dtype_map.get(sample_width, self._np.int16)

        audio_array = self._np.frombuffer(audio_bytes, dtype=dtype)
        rms_value = self._np.sqrt(self._np.mean(audio_array**2))

        return float(rms_value)

    def start_recording(self) -> None:
        """Enable recording."""

        self.is_recording = True
        logger.info("Started recording")

    def stop_recording(self) -> None:
        """Disable recording."""

        self.is_recording = False
        logger.info("Stopped recording")

    def start_receiving(self) -> None:
        """Mark assistant response playback in progress."""

        self.is_receiving = True
        self.is_recording = False
        logger.info("Started receiving assistant response")

    def stop_receiving(self) -> None:
        """Mark assistant response playback complete."""

        self.is_receiving = False
        logger.info("Stopped receiving assistant response")

    def get_audio_data(self) -> bytes | None:
        """Drain queued audio and return concatenated bytes."""

        chunks: list[bytes] = []
        while True:
            try:
                chunks.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return b"".join(chunks) if chunks else None

    def drain_queue(self, max_items: int = 9999) -> int:
        """Remove queued audio chunks."""

        removed = 0
        while removed < max_items:
            try:
                self.queue.get_nowait()
                removed += 1
            except queue.Empty:
                break
        return removed

    def close(self) -> None:
        """Close the audio stream."""

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        logger.info("AsyncMicrophone closed")
