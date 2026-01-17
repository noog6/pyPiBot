"""Async microphone input helper."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import queue
from typing import Any

from interaction.utils import CHANNELS, CHUNK, RATE, resolve_format


class AsyncMicrophone:
    """Asynchronous microphone capture with internal buffering."""

    def __init__(
        self,
        input_device_index: int | None = None,
        input_name_hint: str | None = None,
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
            logging.info("[ASYNC MIC] Listing input devices:")
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                logging.info(
                    "[ASYNC MIC] Device %s: %s | Input Channels: %s",
                    i,
                    info.get("name"),
                    info.get("maxInputChannels"),
                )
            logging.info("[ASYNC MIC] Completed device list")

        if input_device_index is None and input_name_hint:
            input_device_index = self._find_device_index(
                name_hint=input_name_hint,
                require_input=True,
            )

        self.stream = self.p.open(
            format=audio_format,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK,
            stream_callback=self.callback,
        )
        self.queue: queue.Queue[bytes] = queue.Queue(maxsize=50)
        self.is_recording = False
        self.is_receiving = False
        logging.info("AsyncMicrophone initialized")

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

    def _find_device_index(
        self,
        name_hint: str,
        require_input: bool = False,
        require_output: bool = False,
    ) -> int:
        name_hint = name_hint.lower()
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            name = info.get("name", "").lower()
            if name_hint in name:
                if require_input and info.get("maxInputChannels", 0) <= 0:
                    continue
                if require_output and info.get("maxOutputChannels", 0) <= 0:
                    continue
                return i
        raise RuntimeError(f"No device matching '{name_hint}' found")

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
        logging.info("Started recording")

    def stop_recording(self) -> None:
        """Disable recording."""

        self.is_recording = False
        logging.info("Stopped recording")

    def start_receiving(self) -> None:
        """Mark assistant response playback in progress."""

        self.is_receiving = True
        self.is_recording = False
        logging.info("Started receiving assistant response")

    def stop_receiving(self) -> None:
        """Mark assistant response playback complete."""

        self.is_receiving = False
        logging.info("Stopped receiving assistant response")

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
        logging.info("AsyncMicrophone closed")
