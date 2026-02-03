"""Audio playback helper."""

from __future__ import annotations

import importlib
import importlib.util
import queue
import threading
import time
from typing import Any

from core.logging import logger
from interaction.utils import CHANNELS, resolve_device_index, resolve_format


INPUT_RATE = 24000
OUTPUT_RATE = 48000
FRAMES_PER_BUFFER = 16384


class AudioPlayer:
    """Audio playback controller with a background worker."""

    def __init__(
        self,
        on_playback_complete: Any | None = None,
        output_device_name: str | None = None,
    ) -> None:
        if importlib.util.find_spec("pyaudio") is None:
            raise RuntimeError("PyAudio is required for AudioPlayer")

        pyaudio = importlib.import_module("pyaudio")
        audioop_module_name = None
        if importlib.util.find_spec("audioop") is not None:
            audioop_module_name = "audioop"
        elif importlib.util.find_spec("audioop_lts") is not None:
            audioop_module_name = "audioop_lts"

        if audioop_module_name is None:
            raise RuntimeError("audioop or audioop-lts is required for AudioPlayer")

        self._audioop = importlib.import_module(audioop_module_name)
        self.on_playback_complete = on_playback_complete
        self.p = pyaudio.PyAudio()
        audio_format = resolve_format()

        if not output_device_name:
            raise RuntimeError(
                "Audio output device name is required. Set audio.output.device_name"
            )

        output_device_index = resolve_device_index(
            self.p,
            output_device_name,
            require_output=True,
        )
        try:
            info = self.p.get_device_info_by_index(output_device_index)
            logger.info(
                "[AUDIO] Output device (selected): %s idx=%s defaultRate=%s",
                info.get("name"),
                info.get("index"),
                info.get("defaultSampleRate"),
            )
        except Exception:
            logger.info(
                "[AUDIO] Output device selected: %s (idx=%s)",
                output_device_name,
                output_device_index,
            )

        try:
            self.stream = self.p.open(
                format=audio_format,
                channels=CHANNELS,
                rate=OUTPUT_RATE,
                output=True,
                output_device_index=output_device_index,
                frames_per_buffer=FRAMES_PER_BUFFER,
                start=True,
            )
        except Exception as exc:
            self._log_devices(require_output=True)
            raise RuntimeError(
                "Failed to open output audio device "
                f"'{output_device_name}' (idx={output_device_index})"
            ) from exc

        self._q: queue.Queue[bytes | None] = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending = 0
        self._response_closed = False
        self._ratecv_state = None

        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def _log_devices(
        self,
        *,
        require_input: bool = False,
        require_output: bool = False,
    ) -> None:
        logger.info(
            "[AUDIO] Listing audio devices (input=%s output=%s)",
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
                "[AUDIO] Device %s: %s | Input Channels: %s | Output Channels: %s",
                i,
                info.get("name"),
                info.get("maxInputChannels"),
                info.get("maxOutputChannels"),
            )

    def start_response(self) -> None:
        """Call at response.created (or when you begin accepting audio)."""

        self.flush()
        with self._lock:
            self._pending = 0
            self._response_closed = False
            self._ratecv_state = None

    def close_response(self) -> None:
        """Call at response.output_audio.done."""

        with self._lock:
            self._response_closed = True
        self._maybe_fire_complete()

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    audio_data = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if audio_data is None:
                    break

                out_data, self._ratecv_state = self._audioop.ratecv(
                    audio_data,
                    2,
                    1,
                    INPUT_RATE,
                    OUTPUT_RATE,
                    self._ratecv_state,
                )

                chunk_bytes = 16384
                for i in range(0, len(out_data), chunk_bytes):
                    self.stream.write(out_data[i : i + chunk_bytes])

                with self._lock:
                    self._pending = max(0, self._pending - 1)

                self._maybe_fire_complete()

        except Exception:
            logger.exception("Audio output worker crashed")

    def _maybe_fire_complete(self) -> None:
        cb = None
        with self._lock:
            if self._response_closed and self._pending == 0:
                cb = self.on_playback_complete
                self._response_closed = False

        if cb:
            try:
                try:
                    out_lat = float(getattr(self.stream, "get_output_latency", lambda: 0.0)())
                except Exception:
                    out_lat = 0.0

                buffer_secs = FRAMES_PER_BUFFER / OUTPUT_RATE
                time.sleep(out_lat + buffer_secs)

                cb()
            except Exception:
                logger.exception("on_playback_complete callback failed")

    def play_audio(self, audio_data: bytes) -> None:
        """Enqueue audio data for playback."""

        with self._lock:
            self._pending += 1

        try:
            self._q.put_nowait(audio_data)
        except queue.Full:
            with self._lock:
                self._pending -= 1
            logger.warning("Audio queue full; dropping audio")
            self._maybe_fire_complete()

    def flush(self) -> None:
        """Clear queued audio data."""

        removed = 0
        try:
            while True:
                self._q.get_nowait()
                removed += 1
        except queue.Empty:
            pass

        with self._lock:
            self._pending = max(0, self._pending - removed)
            self._ratecv_state = None

    def close(self) -> None:
        """Close the audio output stream."""

        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._t.join(timeout=1.0)
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.p.terminate()
