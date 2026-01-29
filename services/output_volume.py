"""ALSA-backed output volume control."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass


_VOLUME_PATTERN = re.compile(r"\[(\d+)%\]")


@dataclass
class VolumeStatus:
    """Current volume status from ALSA."""

    percent: int
    muted: bool


class OutputVolumeController:
    """Authoritative output volume controller using ALSA amixer."""

    _instance: "OutputVolumeController | None" = None

    def __init__(self, mixer: str = "Master", device: str = "default") -> None:
        self._mixer = mixer
        self._device = device
        self._last_set_time = 0.0

    @classmethod
    def get_instance(cls) -> "OutputVolumeController":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_volume(self) -> VolumeStatus:
        output = self._run_amixer(["sget", self._mixer])
        percent = self._parse_percent(output)
        muted = self._parse_muted(output)
        return VolumeStatus(percent=percent, muted=muted)

    def set_volume(
        self,
        percent: int,
        *,
        emergency: bool = False,
        min_percent: int = 1,
        max_percent: int = 100,
        rate_limit_s: float = 1.0,
    ) -> VolumeStatus:
        if percent < min_percent or percent > max_percent:
            raise ValueError(
                f"Volume percent must be within {min_percent}-{max_percent}, got {percent}."
            )
        now = time.monotonic()
        elapsed = now - self._last_set_time
        if not emergency and elapsed < rate_limit_s:
            raise RuntimeError(
                f"Volume change rate-limited. Retry after {rate_limit_s - elapsed:.2f}s."
            )
        self._run_amixer(["sset", self._mixer, f"{percent}%"])
        self._last_set_time = now
        return self.get_volume()

    def _run_amixer(self, args: list[str]) -> str:
        cmd = ["amixer", "-D", self._device]
        cmd.extend(args)
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout

    def _parse_percent(self, output: str) -> int:
        match = _VOLUME_PATTERN.search(output)
        if not match:
            raise RuntimeError("Unable to parse volume percentage from amixer output.")
        return int(match.group(1))

    def _parse_muted(self, output: str) -> bool:
        return "[off]" in output.lower()
