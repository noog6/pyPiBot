"""Attention state machine for vision-triggered camera behavior."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from core.logging import logger
from vision.detections import DetectionEvent


class AttentionState(Enum):
    """High-level attention modes for Theo's capture loop."""

    IDLE = "idle"
    CURIOUS = "curious"
    ENGAGED = "engaged"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class AttentionConfig:
    """Configuration values for the attention state machine."""

    enabled: bool = True
    curious_timeout_ms: int = 3000
    engage_confirm_ms: int = 1200
    cooldown_timeout_ms: int = 3500
    cooldown_reengage_window_ms: int = 1500
    mad_repeat_count: int = 2
    mad_window_ms: int = 1200
    engaged_capture_period_ms: int = 1500
    burst_enabled: bool = True
    burst_count: int = 3
    burst_cooldown_ms: int = 2500
    interesting_labels: tuple[str, ...] = ("person",)
    min_confidence: float = 0.45


class AttentionController:
    """Singleton attention controller driven by MAD and detection events."""

    _instance: "AttentionController | None" = None

    def __init__(self) -> None:
        if AttentionController._instance is not None:
            raise RuntimeError("You cannot create another AttentionController class")
        self.config = self._load_config()
        self._state = AttentionState.IDLE
        self._state_since_ms = 0
        self._last_interesting_ms = 0
        self._curious_started_ms = 0
        self._mad_hits_ms: deque[int] = deque(maxlen=max(self.config.mad_repeat_count * 2, 4))
        self._last_burst_ms = 0
        self._burst_armed = True
        AttentionController._instance = self

    @classmethod
    def get_instance(cls) -> "AttentionController":
        """Return singleton attention controller."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_state(self) -> AttentionState:
        """Return current attention state."""

        return self._state

    def is_interesting_event(self, detections: DetectionEvent | None) -> bool:
        """Return whether detections include an interesting label above threshold."""

        if detections is None:
            return False
        labels = {item.lower() for item in self.config.interesting_labels}
        for detection in detections.detections:
            if (
                detection.label.lower() in labels
                and float(detection.confidence) >= self.config.min_confidence
            ):
                return True
        return False

    def update(
        self,
        now_ms: int,
        mad_changed: bool,
        detections: DetectionEvent | None,
    ) -> AttentionState:
        """Update attention state using current signals."""

        if not self.config.enabled:
            self._state = AttentionState.IDLE
            return self._state

        if self._state_since_ms == 0:
            self._state_since_ms = now_ms

        interesting = self.is_interesting_event(detections)
        if interesting:
            self._last_interesting_ms = now_ms
        if mad_changed:
            self._mad_hits_ms.append(now_ms)
        mad_repeated = self._has_repeated_mad(now_ms)

        state_before = self._state

        if self._state is AttentionState.IDLE:
            if interesting:
                self._curious_started_ms = now_ms
                self._transition(AttentionState.CURIOUS, now_ms, "interesting detection")
            elif mad_repeated:
                self._curious_started_ms = now_ms
                self._transition(AttentionState.CURIOUS, now_ms, "repeated lores MAD")

        elif self._state is AttentionState.CURIOUS:
            if interesting and (now_ms - self._curious_started_ms) >= self.config.engage_confirm_ms:
                self._transition(AttentionState.ENGAGED, now_ms, "interesting persisted")
            elif (not interesting) and (now_ms - self._state_since_ms) >= self.config.curious_timeout_ms:
                self._transition(AttentionState.IDLE, now_ms, "curious timeout")

        elif self._state is AttentionState.ENGAGED:
            if not interesting and not mad_changed:
                self._transition(AttentionState.COOLDOWN, now_ms, "signals quiet")

        elif self._state is AttentionState.COOLDOWN:
            quiet_for_ms = now_ms - self._state_since_ms
            if interesting and quiet_for_ms <= self.config.cooldown_reengage_window_ms:
                self._transition(AttentionState.ENGAGED, now_ms, "quick re-engage")
            elif quiet_for_ms >= self.config.cooldown_timeout_ms:
                self._transition(AttentionState.IDLE, now_ms, "cooldown elapsed")

        if state_before is not self._state:
            return self._state

        if mad_changed or interesting:
            logger.debug(
                "[ATTENTION] signals state=%s interesting=%s mad_changed=%s",
                self._state.value,
                interesting,
                mad_changed,
            )
        return self._state

    def should_send_image(
        self,
        state: AttentionState,
        mad_changed: bool,
        detections: DetectionEvent | None,
    ) -> bool:
        """Return whether capture/send should be encouraged for this state."""

        interesting = self.is_interesting_event(detections)
        if not self.config.enabled:
            return mad_changed or interesting
        if mad_changed or interesting:
            return True
        return state in (AttentionState.CURIOUS, AttentionState.ENGAGED)

    def get_capture_period_ms(self, state: AttentionState, base_period_ms: int) -> int:
        """Return effective loop period based on attention state."""

        if not self.config.enabled:
            return base_period_ms
        if state is AttentionState.ENGAGED:
            return max(100, self.config.engaged_capture_period_ms)
        return base_period_ms

    def should_burst(self, state: AttentionState) -> bool:
        """Return whether a burst should happen for current engaged state."""

        if not self.config.enabled or not self.config.burst_enabled:
            return False
        if state is not AttentionState.ENGAGED:
            self._burst_armed = True
            return False
        if not self._burst_armed:
            return False
        if (self._state_since_ms - self._last_burst_ms) < self.config.burst_cooldown_ms:
            return False
        self._burst_armed = False
        self._last_burst_ms = self._state_since_ms
        return True

    def get_burst_count(self) -> int:
        """Return number of frames to capture during bursts."""

        return max(1, self.config.burst_count)

    def _load_config(self) -> AttentionConfig:
        try:
            from config import ConfigController

            config = ConfigController.get_instance().get_config()
        except Exception:
            config = {}
        classes_value = config.get("attention_interesting_classes", ["person"])
        if isinstance(classes_value, list):
            labels = tuple(str(item) for item in classes_value)
        else:
            labels = AttentionConfig().interesting_labels

        return AttentionConfig(
            enabled=bool(config.get("attention_enabled", True)),
            curious_timeout_ms=int(config.get("attention_curious_timeout_ms", 3000)),
            engage_confirm_ms=int(config.get("attention_engage_confirm_ms", 1200)),
            cooldown_timeout_ms=int(config.get("attention_cooldown_timeout_ms", 3500)),
            cooldown_reengage_window_ms=int(
                config.get("attention_cooldown_reengage_window_ms", 1500)
            ),
            mad_repeat_count=int(config.get("attention_mad_repeat_count", 2)),
            mad_window_ms=int(config.get("attention_mad_window_ms", 1200)),
            engaged_capture_period_ms=int(
                config.get("attention_engaged_capture_period_ms", 1500)
            ),
            burst_enabled=bool(config.get("attention_burst_enabled", True)),
            burst_count=int(config.get("attention_burst_count", 3)),
            burst_cooldown_ms=int(config.get("attention_burst_cooldown_ms", 2500)),
            interesting_labels=labels,
            min_confidence=float(config.get("attention_min_confidence", 0.45)),
        )

    def _has_repeated_mad(self, now_ms: int) -> bool:
        window_start = now_ms - self.config.mad_window_ms
        while self._mad_hits_ms and self._mad_hits_ms[0] < window_start:
            self._mad_hits_ms.popleft()
        return len(self._mad_hits_ms) >= self.config.mad_repeat_count

    def _transition(self, new_state: AttentionState, now_ms: int, reason: str) -> None:
        old_state = self._state
        if old_state is new_state:
            return
        self._state = new_state
        self._state_since_ms = now_ms
        if new_state is not AttentionState.ENGAGED:
            self._burst_armed = True
        logger.info(
            "[ATTENTION] %s -> %s (%s)",
            old_state.value,
            new_state.value,
            reason,
        )
