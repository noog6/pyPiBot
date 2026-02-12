from vision.attention import AttentionConfig, AttentionController, AttentionState
from vision.detections import Detection, DetectionEvent


def _event(timestamp_ms: int, label: str = "person", confidence: float = 0.9) -> DetectionEvent:
    return DetectionEvent(
        timestamp_ms=timestamp_ms,
        detections=[Detection(label=label, confidence=confidence, bbox=(0.0, 0.0, 1.0, 1.0))],
    )


def _controller(config: AttentionConfig) -> AttentionController:
    AttentionController._instance = None
    controller = AttentionController()
    controller.config = config
    controller._state = AttentionState.IDLE
    controller._state_since_ms = 0
    controller._mad_hits_ms.clear()
    controller._burst_armed = True
    return controller


def test_idle_to_curious_on_repeated_mad() -> None:
    controller = _controller(
        AttentionConfig(mad_repeat_count=2, mad_window_ms=500, engage_confirm_ms=5000)
    )

    assert controller.update(1000, mad_changed=True, detections=None) is AttentionState.IDLE
    assert controller.update(1200, mad_changed=True, detections=None) is AttentionState.CURIOUS


def test_curious_to_engaged_on_persistent_detection() -> None:
    controller = _controller(AttentionConfig(engage_confirm_ms=300))

    assert controller.update(1000, mad_changed=False, detections=_event(1000)) is AttentionState.CURIOUS
    assert controller.update(1250, mad_changed=False, detections=_event(1250)) is AttentionState.CURIOUS
    assert controller.update(1350, mad_changed=False, detections=_event(1350)) is AttentionState.ENGAGED


def test_engaged_to_cooldown_to_idle() -> None:
    controller = _controller(AttentionConfig(engage_confirm_ms=0, cooldown_timeout_ms=400))

    controller.update(1000, mad_changed=False, detections=_event(1000))
    assert controller.update(1001, mad_changed=False, detections=_event(1001)) is AttentionState.ENGAGED
    assert controller.update(1200, mad_changed=False, detections=None) is AttentionState.COOLDOWN
    assert controller.update(1700, mad_changed=False, detections=None) is AttentionState.IDLE


def test_cooldown_immediate_reengage() -> None:
    controller = _controller(
        AttentionConfig(engage_confirm_ms=0, cooldown_reengage_window_ms=600, cooldown_timeout_ms=2000)
    )

    controller.update(1000, mad_changed=False, detections=_event(1000))
    controller.update(1001, mad_changed=False, detections=_event(1001))
    assert controller.update(1300, mad_changed=False, detections=None) is AttentionState.COOLDOWN
    assert controller.update(1600, mad_changed=False, detections=_event(1600)) is AttentionState.ENGAGED


def test_disabled_mode_passthrough_behavior() -> None:
    controller = _controller(AttentionConfig(enabled=False))

    assert controller.update(1000, mad_changed=True, detections=_event(1000)) is AttentionState.IDLE
    assert controller.get_capture_period_ms(AttentionState.ENGAGED, 5000) == 5000
    assert controller.should_send_image(AttentionState.IDLE, mad_changed=False, detections=None) is False
    assert controller.should_send_image(AttentionState.IDLE, mad_changed=True, detections=None) is True
