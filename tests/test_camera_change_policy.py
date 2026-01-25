from hardware.camera_change_policy import CameraChangeConfig, CameraChangePolicy


def _collect_promotions(
    policy: CameraChangePolicy, values: list[float], start_s: float = 0.0, step_s: float = 1.0
) -> list[int]:
    promotions: list[int] = []
    now_s = start_s
    for idx, mad in enumerate(values):
        result = policy.update(mad, now_s)
        if result.promoted:
            promotions.append(idx)
        now_s += step_s
    return promotions


def test_noisy_sequence_promotes_at_most_once() -> None:
    policy = CameraChangePolicy(CameraChangeConfig())
    sequence = [8, 10, 12, 14, 18, 22, 28, 26, 24, 23, 18, 16, 14]
    promotions = _collect_promotions(policy, sequence)
    assert len(promotions) <= 1


def test_alternating_near_threshold_noise_does_not_spam() -> None:
    policy = CameraChangePolicy(CameraChangeConfig())
    sequence = [20, 26, 21, 27, 22, 25, 19, 24]
    promotions = _collect_promotions(policy, sequence)
    assert promotions == []


def test_cooldown_suppresses_additional_promotions() -> None:
    policy = CameraChangePolicy(
        CameraChangeConfig(
            debounce_frames=2,
            cooldown_seconds=10.0,
        )
    )
    sequence = [30, 30, 10, 10, 30, 30, 10, 10, 30, 30]
    promotions = _collect_promotions(policy, sequence)
    assert len(promotions) == 1
