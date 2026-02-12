#!/usr/bin/env python3
"""Manual smoke test for the IMX500 controller skeleton."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.imx500_controller import Detection, Imx500Controller


def _print_update(detections: list[Detection], timestamp: float) -> None:
    print(f"callback ts={timestamp:.3f} detections={len(detections)}")


def main() -> None:
    controller = Imx500Controller.get_instance()
    controller.subscribe(_print_update)

    controller.start()
    controller.start()  # idempotent

    print("available:", controller.is_available())
    print("latest:", controller.get_latest_detections())

    # Internal publish helper exists for future integration points.
    controller._publish_detections(
        [Detection(label="person", confidence=0.9, bbox=(0.1, 0.1, 0.5, 0.8))]
    )
    print("latest_after_publish:", controller.get_latest_detections())

    controller.unsubscribe(_print_update)
    controller.stop()
    controller.stop()  # idempotent


if __name__ == "__main__":
    main()
