"""Gesture library definitions and persistence."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading

from core.logging import logger as LOGGER
from motion.action import Action
from motion.keyframe import Keyframe
from motion.motion_controller import MotionController, millis
from storage.controller import StorageController


@dataclass(frozen=True)
class GestureFrameSpec:
    """Definition for one gesture frame in author-time (spec) coordinates.

    `duration_ms` is the nominal target dwell/move time for this frame in the
    gesture definition. It is copied into `Keyframe.final_target_time` when the
    runtime keyframe chain is built, then converted into runtime timing state
    (`deadline_ms`, `duration_ms`) when the controller initializes the frame.
    """

    name: str
    pan_offset: float
    tilt_offset: float
    duration_ms: int
    absolute_target: bool = False
    roll_offset: float = 0.0
    ear_left_offset: float = 0.0
    ear_right_offset: float = 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        """Return the spec as a serializable dictionary."""

        return {
            "name": self.name,
            "pan_offset": self.pan_offset,
            "tilt_offset": self.tilt_offset,
            "duration_ms": self.duration_ms,
            "absolute_target": self.absolute_target,
            "roll_offset": self.roll_offset,
            "ear_left_offset": self.ear_left_offset,
            "ear_right_offset": self.ear_right_offset,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float | int | str]) -> "GestureFrameSpec":
        """Build a spec from a dictionary payload."""

        return cls(
            name=str(payload["name"]),
            pan_offset=float(payload["pan_offset"]),
            tilt_offset=float(payload["tilt_offset"]),
            duration_ms=int(payload["duration_ms"]),
            absolute_target=bool(payload.get("absolute_target", False)),
            roll_offset=float(payload.get("roll_offset", 0.0)),
            ear_left_offset=float(payload.get("ear_left_offset", 0.0)),
            ear_right_offset=float(payload.get("ear_right_offset", 0.0)),
        )


@dataclass(frozen=True)
class GestureDefinition:
    """Definition for a gesture action."""

    name: str
    priority: int
    frames: tuple[GestureFrameSpec, ...]
    timing_style: str = "neutral"

    def to_dict(self) -> dict[str, object]:
        """Return the definition as a serializable dictionary."""

        return {
            "name": self.name,
            "priority": self.priority,
            "timing_style": self.timing_style,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GestureDefinition":
        """Build a definition from a dictionary payload."""

        frames = tuple(GestureFrameSpec.from_dict(frame) for frame in payload["frames"])
        return cls(
            name=str(payload["name"]),
            priority=int(payload["priority"]),
            timing_style=str(payload.get("timing_style", "neutral")),
            frames=frames,
        )


_LOOK_GESTURE_NAMES = frozenset(
    {
        "gesture_look_up",
        "gesture_look_down",
        "gesture_look_left",
        "gesture_look_right",
        "gesture_look_center",
    }
)
_STYLE_MULTIPLIER = {
    "neutral": 1.0,
    "snap": 0.75,
    "solemn": 1.35,
}
_DISTANCE_SCALE_DEGREES = 18.0
_LOOK_DURATION_MIN_MS = 160
_LOOK_DURATION_MAX_MS = 1400
_CANONICAL_CENTER_PAN_DEGREES = 0.0
_CANONICAL_CENTER_TILT_DEGREES = 0.0


DEFAULT_GESTURES = (
    GestureDefinition(
        name="gesture_idle",
        priority=1,
        frames=(
            GestureFrameSpec(
                name="idle-left",
                pan_offset=-4.0,
                tilt_offset=2.5,
                duration_ms=1200,
            ),
            GestureFrameSpec(
                name="idle-right",
                pan_offset=4.0,
                tilt_offset=-2.5,
                duration_ms=1200,
            ),
            GestureFrameSpec(
                name="idle-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=1000,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_nod",
        priority=2,
        frames=(
            GestureFrameSpec(
                name="nod-down",
                pan_offset=0.0,
                tilt_offset=-10.0,
                duration_ms=350,
                ear_left_offset=2.0,
                ear_right_offset=2.0,
            ),
            GestureFrameSpec(
                name="nod-up",
                pan_offset=0.0,
                tilt_offset=10.0,
                duration_ms=350,
                ear_left_offset=1.0,
                ear_right_offset=1.0,
            ),
            GestureFrameSpec(
                name="nod-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=400,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_no",
        priority=2,
        frames=(
            GestureFrameSpec(
                name="no-left",
                pan_offset=-12.0,
                tilt_offset=0.0,
                duration_ms=300,
                ear_left_offset=1.5,
                ear_right_offset=-1.0,
            ),
            GestureFrameSpec(
                name="no-right",
                pan_offset=12.0,
                tilt_offset=0.0,
                duration_ms=300,
                ear_left_offset=-1.0,
                ear_right_offset=1.5,
            ),
            GestureFrameSpec(
                name="no-left-return",
                pan_offset=-8.0,
                tilt_offset=0.0,
                duration_ms=250,
            ),
            GestureFrameSpec(
                name="no-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=350,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_look_around",
        priority=1,
        frames=(
            GestureFrameSpec(
                name="look-left",
                pan_offset=-16.0,
                tilt_offset=3.0,
                duration_ms=700,
                ear_left_offset=2.0,
                ear_right_offset=1.0,
            ),
            GestureFrameSpec(
                name="look-right",
                pan_offset=16.0,
                tilt_offset=3.0,
                duration_ms=800,
                ear_left_offset=1.0,
                ear_right_offset=2.0,
            ),
            GestureFrameSpec(
                name="look-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=700,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_look_up",
        priority=2,
        timing_style="neutral",
        frames=(
            GestureFrameSpec(
                name="look-up",
                pan_offset=0.0,
                tilt_offset=999.0,
                duration_ms=600,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_look_left",
        priority=2,
        timing_style="neutral",
        frames=(
            GestureFrameSpec(
                name="look-left",
                pan_offset=-999.0,
                tilt_offset=0.0,
                duration_ms=600,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_look_right",
        priority=2,
        timing_style="neutral",
        frames=(
            GestureFrameSpec(
                name="look-right",
                pan_offset=999.0,
                tilt_offset=0.0,
                duration_ms=600,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_look_down",
        priority=2,
        timing_style="neutral",
        frames=(
            GestureFrameSpec(
                name="look-down",
                pan_offset=0.0,
                tilt_offset=-999.0,
                duration_ms=600,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_look_center",
        priority=2,
        timing_style="neutral",
        frames=(
            GestureFrameSpec(
                name="look-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=600,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_curious_tilt",
        priority=1,
        frames=(
            GestureFrameSpec(
                name="tilt-up",
                pan_offset=0.0,
                tilt_offset=7.0,
                duration_ms=500,
                roll_offset=4.0,
                ear_left_offset=3.0,
                ear_right_offset=-2.0,
            ),
            GestureFrameSpec(
                name="tilt-down",
                pan_offset=0.0,
                tilt_offset=-7.0,
                duration_ms=500,
                roll_offset=-3.0,
                ear_left_offset=-2.0,
                ear_right_offset=3.0,
            ),
            GestureFrameSpec(
                name="tilt-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=450,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_startup_presence",
        priority=3,
        frames=(
            GestureFrameSpec(
                name="startup-frame-1",
                pan_offset=0.0,
                tilt_offset=-40.0,
                duration_ms=1500,
                absolute_target=True,
                ear_left_offset=3.0,
                ear_right_offset=3.0,
            ),
            GestureFrameSpec(
                name="startup-frame-2",
                pan_offset=0.0,
                tilt_offset=25.0,
                duration_ms=1500,
                absolute_target=True,
                ear_left_offset=0.0,
                ear_right_offset=0.0,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_shutdown_rest",
        priority=3,
        frames=(
            GestureFrameSpec(
                name="shutdown-frame-1",
                pan_offset=0.0,
                tilt_offset=-40.0,
                duration_ms=1000,
                absolute_target=True,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_attention_snap",
        priority=2,
        frames=(
            GestureFrameSpec(
                name="snap-right",
                pan_offset=10.0,
                tilt_offset=2.0,
                duration_ms=250,
            ),
            GestureFrameSpec(
                name="snap-hold",
                pan_offset=10.0,
                tilt_offset=2.0,
                duration_ms=300,
            ),
            GestureFrameSpec(
                name="snap-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=400,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_attention_hold",
        priority=2,
        frames=(
            GestureFrameSpec(
                name="attention-hold-acquire",
                pan_offset=6.0,
                tilt_offset=1.5,
                duration_ms=260,
                ear_left_offset=4.0,
                ear_right_offset=4.0,
            ),
            GestureFrameSpec(
                name="attention-hold",
                pan_offset=6.0,
                tilt_offset=1.5,
                duration_ms=900,
                ear_left_offset=4.0,
                ear_right_offset=4.0,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_attention_release",
        priority=2,
        frames=(
            GestureFrameSpec(
                name="attention-release-center",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=320,
            ),
            GestureFrameSpec(
                name="attention-release-settle",
                pan_offset=1.0,
                tilt_offset=-0.5,
                duration_ms=180,
            ),
            GestureFrameSpec(
                name="attention-release-neutral",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=220,
                absolute_target=True,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_speaking_posture",
        priority=1,
        frames=(
            GestureFrameSpec(
                name="speaking-posture-acquire",
                pan_offset=0.0,
                tilt_offset=2.5,
                duration_ms=280,
                ear_left_offset=1.5,
                ear_right_offset=1.5,
            ),
            GestureFrameSpec(
                name="speaking-posture-hold",
                pan_offset=0.0,
                tilt_offset=2.5,
                duration_ms=520,
                ear_left_offset=1.0,
                ear_right_offset=1.0,
            ),
        ),
    ),
    GestureDefinition(
        name="gesture_speaking_settle",
        priority=1,
        frames=(
            GestureFrameSpec(
                name="speaking-settle-release",
                pan_offset=0.0,
                tilt_offset=-1.5,
                duration_ms=220,
            ),
            GestureFrameSpec(
                name="speaking-settle-neutral",
                pan_offset=0.0,
                tilt_offset=0.0,
                duration_ms=260,
                absolute_target=True,
            ),
        ),
    ),
)


class GestureLibrary:
    """Singleton library for storing gesture definitions."""

    _instance: "GestureLibrary | None" = None
    _lock = threading.Lock()

    def __init__(self, storage_controller: StorageController | None = None) -> None:
        if GestureLibrary._instance is not None:
            raise RuntimeError("You cannot create another GestureLibrary class")

        self._storage = storage_controller or StorageController.get_instance()
        self._definitions: dict[str, GestureDefinition] = {}
        self._library_path = self._resolve_library_path()
        self._load_library()
        self.ensure_defaults()
        GestureLibrary._instance = self

    @classmethod
    def get_instance(cls) -> "GestureLibrary":
        """Return the singleton instance of the gesture library."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def list_gestures(self) -> list[str]:
        """Return a list of gesture names in the library."""

        return sorted(self._definitions.keys())

    def get(self, name: str) -> GestureDefinition:
        """Return a gesture definition by name."""

        return self._definitions[name]

    def register(self, definition: GestureDefinition, persist: bool = True) -> None:
        """Register a gesture definition in the library."""

        self._definitions[definition.name] = definition
        if persist:
            self._persist_library()

    def ensure_defaults(self) -> None:
        """Ensure bundled default gestures are present and current."""

        changed = False
        for definition in DEFAULT_GESTURES:
            if self._definitions.get(definition.name) != definition:
                self._definitions[definition.name] = definition
                changed = True
        if changed:
            self._persist_library()

    def build_action(
        self,
        name: str,
        delay_ms: int = 0,
        intensity: float = 1.0,
        style: str | None = None,
    ) -> Action:
        """Build an action from a gesture definition.

        Timing note: gesture frame `duration_ms` values are treated as nominal
        spec durations. Runtime completion still depends on controller
        termination rules (`_frame_done`), not this build-time conversion alone.
        """

        definition = self.get(name)
        controller = MotionController.get_instance()
        if hasattr(controller, "get_current_logical_pose"):
            current_pose = controller.get_current_logical_pose()
        else:
            current_pose = getattr(
                controller, "current_servo_position", {"pan": 0.0, "tilt": 0.0}
            )
        current_pan = float(current_pose.get("pan", 0.0))
        current_tilt = float(current_pose.get("tilt", 0.0))
        current_roll = float(current_pose.get("roll", 0.0))
        current_ear_left = float(current_pose.get("ear_left", 0.0))
        current_ear_right = float(current_pose.get("ear_right", 0.0))

        frames = self._build_keyframe_chain(
            controller=controller,
            definition=definition,
            base_pan=current_pan,
            base_tilt=current_tilt,
            base_roll=current_roll,
            base_ear_left=current_ear_left,
            base_ear_right=current_ear_right,
            intensity=float(intensity),
            style=(style or definition.timing_style),
        )
        return Action(
            priority=definition.priority,
            timestamp=millis() + delay_ms,
            name=definition.name,
            frames=frames,
        )

    def _resolve_library_path(self) -> Path:
        if not hasattr(self._storage, "get_storage_info"):
            return Path("var") / "gesture_library.json"
        storage_info = self._storage.get_storage_info()
        return storage_info.log_dir / "gesture_library.json"

    def _load_library(self) -> None:
        if not self._library_path.exists():
            return

        try:
            payload = json.loads(self._library_path.read_text(encoding="utf-8"))
            gestures = payload.get("gestures", [])
            for gesture_payload in gestures:
                definition = GestureDefinition.from_dict(gesture_payload)
                self._definitions[definition.name] = definition
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Failed to load gesture library: %s", exc)

    def _persist_library(self) -> None:
        with self._lock:
            payload = {
                "gestures": [
                    definition.to_dict() for definition in self._definitions.values()
                ]
            }
            try:
                self._library_path.parent.mkdir(parents=True, exist_ok=True)
                self._library_path.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
            except OSError as exc:
                LOGGER.warning("Failed to persist gesture library: %s", exc)

    def _get_servo_limits(
        self, controller: MotionController, name: str
    ) -> tuple[float, float]:
        if name == "tilt":
            if "tilt" in controller.servo_registry.servos:
                servo = controller.servo_registry.servos["tilt"]
                return float(servo.min_angle), float(servo.max_angle)
            left = controller.servo_registry.servos["tilt_left"]
            right = controller.servo_registry.servos["tilt_right"]
            return (
                max(float(left.min_angle), float(right.min_angle)),
                min(float(left.max_angle), float(right.max_angle)),
            )
        servo = controller.servo_registry.servos[name]
        return float(servo.min_angle), float(servo.max_angle)

    def _clamp(self, value: float, minimum: float, maximum: float) -> float:
        if value < minimum:
            return minimum
        if value > maximum:
            return maximum
        return value

    def _build_keyframe_chain(
        self,
        controller: MotionController,
        definition: GestureDefinition,
        base_pan: float,
        base_tilt: float,
        base_roll: float,
        base_ear_left: float,
        base_ear_right: float,
        intensity: float,
        style: str,
    ) -> Keyframe:
        iterator = iter(definition.frames)
        transition_pan = base_pan
        transition_tilt = base_tilt
        first_spec = next(iterator)
        first_frame = self._create_keyframe(
            controller,
            definition,
            first_spec,
            base_pan=base_pan,
            base_tilt=base_tilt,
            base_roll=base_roll,
            base_ear_left=base_ear_left,
            base_ear_right=base_ear_right,
            transition_pan=transition_pan,
            transition_tilt=transition_tilt,
            intensity=intensity,
            style=style,
        )
        transition_pan = float(first_frame.servo_destination["pan"])
        transition_tilt = float(first_frame.servo_destination["tilt"])
        current = first_frame

        for spec in iterator:
            next_frame = self._create_keyframe(
                controller,
                definition,
                spec,
                base_pan=base_pan,
                base_tilt=base_tilt,
                base_roll=base_roll,
                base_ear_left=base_ear_left,
                base_ear_right=base_ear_right,
                transition_pan=transition_pan,
                transition_tilt=transition_tilt,
                intensity=intensity,
                style=style,
            )
            current.next = next_frame
            current = next_frame
            transition_pan = float(next_frame.servo_destination["pan"])
            transition_tilt = float(next_frame.servo_destination["tilt"])

        return first_frame

    def _duration_for_frame(
        self,
        *,
        definition: GestureDefinition,
        spec: GestureFrameSpec,
        target_pan: float,
        target_tilt: float,
        transition_pan: float,
        transition_tilt: float,
        style: str,
    ) -> int:
        if definition.name not in _LOOK_GESTURE_NAMES:
            return int(spec.duration_ms)

        pan_delta = abs(target_pan - transition_pan)
        tilt_delta = abs(target_tilt - transition_tilt)
        distance = pan_delta + tilt_delta
        baseline_ms = int(
            round(
                float(spec.duration_ms) * max(distance / _DISTANCE_SCALE_DEGREES, 0.35)
            )
        )

        multiplier = _STYLE_MULTIPLIER.get(style, _STYLE_MULTIPLIER["neutral"])
        styled_ms = int(round(baseline_ms * multiplier))
        return int(self._clamp(styled_ms, _LOOK_DURATION_MIN_MS, _LOOK_DURATION_MAX_MS))

    def _create_keyframe(
        self,
        controller: MotionController,
        definition: GestureDefinition,
        spec: GestureFrameSpec,
        base_pan: float,
        base_tilt: float,
        transition_pan: float,
        transition_tilt: float,
        intensity: float,
        style: str,
        base_roll: float = 0.0,
        base_ear_left: float = 0.0,
        base_ear_right: float = 0.0,
    ) -> Keyframe:
        """Create a runtime keyframe from a gesture spec frame.

        The authored `spec.duration_ms` is copied into
        `Keyframe.final_target_time` as the per-frame nominal target time.
        Absolute runtime deadlines are derived later by `MotionController`.
        """
        pan_min, pan_max = self._get_servo_limits(controller, "pan")
        tilt_min, tilt_max = self._get_servo_limits(controller, "tilt")
        if definition.name == "gesture_look_center":
            target_pan = self._clamp(_CANONICAL_CENTER_PAN_DEGREES, pan_min, pan_max)
            target_tilt = self._clamp(
                _CANONICAL_CENTER_TILT_DEGREES, tilt_min, tilt_max
            )
            target_roll = 0.0
            target_ear_left = 0.0
            target_ear_right = 0.0
        else:
            if spec.absolute_target:
                target_pan = self._clamp(spec.pan_offset, pan_min, pan_max)
                target_tilt = self._clamp(spec.tilt_offset, tilt_min, tilt_max)
                target_roll = spec.roll_offset
                target_ear_left = spec.ear_left_offset
                target_ear_right = spec.ear_right_offset
            else:
                target_pan = self._clamp(
                    base_pan + spec.pan_offset * intensity, pan_min, pan_max
                )
                target_tilt = self._clamp(
                    base_tilt + spec.tilt_offset * intensity, tilt_min, tilt_max
                )
                target_roll = base_roll + spec.roll_offset * intensity
                target_ear_left = base_ear_left + spec.ear_left_offset * intensity
                target_ear_right = base_ear_right + spec.ear_right_offset * intensity
        frame = controller.generate_base_keyframe(
            pan_degrees=target_pan,
            tilt_degrees=target_tilt,
            roll_degrees=target_roll,
            ear_left_degrees=target_ear_left,
            ear_right_degrees=target_ear_right,
        )
        frame.name = spec.name
        frame.final_target_time = self._duration_for_frame(
            definition=definition,
            spec=spec,
            target_pan=target_pan,
            target_tilt=target_tilt,
            transition_pan=transition_pan,
            transition_tilt=transition_tilt,
            style=style,
        )
        return frame
