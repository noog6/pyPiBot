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
    """Definition for a single gesture keyframe offset."""

    name: str
    pan_offset: float
    tilt_offset: float
    duration_ms: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return the spec as a serializable dictionary."""

        return {
            "name": self.name,
            "pan_offset": self.pan_offset,
            "tilt_offset": self.tilt_offset,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float | int | str]) -> "GestureFrameSpec":
        """Build a spec from a dictionary payload."""

        return cls(
            name=str(payload["name"]),
            pan_offset=float(payload["pan_offset"]),
            tilt_offset=float(payload["tilt_offset"]),
            duration_ms=int(payload["duration_ms"]),
        )


@dataclass(frozen=True)
class GestureDefinition:
    """Definition for a gesture action."""

    name: str
    priority: int
    frames: tuple[GestureFrameSpec, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the definition as a serializable dictionary."""

        return {
            "name": self.name,
            "priority": self.priority,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GestureDefinition":
        """Build a definition from a dictionary payload."""

        frames = tuple(GestureFrameSpec.from_dict(frame) for frame in payload["frames"])
        return cls(
            name=str(payload["name"]),
            priority=int(payload["priority"]),
            frames=frames,
        )


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
            ),
            GestureFrameSpec(
                name="nod-up",
                pan_offset=0.0,
                tilt_offset=10.0,
                duration_ms=350,
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
            ),
            GestureFrameSpec(
                name="no-right",
                pan_offset=12.0,
                tilt_offset=0.0,
                duration_ms=300,
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
            ),
            GestureFrameSpec(
                name="look-right",
                pan_offset=16.0,
                tilt_offset=3.0,
                duration_ms=800,
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
        name="gesture_curious_tilt",
        priority=1,
        frames=(
            GestureFrameSpec(
                name="tilt-up",
                pan_offset=0.0,
                tilt_offset=8.0,
                duration_ms=500,
            ),
            GestureFrameSpec(
                name="tilt-down",
                pan_offset=0.0,
                tilt_offset=-8.0,
                duration_ms=500,
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
        """Ensure the base gestures are present in the library."""

        added = False
        for definition in DEFAULT_GESTURES:
            if definition.name not in self._definitions:
                self._definitions[definition.name] = definition
                added = True
        if added:
            self._persist_library()

    def build_action(self, name: str, delay_ms: int = 0, intensity: float = 1.0) -> Action:
        """Build an action from a gesture definition."""

        definition = self.get(name)
        controller = MotionController.get_instance()
        pan_servo = controller.servo_registry.servos["pan"]
        tilt_servo = controller.servo_registry.servos["tilt"]
        current_pan = float(pan_servo.read_value())
        current_tilt = float(tilt_servo.read_value())

        frames = self._build_keyframe_chain(
            controller=controller,
            definition=definition,
            base_pan=current_pan,
            base_tilt=current_tilt,
            intensity=float(intensity),
        )
        return Action(
            priority=definition.priority,
            timestamp=millis() + delay_ms,
            name=definition.name,
            frames=frames,
        )

    def _resolve_library_path(self) -> Path:
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
                "gestures": [definition.to_dict() for definition in self._definitions.values()]
            }
            try:
                self._library_path.parent.mkdir(parents=True, exist_ok=True)
                self._library_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError as exc:
                LOGGER.warning("Failed to persist gesture library: %s", exc)

    def _get_servo_limits(
        self, controller: MotionController, name: str
    ) -> tuple[float, float]:
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
        intensity: float,
    ) -> Keyframe:
        iterator = iter(definition.frames)
        first_spec = next(iterator)
        first_frame = self._create_keyframe(
            controller,
            first_spec,
            base_pan=base_pan,
            base_tilt=base_tilt,
            intensity=intensity,
        )
        current = first_frame

        for spec in iterator:
            next_frame = self._create_keyframe(
                controller,
                spec,
                base_pan=base_pan,
                base_tilt=base_tilt,
                intensity=intensity,
            )
            current.next = next_frame
            current = next_frame

        return first_frame

    def _create_keyframe(
        self,
        controller: MotionController,
        spec: GestureFrameSpec,
        base_pan: float,
        base_tilt: float,
        intensity: float,
    ) -> Keyframe:
        pan_min, pan_max = self._get_servo_limits(controller, "pan")
        tilt_min, tilt_max = self._get_servo_limits(controller, "tilt")
        frame = controller.generate_base_keyframe(
            pan_degrees=self._clamp(base_pan + spec.pan_offset * intensity, pan_min, pan_max),
            tilt_degrees=self._clamp(base_tilt + spec.tilt_offset * intensity, tilt_min, tilt_max),
        )
        frame.name = spec.name
        frame.final_target_time = spec.duration_ms
        return frame
