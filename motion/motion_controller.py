"""Motion controller for servo-driven keyframe actions."""

from __future__ import annotations

import heapq
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

from ai.embodiment_policy import EmbodimentPolicy, LifecyclePostureEvent
from hardware.servo_registry import ServoRegistry
from motion.action import Action
from motion.keyframe import Keyframe
from motion.logging import log_debug, log_error, log_info, log_warning


def millis() -> int:
    return int(time.monotonic() * 1000)


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class MotionTuning:
    dt_min_s: float = 0.005
    dt_max_s: float = 0.05
    pan_step_min_deg: float = 0.2
    pan_step_max_deg: float = 1.2
    pan_step_scale_deg: float = 70.0
    tilt_step_min_deg: float = 0.25
    tilt_step_max_deg: float = 2.0
    tilt_step_scale_deg: float = 45.0
    pan_a_max: float = 165.0
    tilt_a_max: float = 525.0
    v_max_smoothing_tau_s: float = 0.02
    position_eps_deg: float = 0.05
    at_dest_eps_deg: float = 0.5
    debug_motion: bool = False
    debug_motion_log_interval_ms: int = 200


TUNING = MotionTuning()

# Isolate logical roll sign convention here so hardware/IMU validation can flip it.
ROLL_TO_TILT_SIGN = 1.0
LOGICAL_POSE_AXES = ("pan", "tilt", "roll", "ear_left", "ear_right")


def limit_step(
    current: float,
    target: float,
    v_state: dict[str, float],
    axis: str,
    dt_s: float,
    v_max: float,
    a_max: float,
    eps: float = 1e-3,
) -> float:
    """Acceleration-limited velocity follower."""

    v = float(v_state.get(axis, 0.0))
    err = target - current

    if abs(err) <= eps:
        v_state[axis] = 0.0
        return target

    v_des = err / max(dt_s, 1e-6)
    if v_des > v_max:
        v_des = v_max
    if v_des < -v_max:
        v_des = -v_max

    dv_max = a_max * dt_s
    dv = v_des - v
    if dv > dv_max:
        dv = dv_max
    if dv < -dv_max:
        dv = -dv_max
    v = v + dv

    if v > v_max:
        v = v_max
    if v < -v_max:
        v = -v_max

    nxt = current + v * dt_s

    if (target - current) * (target - nxt) <= 0.0:
        v_state[axis] = 0.0
        return target

    v_state[axis] = v
    return nxt


def scaled_step(
    dist_deg: float, step_min: float, step_max: float, scale_deg: float
) -> float:
    ratio = clamp01(abs(dist_deg) / max(scale_deg, 1e-6))
    ratio = smoothstep(ratio)
    return step_min + (step_max - step_min) * ratio


def axis_step_v_max(
    axis: str, dist_deg: float, nominal_dt_s: float, tuning: MotionTuning
) -> float:
    if axis == "pan":
        step_deg = scaled_step(
            dist_deg,
            tuning.pan_step_min_deg,
            tuning.pan_step_max_deg,
            tuning.pan_step_scale_deg,
        )
    elif axis == "tilt":
        step_deg = scaled_step(
            dist_deg,
            tuning.tilt_step_min_deg,
            tuning.tilt_step_max_deg,
            tuning.tilt_step_scale_deg,
        )
    else:
        raise ValueError(f"Unsupported axis for shaped step: {axis}")

    return step_deg / max(nominal_dt_s, 1e-6)


class MotionController:
    """Singleton controller for motion sequences."""

    _instance: "MotionController | None" = None

    def __init__(self) -> None:
        if MotionController._instance is not None:
            raise RuntimeError("You cannot create another MotionController class")

        self._control_loop_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._moving_event = threading.Event()
        self._queue_lock = threading.Lock()
        self.control_loop_index = 0
        self.control_loop_function = None
        self.control_loop_period_ms = 100
        self.control_loop_start_time = [0] * 100
        self.transition_time = 1500
        self.servo_registry = ServoRegistry.get_instance()
        self.current_logical_pose = {axis: 0.0 for axis in LOGICAL_POSE_AXES}
        # Compatibility alias for older call sites/tests; this now stores logical pose,
        # not raw physical servo positions.
        self.current_servo_position = self.current_logical_pose
        self.axis_v = {axis: 0.0 for axis in LOGICAL_POSE_AXES}
        self.axis_v_max = {axis: 0.0 for axis in LOGICAL_POSE_AXES}
        self.action_queue: list[Action] = []
        self.current_action: Action | None = None
        self._action_lifecycle_callbacks: list[Callable[[str, Action], None]] = []
        self._last_update_ms: int | None = None
        self._last_motion_debug_log_ms: int | None = None
        MotionController._instance = self

    @classmethod
    def get_instance(cls) -> "MotionController":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_control_loop_alive(self) -> bool:
        if self._control_loop_thread is None:
            return False

        return self._control_loop_thread.is_alive()

    def toggle_control_loop(self) -> None:
        if self.is_control_loop_alive():
            self.stop_control_loop()
        else:
            self.start_control_loop()

    def start_control_loop(self, control_loop_period_ms: int = 20) -> None:
        if (
            self._control_loop_thread is None
            or not self._control_loop_thread.is_alive()
        ):
            self.control_loop_period_ms = control_loop_period_ms
            self._run_lifecycle_posture_blocking(event=LifecyclePostureEvent.STARTUP)
            self._stop_event.clear()
            self.control_loop_period_ms = control_loop_period_ms
            self._control_loop_thread = threading.Thread(
                target=self._control_loop, daemon=True
            )
            self._control_loop_thread.start()

    def stop_control_loop(self) -> None:
        if self._control_loop_thread is not None:
            self._stop_event.set()
            self._control_loop_thread.join()
            self._control_loop_thread = None
            self._run_lifecycle_posture_blocking(event=LifecyclePostureEvent.SHUTDOWN)
            self.relax_all_servos()
            log_info(
                f"[MOTION] control loop stopped at index: {self.control_loop_index}"
            )
            self.control_loop_index = 0

    def _run_lifecycle_posture_blocking(self, *, event: LifecyclePostureEvent) -> None:
        decision = EmbodimentPolicy().decide_lifecycle_posture(event=event)
        log_info(
            "[MOTION] lifecycle posture decision event=%s cue=%s reason_codes=%s",
            event.value,
            decision.cue_name,
            list(decision.reason_codes),
        )
        if decision.cue_name is None:
            return
        self._run_lifecycle_gesture_blocking(gesture_name=decision.cue_name)

    def _run_lifecycle_gesture_blocking(self, *, gesture_name: str) -> None:
        from motion.gesture_library import GestureLibrary

        if gesture_name not in {"gesture_startup_presence", "gesture_shutdown_rest"}:
            raise ValueError(f"Unsupported lifecycle gesture: {gesture_name}")

        action = GestureLibrary.get_instance().build_action(gesture_name)
        log_info("[MOTION] lifecycle gesture dispatch gesture=%s", gesture_name)
        frame = action.current_frame
        while frame is not None:
            while not self.move_to_keyframe(frame):
                time.sleep(0.02)
            frame = frame.next

    def _control_loop(self) -> None:
        next_control_loop_time = millis()

        while not self._stop_event.is_set():
            current_time = millis()
            while current_time >= next_control_loop_time:
                self.control_loop_index += 1

                try:
                    self.update_pose()
                except Exception as exc:
                    log_error(f"[MOTION] Error in control loop (retrying): {exc}")
                    traceback.print_exc()

                self.control_loop_start_time.append(
                    current_time - next_control_loop_time
                )
                if len(self.control_loop_start_time) > 100:
                    self.control_loop_start_time.pop(0)
                next_control_loop_time += self.control_loop_period_ms
                current_time = millis()
            else:
                time.sleep(0.001)

    def update_pose(self) -> None:
        if not self.current_action:
            self.current_action = self.get_next_action()

        if self.current_action:
            if self.move_to_keyframe(self.current_action.current_frame):
                self.current_action.current_frame = (
                    self.current_action.current_frame.next
                )

                if self.current_action.current_frame is None:
                    completed_action = self.current_action
                    self._emit_action_lifecycle("completed", completed_action)
                    self.current_action = self.get_next_action()

    def _clamp_servo_target(self, servo_name: str, value: float) -> float:
        servo = self.servo_registry.servos[servo_name]
        minimum = float(servo.min_angle)
        maximum = float(servo.max_angle)
        raw_value = float(value)
        clamped = max(minimum, min(maximum, raw_value))
        if clamped != raw_value:
            log_warning(
                "[MOTION] physical servo target clamped servo=%s requested=%.3f clamped=%.3f min=%.3f max=%.3f",
                servo_name,
                raw_value,
                clamped,
                minimum,
                maximum,
            )
        return clamped

    def _logical_pose_to_servo_targets(
        self, logical_pose: dict[str, float]
    ) -> dict[str, float]:
        """Translate public logical pose axes to private physical servo targets."""

        pan = float(logical_pose.get("pan", 0.0))
        tilt = float(logical_pose.get("tilt", 0.0))
        roll = float(logical_pose.get("roll", 0.0)) * ROLL_TO_TILT_SIGN
        targets = {
            "pan": pan,
            "tilt_left": tilt + roll,
            "tilt_right": tilt - roll,
            "ear_left": float(logical_pose.get("ear_left", 0.0)),
            "ear_right": float(logical_pose.get("ear_right", 0.0)),
        }
        return {
            name: self._clamp_servo_target(name, value)
            for name, value in targets.items()
        }

    def get_current_logical_pose(self) -> dict[str, float]:
        """Return the current public/logical motion pose."""

        return {
            axis: float(self.current_logical_pose.get(axis, 0.0))
            for axis in LOGICAL_POSE_AXES
        }

    def _axis_v_max_for_remaining(
        self, axis: str, remaining: float, nominal_dt_s: float
    ) -> float:
        shaped_axis = "pan" if axis == "pan" else "tilt"
        return axis_step_v_max(shaped_axis, remaining, nominal_dt_s, TUNING)

    def _axis_a_max(self, axis: str) -> float:
        return TUNING.pan_a_max if axis == "pan" else TUNING.tilt_a_max

    def move_to_keyframe(self, new_frame: Keyframe) -> bool:
        self._moving_event.set()
        now_ms = millis()

        if not new_frame.is_initialized:
            self._init_frame(new_frame, now_ms)

        dt_s = self._compute_dt_s(now_ms)
        nominal_dt_s = self._compute_nominal_dt_s()
        desired_pose = {
            axis: float(new_frame.servo_destination.get(axis, 0.0))
            for axis in LOGICAL_POSE_AXES
        }

        limited_pose: dict[str, float] = {}
        axis_v_max_by_axis: dict[str, float] = {}
        for axis, desired_value in desired_pose.items():
            remaining = desired_value - self.current_logical_pose[axis]
            v_max_raw = self._axis_v_max_for_remaining(axis, remaining, nominal_dt_s)
            v_max = self._smooth_v_max(axis, v_max_raw, nominal_dt_s)
            axis_v_max_by_axis[axis] = v_max
            limited_pose[axis] = limit_step(
                self.current_logical_pose[axis],
                desired_value,
                self.axis_v,
                axis,
                dt_s,
                v_max,
                self._axis_a_max(axis),
                eps=TUNING.position_eps_deg,
            )

        physical_targets = self._logical_pose_to_servo_targets(limited_pose)

        if abs(limited_pose["pan"] - self.current_logical_pose["pan"]) > 1.0:
            last_pan_log_ms = new_frame.last_pan_log_ms
            if last_pan_log_ms is None or now_ms - last_pan_log_ms >= 200:
                log_info(
                    "[MOTION] [%s] 'pan' servo to (%.2f) (wanted: %.2f) (PAN_V_MAX:%.3f) (Elapsed ms:%s)",
                    new_frame.name,
                    limited_pose["pan"],
                    desired_pose["pan"],
                    axis_v_max_by_axis["pan"],
                    now_ms - new_frame.start_time_ms,
                )
                new_frame.last_pan_log_ms = now_ms

        self.current_logical_pose.update(limited_pose)
        for servo_name, target in physical_targets.items():
            self.servo_registry.servos[servo_name].write_value(target)

        eps = TUNING.at_dest_eps_deg
        at_dest = all(
            abs(self.current_logical_pose[axis] - desired_pose[axis]) <= eps
            for axis in LOGICAL_POSE_AXES
        )

        if self._should_emit_motion_debug_log(now_ms):
            log_info(
                "[MOTION][debug] dt=%.4f pan_v=%.2f pan_vmax=%.2f tilt_v=%.2f tilt_vmax=%.2f roll_v=%.2f roll_vmax=%.2f pan_err=%.2f tilt_err=%.2f roll_err=%.2f",
                dt_s,
                self.axis_v["pan"],
                axis_v_max_by_axis["pan"],
                self.axis_v["tilt"],
                axis_v_max_by_axis["tilt"],
                self.axis_v["roll"],
                axis_v_max_by_axis["roll"],
                desired_pose["pan"] - self.current_logical_pose["pan"],
                desired_pose["tilt"] - self.current_logical_pose["tilt"],
                desired_pose["roll"] - self.current_logical_pose["roll"],
            )

        done = self._frame_done(new_frame, at_dest, now_ms)
        if done:
            self.current_logical_pose.update(desired_pose)
            for servo_name, target in self._logical_pose_to_servo_targets(
                desired_pose
            ).items():
                self.servo_registry.servos[servo_name].write_value(target)
            elapsed_ms = now_ms - new_frame.start_time_ms
            log_debug(
                "[MOTION] logical pose move completed pan=%.3f tilt=%.3f roll=%.3f ear_left=%.3f ear_right=%.3f elapsed_ms=%s",
                desired_pose["pan"],
                desired_pose["tilt"],
                desired_pose["roll"],
                desired_pose["ear_left"],
                desired_pose["ear_right"],
                elapsed_ms,
            )
            self._moving_event.clear()
            return True

        return False

    def _should_emit_motion_debug_log(self, now_ms: int) -> bool:
        if not TUNING.debug_motion:
            return False

        interval_ms = max(0, int(TUNING.debug_motion_log_interval_ms))
        if interval_ms == 0:
            return True

        last_log_ms = self._last_motion_debug_log_ms
        if last_log_ms is None or now_ms - last_log_ms >= interval_ms:
            self._last_motion_debug_log_ms = now_ms
            return True
        return False

    def _compute_dt_s(self, now_ms: int) -> float:
        if self._last_update_ms is None:
            dt_ms = max(self.control_loop_period_ms, 1)
        else:
            dt_ms = max(now_ms - self._last_update_ms, 1)
        self._last_update_ms = now_ms
        dt_s = dt_ms / 1000.0
        if dt_s < TUNING.dt_min_s:
            return TUNING.dt_min_s
        if dt_s > TUNING.dt_max_s:
            return TUNING.dt_max_s
        return dt_s

    def _smooth_v_max(self, axis: str, v_max: float, dt_s: float) -> float:
        tau_s = TUNING.v_max_smoothing_tau_s
        if tau_s <= 0.0:
            self.axis_v_max[axis] = max(v_max, 0.0)
            return v_max
        prev = self.axis_v_max.get(axis, 0.0)
        alpha = dt_s / (tau_s + dt_s)
        smoothed = prev + alpha * (v_max - prev)
        smoothed = max(smoothed, 0.0)
        self.axis_v_max[axis] = smoothed
        return smoothed

    def _compute_nominal_dt_s(self) -> float:
        dt_s = max(self.control_loop_period_ms, 1) / 1000.0
        if dt_s < TUNING.dt_min_s:
            return TUNING.dt_min_s
        if dt_s > TUNING.dt_max_s:
            return TUNING.dt_max_s
        return dt_s

    def _init_frame(self, frame: Keyframe, now_ms: int) -> None:
        """Initialize runtime timing/position state for the active frame.

        `frame.final_target_time` is treated as a nominal per-frame target
        duration (ms). On initialization we derive:
        - `deadline_ms`: absolute cutoff timestamp (`now_ms + nominal_duration`)
        - `duration_ms`: normalized runtime duration for reporting/logs
        """
        frame.start_time_ms = now_ms
        self._last_update_ms = now_ms - max(self.control_loop_period_ms, 1)

        dur = max(0, int(frame.final_target_time))
        frame.deadline_ms = now_ms + dur
        frame.duration_ms = max(1, dur)

        frame.start_pos = {
            axis: float(self.current_logical_pose.get(axis, 0.0))
            for axis in LOGICAL_POSE_AXES
        }
        frame.delta_pos = {
            axis: float(frame.servo_destination.get(axis, 0.0)) - frame.start_pos[axis]
            for axis in LOGICAL_POSE_AXES
        }

        log_info(
            "[MOTION] New motion frame started name=%s duration_ms=%s "
            "pan=%.2f->%.2f tilt=%.2f->%.2f roll=%.2f->%.2f "
            "ear_left=%.2f->%.2f ear_right=%.2f->%.2f deadline_ms=%s",
            frame.name,
            frame.duration_ms,
            self.current_logical_pose["pan"],
            frame.servo_destination.get("pan", 0.0),
            self.current_logical_pose["tilt"],
            frame.servo_destination.get("tilt", 0.0),
            self.current_logical_pose["roll"],
            frame.servo_destination.get("roll", 0.0),
            self.current_logical_pose["ear_left"],
            frame.servo_destination.get("ear_left", 0.0),
            self.current_logical_pose["ear_right"],
            frame.servo_destination.get("ear_right", 0.0),
            frame.deadline_ms,
        )
        frame.is_initialized = True

    def _frame_done(self, frame: Keyframe, at_dest: bool, now_ms: int) -> bool:
        """Return whether the frame should advance under current timing rules.

        Current contract intentionally requires BOTH:
        1) destination reached (`at_dest`), and
        2) nominal time budget elapsed (`now_ms >= deadline_ms`).

        If no deadline exists, completion is purely `at_dest`.
        """
        fail_open_on_deadline = False

        if not frame.has_deadline():
            return at_dest

        time_up = now_ms >= frame.deadline_ms

        if at_dest:
            return time_up

        if fail_open_on_deadline and time_up:
            log_warning(
                f"[MOTION] Frame '{frame.name}' missed destination before deadline; "
                "advancing anyway."
            )
            return True

        return False

    def generate_base_keyframe(
        self,
        pan_degrees: int,
        tilt_degrees: int,
        roll_degrees: int = 0,
        ear_left_degrees: int = 0,
        ear_right_degrees: int = 0,
    ) -> Keyframe:
        new_frame = Keyframe(final_target_time=self.transition_time, name="base")
        new_frame.servo_destination.update(
            {
                "pan": pan_degrees,
                "tilt": tilt_degrees,
                "roll": roll_degrees,
                "ear_left": ear_left_degrees,
                "ear_right": ear_right_degrees,
            }
        )
        return new_frame

    def relax_all_servos(self) -> None:
        for servo in self.servo_registry.servos.values():
            servo.relax()

    def get_next_action(self) -> Action | None:
        next_action = None
        current_time = millis()
        with self._queue_lock:
            if self.action_queue and self.action_queue[0].timestamp <= current_time:
                next_action = heapq.heappop(self.action_queue)
        if next_action:
            next_action.set_frame_times(current_time)
            self._emit_action_lifecycle("started", next_action)
        return next_action

    def add_action_to_queue(self, new_action: Action) -> None:
        with self._queue_lock:
            heapq.heappush(self.action_queue, new_action)
        self._emit_action_lifecycle("queued", new_action)

    def register_action_lifecycle_callback(
        self, callback: Callable[[str, Action], None]
    ) -> None:
        if not callable(callback):
            return
        if callback not in self._action_lifecycle_callbacks:
            self._action_lifecycle_callbacks.append(callback)

    def _emit_action_lifecycle(self, event: str, action: Action) -> None:
        if not self._action_lifecycle_callbacks:
            return
        for callback in list(self._action_lifecycle_callbacks):
            try:
                callback(event, action)
            except Exception as exc:
                log_warning(
                    "[MOTION] action lifecycle callback failed event=%s action=%s error=%s",
                    event,
                    str(getattr(action, "name", "unknown") or "unknown"),
                    exc,
                )

    def is_moving(self) -> bool:
        return self._moving_event.is_set()
