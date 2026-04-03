"""Motion controller for servo-driven keyframe actions."""

from __future__ import annotations

import heapq
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Callable

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


def scaled_step(dist_deg: float, step_min: float, step_max: float, scale_deg: float) -> float:
    ratio = clamp01(abs(dist_deg) / max(scale_deg, 1e-6))
    ratio = smoothstep(ratio)
    return step_min + (step_max - step_min) * ratio


def axis_step_v_max(axis: str, dist_deg: float, nominal_dt_s: float, tuning: MotionTuning) -> float:
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
        self.current_servo_position = {"pan": 0.0, "tilt": 0.0}
        self.axis_v = {"pan": 0.0, "tilt": 0.0}
        self.axis_v_max = {"pan": 0.0, "tilt": 0.0}
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
        if self._control_loop_thread is None or not self._control_loop_thread.is_alive():
            self.control_loop_period_ms = control_loop_period_ms
            starting_frame = self.generate_base_keyframe(pan_degrees=0, tilt_degrees=-40)
            starting_frame.name = "Starting Frame - 1"
            while not self.move_to_keyframe(starting_frame):
                time.sleep(0.02)
            time.sleep(1.0)
            starting_frame = self.generate_base_keyframe(pan_degrees=0, tilt_degrees=25)
            starting_frame.name = "Starting Frame - 2"
            while not self.move_to_keyframe(starting_frame):
                time.sleep(0.02)
            self._stop_event.clear()
            self.control_loop_period_ms = control_loop_period_ms
            self._control_loop_thread = threading.Thread(target=self._control_loop, daemon=True)
            self._control_loop_thread.start()

    def stop_control_loop(self) -> None:
        if self._control_loop_thread is not None:
            self._stop_event.set()
            self._control_loop_thread.join()
            self._control_loop_thread = None
            sit_frame = self.generate_base_keyframe(pan_degrees=0, tilt_degrees=-40)
            sit_frame.name = "Ending Frame - 1"
            sit_frame.final_target_time = 1000
            sit_frame.deadline_ms = None
            while not self.move_to_keyframe(sit_frame):
                time.sleep(0.02)
            self.relax_all_servos()
            log_info(f"[MOTION] control loop stopped at index: {self.control_loop_index}")
            self.control_loop_index = 0

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

                self.control_loop_start_time.append(current_time - next_control_loop_time)
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
                self.current_action.current_frame = self.current_action.current_frame.next

                if self.current_action.current_frame is None:
                    completed_action = self.current_action
                    self._emit_action_lifecycle("completed", completed_action)
                    self.current_action = self.get_next_action()

    def move_to_keyframe(self, new_frame: Keyframe) -> bool:
        self._moving_event.set()
        now_ms = millis()

        if not new_frame.is_initialized:
            self._init_frame(new_frame, now_ms)

        dt_s = self._compute_dt_s(now_ms)
        nominal_dt_s = self._compute_nominal_dt_s()
        desired_pan = new_frame.servo_destination["pan"]
        desired_tilt = new_frame.servo_destination["tilt"]
        pan_remaining = desired_pan - self.current_servo_position["pan"]
        tilt_remaining = desired_tilt - self.current_servo_position["tilt"]
        pan_v_max_raw = axis_step_v_max("pan", pan_remaining, nominal_dt_s, TUNING)
        tilt_v_max_raw = axis_step_v_max("tilt", tilt_remaining, nominal_dt_s, TUNING)
        pan_v_max = self._smooth_v_max("pan", pan_v_max_raw, nominal_dt_s)
        tilt_v_max = self._smooth_v_max("tilt", tilt_v_max_raw, nominal_dt_s)
        pan_a_max = TUNING.pan_a_max
        tilt_a_max = TUNING.tilt_a_max

        limited_pan = limit_step(
            self.current_servo_position["pan"],
            desired_pan,
            self.axis_v,
            "pan",
            dt_s,
            pan_v_max,
            pan_a_max,
            eps=TUNING.position_eps_deg,
        )

        limited_tilt = limit_step(
            self.current_servo_position["tilt"],
            desired_tilt,
            self.axis_v,
            "tilt",
            dt_s,
            tilt_v_max,
            tilt_a_max,
            eps=TUNING.position_eps_deg,
        )

        if abs(limited_pan - self.current_servo_position["pan"]) > 1.0:
            last_pan_log_ms = new_frame.last_pan_log_ms
            if last_pan_log_ms is None or now_ms - last_pan_log_ms >= 200:
                log_info(
                    "[MOTION] [%s] 'pan' servo to (%.2f) (wanted: %.2f) "
                    "(PAN_V_MAX:%.3f) (Elapsed ms:%s)",
                    new_frame.name,
                    limited_pan,
                    desired_pan,
                    pan_v_max,
                    now_ms - new_frame.start_time_ms,
                )
                new_frame.last_pan_log_ms = now_ms

        self.current_servo_position["pan"] = limited_pan
        self.current_servo_position["tilt"] = limited_tilt
        self.servo_registry.servos["pan"].write_value(limited_pan)
        self.servo_registry.servos["tilt"].write_value(limited_tilt)

        eps = TUNING.at_dest_eps_deg
        at_dest = (
            abs(self.current_servo_position["pan"] - desired_pan) <= eps
            and abs(self.current_servo_position["tilt"] - desired_tilt) <= eps
        )

        if self._should_emit_motion_debug_log(now_ms):
            log_info(
                "[MOTION][debug] dt=%.4f pan_v=%.2f pan_vmax=%.2f tilt_v=%.2f tilt_vmax=%.2f "
                "pan_err=%.2f tilt_err=%.2f",
                dt_s,
                self.axis_v["pan"],
                pan_v_max,
                self.axis_v["tilt"],
                tilt_v_max,
                desired_pan - self.current_servo_position["pan"],
                desired_tilt - self.current_servo_position["tilt"],
            )

        done = self._frame_done(new_frame, at_dest, now_ms)

        if done:
            self.current_servo_position["pan"] = desired_pan
            self.current_servo_position["tilt"] = desired_tilt
            self.servo_registry.servos["pan"].write_value(desired_pan)
            self.servo_registry.servos["tilt"].write_value(desired_tilt)

            elapsed_ms = now_ms - new_frame.start_time_ms
            log_debug(
                "[MOTION] 'pan' servo move completed (Cmd: %.3f) "
                "(Position: %.2f) (Elapsed ms: %s)",
                desired_pan,
                desired_pan,
                elapsed_ms,
            )
            log_debug(
                "[MOTION] 'tilt' servo move completed (Cmd: %.3f) "
                "(Position: %.2f) (Elapsed ms: %s)",
                desired_tilt,
                desired_tilt,
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
            "pan": float(self.current_servo_position["pan"]),
            "tilt": float(self.current_servo_position["tilt"]),
        }
        frame.delta_pos = {
            "pan": float(frame.servo_destination["pan"]) - frame.start_pos["pan"],
            "tilt": float(frame.servo_destination["tilt"]) - frame.start_pos["tilt"],
        }

        log_info(
            "[MOTION] New motion frame started (Name:%s) (Duration:%sms) "
            "pan %.2f->%.2f tilt %.2f->%.2f (deadline_ms:%s)",
            frame.name,
            frame.duration_ms,
            self.current_servo_position["pan"],
            frame.servo_destination["pan"],
            self.current_servo_position["tilt"],
            frame.servo_destination["tilt"],
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

    def generate_base_keyframe(self, pan_degrees: int, tilt_degrees: int) -> Keyframe:
        new_frame = Keyframe(final_target_time=self.transition_time, name="base")
        new_frame.servo_destination["pan"] = pan_degrees
        new_frame.servo_destination["tilt"] = tilt_degrees
        return new_frame

    def relax_all_servos(self) -> None:
        self.servo_registry.servos["pan"].relax()
        self.servo_registry.servos["tilt"].relax()

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

    def register_action_lifecycle_callback(self, callback: Callable[[str, Action], None]) -> None:
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
