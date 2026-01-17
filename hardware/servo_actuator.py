"""Servo actuator helper."""

from __future__ import annotations

from dataclasses import dataclass

from hardware.pca9685_servo_controller import PCA9685Actuator


def map_range(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map a value from one range to another."""

    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


@dataclass
class ServoActuator:
    """Represents a single servo driven by a PCA9685 controller."""

    id: int = 0
    name: str = ""
    min_angle: float = -90
    max_angle: float = 90
    offset: float = 0
    neutral_angle: float = 0
    is_reversed: bool = False
    pwm: PCA9685Actuator | None = None

    SERVO_MIN_PWM: int = 500
    SERVO_MAX_PWM: int = 2500
    SERVO_NEUTRAL: int = 1500

    SERVO_MAX_DEG: int = 90
    SERVO_MIN_DEG: int = -90

    def __post_init__(self) -> None:
        self.is_connected = True
        self.current_angle = self.neutral_angle
        if self.pwm is None:
            new_pwm = PCA9685Actuator(0x40, debug=False)
            new_pwm.setPWMFreq(50)
            self.pwm = new_pwm

    def initialize(self) -> None:
        """Initialize the servo (placeholder)."""

    def write_value(self, new_angle: float = 0) -> None:
        """Write a new target angle to the servo."""

        if not self.is_connected:
            return

        if new_angle > self.max_angle:
            self.current_angle = self.max_angle
        elif new_angle < self.min_angle:
            self.current_angle = self.min_angle
        else:
            self.current_angle = new_angle

        if self.is_reversed:
            angle_offset = map_range(
                self.current_angle - self.neutral_angle,
                self.SERVO_MIN_DEG,
                self.SERVO_MAX_DEG,
                self.SERVO_MIN_PWM,
                self.SERVO_MAX_PWM,
            )
        else:
            angle_offset = map_range(
                self.current_angle + self.neutral_angle,
                self.SERVO_MAX_DEG,
                self.SERVO_MIN_DEG,
                self.SERVO_MIN_PWM,
                self.SERVO_MAX_PWM,
            )
        output_pulse = self.offset + angle_offset

        if output_pulse > self.SERVO_MAX_PWM:
            output_pulse = self.SERVO_MAX_PWM
        elif output_pulse < self.SERVO_MIN_PWM:
            output_pulse = self.SERVO_MIN_PWM

        self.pwm.write_value(self.id, int(output_pulse))

    def relax(self) -> None:
        """Relax the servo (disable PWM)."""

        self.pwm.setPWM(self.id, 0, 0)

    def neutral_position(self, neutral_offset: float = 0) -> None:
        """Move servo to neutral position."""

        self.write_value(self.neutral_angle + neutral_offset)

    def read_value(self) -> float:
        """Return the current servo angle."""

        return self.current_angle
