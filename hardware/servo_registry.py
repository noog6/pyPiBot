"""Registry for servo actuators."""

from __future__ import annotations

from hardware.pca9685_servo_controller import PCA9685Actuator
from hardware.servo_actuator import ServoActuator


class ServoRegistry:
    """Singleton registry for servo actuators."""

    _instance: "ServoRegistry | None" = None

    def __init__(self) -> None:
        if ServoRegistry._instance is not None:
            raise RuntimeError("You cannot create another ServoRegistry class")

        self.pwm = PCA9685Actuator(0x40, debug=False)
        self.pwm.setPWMFreq(50)
        self.servos = self._create_servos()
        ServoRegistry._instance = self

    def _create_servos(self) -> dict[str, ServoActuator]:
        servos = {
            "pan": ServoActuator(
                id=9,
                name="Pan_Servo",
                min_angle=-90,
                max_angle=90,
                offset=0,
                neutral_angle=0,
                is_reversed=False,
                pwm=self.pwm,
            ),
            "tilt": ServoActuator(
                id=8,
                name="Tilt_Servo",
                min_angle=-45,
                max_angle=45,
                offset=0,
                neutral_angle=0,
                is_reversed=False,
                pwm=self.pwm,
            ),
        }

        return servos

    def get_servos(self) -> dict[str, ServoActuator]:
        """Return the existing servo instances."""

        return self.servos

    @classmethod
    def get_instance(cls) -> "ServoRegistry":
        """Return the singleton instance of the registry."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_servo_names(self) -> list[str]:
        """Return a list of the names of all servos in the registry."""

        return list(self.get_servos().keys())

    def refresh_servos(self) -> None:
        """Destroy existing servo objects and recreate servos."""

        self.servos.clear()
        self.servos = self._create_servos()
