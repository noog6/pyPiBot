"""Hardware controller package."""

from __future__ import annotations

__all__ = [
    "ADS1015Sensor",
    "CameraController",
    "ICM20948Sensor",
    "LPS22HBSensor",
    "PCA9685Actuator",
    "ServoActuator",
    "ServoRegistry",
]

_IMPORTS = {
    "ADS1015Sensor": ("hardware.ads1015_sensor", "ADS1015Sensor"),
    "CameraController": ("hardware.camera_controller", "CameraController"),
    "ICM20948Sensor": ("hardware.icm20948_sensor", "ICM20948Sensor"),
    "LPS22HBSensor": ("hardware.lps22hb_sensor", "LPS22HBSensor"),
    "PCA9685Actuator": ("hardware.pca9685_servo_controller", "PCA9685Actuator"),
    "ServoActuator": ("hardware.servo_actuator", "ServoActuator"),
    "ServoRegistry": ("hardware.servo_registry", "ServoRegistry"),
}


def __getattr__(name: str):
    if name not in _IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _IMPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
