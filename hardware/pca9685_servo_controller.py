"""PCA9685 servo controller."""

from __future__ import annotations

import importlib
import importlib.util
import math
import time
from typing import Any

from core.logging import logger as LOGGER


class PCA9685Actuator:
    """Low-level controller for the PCA9685 PWM driver."""

    __SUBADR1 = 0x02
    __SUBADR2 = 0x03
    __SUBADR3 = 0x04
    __MODE1 = 0x00
    __PRESCALE = 0xFE
    __LED0_ON_L = 0x06
    __LED0_ON_H = 0x07
    __LED0_OFF_L = 0x08
    __LED0_OFF_H = 0x09
    __ALLLED_ON_L = 0xFA
    __ALLLED_ON_H = 0xFB
    __ALLLED_OFF_L = 0xFC
    __ALLLED_OFF_H = 0xFD

    def __init__(self, address: int = 0x40, debug: bool = False) -> None:
        if importlib.util.find_spec("smbus") is None:
            raise RuntimeError("smbus is required for PCA9685Actuator")

        smbus = importlib.import_module("smbus")
        self.bus = smbus.SMBus(1)
        self.address = address
        self.debug = debug
        if self.debug:
            LOGGER.info("Resetting PCA9685")
        self.write(self.__MODE1, 0x00)

    def write(self, reg: int, value: int) -> None:
        """Write an 8-bit value to the specified register/address."""

        self.bus.write_byte_data(self.address, reg, value)
        if self.debug:
            LOGGER.info("I2C: Write 0x%02X to register 0x%02X", value, reg)

    def read(self, reg: int) -> int:
        """Read an unsigned byte from the I2C device."""

        result = self.bus.read_byte_data(self.address, reg)
        if self.debug:
            LOGGER.info(
                "I2C: Device 0x%02X returned 0x%02X from reg 0x%02X",
                self.address,
                result & 0xFF,
                reg,
            )
        return result

    def setPWMFreq(self, freq: float) -> None:
        """Set the PWM frequency."""

        prescaleval = 25000000.0
        prescaleval /= 4096.0
        prescaleval /= float(freq)
        prescaleval -= 1.0
        if self.debug:
            LOGGER.info("Setting PWM frequency to %s Hz", freq)
            LOGGER.info("Estimated pre-scale: %s", prescaleval)
        prescale = math.floor(prescaleval + 0.5)
        if self.debug:
            LOGGER.info("Final pre-scale: %s", prescale)

        oldmode = self.read(self.__MODE1)
        newmode = (oldmode & 0x7F) | 0x10
        self.write(self.__MODE1, newmode)
        self.write(self.__PRESCALE, int(math.floor(prescale)))
        self.write(self.__MODE1, oldmode)
        time.sleep(0.005)
        self.write(self.__MODE1, oldmode | 0x80)

    def setPWM(self, channel: int, on: int, off: int) -> None:
        """Set a single PWM channel."""

        self.write(self.__LED0_ON_L + 4 * channel, on & 0xFF)
        self.write(self.__LED0_ON_H + 4 * channel, on >> 8)
        self.write(self.__LED0_OFF_L + 4 * channel, off & 0xFF)
        self.write(self.__LED0_OFF_H + 4 * channel, off >> 8)
        if self.debug:
            LOGGER.info("channel: %s  LED_ON: %s LED_OFF: %s", channel, on, off)

    def setServoPulse(self, channel: int, pulse: float) -> None:
        """Set the servo pulse (PWM frequency must be 50Hz)."""

        pulse = pulse * 4096 / 20000
        self.setPWM(channel, 0, int(pulse))

    def initialize(self) -> None:
        """Reinitialize the controller."""

        smbus = importlib.import_module("smbus")
        self.bus = smbus.SMBus(1)
        if self.debug:
            LOGGER.info("Resetting PCA9685")
        self.write(self.__MODE1, 0x00)

    def write_value(self, channel: int, pulse: float) -> None:
        """Write a pulse value to the target channel."""

        self.setServoPulse(channel, pulse)


def _demo() -> None:
    pwm = PCA9685Actuator(0x40, debug=False)
    pwm.setPWMFreq(50)
    target_servos = list(range(16))
    while True:
        for servo_id in target_servos:
            LOGGER.info("Setting servo (%s) to neutral position", servo_id)
            pwm.write_value(servo_id, 1500)
        time.sleep(0.02)


if __name__ == "__main__":
    _demo()
