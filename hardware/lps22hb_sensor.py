"""LPS22HB pressure/temperature sensor controller."""

from __future__ import annotations

import importlib
import importlib.util
import time

LPS22HB_I2C_ADDRESS = 0x5C

LPS_ID = 0xB1
LPS_INT_CFG = 0x0B
LPS_THS_P_L = 0x0C
LPS_THS_P_H = 0x0D
LPS_WHO_AM_I = 0x0F
LPS_CTRL_REG1 = 0x10
LPS_CTRL_REG2 = 0x11
LPS_CTRL_REG3 = 0x12
LPS_FIFO_CTRL = 0x14
LPS_REF_P_XL = 0x15
LPS_REF_P_L = 0x16
LPS_REF_P_H = 0x17
LPS_RPDS_L = 0x18
LPS_RPDS_H = 0x19
LPS_RES_CONF = 0x1A
LPS_INT_SOURCE = 0x25
LPS_FIFO_STATUS = 0x26
LPS_STATUS = 0x27
LPS_PRESS_OUT_XL = 0x28
LPS_PRESS_OUT_L = 0x29
LPS_PRESS_OUT_H = 0x2A
LPS_TEMP_OUT_L = 0x2B
LPS_TEMP_OUT_H = 0x2C
LPS_RES = 0x33


class LPS22HBSensor:
    """Singleton controller for the LPS22HB sensor."""

    _instance: "LPS22HBSensor | None" = None

    def __init__(self, address: int = LPS22HB_I2C_ADDRESS) -> None:
        if LPS22HBSensor._instance is not None:
            raise RuntimeError("You cannot create another LPS22HBSensor class")

        if importlib.util.find_spec("smbus") is None:
            raise RuntimeError("smbus is required for LPS22HBSensor")

        smbus = importlib.import_module("smbus")
        self._address = address
        self._bus = smbus.SMBus(1)
        self.air_pressure = 0.0
        self.air_temperature = 0.0
        self.reset()
        self._write_byte(
            LPS_CTRL_REG1,
            0x02,
        )

        LPS22HBSensor._instance = self

    @classmethod
    def get_instance(cls) -> "LPS22HBSensor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        buf = self._read_u16(LPS_CTRL_REG2)
        buf |= 0x04
        self._write_byte(LPS_CTRL_REG2, buf)
        while buf:
            buf = self._read_u16(LPS_CTRL_REG2)
            buf &= 0x04

    def initialize(self) -> None:
        self.reset()

    def read_value(self) -> tuple[float, float]:
        self.start_oneshot()
        self._wait_for_data_ready(timeout_s=0.1)
        air_pressure = self.get_pressure()
        air_temperature = self.get_temperature()
        return air_pressure, air_temperature

    def _wait_for_data_ready(self, timeout_s: float = 0.1) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self._read_byte(LPS_STATUS)
            if (status & 0x03) == 0x03:
                return
            time.sleep(0.005)

    def start_oneshot(self) -> None:
        buf = self._read_u16(LPS_CTRL_REG2)
        buf |= 0x01
        self._write_byte(LPS_CTRL_REG2, buf)

    def _read_byte(self, cmd: int) -> int:
        return self._bus.read_byte_data(self._address, cmd)

    def _read_u16(self, cmd: int) -> int:
        lsb = self._bus.read_byte_data(self._address, cmd)
        msb = self._bus.read_byte_data(self._address, cmd + 1)
        return (msb << 8) + lsb

    def _write_byte(self, cmd: int, val: int) -> None:
        self._bus.write_byte_data(self._address, cmd, val)

    def get_pressure(self) -> float:
        u8buf = [0, 0, 0]
        if (self._read_byte(LPS_STATUS) & 0x01) == 0x01:
            u8buf[0] = self._read_byte(LPS_PRESS_OUT_XL)
            u8buf[1] = self._read_byte(LPS_PRESS_OUT_L)
            u8buf[2] = self._read_byte(LPS_PRESS_OUT_H)
            self.air_pressure = ((u8buf[2] << 16) + (u8buf[1] << 8) + u8buf[0]) / 4096.0
        return self.air_pressure

    def convert_hpa_to_altitude(self, pressure: float) -> float:
        altitude = 44307.692 * (1 - (pressure / 1013.25) ** 0.190284)
        return altitude

    def get_temperature(self) -> float:
        u8buf = [0, 0]
        if (self._read_byte(LPS_STATUS) & 0x02) == 0x02:
            u8buf[0] = self._read_byte(LPS_TEMP_OUT_L)
            u8buf[1] = self._read_byte(LPS_TEMP_OUT_H)
            self.air_temperature = ((u8buf[1] << 8) + u8buf[0]) / 100.0
        return self.air_temperature
