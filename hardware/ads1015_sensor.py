"""ADS1015 sensor controller."""

from __future__ import annotations

import importlib
import importlib.util
import time

from core.logging import logger as LOGGER

ADS_I2C_ADDRESS = 0x48

ADS_POINTER_CONVERT = 0x00
ADS_POINTER_CONFIG = 0x01
ADS_POINTER_LOWTHRESH = 0x02
ADS_POINTER_HIGHTHRESH = 0x03

ADS_CONFIG_OS_BUSY = 0x0000
ADS_CONFIG_OS_NOBUSY = 0x8000
ADS_CONFIG_OS_SINGLE_CONVERT = 0x8000
ADS_CONFIG_OS_NO_EFFECT = 0x0000
ADS_CONFIG_MUX_MUL_0_1 = 0x0000
ADS_CONFIG_MUX_MUL_0_3 = 0x1000
ADS_CONFIG_MUX_MUL_1_3 = 0x2000
ADS_CONFIG_MUX_MUL_2_3 = 0x3000
ADS_CONFIG_MUX_SINGLE_0 = 0x4000
ADS_CONFIG_MUX_SINGLE_1 = 0x5000
ADS_CONFIG_MUX_SINGLE_2 = 0x6000
ADS_CONFIG_MUX_SINGLE_3 = 0x7000
ADS_CONFIG_PGA_6144 = 0x0000
ADS_CONFIG_PGA_4096 = 0x0200
ADS_CONFIG_PGA_2048 = 0x0400
ADS_CONFIG_PGA_1024 = 0x0600
ADS_CONFIG_PGA_512 = 0x0800
ADS_CONFIG_PGA_256 = 0x0A00
ADS_CONFIG_MODE_CONTINUOUS = 0x0000
ADS_CONFIG_MODE_NOCONTINUOUS = 0x0100
ADS_CONFIG_DR_RATE_128 = 0x0000
ADS_CONFIG_DR_RATE_250 = 0x0020
ADS_CONFIG_DR_RATE_490 = 0x0040
ADS_CONFIG_DR_RATE_920 = 0x0060
ADS_CONFIG_DR_RATE_1600 = 0x0080
ADS_CONFIG_DR_RATE_2400 = 0x00A0
ADS_CONFIG_DR_RATE_3300 = 0x00C0
ADS_CONFIG_COMP_MODE_WINDOW = 0x0010
ADS_CONFIG_COMP_MODE_TRADITIONAL = 0x0000
ADS_CONFIG_COMP_POL_LOW = 0x0000
ADS_CONFIG_COMP_POL_HIGH = 0x0008
ADS_CONFIG_COMP_LAT = 0x0004
ADS_CONFIG_COMP_NONLAT = 0x0000
ADS_CONFIG_COMP_QUE_ONE = 0x0000
ADS_CONFIG_COMP_QUE_TWO = 0x0001
ADS_CONFIG_COMP_QUE_FOUR = 0x0002
ADS_CONFIG_COMP_QUE_NON = 0x0003

ACS711_31AB_ZERO_VOLTAGE = 1.65
ACS711_31AB_SENSITIVITY_V_PER_A = 0.045
ADS1015_LSB_VOLTS = 0.002
MUX_MASK = 0x7000

class ADS1015Sensor:
    """Singleton controller for ADS1015 sensor readings."""

    _instance: "ADS1015Sensor | None" = None

    def __init__(self, address: int = ADS_I2C_ADDRESS) -> None:
        if ADS1015Sensor._instance is not None:
            raise RuntimeError("You cannot create another ADS1015Sensor class")

        if importlib.util.find_spec("smbus") is None:
            raise RuntimeError("smbus is required for ADS1015Sensor")

        smbus = importlib.import_module("smbus")
        self._address = address
        self._bus = smbus.SMBus(1)
        self.Config_Set = 0
        self.initialize()
        self.read_value()

        ADS1015Sensor._instance = self

    @classmethod
    def get_instance(cls) -> "ADS1015Sensor":
        """Return the singleton instance of the controller."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def read_value(self) -> list[int]:
        """Read all channels and return their values."""

        values: list[int] = []
        for channel in range(4):
            values.append(self.single_read(channel))
        return values

    def _read_u16(self, cmd: int) -> int:
        data = self._bus.read_i2c_block_data(self._address, cmd, 2)
        return (data[0] << 8) | data[1]

    def _write_word(self, cmd: int, val: int) -> None:
        val_h = val & 0xFF
        val_l = val >> 8
        val = (val_h << 8) | val_l
        self._bus.write_word_data(self._address, cmd, val)

    def initialize(self) -> None:
        """Initialize ADS1015 settings."""

        self.Config_Set = (
            ADS_CONFIG_MODE_NOCONTINUOUS
            | ADS_CONFIG_PGA_4096
            | ADS_CONFIG_COMP_QUE_NON
            | ADS_CONFIG_COMP_NONLAT
            | ADS_CONFIG_COMP_POL_LOW
            | ADS_CONFIG_COMP_MODE_TRADITIONAL
            | ADS_CONFIG_DR_RATE_1600
        )
        self._write_word(ADS_POINTER_CONFIG, self.Config_Set)
        time.sleep(0.01)

    def single_read(self, channel: int) -> int:
        """Read a single channel data value."""
        config = self.Config_Set & ~MUX_MASK
        
        if channel == 0:
            config |= ADS_CONFIG_MUX_SINGLE_0
        elif channel == 1:
            config |= ADS_CONFIG_MUX_SINGLE_1
        elif channel == 2:
            config |= ADS_CONFIG_MUX_SINGLE_2
        elif channel == 3:
            config |= ADS_CONFIG_MUX_SINGLE_3
        else:
            raise ValueError(f"Invalid ADS1015 channel: {channel}")
            
        config |= ADS_CONFIG_OS_SINGLE_CONVERT
        self._write_word(ADS_POINTER_CONFIG, config)
        time.sleep(0.01)
        data = self._read_u16(ADS_POINTER_CONVERT) >> 4
        return data
    
    def read_channel_voltage(self, channel: int) -> float:
        data = self.single_read(channel)
        return data * ADS1015_LSB_VOLTS
    
    def read_battery_voltage(self) -> float:
        """Read battery voltage from channel 3."""

        resistor_r1 = 9750
        resistor_r2 = 6770

        analog_reading = self.read_channel_voltage(3)
        battery_voltage = round(analog_reading * ((resistor_r1 + resistor_r2) / resistor_r2), 2)

        return battery_voltage

    def read_system_amperage(self) -> float:
        sensor_voltage = self.read_channel_voltage(0)
        current_amps = (sensor_voltage - ACS711_31AB_ZERO_VOLTAGE) / ACS711_31AB_SENSITIVITY_V_PER_A
        return round(current_amps, 2)

def _demo() -> None:
    print("ADS1015 Test Program")
    ads1015 = ADS1015Sensor()
    while True:
        time.sleep(0.5)
        values = ads1015.read_value()
        battery_voltage = ads1015.read_battery_voltage()
        system_amperage = ads1015.read_system_amperage()
        print(f"AIN0={values[0]} AIN1={values[1]} AIN2={values[2]} AIN3={values[3]} | battery={battery_voltage}V | current={system_amperage}A")


if __name__ == "__main__":
    _demo()
