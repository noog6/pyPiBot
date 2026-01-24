"""Tool definitions for realtime API calls."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from hardware import ADS1015Sensor


ToolFn = Callable[..., Awaitable[Any]]

function_map: dict[str, ToolFn] = {}

tools: list[dict[str, Any]] = []


async def read_battery_voltage() -> dict[str, Any]:
    """Return the current LiPo battery voltage via the ADS1015 sensor."""

    sensor = ADS1015Sensor.get_instance()
    voltage = sensor.read_battery_voltage()
    return {
        "voltage": voltage,
        "unit": "V",
        "min_voltage": 7.0,
        "max_voltage": 8.4,
    }


tools.append(
    {
        "type": "function",
        "name": "read_battery_voltage",
        "description": (
            "Fetch the current voltage of the onboard 2S LiPo battery. "
            "Safe operating range is 7.0V to 8.4V. If the reading is within "
            "0.5V of the minimum voltage, complain about it; being near the max is fine."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
)

function_map["read_battery_voltage"] = read_battery_voltage
