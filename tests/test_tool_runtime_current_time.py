"""Focused tests for get_current_time runtime payload contract."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from services import tool_runtime


def test_get_current_time_success_payload_contains_required_fields() -> None:
    payload = tool_runtime.get_current_time()

    assert payload["status"] == "ok"
    required = {
        "status",
        "local_datetime_iso",
        "local_date",
        "local_time",
        "timezone_name",
        "utc_offset",
        "weekday",
        "unix_epoch_ms",
    }
    assert required <= set(payload)
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", payload["local_date"])
    assert re.match(r"^\d{2}:\d{2}:\d{2}$", payload["local_time"])
    assert re.match(r"^[+-]\d{2}:\d{2}$", payload["utc_offset"])
    assert isinstance(payload["unix_epoch_ms"], int)


def test_get_current_time_timezone_override_works() -> None:
    payload = tool_runtime.get_current_time({"timezone": "UTC"})

    assert payload["status"] == "ok"
    assert payload["timezone_source"] == "request"
    assert payload["requested_timezone"] == "UTC"
    assert payload["timezone_name"] == "UTC"
    assert payload["utc_offset"] == "+00:00"
    parsed = datetime.fromisoformat(payload["local_datetime_iso"])
    assert parsed.tzinfo is not None
    assert parsed.astimezone(timezone.utc).utcoffset().total_seconds() == 0


def test_get_current_time_invalid_timezone_returns_deterministic_error() -> None:
    payload = tool_runtime.get_current_time({"timezone": "Not/A_Real_Zone"})

    assert payload == {
        "status": "error",
        "error_code": "invalid_timezone",
        "message": "Unknown timezone: Not/A_Real_Zone",
        "requested_timezone": "Not/A_Real_Zone",
        "timezone_source": "request",
    }


def test_get_current_time_can_skip_period_of_day() -> None:
    payload = tool_runtime.get_current_time({"timezone": "UTC", "include_period_of_day": False})

    assert payload["status"] == "ok"
    assert "period_of_day" not in payload
