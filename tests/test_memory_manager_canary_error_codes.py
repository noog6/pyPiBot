"""Unit tests for semantic canary error code normalization."""

from __future__ import annotations

from services.memory_manager import _normalize_canary_error_code


def test_normalize_canary_error_code_request_timeout_maps_to_timeout() -> None:
    normalized = _normalize_canary_error_code("request_timeout")
    assert normalized == "timeout"
    assert normalized != "connection"


def test_normalize_canary_error_code_auth_forbidden_maps_to_auth() -> None:
    normalized = _normalize_canary_error_code("auth_forbidden")
    assert normalized == "auth"
    assert normalized != "connection"


def test_normalize_canary_error_code_provider_http_error_maps_to_http() -> None:
    normalized = _normalize_canary_error_code("provider_http_error")
    assert normalized == "http"
    assert normalized != "connection"


def test_normalize_canary_error_code_unknown_provider_code_is_preserved() -> None:
    normalized = _normalize_canary_error_code("strange_error_xyz")
    assert normalized == "strange_error_xyz"
    assert normalized != "connection"


def test_normalize_canary_error_code_empty_value_without_exception_maps_to_unknown() -> None:
    normalized = _normalize_canary_error_code(None)
    assert normalized == "unknown"
    assert normalized != "connection"

