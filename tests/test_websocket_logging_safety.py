"""Regression coverage for websocket logging safety defaults."""

from __future__ import annotations

import logging

import main
from core import logging as core_logging


NOISY_LOGGERS = ("websockets.client", "websockets.protocol")


def test_configure_logging_installs_websocket_redaction_filter() -> None:
    main.configure_logging("DEBUG")

    for logger_name in NOISY_LOGGERS:
        filters = logging.getLogger(logger_name).filters
        assert any(
            isinstance(flt, core_logging.WebsocketAuthorizationRedactionFilter)
            for flt in filters
        )


def test_websocket_filter_redacts_bearer_authorization_value(
    caplog,
) -> None:
    core_logging.configure_websocket_library_logging()
    websocket_logger = logging.getLogger("websockets.client")
    original_level = websocket_logger.level
    original_propagate = websocket_logger.propagate

    try:
        websocket_logger.setLevel(logging.DEBUG)
        websocket_logger.propagate = True

        caplog.set_level(logging.DEBUG)
        websocket_logger.debug("> Authorization: Bearer sk-secret-123456789")

        messages = [record.getMessage() for record in caplog.records]

        assert any("> Authorization: Bearer <redacted>" in message for message in messages)
        assert all("sk-secret-123456789" not in message for message in messages)
    finally:
        websocket_logger.setLevel(original_level)
        websocket_logger.propagate = original_propagate
