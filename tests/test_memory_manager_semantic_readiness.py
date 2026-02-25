from __future__ import annotations

from types import SimpleNamespace

from services.embedding_provider import NoopEmbeddingProvider
from services.memory_manager import MemoryManager


class _WorkingProvider:
    def embed_text(self, text: str):
        return text


class _InvalidProvider:
    pass


def _make_manager(*, provider_name: str = "openai") -> MemoryManager:
    manager = MemoryManager.__new__(MemoryManager)
    manager._semantic_config = SimpleNamespace(provider=provider_name)
    manager._semantic_provider_enabled = {"openai": True}
    manager._semantic_canary_bypass = False
    manager._semantic_canary_last = {"canary_success": False, "error_code": "not_run"}
    manager._embedding_provider = _WorkingProvider()
    return manager


def test_provider_none_or_noop_is_not_configured() -> None:
    manager_none = _make_manager(provider_name="none")
    manager_noop = _make_manager(provider_name="noop")

    assert manager_none._is_semantic_provider_ready() == (False, "provider_not_configured")
    assert manager_noop._is_semantic_provider_ready() == (False, "provider_not_configured")


def test_provider_missing_reports_provider_missing() -> None:
    manager = _make_manager()
    manager._embedding_provider = None

    assert manager._is_semantic_provider_ready() == (False, "provider_missing")


def test_noop_provider_reports_provider_unavailable() -> None:
    manager = _make_manager()
    manager._embedding_provider = NoopEmbeddingProvider()

    assert manager._is_semantic_provider_ready() == (False, "provider_unavailable")


def test_canary_bypass_reports_bypassed_and_ready() -> None:
    manager = _make_manager()
    manager._semantic_canary_bypass = True

    assert manager._is_semantic_provider_ready() == (True, "canary_bypassed")


def test_canary_state_absent_or_empty_reports_not_run() -> None:
    manager_absent = _make_manager()
    manager_empty = _make_manager()
    delattr(manager_absent, "_semantic_canary_last")
    manager_empty._semantic_canary_last = {}

    assert manager_absent._is_semantic_provider_ready() == (False, "canary_not_run")
    assert manager_empty._is_semantic_provider_ready() == (False, "canary_not_run")


def test_canary_error_code_not_run_reports_not_run() -> None:
    manager = _make_manager()
    manager._semantic_canary_last = {"canary_success": False, "error_code": "not_run"}

    assert manager._is_semantic_provider_ready() == (False, "canary_not_run")


def test_canary_success_reports_ready() -> None:
    manager = _make_manager()
    manager._semantic_canary_last = {"canary_success": True, "error_code": "none"}

    assert manager._is_semantic_provider_ready() == (True, "canary_ready")


def test_canary_timeout_failure_reports_known_class() -> None:
    manager = _make_manager()
    manager._semantic_canary_last = {"canary_success": False, "error_code": "timeout"}

    assert manager._is_semantic_provider_ready() == (False, "canary_failed:timeout")


def test_canary_uncategorized_failure_reports_other() -> None:
    manager = _make_manager()
    manager._semantic_canary_last = {"canary_success": False, "error_code": "strange_error_xyz"}

    assert manager._is_semantic_provider_ready() == (False, "canary_failed:other")


def test_canary_unknown_failure_reports_unknown() -> None:
    manager = _make_manager()
    manager._semantic_canary_last = {"canary_success": False, "error_code": "unknown"}

    assert manager._is_semantic_provider_ready() == (False, "canary_failed:unknown")


def test_provider_disabled_reports_provider_disabled() -> None:
    manager = _make_manager()
    manager._semantic_provider_enabled = {"openai": False}

    assert manager._is_semantic_provider_ready() == (False, "provider_disabled")


def test_provider_missing_embed_text_reports_provider_invalid() -> None:
    manager = _make_manager()
    manager._embedding_provider = _InvalidProvider()

    assert manager._is_semantic_provider_ready() == (False, "provider_invalid")
