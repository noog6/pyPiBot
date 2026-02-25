"""Tests for one-shot embedding probe diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

from services.embedding_provider import EmbeddingProvider, EmbeddingResult
from services.memory_manager import run_embedding_probe_once


class _ProbeErrorProvider(EmbeddingProvider):
    def __init__(self, *, error_code: str, timeout_budget_ms: int | None = None) -> None:
        self._error_code = error_code
        self._timeout_budget_ms = timeout_budget_ms

    def embed_text(self, text: str) -> EmbeddingResult:
        return SimpleNamespace(
            vector=b"",
            dimension=0,
            model="stub-probe",
            model_version="v1",
            vector_norm=None,
            provider="openai",
            status="error",
            error_code=self._error_code,
            error_class="ProviderTimeout",
            timeout_budget_ms=self._timeout_budget_ms,
            timer_start="provider_wait_start",
            observed_elapsed_ms_at_timeout=432,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]


def test_run_embedding_probe_once_normalizes_legacy_wrapper_timeout_codes() -> None:
    probe = run_embedding_probe_once(
        provider=_ProbeErrorProvider(error_code="canary_timeout", timeout_budget_ms=321),
        enabled=True,
        bypass=False,
        timeout_ms=800,
    )

    assert probe["canary_success"] is False
    assert probe["error_code"] == "timeout_wrapper"
    assert probe["timeout_triggered"] == "none"
    assert probe["timeout_budget_ms"] == 321
    assert probe["timer_start"] == "provider_wait_start"
    assert probe["observed_elapsed_ms_at_timeout"] == 432


def test_run_embedding_probe_once_marks_provider_timeout_triggered_from_normalized_code() -> None:
    probe = run_embedding_probe_once(
        provider=_ProbeErrorProvider(error_code="request_timeout"),
        enabled=True,
        bypass=False,
        timeout_ms=800,
    )

    assert probe["canary_success"] is False
    assert probe["error_code"] == "timeout_provider"
    assert probe["timeout_triggered"] == "provider"
    assert probe["timeout_budget_ms"] == 800
    assert probe["timer_start"] == "provider_wait_start"
    assert probe["observed_elapsed_ms_at_timeout"] == 432
