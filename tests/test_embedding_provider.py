"""Tests for embedding provider normalization and fallback behavior."""

from __future__ import annotations

import json
import socket
from urllib import error

from services.embedding_provider import (
    NoopEmbeddingProvider,
    OpenAIEmbeddingProvider,
    build_embedding_provider,
)


def test_noop_provider_returns_structured_unavailable() -> None:
    provider = NoopEmbeddingProvider()

    result = provider.embed_text("hello")

    assert result.status == "unavailable"
    assert result.error_code == "provider_unavailable"
    assert result.dimension == 0
    assert result.vector == b""


def test_openai_provider_returns_missing_key_error_for_batch() -> None:
    provider = OpenAIEmbeddingProvider(api_key="", enabled=True)

    results = provider.embed_batch(["alpha", "beta"])

    assert len(results) == 2
    assert all(item.status == "error" for item in results)
    assert all(item.error_code == "missing_api_key" for item in results)


def test_openai_provider_normalizes_float_vector(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "data": [
                        {
                            "embedding": [3.0, 4.0],
                            "model": "text-embedding-3-small",
                        }
                    ]
                }
            ).encode("utf-8")

    def _fake_urlopen(req, timeout):  # noqa: ANN001
        return _Response()

    monkeypatch.setattr("services.embedding_provider.request.urlopen", _fake_urlopen)
    provider = OpenAIEmbeddingProvider(api_key="key", enabled=True)

    result = provider.embed_text("hello")

    assert result.status == "ready"
    assert result.dimension == 2
    assert len(result.vector) == 8
    assert result.vector_norm == 5.0


def test_openai_provider_handles_request_error(monkeypatch) -> None:
    def _failing_urlopen(req, timeout):  # noqa: ANN001
        raise error.URLError("boom")

    monkeypatch.setattr("services.embedding_provider.request.urlopen", _failing_urlopen)
    provider = OpenAIEmbeddingProvider(api_key="key", enabled=True)

    result = provider.embed_text("hello")

    assert result.status == "error"
    assert result.error_code == "connection_error"


def test_build_embedding_provider_uses_openai_when_configured() -> None:
    provider = build_embedding_provider(
        {
            "memory_semantic": {
                "enabled": True,
                "provider": "openai",
                "openai": {
                    "enabled": True,
                    "api_key": "test",
                    "model": "text-embedding-3-small",
                },
            }
        }
    )

    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_build_embedding_provider_requires_explicit_openai_enable() -> None:
    provider = build_embedding_provider(
        {
            "memory_semantic": {
                "enabled": True,
                "provider": "openai",
                "openai": {
                    "enabled": False,
                    "api_key": "test",
                },
            }
        }
    )

    result = provider.embed_text("hello")

    assert isinstance(provider, OpenAIEmbeddingProvider)
    assert result.status == "error"
    assert result.error_code == "provider_disabled"


def test_build_embedding_provider_surfaces_unsupported_provider() -> None:
    provider = build_embedding_provider(
        {"memory_semantic": {"provider": "local_light", "enabled": True}}
    )

    result = provider.embed_text("hello")

    assert isinstance(provider, NoopEmbeddingProvider)
    assert result.status == "unavailable"
    assert result.error_code == "unsupported_provider"
    assert "local_light" in (result.error_message or "")


def test_openai_provider_classifies_http_auth_error(monkeypatch) -> None:
    def _failing_urlopen(req, timeout):  # noqa: ANN001
        raise error.HTTPError(url="https://api.openai.com/v1/embeddings", code=401, msg="unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr("services.embedding_provider.request.urlopen", _failing_urlopen)
    provider = OpenAIEmbeddingProvider(api_key="key", enabled=True)

    result = provider.embed_text("hello")

    assert result.status == "error"
    assert result.error_code == "auth_forbidden"


def test_openai_provider_classifies_model_not_found_error(monkeypatch) -> None:
    def _failing_urlopen(req, timeout):  # noqa: ANN001
        raise error.HTTPError(url="https://api.openai.com/v1/embeddings", code=404, msg="not found", hdrs=None, fp=None)

    monkeypatch.setattr("services.embedding_provider.request.urlopen", _failing_urlopen)
    provider = OpenAIEmbeddingProvider(api_key="key", enabled=True)

    result = provider.embed_text("hello")

    assert result.status == "error"
    assert result.error_code == "model_not_found"


def test_openai_provider_classifies_timeout_error(monkeypatch) -> None:
    def _failing_urlopen(req, timeout):  # noqa: ANN001
        raise socket.timeout("timed out")

    monkeypatch.setattr("services.embedding_provider.request.urlopen", _failing_urlopen)
    provider = OpenAIEmbeddingProvider(api_key="key", enabled=True)

    result = provider.embed_text("hello")

    assert result.status == "error"
    assert result.error_code == "request_timeout"


def test_openai_provider_classifies_connection_refused(monkeypatch) -> None:
    def _failing_urlopen(req, timeout):  # noqa: ANN001
        raise error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr("services.embedding_provider.request.urlopen", _failing_urlopen)
    provider = OpenAIEmbeddingProvider(api_key="key", enabled=True)

    result = provider.embed_text("hello")

    assert result.status == "error"
    assert result.error_code == "connection_refused"
