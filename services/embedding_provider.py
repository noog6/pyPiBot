"""Embedding provider interfaces and minimal implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import math
import os
import struct
from typing import Any
from urllib import request


@dataclass(frozen=True)
class EmbeddingResult:
    """Normalized embedding payload for storage and retrieval."""

    vector: bytes
    dimension: int
    model: str
    model_version: str | None
    vector_norm: float | None
    provider: str
    status: str
    error_code: str | None = None
    error_message: str | None = None


class EmbeddingProvider(ABC):
    """Base interface for text embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text input."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a list of text inputs."""


class NoopEmbeddingProvider(EmbeddingProvider):
    """Provider that always returns a structured unavailable state."""

    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=b"",
            dimension=0,
            model="noop",
            model_version="v1",
            vector_norm=None,
            provider="noop",
            status="unavailable",
            error_code="provider_unavailable",
            error_message="Embedding provider is disabled or not configured.",
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider that degrades to structured errors."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout_s: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self._api_key = api_key.strip()
        self._model = model.strip() or "text-embedding-3-small"
        self._timeout_s = max(1.0, float(timeout_s))
        self._enabled = bool(enabled)

    def embed_text(self, text: str) -> EmbeddingResult:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        if not texts:
            return []
        if not self._enabled:
            return [self._failure_result(code="provider_disabled", message="OpenAI embedding provider is disabled.") for _ in texts]
        if not self._api_key:
            return [self._failure_result(code="missing_api_key", message="OPENAI_API_KEY is not set.") for _ in texts]

        try:
            payload = json.dumps(
                {
                    "model": self._model,
                    "input": texts,
                    "encoding_format": "float",
                }
            ).encode("utf-8")
            req = request.Request(
                "https://api.openai.com/v1/embeddings",
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            with request.urlopen(req, timeout=self._timeout_s) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            vectors = parsed.get("data") or []
            if len(vectors) != len(texts):
                return [
                    self._failure_result(
                        code="invalid_response_count",
                        message="Embedding response count did not match input count.",
                    )
                    for _ in texts
                ]
            return [self._success_result(item) for item in vectors]
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            return [
                self._failure_result(
                    code="provider_request_failed",
                    message=f"OpenAI embeddings request failed: {error_type}",
                )
                for _ in texts
            ]

    def _success_result(self, payload: dict[str, Any]) -> EmbeddingResult:
        embedding = payload.get("embedding") or []
        if not isinstance(embedding, list) or not all(isinstance(v, (int, float)) for v in embedding):
            return self._failure_result(
                code="invalid_embedding_vector",
                message="Provider returned a non-numeric embedding vector.",
            )
        vector = [float(value) for value in embedding]
        vector_bytes = _encode_float32_vector(vector)
        norm = _vector_norm(vector)
        return EmbeddingResult(
            vector=vector_bytes,
            dimension=len(vector),
            model=str(payload.get("model") or self._model),
            model_version=str(payload.get("model") or self._model),
            vector_norm=norm,
            provider="openai",
            status="ready",
        )

    def _failure_result(self, *, code: str, message: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=b"",
            dimension=0,
            model=self._model,
            model_version=None,
            vector_norm=None,
            provider="openai",
            status="error",
            error_code=code,
            error_message=message,
        )


def build_embedding_provider(config: dict[str, Any] | None = None) -> EmbeddingProvider:
    """Build a provider from config, with safe fallback to noop."""

    semantic_cfg = dict((config or {}).get("memory_semantic") or {})
    provider_name = str(semantic_cfg.get("provider", "none")).strip().lower()
    if provider_name not in {"openai"}:
        return NoopEmbeddingProvider()

    openai_cfg = dict(semantic_cfg.get("openai") or {})
    enabled = bool(openai_cfg.get("enabled", False)) and bool(semantic_cfg.get("enabled", False))
    model = str(openai_cfg.get("model", "text-embedding-3-small"))
    timeout_s = float(openai_cfg.get("timeout_s", 10.0))
    api_key = str(openai_cfg.get("api_key") or os.getenv("OPENAI_API_KEY", ""))
    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=model,
        timeout_s=timeout_s,
        enabled=enabled,
    )


def _encode_float32_vector(vector: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in vector)


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))
