"""Tests for semantic embedding canary CLI diagnostics."""

from __future__ import annotations

from pathlib import Path

from config.controller import ConfigController
from services.embedding_provider import EmbeddingProvider, EmbeddingResult
from services.memory_manager import MemoryManager, run_embed_canary_cli


class _ReadyProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=b"\x00\x00\x80?\x00\x00\x80?",
            dimension=2,
            model="stub-ready",
            model_version="v1",
            vector_norm=1.0,
            provider="openai",
            status="ready",
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]


class _FailingProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> EmbeddingResult:
        return EmbeddingResult(
            vector=b"",
            dimension=0,
            model="stub-fail",
            model_version=None,
            vector_norm=None,
            provider="openai",
            status="error",
            error_code="missing_api_key",
            error_message="missing key",
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed_text(text) for text in texts]


def _write_config(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        """memory_semantic:
  enabled: true
  provider: openai
  startup_canary_timeout_ms: 800
  openai:
    enabled: true
    model: text-embedding-3-small
    timeout_s: 3.5
""",
        encoding="utf-8",
    )


def _reset_singletons() -> None:
    ConfigController._instance = None
    MemoryManager._instance = None


def test_embed_canary_cli_success_output_and_exit_code(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    exit_code = run_embed_canary_cli(
        ["--embed-canary"],
        provider_factory=lambda _config: _ReadyProvider(),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "provider=openai" in output
    assert "model=text-embedding-3-small" in output
    assert "timeout_s=3.5" in output
    assert "startup_canary_timeout_ms=800" in output
    assert "api_key_present=False" in output
    assert "canary_success=True" in output
    assert "dimension=2" in output
    assert "error_code=none" in output


def test_embed_canary_cli_failure_output_and_exit_code(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    exit_code = run_embed_canary_cli(
        ["--embed-canary"],
        provider_factory=lambda _config: _FailingProvider(),
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "canary_success=False" in output
    assert "error_code=auth" in output


def test_embed_canary_cli_bypass_mode_succeeds_for_offline_validation(tmp_path: Path, monkeypatch, capsys) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYPIBOT_SEMANTIC_CANARY_BYPASS", "1")
    _reset_singletons()

    exit_code = run_embed_canary_cli(
        ["--embed-canary"],
        provider_factory=lambda _config: _FailingProvider(),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "canary_success=True" in output
    assert "error_code=bypassed" in output
