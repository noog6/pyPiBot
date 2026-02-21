"""Tests for memory_semantic configuration normalization."""

from __future__ import annotations

from pathlib import Path

from config.controller import ConfigController


def _reset_singletons() -> None:
    ConfigController._instance = None


def test_config_controller_sets_memory_semantic_defaults(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("assistant_name: Theo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    semantic_cfg = ConfigController.get_instance().get_config()["memory_semantic"]

    assert semantic_cfg["enabled"] is False
    assert semantic_cfg["provider"] == "none"
    assert semantic_cfg["rerank_enabled"] is False
    assert semantic_cfg["max_candidates_for_semantic"] == 64
    assert semantic_cfg["min_similarity"] == 0.25
    assert semantic_cfg["rerank_influence_min_cosine"] == 0.25
    assert semantic_cfg["dedupe_strong_match_cosine"] is None
    assert semantic_cfg["background_embedding_enabled"] is True
    assert semantic_cfg["rolling_backfill_batch_size"] == 4
    assert semantic_cfg["rolling_backfill_interval_idle_cycles"] == 15
    assert semantic_cfg["max_embedding_retries"] == 8
    assert semantic_cfg["write_timeout_ms"] == 75
    assert semantic_cfg["query_timeout_ms"] == 2000
    assert semantic_cfg["startup_canary_timeout_ms"] == 1500
    assert semantic_cfg["startup_canary_bypass"] is False
    assert semantic_cfg["max_writes_per_minute"] == 120
    assert semantic_cfg["max_queries_per_minute"] == 240
    assert semantic_cfg["openai"]["enabled"] is False
    assert semantic_cfg["openai"]["model"] == "text-embedding-3-small"
    assert semantic_cfg["openai"]["timeout_s"] == 10.0


def test_config_controller_clamps_semantic_startup_canary_timeout_floor(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        """memory_semantic:
  startup_canary_timeout_ms: 25
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    semantic_cfg = ConfigController.get_instance().get_config()["memory_semantic"]

    assert semantic_cfg["startup_canary_timeout_ms"] == 500


def test_repo_default_config_uses_supported_semantic_provider() -> None:
    config_text = Path("config/default.yaml").read_text(encoding="utf-8")

    assert 'provider: "openai"' in config_text


def test_config_controller_clamps_semantic_query_timeout_and_provider_timeout(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        """memory_semantic:
  query_timeout_ms: 50
  openai:
    timeout_s: 0.01
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    semantic_cfg = ConfigController.get_instance().get_config()["memory_semantic"]

    assert semantic_cfg["query_timeout_ms"] == 100
    assert semantic_cfg["openai"]["timeout_s"] > 0.1
    assert semantic_cfg["openai"]["timeout_s"] * 1000 > semantic_cfg["query_timeout_ms"]
