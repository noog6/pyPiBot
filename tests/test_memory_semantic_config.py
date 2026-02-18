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
    assert semantic_cfg["background_embedding_enabled"] is True
    assert semantic_cfg["write_timeout_ms"] == 75
    assert semantic_cfg["query_timeout_ms"] == 40
    assert semantic_cfg["max_writes_per_minute"] == 120
    assert semantic_cfg["max_queries_per_minute"] == 240
