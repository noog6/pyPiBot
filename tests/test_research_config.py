"""Tests for research configuration defaults and normalization."""

from __future__ import annotations

from pathlib import Path

from config.controller import ConfigController


def _reset_singletons() -> None:
    ConfigController._instance = None


def test_config_controller_sets_safe_research_defaults(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text("assistant_name: Theo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    research_cfg = ConfigController.get_instance().get_config()["research"]

    assert research_cfg["enabled"] is False
    assert research_cfg["provider"] == "null"
    assert research_cfg["packet_schema"] == "research_packet_v1"
    assert research_cfg["permission_required"] is True
    assert research_cfg["openai"]["enabled"] is False
    assert research_cfg["openai"]["model"] == "gpt-4.1-mini"
    assert research_cfg["firecrawl"]["enabled"] is False
    assert research_cfg["firecrawl"]["max_pages"] == 1
    assert research_cfg["firecrawl"]["allowlist_mode"] == "public"
    assert research_cfg["firecrawl"]["allowlist_domains"] == []
    assert research_cfg["budget"]["daily_limit"] == 20
    assert research_cfg["cache"]["ttl_hours"] == 24
    assert research_cfg["escalation"]["max_rounds"] == 1
