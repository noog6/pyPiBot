"""Tests for research configuration defaults and normalization."""

from __future__ import annotations

import logging
from pathlib import Path

from config.controller import ConfigController
from services.research.openai_service import OpenAIResearchService, build_openai_service_or_null


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
    assert research_cfg["research_mode"] == "ask_on_assistant_or_unknown"
    assert research_cfg["openai"]["enabled"] is False
    assert research_cfg["openai"]["model"] == "gpt-4.1-mini"
    assert research_cfg["firecrawl"]["enabled"] is False
    assert research_cfg["firecrawl"]["pdf_ingestion_enabled"] is False
    assert research_cfg["firecrawl"]["timeout_s"] == 15.0
    assert research_cfg["firecrawl"]["max_pages"] == 1
    assert research_cfg["firecrawl"]["allowlist_mode"] == "public"
    assert research_cfg["firecrawl"]["allowlist_domains"] == []
    assert research_cfg["budget"]["daily_limit"] == 50
    assert research_cfg["cache"]["ttl_hours"] == 24
    assert research_cfg["escalation"]["max_rounds"] == 1


def test_config_controller_accepts_legacy_budget_state_file_with_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "default.yaml").write_text(
        """
assistant_name: Theo
research:
  enabled: true
  provider: openai
  openai:
    enabled: false
  budget:
    daily_limit: 5
    state_file: ./var/custom-budget.json
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    _reset_singletons()

    with caplog.at_level(logging.WARNING):
        research_cfg = ConfigController.get_instance().get_config()["research"]

    assert research_cfg["budget"]["daily_limit"] == 5
    assert research_cfg["budget"]["state_file"] == "./var/custom-budget.json"
    assert any(
        "research.budget.state_file is deprecated and ignored" in record.message
        for record in caplog.records
    )


def test_build_openai_service_ignores_legacy_budget_state_file_config() -> None:
    service = build_openai_service_or_null(
        {
            "research": {
                "enabled": True,
                "provider": "openai",
                "openai": {"enabled": False},
                "budget": {"daily_limit": 5, "state_file": "./var/custom-budget.json"},
            }
        }
    )

    assert isinstance(service, OpenAIResearchService)
    assert str(service._budget._legacy_state_file) == "var/research_budget.json"


def test_build_openai_service_logs_budget_startup_info(monkeypatch) -> None:
    info_calls: list[str] = []

    def _fake_info(msg: str, *args) -> None:
        info_calls.append(msg % args if args else msg)

    monkeypatch.setattr("services.research.openai_service.LOGGER.info", _fake_info)

    service = build_openai_service_or_null(
        {
            "research": {
                "enabled": True,
                "provider": "openai",
                "openai": {"enabled": False},
                "budget": {"daily_limit": 7},
            }
        }
    )

    assert isinstance(service, OpenAIResearchService)
    assert any(
        "[Research] startup daily_limit=7 authority=sqlite_storage_controller" in line
        for line in info_calls
    )


def test_build_openai_service_logs_deprecated_budget_state_file_signal(monkeypatch) -> None:
    info_calls: list[str] = []
    warning_calls: list[str] = []

    def _fake_info(msg: str, *args) -> None:
        info_calls.append(msg % args if args else msg)

    def _fake_warning(msg: str, *args) -> None:
        warning_calls.append(msg % args if args else msg)

    monkeypatch.setattr("services.research.openai_service.LOGGER.info", _fake_info)
    monkeypatch.setattr("services.research.openai_service.LOGGER.warning", _fake_warning)

    service = build_openai_service_or_null(
        {
            "research": {
                "enabled": True,
                "provider": "openai",
                "openai": {"enabled": False},
                "budget": {"daily_limit": 5, "state_file": "./var/legacy.json"},
            }
        }
    )

    assert isinstance(service, OpenAIResearchService)
    assert any("Ignoring deprecated research.budget.state_file" in line for line in warning_calls)
    assert any(
        "[Research] startup daily_limit=5 authority=sqlite_storage_controller" in line
        for line in info_calls
    )
