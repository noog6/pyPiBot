"""Configuration controller for YAML-based settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MEMORY_SEMANTIC_QUERY_TIMEOUT_FLOOR_MS = 100
MEMORY_SEMANTIC_QUERY_TIMEOUT_DEFAULT_MS = 2000
MEMORY_SEMANTIC_OPENAI_TIMEOUT_DEFAULT_S = 10.0
MEMORY_SEMANTIC_STARTUP_CANARY_TIMEOUT_FLOOR_MS = 500
MEMORY_SEMANTIC_STARTUP_CANARY_TIMEOUT_DEFAULT_MS = 1500


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigPaths:
    """Filesystem paths for configuration files."""

    config_dir: Path
    config_file: Path
    override_file: Path


class ConfigController:
    """Singleton controller for loading and updating configuration."""

    _instance: "ConfigController | None" = None

    def __init__(self, config_file: str = "default.yaml") -> None:
        if ConfigController._instance is not None:
            raise RuntimeError("You cannot create another ConfigController class")

        config_dir = Path("config")
        self.paths = ConfigPaths(
            config_dir=config_dir,
            config_file=config_dir / config_file,
            override_file=config_dir / "override.yaml",
        )
        self.config: dict[str, Any] = {}
        self.load_config()

    @classmethod
    def get_instance(cls) -> "ConfigController":
        """Return the singleton instance of the controller."""

        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_config(self) -> None:
        """Load configuration from default and override YAML files."""

        with self.paths.config_file.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        if self.paths.override_file.exists():
            with self.paths.override_file.open("r", encoding="utf-8") as file:
                override_config = yaml.safe_load(file) or {}
            if override_config:
                config = self._deep_merge(config, override_config)

        self.config = self._normalize_legacy_config(config)

    def save_config(self, config: dict[str, Any]) -> None:
        """Persist configuration to override.yaml, archiving previous overrides."""

        if self.paths.override_file.exists():
            archive_index = 1
            archive_file = self._archive_path(archive_index)
            while archive_file.exists():
                archive_index += 1
                archive_file = self._archive_path(archive_index)
            self.paths.override_file.rename(archive_file)

        with self.paths.override_file.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file)

    def get_config(self) -> dict[str, Any]:
        """Return the currently loaded configuration."""

        return dict(self.config)

    def set_config(self, config: dict[str, Any]) -> None:
        """Set and persist configuration values."""

        self.config = dict(config)
        self.save_config(self.config)

    def _archive_path(self, index: int) -> Path:
        """Return the archive path for a given override index."""

        filename = f"override_{index:04d}.yaml"
        return self.paths.config_dir / filename

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge dictionaries, overriding base values with override values."""

        merged = dict(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _normalize_legacy_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Normalize config while preserving backwards-compatible battery keys."""

        normalized = dict(config)
        battery_cfg = dict(normalized.get("battery") or {})
        response_cfg = dict(battery_cfg.get("response") or {})

        battery_cfg["voltage_min"] = float(
            battery_cfg.get("voltage_min", normalized.get("battery_voltage_min", 7.0))
        )
        battery_cfg["voltage_max"] = float(
            battery_cfg.get("voltage_max", normalized.get("battery_voltage_max", 8.4))
        )
        battery_cfg["warning_percent"] = float(
            battery_cfg.get("warning_percent", normalized.get("battery_warning_percent", 50.0))
        )
        battery_cfg["critical_percent"] = float(
            battery_cfg.get("critical_percent", normalized.get("battery_critical_percent", 25.0))
        )
        battery_cfg["hysteresis_percent"] = float(
            battery_cfg.get("hysteresis_percent", normalized.get("battery_hysteresis_percent", 0.0))
        )

        response_cfg["enabled"] = bool(
            response_cfg.get("enabled", normalized.get("battery_response_enabled", True))
        )
        response_cfg["cooldown_s"] = float(
            response_cfg.get("cooldown_s", normalized.get("battery_response_cooldown_s", 60.0))
        )
        response_cfg["allow_warning"] = bool(
            response_cfg.get("allow_warning", normalized.get("battery_response_allow_warning", True))
        )
        response_cfg["allow_critical"] = bool(
            response_cfg.get("allow_critical", normalized.get("battery_response_allow_critical", True))
        )
        response_cfg["require_transition"] = bool(
            response_cfg.get(
                "require_transition",
                normalized.get("battery_response_require_transition", False),
            )
        )

        battery_cfg["response"] = response_cfg
        normalized["battery"] = battery_cfg

        research_cfg = dict(normalized.get("research") or {})
        research_cfg["enabled"] = bool(research_cfg.get("enabled", False))
        research_cfg["provider"] = str(research_cfg.get("provider", "null"))
        research_cfg["packet_schema"] = str(
            research_cfg.get("packet_schema", "research_packet_v1")
        )
        research_cfg["permission_required"] = bool(
            research_cfg.get("permission_required", True)
        )
        requested_research_mode = str(research_cfg.get("research_mode", "")).strip().lower()
        if requested_research_mode not in {"auto", "ask", "disabled"}:
            requested_research_mode = "ask" if research_cfg["permission_required"] else "auto"
        research_cfg["research_mode"] = requested_research_mode
        openai_cfg = dict(research_cfg.get("openai") or {})
        openai_cfg["enabled"] = bool(openai_cfg.get("enabled", False))
        openai_cfg["model"] = str(openai_cfg.get("model", "gpt-4.1-mini"))
        openai_cfg["timeout_s"] = float(openai_cfg.get("timeout_s", 30.0))
        openai_cfg["max_output_chars"] = int(openai_cfg.get("max_output_chars", 2400))
        openai_cfg["max_facts"] = int(openai_cfg.get("max_facts", 8))
        openai_cfg["max_sources"] = int(openai_cfg.get("max_sources", 6))
        research_cfg["openai"] = openai_cfg

        firecrawl_cfg = dict(research_cfg.get("firecrawl") or {})
        firecrawl_cfg["enabled"] = bool(firecrawl_cfg.get("enabled", False))
        firecrawl_cfg["pdf_ingestion_enabled"] = bool(
            firecrawl_cfg.get("pdf_ingestion_enabled", False)
        )
        firecrawl_cfg["timeout_s"] = float(firecrawl_cfg.get("timeout_s", 15.0))
        firecrawl_cfg["api_key"] = str(firecrawl_cfg.get("api_key", ""))
        firecrawl_cfg["max_pages"] = int(firecrawl_cfg.get("max_pages", 1))
        firecrawl_cfg["max_markdown_chars"] = int(
            firecrawl_cfg.get("max_markdown_chars", 20000)
        )
        firecrawl_cfg["cache_dir"] = str(
            firecrawl_cfg.get("cache_dir", "./var/research_cache")
        )
        firecrawl_cfg["cache_ttl_hours"] = int(
            firecrawl_cfg.get("cache_ttl_hours", 24)
        )
        firecrawl_cfg["allowlist_mode"] = str(
            firecrawl_cfg.get("allowlist_mode", "public")
        )
        allowlist_domains = firecrawl_cfg.get("allowlist_domains") or []
        firecrawl_cfg["allowlist_domains"] = [
            str(item)
            for item in allowlist_domains
            if isinstance(item, str) and item.strip()
        ]
        research_cfg["firecrawl"] = firecrawl_cfg

        budget_cfg = dict(research_cfg.get("budget") or {})
        budget_cfg["daily_limit"] = int(budget_cfg.get("daily_limit", 20))
        budget_cfg["state_file"] = str(
            budget_cfg.get("state_file", "./var/research_budget.json")
        )
        research_cfg["budget"] = budget_cfg

        cache_cfg = dict(research_cfg.get("cache") or {})
        cache_cfg["dir"] = str(cache_cfg.get("dir", "./var/research_cache"))
        cache_cfg["ttl_hours"] = int(cache_cfg.get("ttl_hours", 24))
        research_cfg["cache"] = cache_cfg

        escalation_cfg = dict(research_cfg.get("escalation") or {})
        escalation_cfg["enabled"] = bool(escalation_cfg.get("enabled", False))
        escalation_cfg["max_rounds"] = int(escalation_cfg.get("max_rounds", 1))
        research_cfg["escalation"] = escalation_cfg

        normalized["research"] = research_cfg

        memory_semantic_cfg = dict(normalized.get("memory_semantic") or {})
        memory_semantic_cfg["enabled"] = bool(memory_semantic_cfg.get("enabled", False))
        memory_semantic_cfg["provider"] = str(memory_semantic_cfg.get("provider", "none"))
        memory_semantic_cfg["rerank_enabled"] = bool(
            memory_semantic_cfg.get("rerank_enabled", False)
        )
        memory_semantic_cfg["max_candidates_for_semantic"] = int(
            memory_semantic_cfg.get("max_candidates_for_semantic", 64)
        )
        memory_semantic_cfg["min_similarity"] = float(
            memory_semantic_cfg.get("min_similarity", 0.25)
        )
        memory_semantic_cfg["rerank_influence_min_cosine"] = float(
            memory_semantic_cfg.get("rerank_influence_min_cosine", 0.25)
        )
        dedupe_cosine = memory_semantic_cfg.get("dedupe_strong_match_cosine")
        memory_semantic_cfg["dedupe_strong_match_cosine"] = (
            float(dedupe_cosine)
            if dedupe_cosine is not None
            else None
        )
        memory_semantic_cfg["background_embedding_enabled"] = bool(
            memory_semantic_cfg.get("background_embedding_enabled", True)
        )
        memory_semantic_cfg["inline_embedding_on_write_when_background_disabled"] = bool(
            memory_semantic_cfg.get("inline_embedding_on_write_when_background_disabled", False)
        )
        memory_semantic_cfg["rolling_backfill_batch_size"] = max(
            1,
            int(memory_semantic_cfg.get("rolling_backfill_batch_size", 4)),
        )
        memory_semantic_cfg["rolling_backfill_interval_idle_cycles"] = max(
            1,
            int(memory_semantic_cfg.get("rolling_backfill_interval_idle_cycles", 15)),
        )
        memory_semantic_cfg["max_embedding_retries"] = max(
            1,
            int(memory_semantic_cfg.get("max_embedding_retries", 8)),
        )
        memory_semantic_cfg["write_timeout_ms"] = max(
            1,
            int(memory_semantic_cfg.get("write_timeout_ms", 75)),
        )
        memory_semantic_cfg["query_timeout_ms"] = max(
            MEMORY_SEMANTIC_QUERY_TIMEOUT_FLOOR_MS,
            int(
                memory_semantic_cfg.get(
                    "query_timeout_ms",
                    MEMORY_SEMANTIC_QUERY_TIMEOUT_DEFAULT_MS,
                )
            ),
        )
        memory_semantic_cfg["startup_canary_timeout_ms"] = max(
            MEMORY_SEMANTIC_STARTUP_CANARY_TIMEOUT_FLOOR_MS,
            int(
                memory_semantic_cfg.get(
                    "startup_canary_timeout_ms",
                    MEMORY_SEMANTIC_STARTUP_CANARY_TIMEOUT_DEFAULT_MS,
                )
            ),
        )
        # Keep canary timeout independent from write timeout: startup canary validates
        # provider/network readiness and should tolerate cold-start jitter.
        memory_semantic_cfg["startup_canary_bypass"] = bool(
            memory_semantic_cfg.get("startup_canary_bypass", False)
        )
        memory_semantic_cfg["max_writes_per_minute"] = int(
            memory_semantic_cfg.get("max_writes_per_minute", 120)
        )
        memory_semantic_cfg["max_queries_per_minute"] = int(
            memory_semantic_cfg.get("max_queries_per_minute", 240)
        )
        memory_openai_cfg = dict(memory_semantic_cfg.get("openai") or {})
        memory_openai_cfg["enabled"] = bool(memory_openai_cfg.get("enabled", False))
        memory_openai_cfg["model"] = str(
            memory_openai_cfg.get("model", "text-embedding-3-small")
        )
        memory_openai_cfg["timeout_s"] = max(
            0.1,
            float(memory_openai_cfg.get("timeout_s", MEMORY_SEMANTIC_OPENAI_TIMEOUT_DEFAULT_S)),
        )
        # Semantic timeout relationships are normalized to keep deterministic guardrails:
        # write_timeout_ms <= query_timeout_ms < provider_timeout_ms and
        # startup_canary_timeout_ms < provider_timeout_ms.
        if memory_semantic_cfg["write_timeout_ms"] > memory_semantic_cfg["query_timeout_ms"]:
            logger.warning(
                "memory_semantic.write_timeout_ms exceeds query_timeout_ms; clamping write timeout to query timeout.",
                extra={
                    "configured_write_timeout_ms": memory_semantic_cfg["write_timeout_ms"],
                    "configured_query_timeout_ms": memory_semantic_cfg["query_timeout_ms"],
                },
            )
            memory_semantic_cfg["write_timeout_ms"] = memory_semantic_cfg["query_timeout_ms"]

        # Query timeout is bounded to a practical floor and should remain below provider timeout
        # so semantic retrieval can fail open before provider-level request timeout is hit.
        provider_timeout_ms = int(memory_openai_cfg["timeout_s"] * 1000)
        min_provider_timeout_ms = (
            max(
                memory_semantic_cfg["query_timeout_ms"],
                memory_semantic_cfg["startup_canary_timeout_ms"],
            )
            + 1
        )
        if provider_timeout_ms < min_provider_timeout_ms:
            logger.warning(
                "memory_semantic.openai.timeout_s too low for configured semantic timeout budget; increasing provider timeout.",
                extra={
                    "configured_provider_timeout_ms": provider_timeout_ms,
                    "required_min_provider_timeout_ms": min_provider_timeout_ms,
                    "query_timeout_ms": memory_semantic_cfg["query_timeout_ms"],
                    "startup_canary_timeout_ms": memory_semantic_cfg["startup_canary_timeout_ms"],
                },
            )
            provider_timeout_ms = min_provider_timeout_ms
            memory_openai_cfg["timeout_s"] = provider_timeout_ms / 1000.0
        memory_semantic_cfg["openai"] = memory_openai_cfg
        normalized["memory_semantic"] = memory_semantic_cfg

        logging_cfg = dict(normalized.get("logging") or {})
        user_transcripts_cfg = dict(logging_cfg.get("user_transcripts") or {})
        user_transcripts_cfg["enabled"] = bool(user_transcripts_cfg.get("enabled", False))

        partials_cfg = dict(user_transcripts_cfg.get("partials") or {})
        partials_cfg["enabled"] = bool(partials_cfg.get("enabled", False))
        partials_cfg["min_chars_delta"] = max(1, int(partials_cfg.get("min_chars_delta", 8)))
        user_transcripts_cfg["partials"] = partials_cfg

        redact_cfg = dict(user_transcripts_cfg.get("redact") or {})
        redact_cfg["enabled"] = bool(redact_cfg.get("enabled", True))
        user_transcripts_cfg["redact"] = redact_cfg

        logging_cfg["user_transcripts"] = user_transcripts_cfg
        normalized["logging"] = logging_cfg

        return normalized
