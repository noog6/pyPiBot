"""Configuration controller for YAML-based settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
        return normalized
