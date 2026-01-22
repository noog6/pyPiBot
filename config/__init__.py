"""Configuration package utilities."""

__all__ = ["ConfigController"]


def __getattr__(name: str):
    if name == "ConfigController":
        from config.controller import ConfigController

        return ConfigController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
