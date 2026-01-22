"""Storage package utilities."""

__all__ = ["StorageController", "probe"]


def __getattr__(name: str):
    if name == "StorageController":
        from storage.controller import StorageController

        return StorageController
    if name == "probe":
        from storage.diagnostics import probe

        return probe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
