"""Storage package utilities."""

__all__ = [
    "StorageController",
    "create_memory_store",
    "create_research_budget_store",
    "create_user_profile_store",
    "probe",
]


def __getattr__(name: str):
    if name == "StorageController":
        from storage.controller import StorageController

        return StorageController
    if name == "create_memory_store":
        from storage.factories import create_memory_store

        return create_memory_store
    if name == "create_user_profile_store":
        from storage.factories import create_user_profile_store

        return create_user_profile_store
    if name == "create_research_budget_store":
        from storage.factories import create_research_budget_store

        return create_research_budget_store
    if name == "probe":
        from storage.diagnostics import probe

        return probe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
