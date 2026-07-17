"""Authoritative tool capability helpers shared by governance-adjacent runtime seams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


READ_PURPOSE_USER_REQUESTED_RESULT = "user_requested_result"
READ_PURPOSE_COMPOUND_INTERMEDIATE = "compound_intermediate"
READ_PURPOSE_RUNTIME_CONTEXT = "runtime_context"
READ_PURPOSE_BACKGROUND_OBSERVATION = "background_observation"
READ_PURPOSE_INTERNAL_HEALTH = "internal_health"
READ_PURPOSE_NOT_READ = "not_read_helper"

_READ_PURPOSES = {
    READ_PURPOSE_USER_REQUESTED_RESULT,
    READ_PURPOSE_COMPOUND_INTERMEDIATE,
    READ_PURPOSE_RUNTIME_CONTEXT,
    READ_PURPOSE_BACKGROUND_OBSERVATION,
    READ_PURPOSE_INTERNAL_HEALTH,
}

_ALLOWED_OPERATION_KINDS = {"read", "write", "action"}
_ALLOWED_RESULT_VISIBILITIES = {"user_facing", "internal", "compound", "background"}


@dataclass(frozen=True)
class ToolCapabilities:
    tool_name: str
    operation_kind: str = "action"
    result_visibility: str = "internal"
    user_result_eligible: bool = False
    compound_intermediate_eligible: bool = False
    runtime_only_background: bool = False
    capability_source: str = "default_conservative"
    governance_classification: str = ""

    @property
    def is_read_only(self) -> bool:
        return self.operation_kind == "read"


def _normalize_bool(value: Any, *, field_name: str, tool_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Tool capability '{field_name}' for '{tool_name}' must be a boolean.")


def capabilities_from_governance_payload(
    tool_name: str,
    payload: Mapping[str, Any] | None,
    *,
    require_declared: bool = False,
) -> ToolCapabilities:
    """Build capabilities from the single governance tool-spec declaration.

    Missing declarations are intentionally conservative: runtime treats them as
    non-read internal actions. Startup validation passes ``require_declared`` so
    registered tools fail fast instead of silently becoming user-facing reads.
    """

    normalized_name = str(tool_name or "").strip().lower()
    raw = payload or {}
    raw_capabilities = raw.get("capabilities")
    confirm_reason = str(raw.get("confirm_reason") or "").strip()
    if raw_capabilities is None:
        if require_declared:
            raise ValueError(f"Tool capabilities missing for registered tool '{normalized_name}'.")
        return ToolCapabilities(
            tool_name=normalized_name,
            governance_classification=confirm_reason,
        )
    if not isinstance(raw_capabilities, Mapping):
        raise ValueError(f"Tool capabilities for '{normalized_name}' must be a mapping.")

    missing = [
        field
        for field in (
            "operation_kind",
            "result_visibility",
            "user_result_eligible",
            "compound_intermediate_eligible",
            "runtime_only_background",
        )
        if field not in raw_capabilities
    ]
    if missing:
        raise ValueError(
            f"Tool capabilities for '{normalized_name}' missing required fields: "
            + ", ".join(missing)
        )

    operation_kind = str(raw_capabilities.get("operation_kind") or "").strip().lower()
    if operation_kind not in _ALLOWED_OPERATION_KINDS:
        raise ValueError(
            f"Tool capability operation_kind for '{normalized_name}' must be one of "
            f"{sorted(_ALLOWED_OPERATION_KINDS)}."
        )
    result_visibility = str(raw_capabilities.get("result_visibility") or "").strip().lower()
    if result_visibility not in _ALLOWED_RESULT_VISIBILITIES:
        raise ValueError(
            f"Tool capability result_visibility for '{normalized_name}' must be one of "
            f"{sorted(_ALLOWED_RESULT_VISIBILITIES)}."
        )

    return ToolCapabilities(
        tool_name=normalized_name,
        operation_kind=operation_kind,
        result_visibility=result_visibility,
        user_result_eligible=_normalize_bool(
            raw_capabilities.get("user_result_eligible"),
            field_name="user_result_eligible",
            tool_name=normalized_name,
        ),
        compound_intermediate_eligible=_normalize_bool(
            raw_capabilities.get("compound_intermediate_eligible"),
            field_name="compound_intermediate_eligible",
            tool_name=normalized_name,
        ),
        runtime_only_background=_normalize_bool(
            raw_capabilities.get("runtime_only_background"),
            field_name="runtime_only_background",
            tool_name=normalized_name,
        ),
        capability_source="governance.tool_specs.capabilities",
        governance_classification=confirm_reason,
    )


def build_tool_capability_map(
    raw_tool_specs: Mapping[str, Mapping[str, Any]] | None,
    *,
    registered_tool_names: set[str] | None = None,
    require_declared: bool = False,
) -> dict[str, ToolCapabilities]:
    raw_tool_specs = raw_tool_specs or {}
    names = {str(name).strip().lower() for name in raw_tool_specs}
    if registered_tool_names is not None:
        names = {str(name).strip().lower() for name in registered_tool_names}
    return {
        name: capabilities_from_governance_payload(
            name,
            raw_tool_specs.get(name) or raw_tool_specs.get(str(name)),
            require_declared=require_declared,
        )
        for name in sorted(names)
    }


def validate_tool_capability_declarations(
    raw_tool_specs: Mapping[str, Mapping[str, Any]] | None,
    *,
    registered_tool_names: set[str],
) -> None:
    build_tool_capability_map(
        raw_tool_specs,
        registered_tool_names=registered_tool_names,
        require_declared=True,
    )


def normalize_read_purpose(value: Any) -> str:
    purpose = str(value or "").strip().lower()
    if purpose == "internal_runtime_context":
        return READ_PURPOSE_RUNTIME_CONTEXT
    if purpose == "model_context_only":
        return READ_PURPOSE_RUNTIME_CONTEXT
    return purpose if purpose in _READ_PURPOSES else ""
