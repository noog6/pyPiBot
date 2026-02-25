"""Validation helpers for docs/tool_inventory.md governance metadata parity."""

from __future__ import annotations

from pathlib import Path

import yaml

_TRACKED_COLUMNS = {
    "Governance Tier": "governance_tier",
    "Side Effects": "side_effects",
    "Sensitivity": "sensitivity",
    "Default Confirmation": "default_confirmation",
}


def _parse_tool_inventory_rows(inventory_text: str) -> dict[str, dict[str, str]]:
    """Parse tool metadata rows from the tool inventory markdown table."""
    lines = [line.strip() for line in inventory_text.splitlines() if line.strip()]

    try:
        table_start = lines.index("| Tool | Governance Tier | Side Effects | Sensitivity | Default Confirmation |")
    except ValueError as exc:
        raise ValueError("Tool inventory table header is missing or changed.") from exc

    rows: dict[str, dict[str, str]] = {}
    for line in lines[table_start + 2 :]:
        if not line.startswith("|"):
            continue

        raw_cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(raw_cells) != 5:
            continue

        tool_cell = raw_cells[0]
        if not (tool_cell.startswith("`") and tool_cell.endswith("`")):
            continue

        tool_name = tool_cell.strip("`")
        rows[tool_name] = {
            "governance_tier": raw_cells[1].upper(),
            "side_effects": raw_cells[2].upper(),
            "sensitivity": raw_cells[3].upper(),
            "default_confirmation": raw_cells[4].upper(),
        }

    if not rows:
        raise ValueError("No tool rows were parsed from docs/tool_inventory.md.")

    return rows


def collect_tool_inventory_mismatches(
    inventory_path: Path = Path("docs/tool_inventory.md"),
    config_path: Path = Path("config/default.yaml"),
) -> list[str]:
    """Return parity mismatches between docs/tool_inventory.md and config tool specs."""
    inventory_rows = _parse_tool_inventory_rows(inventory_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tool_specs = config["governance"]["tool_specs"]

    mismatches: list[str] = []

    inventory_tools = set(inventory_rows)
    spec_tools = set(tool_specs)

    missing_in_inventory = sorted(spec_tools - inventory_tools)
    if missing_in_inventory:
        mismatches.append(
            "docs/tool_inventory.md is missing tool rows for: "
            + ", ".join(missing_in_inventory)
        )

    unknown_in_inventory = sorted(inventory_tools - spec_tools)
    if unknown_in_inventory:
        mismatches.append(
            "docs/tool_inventory.md contains unknown tools: " + ", ".join(unknown_in_inventory)
        )

    for tool_name in sorted(inventory_tools & spec_tools):
        row = inventory_rows[tool_name]
        spec = tool_specs[tool_name]
        for display_name, spec_key in _TRACKED_COLUMNS.items():
            expected = str(spec[spec_key]).upper()
            actual = row[spec_key]
            if actual != expected:
                mismatches.append(
                    f"{tool_name}: {display_name} is '{actual}' in docs/tool_inventory.md "
                    f"but '{expected}' in config/default.yaml"
                )

    return mismatches


def assert_tool_inventory_matches_config(
    inventory_path: Path = Path("docs/tool_inventory.md"),
    config_path: Path = Path("config/default.yaml"),
) -> None:
    """Raise ValueError when docs tool inventory diverges from canonical config."""
    mismatches = collect_tool_inventory_mismatches(inventory_path=inventory_path, config_path=config_path)
    if mismatches:
        raise ValueError("\n".join(mismatches))
