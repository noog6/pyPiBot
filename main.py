"""Command-line entry point for the Theo robot runtime."""

from __future__ import annotations

import argparse
import logging
import sys

from core.app import build_config, run


def configure_logging() -> None:
    """Configure application logging."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Raw command-line arguments.

    Returns:
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(description="Theo robot runtime")
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=[],
        help="Optional startup prompts for the assistant.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Application entry point.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code.
    """

    if argv is None:
        argv = sys.argv[1:]

    configure_logging()
    args = parse_args(argv)
    config = build_config(args.prompts)
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
