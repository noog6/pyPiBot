"""Command-line entry point for running diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from config.diagnostics import probe as config_probe
from diagnostics.models import DiagnosticStatus
from diagnostics.runner import format_results, run_diagnostics
from ai.diagnostics import probe as ai_probe
from core.diagnostics import probe as core_probe
from interaction.audio_hal import FakeAudioBackend
from interaction.diagnostics import probe as audio_probe
from interaction.microphone_hal import FakeInputBackend
from interaction.microphone_diagnostics import probe as microphone_probe
from hardware.diagnostics import HardwareProbeConfig, probe as hardware_probe
from motion.diagnostics import MotionProbeConfig, probe as motion_probe
from services.diagnostics import probe as services_probe
from storage.diagnostics import probe as storage_probe


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run diagnostics probes.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run probes against a temporary offline directory.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Optional base directory for offline diagnostics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run diagnostics and return an exit code."""

    args = parse_args(argv)
    base_dir = args.base_dir

    if args.offline and base_dir is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_base = Path(tmp_dir)

            config_dir = tmp_base / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "default.yaml").write_text("{}", encoding="utf-8")

            services_dir = tmp_base / "services"
            services_dir.mkdir(parents=True, exist_ok=True)
            (services_dir / "__init__.py").write_text("", encoding="utf-8")
            (services_dir / "offline_service.py").write_text("# offline", encoding="utf-8")

            def config_probe_offline():
                return config_probe(base_dir=tmp_base)

            def ai_probe_offline():
                return ai_probe(api_key="offline-test", require_websockets=False)

            def core_probe_offline():
                return core_probe()

            def audio_probe_offline():
                return audio_probe(backend=FakeAudioBackend())

            def microphone_probe_offline():
                return microphone_probe(backend=FakeInputBackend())

            def hardware_probe_offline():
                return hardware_probe(
                    config=HardwareProbeConfig(require_all=False),
                    available_modules={"smbus", "numpy", "PIL"},
                )

            def motion_probe_offline():
                return motion_probe(
                    servo_names=["pan", "tilt"],
                    config=MotionProbeConfig(),
                )

            def services_probe_offline():
                return services_probe(base_dir=tmp_base)

            def storage_probe_offline():
                return storage_probe(base_dir=tmp_base)

            results = run_diagnostics(
                [
                    config_probe_offline,
                    ai_probe_offline,
                    core_probe_offline,
                    audio_probe_offline,
                    microphone_probe_offline,
                    hardware_probe_offline,
                    motion_probe_offline,
                    services_probe_offline,
                    storage_probe_offline,
                ]
            )
    else:
        def config_probe_with_base():
            return config_probe(base_dir=base_dir)

        def ai_probe_live():
            return ai_probe()

        def core_probe_live():
            return core_probe()

        def audio_probe_live():
            return audio_probe()

        def microphone_probe_live():
            return microphone_probe()

        def hardware_probe_live():
            return hardware_probe(config=HardwareProbeConfig(require_all=False))

        def motion_probe_live():
            return motion_probe()

        def services_probe_with_base():
            return services_probe(base_dir=base_dir)

        def storage_probe_with_base():
            return storage_probe(base_dir=base_dir)

        results = run_diagnostics(
            [
                config_probe_with_base,
                ai_probe_live,
                core_probe_live,
                audio_probe_live,
                microphone_probe_live,
                hardware_probe_live,
                motion_probe_live,
                services_probe_with_base,
                storage_probe_with_base,
            ]
        )

    print(format_results(results))

    has_failures = any(result.status is DiagnosticStatus.FAIL for result in results)
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
