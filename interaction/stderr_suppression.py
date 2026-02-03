"""Helpers to temporarily suppress noisy stderr output during audio init."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import threading
from typing import Iterator

from core.logging import logger as default_logger


_SUPPRESS_LOCK = threading.RLock()


def _audio_debug_enabled(env_var: str) -> bool:
    value = os.getenv(env_var, "")
    return value.lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def suppress_noisy_stderr(
    reason: str,
    *,
    env_var: str = "THEO_AUDIO_DEBUG",
    logger=default_logger,
) -> Iterator[None]:
    """Temporarily silence stderr (including fd=2) for noisy audio init.

    Uses fd-level redirection so C libraries (ALSA/JACK/PortAudio) are silenced.
    Note: suppression is process-wide; other threads writing to stderr during the
    window may have their output hidden, so keep the scope tight.
    """

    if _audio_debug_enabled(env_var):
        yield
        return

    with _SUPPRESS_LOCK:
        saved_stderr = sys.stderr
        saved_fd = os.dup(2)
        temp_file = tempfile.TemporaryFile(mode="w+b")
        exc: BaseException | None = None

        try:
            saved_stderr.flush()
            os.dup2(temp_file.fileno(), 2)
            sys.stderr = io.TextIOWrapper(
                os.fdopen(2, "wb", closefd=False),
                encoding=getattr(saved_stderr, "encoding", "utf-8"),
                errors="replace",
                line_buffering=True,
            )
            yield
        except BaseException as err:
            exc = err
            raise
        finally:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            sys.stderr = saved_stderr

            if exc and logger is not None:
                temp_file.seek(0)
                suppressed = temp_file.read().decode(errors="replace")
                if suppressed.strip():
                    logger.debug(
                        "Suppressed stderr during %s:\n%s",
                        reason,
                        suppressed.rstrip(),
                    )
            temp_file.close()
