"""Centralized structlog configuration."""

from __future__ import annotations

import logging
import os
import sys

import structlog

_CONFIGURED = False


# ─────────────────────────────────────────────────────────────────────────────
# stderr filter — drops a small allow-list of cosmetic third-party warnings
# that bypass the logging system entirely (they go through ``sys.stderr.write``
# directly from inside huggingface_hub / sentence-transformers / tqdm).
# Anything not on the suppress list passes through untouched.
# ─────────────────────────────────────────────────────────────────────────────


_SUPPRESS_SUBSTRINGS = (
    "Please set a HF_TOKEN to enable higher rate limits",
    "Could not cache non-existence of file",
    "No device provided, using",
    "TOKENIZERS_PARALLELISM",
)


class _FilteredStderr:
    def __init__(self, real_stderr) -> None:
        self._real = real_stderr
        self._buffer = ""

    def write(self, data):  # pragma: no cover — passthrough wrapper
        if not data:
            return 0
        self._buffer += data
        # Flush complete lines; hold trailing partials in the buffer.
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not any(s in line for s in _SUPPRESS_SUBSTRINGS):
                self._real.write(line + "\n")
        return len(data)

    def flush(self):  # pragma: no cover
        if self._buffer and not any(s in self._buffer for s in _SUPPRESS_SUBSTRINGS):
            self._real.write(self._buffer)
        self._buffer = ""
        self._real.flush()

    def __getattr__(self, name):  # pragma: no cover
        return getattr(self._real, name)


def _install_stderr_filter() -> None:
    if not isinstance(sys.stderr, _FilteredStderr):
        sys.stderr = _FilteredStderr(sys.stderr)


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    _install_stderr_filter()
    lvl = (level or os.environ.get("HALLUDETECT_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, lvl, logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, lvl, logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Mute the verbose third-party loggers — their INFO output dominates the
    # console (httpx logs every single retrieval call, transformers spams
    # download progress, etc.) without adding signal beyond what our own
    # structlog records emit.
    for noisy in (
        "httpx",
        "httpcore",
        "transformers",
        "transformers.modeling_utils",
        "transformers.tokenization_utils",
        "transformers.configuration_utils",
        "sentence_transformers",
        "huggingface_hub",
        "filelock",
        "urllib3",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Hugging Face emits its own progress/warning prints via env-controllable
    # globals. Silence them where the SDK supports it.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("TQDM_DISABLE", "1")  # global tqdm kill-switch

    # Best-effort runtime calls — env vars alone don't always cover every
    # internal progress bar (e.g. ``transformers.modeling_utils``'s
    # safetensors loader). These functions exist on all v4.30+ / v0.20+ SDKs.
    try:
        from huggingface_hub.utils import disable_progress_bars  # type: ignore

        disable_progress_bars()
    except Exception:  # pragma: no cover
        pass
    try:
        from transformers.utils import logging as _hf_logging  # type: ignore

        _hf_logging.set_verbosity_error()
        _hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover
        pass


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    setup_logging()
    return structlog.get_logger(name)
