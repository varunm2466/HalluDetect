"""Centralized structlog configuration."""

from __future__ import annotations

import logging
import os
import sys

import structlog

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
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


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    setup_logging()
    return structlog.get_logger(name)
