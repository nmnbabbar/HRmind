"""
backend/utils/log.py
=====================
Structured logging setup via structlog.

Named log.py (not logging.py) to avoid shadowing Python's stdlib logging module.

Features
--------
- JSON output in production (LOG_LEVEL != DEBUG)
- Human-readable console renderer in DEBUG mode (coloured, aligned)
- Automatically merges structlog context variables (e.g. session_id)
- Standard stdlib logging is forwarded to structlog processors

Usage
-----
    from backend.utils.log import get_logger, configure_logging

    # Call once at application startup (FastAPI lifespan):
    configure_logging(log_level="INFO")

    # In any module:
    logger = get_logger(__name__)
    logger.info("agent_started", agent="rag", session_id="abc-123")
    logger.error("guardrail_blocked", reason="non-HR query", agent="sql")
"""

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog + stdlib logging.

    Must be called once at application startup before any loggers are used.
    Subsequent calls are safe (idempotent structlog configuration).

    Parameters
    ----------
    log_level : str
        Log level string: "DEBUG" | "INFO" | "WARNING" | "ERROR"
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Shared processors applied to every log event
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_level.upper() == "DEBUG":
        # Developer-friendly: coloured, aligned, human-readable
        renderer = structlog.dev.ConsoleRenderer()
    else:
        # Production: machine-parseable JSON (e.g. for log aggregators)
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to go through structlog formatters
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Usage::
        logger = get_logger(__name__)
        logger.info("event_name", key="value")
    """
    return structlog.get_logger(name)
