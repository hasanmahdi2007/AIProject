"""structlog configuration: JSON in production, readable colour in dev.

Every log line carries the current trace ID when one exists, so a log line and
a span in Phoenix can be joined. The processor is written now and starts
returning real IDs as soon as C3 installs OpenTelemetry - no later edit here.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from beacon_core.settings import get_settings

__all__ = ["add_trace_context", "configure_logging", "get_logger"]

_configured = False


def add_trace_context(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Attach ``trace_id`` / ``span_id`` when an OTel span is active.

    Stubbed until C3 wires OpenTelemetry: if the package is absent, or no span
    is recording, the line is passed through untouched.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return event_dict

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if getattr(ctx, "is_valid", False):
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def configure_logging(*, force: bool = False) -> None:
    """Idempotent. Safe to call from every service entry point."""
    global _configured
    if _configured and not force:
        return

    cfg = get_settings()
    level = getattr(logging, cfg.log_level.upper(), logging.INFO)

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_trace_context,
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if cfg.log_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    configure_logging()
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
