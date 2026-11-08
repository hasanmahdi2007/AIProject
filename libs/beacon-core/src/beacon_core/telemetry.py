"""OpenTelemetry setup - a deliberate no-op stub until C3.

C3 (Observability) owns the real implementation and runs with no other agent
active, because tracing touches every package. The signatures are frozen here
so services can call ``setup_tracing(...)`` from Wave 1 and get real traces
later without an edit.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from beacon_core.settings import Settings, get_settings

__all__ = ["current_traceparent", "setup_tracing", "span"]


def setup_tracing(service_name: str, settings: Settings | None = None) -> None:
    """Install the tracer provider and exporters for one service.

    Stub: records nothing. Calling it is already correct, so Wave 1 services
    wire it into their startup and C3 fills in the body.
    """
    _ = service_name, settings or get_settings()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Context manager for one span. Stub: yields immediately."""
    _ = name, attributes
    yield


def current_traceparent() -> str | None:
    """W3C ``traceparent`` for the active span, for Kafka headers.

    Returns ``None`` while tracing is stubbed, which
    ``beacon_core.kafka.headers_from_trace`` handles by emitting no header.
    """
    return None
