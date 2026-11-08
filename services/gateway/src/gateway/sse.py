"""Server-sent event framing.

Every byte a streaming endpoint emits goes through :func:`format_sse`, which is
the whole fix for defect 3: an error is a typed ``error`` event with a
machine-readable ``code``, never a string smuggled into a ``token`` payload
where a browser would render it as if Claude had said it.

Event vocabulary:

``token``
    One text delta of an assistant reply. ``{"text": "..."}``.
``status`` / ``progress`` / ``completed``
    Match-pipeline updates, forwarded from the ``chan:match:{id}`` channel.
``error``
    Terminal failure. ``{"code": ..., "message": ...}``.
``done``
    Terminal success.

A stream emits exactly one terminal event - ``error`` or ``done``, never both -
so a client needs no timeout to know the reply finished.
"""

from __future__ import annotations

import json
from typing import Any, Final, Literal

__all__ = [
    "SSE_HEADERS",
    "SSE_MEDIA_TYPE",
    "EventName",
    "format_sse",
    "heartbeat",
]

SSE_MEDIA_TYPE: Final = "text/event-stream"

EventName = Literal["token", "status", "progress", "completed", "error", "done"]

SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    # Nginx-style proxies buffer by default, which turns a token stream into
    # one lump delivered at the end.
    "X-Accel-Buffering": "no",
}


def format_sse(event: EventName, data: dict[str, Any]) -> str:
    """Frame one typed event.

    ``data`` is always a JSON object, so a client parses every event the same
    way instead of sniffing whether this particular frame happens to be prose.
    """
    body = json.dumps(data, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {body}\n\n"


def heartbeat() -> str:
    """An SSE comment. Keeps idle proxies from closing a quiet stream."""
    return ": keep-alive\n\n"
