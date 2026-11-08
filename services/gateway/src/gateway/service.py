"""Streaming chat against Claude.

Two of the three spike defects live and die in this module.

**Defect 1 - poisoned history.** The spike did
``history.append({"role": "user", ...})`` *before* opening the stream, against
the same list object it stored. A rate limit, a validation error or a dropped
connection therefore left the user's turn behind with no reply, and the next
request replayed a history ending in a dangling user message - which Claude
either answers twice or rejects outright. The fix: build the outgoing message
list as a **local** value, and persist the user and assistant turns *together*,
once, only after the stream has completed successfully.

**Defect 3 - errors as prose.** The spike yielded ``f"[stream error: ...]"``
into the same byte stream as the reply, so the only thing distinguishing "Claude
said this" from "Claude failed" was square brackets in the text. Everything
here is framed by :func:`gateway.sse.format_sse`, and failures become a typed
``error`` event.

What the spike got right and this keeps: the Claude request is issued *before*
the response object is returned, so auth, rate-limit and validation failures
raise inside the request handler and become real HTTP status codes rather than
a 200 whose body happens to contain an apology.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any, Final, cast

import anthropic
from anthropic.types import MessageParam

from beacon_core.protocols import ConversationStore
from gateway.sse import format_sse

__all__ = ["MAX_TOKENS", "MODEL", "start_reply"]

MODEL: Final = "claude-haiku-5-5"
"""Verified against platform.claude.com on 2026-11-07. Do not "correct" it."""

MAX_TOKENS: Final = 16_000

SYSTEM_PROMPT: Final = (
    "You are Beacon's clinical-trial matching assistant. Answer questions about "
    "a match run's results using only what the user tells you. You are a "
    "research tool, not a clinician: never give medical advice, and say plainly "
    "when you do not know something."
)


async def start_reply(
    conversation_id: str,
    text: str,
    *,
    store: ConversationStore,
    client: anthropic.AsyncAnthropic,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
) -> AsyncIterator[str]:
    """Open a Claude stream and return an iterator of framed SSE events.

    The ``await`` on ``client.messages.stream`` happens here, before the caller
    builds its response, so ``anthropic`` exceptions propagate to the route and
    turn into HTTP status codes.

    Nothing is written to ``store`` until the reply is complete. A refusal or a
    mid-stream API error leaves the conversation exactly as it was - the caller
    can retry the same message and get a clean history.
    """
    history = await store.get(conversation_id)
    user_turn: dict[str, Any] = {"role": "user", "content": text}

    # Local, not the stored list: nothing here is persisted yet.
    outgoing: list[dict[str, Any]] = [*history, user_turn]

    stack = AsyncExitStack()
    stream = await stack.enter_async_context(
        client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            # Turns come back from Redis as plain dicts; the SDK's TypedDict is
            # structurally the same thing.
            messages=cast("list[MessageParam]", outgoing),
        )
    )

    async def events() -> AsyncIterator[str]:
        parts: list[str] = []
        try:
            async for chunk in stream.text_stream:
                parts.append(chunk)
                yield format_sse("token", {"text": chunk})

            final = await stream.get_final_message()

            if final.stop_reason == "refusal":
                # A refusal is not an assistant turn worth replaying, and it is
                # not a token either: it is a typed terminal event.
                yield format_sse(
                    "error",
                    {
                        "code": "refusal",
                        "message": "Claude declined to respond to this message.",
                        "conversation_id": conversation_id,
                    },
                )
                return

            # The only write, and it carries both turns at once.
            await store.append(
                conversation_id,
                user_turn,
                {"role": "assistant", "content": "".join(parts)},
            )
            yield format_sse(
                "done",
                {
                    "conversation_id": conversation_id,
                    "stop_reason": final.stop_reason,
                    "turns": len(outgoing) + 1,
                },
            )
        except anthropic.APIError as exc:
            yield format_sse(
                "error",
                {
                    "code": _error_code(exc),
                    "message": _error_message(exc),
                    "conversation_id": conversation_id,
                },
            )
        finally:
            await stack.aclose()

    return events()


def _error_code(exc: anthropic.APIError) -> str:
    """A stable, machine-readable code. Clients branch on this, not on prose."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return f"claude_http_{status}"
    if isinstance(exc, anthropic.APIConnectionError):
        return "claude_unreachable"
    return "claude_stream_failed"


def _error_message(exc: anthropic.APIError) -> str:
    message = getattr(exc, "message", "") or str(exc)
    return message or exc.__class__.__name__
