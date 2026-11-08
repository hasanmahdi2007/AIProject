"""Test doubles for the gateway's external seams.

Imports nothing from pytest and is never imported by the routes: it lives in
the package, beside the code it impersonates, because a shared module under
``tests/`` cannot be imported by both pytest's ``importlib`` mode and ``mypy``
without a second ``conftest.py``, and a second ``conftest.py`` collides with
the repo-root one under ``mypy .``. See the PR description.

Note which fake proves what. ``InMemoryConversationStore`` only ever asserts
what the *service layer* does and does not write; the claim that history
survives a restart is proved in ``test_conversation_store.py`` against a real
Redis and a real second process, because an in-memory fake cannot possibly
prove it.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, Self

__all__ = [
    "INTEGRATION_HINT",
    "FakeAnthropic",
    "FakeFinalMessage",
    "FakeMessages",
    "FakePubSub",
    "FakeRedis",
    "FakeRequest",
    "FakeSession",
    "FakeSessionmaker",
    "FakeStream",
    "FakeStreamManager",
    "FakeTransaction",
    "InMemoryConversationStore",
    "child_env",
]


# ------------------------------------------------------- conversation store --


class InMemoryConversationStore:
    """A ``ConversationStore`` that records how it was called.

    ``appends`` counts *calls*, not turns, which is what the defect-1 tests
    assert on: the user and assistant turns must arrive in one call, after the
    stream, or not at all.
    """

    def __init__(self, seed: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self._turns: dict[str, list[dict[str, Any]]] = {
            cid: list(turns) for cid, turns in (seed or {}).items()
        }
        self.appends: list[tuple[str, tuple[dict[str, Any], ...]]] = []

    async def get(self, conversation_id: str) -> list[dict[str, Any]]:
        return list(self._turns.get(conversation_id, []))

    async def append(self, conversation_id: str, *turns: dict[str, Any]) -> None:
        self.appends.append((conversation_id, turns))
        self._turns.setdefault(conversation_id, []).extend(turns)

    async def clear(self, conversation_id: str) -> None:
        self._turns.pop(conversation_id, None)

    def snapshot(self, conversation_id: str) -> list[dict[str, Any]]:
        """Synchronous peek, for assertions taken mid-test."""
        return list(self._turns.get(conversation_id, []))


# -------------------------------------------------------------- fake Claude --


class FakeFinalMessage:
    def __init__(self, stop_reason: str | None = "end_turn") -> None:
        self.stop_reason = stop_reason


class FakeStream:
    def __init__(
        self,
        chunks: list[str],
        *,
        stop_reason: str | None = "end_turn",
        fail_after: int | None = None,
        failure: BaseException | None = None,
    ) -> None:
        self._chunks = chunks
        self._stop_reason = stop_reason
        self._fail_after = fail_after
        self._failure = failure

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def generate() -> AsyncIterator[str]:
            for index, chunk in enumerate(self._chunks):
                if self._fail_after is not None and index == self._fail_after:
                    assert self._failure is not None
                    raise self._failure
                yield chunk
            if self._fail_after is not None and self._fail_after >= len(self._chunks):
                assert self._failure is not None
                raise self._failure

        return generate()

    async def get_final_message(self) -> FakeFinalMessage:
        return FakeFinalMessage(self._stop_reason)


class FakeStreamManager:
    """Mirrors ``AsyncMessageStreamManager``: the HTTP request happens in
    ``__aenter__``, which is why a 429 reaches the route and becomes a 429."""

    def __init__(self, stream: FakeStream, raises: BaseException | None) -> None:
        self._stream = stream
        self._raises = raises
        self.closed = False

    async def __aenter__(self) -> FakeStream:
        if self._raises is not None:
            raise self._raises
        return self._stream

    async def __aexit__(self, *exc_info: object) -> bool:
        self.closed = True
        return False


class FakeMessages:
    def __init__(self, parent: FakeAnthropic) -> None:
        self._parent = parent

    def stream(self, **kwargs: Any) -> FakeStreamManager:
        self._parent.calls.append(kwargs)
        manager = FakeStreamManager(self._parent.stream_factory(), self._parent.raises)
        self._parent.managers.append(manager)
        return manager


class FakeAnthropic:
    """Stands in for ``anthropic.AsyncAnthropic`` at the one seam we use."""

    def __init__(
        self,
        *,
        chunks: list[str] | None = None,
        stop_reason: str | None = "end_turn",
        fail_after: int | None = None,
        failure: BaseException | None = None,
        raises: BaseException | None = None,
        on_stream: Any = None,
    ) -> None:
        self._chunks = chunks if chunks is not None else ["Hello", ", ", "world"]
        self._stop_reason = stop_reason
        self._fail_after = fail_after
        self._failure = failure
        self.raises = raises
        self.on_stream = on_stream
        self.calls: list[dict[str, Any]] = []
        self.managers: list[FakeStreamManager] = []
        self.messages = FakeMessages(self)

    def stream_factory(self) -> FakeStream:
        if self.on_stream is not None:
            self.on_stream()
        return FakeStream(
            list(self._chunks),
            stop_reason=self._stop_reason,
            fail_after=self._fail_after,
            failure=self._failure,
        )


# ---------------------------------------------------------------- fake db ----


class FakeTransaction:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> Self:
        self._session.begins += 1
        return self

    async def __aexit__(self, exc_type: object, *_: object) -> bool:
        if exc_type is None:
            self._session.commits += 1
        else:
            self._session.rollbacks += 1
        return False


class FakeSession:
    """Just enough ``AsyncSession`` to assert *how many* transactions a route
    opens and *what* it writes inside them."""

    def __init__(self, rows: dict[tuple[str, str], Any] | None = None) -> None:
        self.rows = dict(rows or {})
        self.added: list[Any] = []
        self.begins = 0
        self.commits = 0
        self.rollbacks = 0
        self.flushes = 0
        self.executed: list[Any] = []
        self.fail_on_execute: BaseException | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    async def get(self, model: type[Any], primary_key: str) -> Any:
        return self.rows.get((model.__name__, primary_key))

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        # Mirror the real flush: a row added in this transaction is visible to
        # a later ``get`` in the same transaction.
        pk = getattr(obj, "id", None)
        if isinstance(pk, str):
            self.rows[(type(obj).__name__, pk)] = obj

    async def execute(self, statement: Any) -> None:
        if self.fail_on_execute is not None:
            raise self.fail_on_execute
        self.executed.append(statement)

    async def flush(self) -> None:
        """A flush is not a commit: it writes inside the open transaction."""
        self.flushes += 1

    def added_of(self, name: str) -> list[Any]:
        return [obj for obj in self.added if type(obj).__name__ == name]


class FakeSessionmaker:
    def __init__(self, session: FakeSession) -> None:
        self.session = session

    def __call__(self) -> FakeSession:
        return self.session


class FakePubSub:
    """Replays a fixed script of pub/sub messages. ``None`` means "nothing
    arrived before the timeout", which is what makes the stream heartbeat."""

    def __init__(self, messages: list[dict[str, Any] | None] | None = None) -> None:
        self._messages: list[dict[str, Any] | None] = list(messages or [])
        self.subscribed: list[str] = []
        self.unsubscribed: list[str] = []
        self.closed = False

    async def subscribe(self, *channels: str) -> None:
        self.subscribed.extend(channels)

    async def get_message(
        self,
        *,
        ignore_subscribe_messages: bool = False,
        timeout: float | None = None,  # noqa: ASYNC109 - mirrors redis-py's signature
    ) -> dict[str, Any] | None:
        if not self._messages:
            return None
        return self._messages.pop(0)

    async def unsubscribe(self, *channels: str) -> None:
        self.unsubscribed.extend(channels)

    async def aclose(self) -> None:
        self.closed = True


class FakeRequest:
    """A ``Request`` with only the one method the SSE loop calls. The counter
    is how a test ends an otherwise infinite stream."""

    def __init__(self, *, disconnect_after: int = 1000) -> None:
        self.polls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self.polls += 1
        return self.polls > self._disconnect_after


class FakeRedis:
    def __init__(
        self,
        *,
        ping_error: BaseException | None = None,
        pubsub_messages: list[dict[str, Any] | None] | None = None,
    ) -> None:
        self.ping_error = ping_error
        self.pings = 0
        self.pubsubs: list[FakePubSub] = []
        self._pubsub_messages = pubsub_messages

    async def ping(self) -> bool:
        self.pings += 1
        if self.ping_error is not None:
            raise self.ping_error
        return True

    def pubsub(self) -> FakePubSub:
        created = FakePubSub(self._pubsub_messages)
        self.pubsubs.append(created)
        return created


# ------------------------------------------------------------------- misc ----

INTEGRATION_HINT = (
    "Point BEACON_POSTGRES__PORT / BEACON_REDIS__PORT at an ad-hoc stack "
    "(see the A2 brief) and run `alembic upgrade head` first."
)


def child_env() -> dict[str, str]:
    """A copy of the environment, for handing to a subprocess unchanged."""
    return dict(os.environ)
