"""Defect 2: conversation history lives in Redis, not in process memory.

These tests are marked ``integration`` and talk to a real Redis, because the
claim being made - "history survives a restart and is visible to another
replica" - is exactly the claim an in-memory fake cannot support. The decisive
test launches a **separate Python process** that builds its own client and its
own store and reads the same key back.

Run them against an ad-hoc Redis, e.g.::

    docker run -d --name a2-redis -p 56379:6379 redis:8
    BEACON_REDIS__PORT=56379 uv run pytest services/gateway -m integration
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis

from beacon_core.redis import key_conversation
from beacon_core.settings import Settings
from gateway.conversations import RedisConversationStore
from gateway.testing import INTEGRATION_HINT, child_env

pytestmark = pytest.mark.integration


@pytest.fixture
def redis_url() -> str:
    return Settings().redis.url


@pytest.fixture
async def live_redis(redis_url: str) -> AsyncIterator[Redis]:
    redis: Redis = Redis.from_url(redis_url, decode_responses=True)
    try:
        await redis.ping()
    except Exception as exc:
        await redis.aclose()
        pytest.skip(f"Redis unreachable: {exc}. {INTEGRATION_HINT}")
    try:
        yield redis
    finally:
        await redis.aclose()


TURNS = [
    {"role": "user", "content": "Why was NCT04567890 excluded?"},
    {"role": "assistant", "content": "Its exclusion criterion 3 requires no prior anthracycline."},
]

# Read the history with a client this process never created, in an interpreter
# that never saw the writer's memory. Anything short of this proves nothing
# about surviving a restart.
FRESH_PROCESS_READER = """
import asyncio
import json
import sys

from redis.asyncio import Redis

from gateway.conversations import RedisConversationStore


async def main() -> None:
    redis = Redis.from_url(sys.argv[1], decode_responses=True)
    store = RedisConversationStore(redis, ttl_seconds=60)
    try:
        print(json.dumps(await store.get(sys.argv[2])))
    finally:
        await redis.aclose()


asyncio.run(main())
"""


async def test_append_then_get_roundtrips(live_redis: Redis) -> None:
    conversation_id = f"test-{uuid.uuid4()}"
    store = RedisConversationStore(live_redis, ttl_seconds=60)

    assert await store.get(conversation_id) == []

    await store.append(conversation_id, *TURNS)

    assert await store.get(conversation_id) == TURNS


async def test_turns_stay_oldest_first_across_appends(live_redis: Redis) -> None:
    conversation_id = f"test-{uuid.uuid4()}"
    store = RedisConversationStore(live_redis, ttl_seconds=60)

    await store.append(conversation_id, *TURNS)
    await store.append(
        conversation_id,
        {"role": "user", "content": "And NCT01234567?"},
        {"role": "assistant", "content": "That one is still open."},
    )

    contents = [turn["content"] for turn in await store.get(conversation_id)]
    assert contents == [
        TURNS[0]["content"],
        TURNS[1]["content"],
        "And NCT01234567?",
        "That one is still open.",
    ]


async def test_the_contract_key_format_is_used(live_redis: Redis) -> None:
    """The key is ``chat:{id}``, built by ``beacon_core.redis``: another replica
    reads the same key only if nobody invents their own format."""
    conversation_id = f"test-{uuid.uuid4()}"
    store = RedisConversationStore(live_redis, ttl_seconds=60)

    await store.append(conversation_id, *TURNS)

    key = key_conversation(conversation_id)
    assert key == f"chat:{conversation_id}"
    assert await live_redis.exists(key) == 1
    assert await live_redis.type(key) == "list"


async def test_every_append_refreshes_the_ttl(live_redis: Redis) -> None:
    conversation_id = f"test-{uuid.uuid4()}"
    store = RedisConversationStore(live_redis, ttl_seconds=120)

    await store.append(conversation_id, *TURNS)
    key = key_conversation(conversation_id)

    ttl = await live_redis.ttl(key)
    assert 0 < ttl <= 120, "an abandoned conversation must expire, not accumulate"


async def test_clear_removes_the_history(live_redis: Redis) -> None:
    conversation_id = f"test-{uuid.uuid4()}"
    store = RedisConversationStore(live_redis, ttl_seconds=60)
    await store.append(conversation_id, *TURNS)

    await store.clear(conversation_id)

    assert await store.get(conversation_id) == []
    assert await live_redis.exists(key_conversation(conversation_id)) == 0


async def test_history_survives_a_fresh_process(live_redis: Redis, redis_url: str) -> None:
    """The defect-2 regression test.

    A second interpreter - no shared memory, its own Redis client, its own
    store instance - reads back what this one wrote. Under the old
    module-level dict this could not return anything at all.
    """
    conversation_id = f"test-{uuid.uuid4()}"
    store = RedisConversationStore(live_redis, ttl_seconds=300)
    await store.append(conversation_id, *TURNS)

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        FRESH_PROCESS_READER,
        redis_url,
        conversation_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_env(),
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)

    assert process.returncode == 0, stderr.decode()
    assert json.loads(stdout.decode().strip()) == TURNS
