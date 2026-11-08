"""Tests for the one Settings object, plus the Redis key namespace."""

from __future__ import annotations

import pytest

from beacon_core import redis as beacon_redis
from beacon_core.settings import Settings, get_settings


class TestDefaults:
    def test_query_count_is_three_and_configurable(self) -> None:
        # Decided 2026-11-08, overriding the guide's 5-10: three facet-diverse
        # queries, tuned by the Day 3 sweep over {1, 3, 5, 10}. It must stay a
        # setting, never a literal inside the generation prompt.
        assert get_settings().retrieval.query_count == 3

    def test_bm25_idf_modifier_is_on(self) -> None:
        # Qdrant computes corpus-level IDF only when the sparse field declares
        # the IDF modifier. Without it no error is raised and rare-term
        # matching - the entire reason BM25 is here - silently degrades.
        assert get_settings().qdrant.sparse_idf_modifier is True

    def test_embedding_dimensions_match_the_model(self) -> None:
        q = get_settings().qdrant
        assert q.dense_model == "BAAI/bge-small-en-v1.5"
        assert q.dense_dim == 384
        assert q.dense_max_tokens == 512, "bge-small truncates silently past 512"

    def test_models_are_the_verified_ids(self) -> None:
        a = get_settings().anthropic
        assert a.cheap_model == "claude-haiku-5-5"
        assert a.headline_model == "claude-sonnet-5-5"
        # The judge must differ from the model being judged, or it grades
        # itself. The headline run is Sonnet 5.5, so the judge is Opus 5.5.
        assert a.judge_model == "claude-opus-5-5"
        assert a.judge_model != a.headline_model

    def test_haiku_pricing_cliff_guard_is_set(self) -> None:
        # Above 100K *prompt* tokens Haiku 5.5 prices jump 5x, counting cache
        # reads. A "batch all 20 trials into one call" refactor must fail
        # loudly rather than quintuple the bill in silence.
        assert get_settings().anthropic.haiku_prompt_token_ceiling == 100_000

    def test_rrf_and_funnel_defaults(self) -> None:
        r = get_settings().retrieval
        assert (r.fuse_top_k, r.rerank_top_k, r.rrf_k) == (100, 20, 60)


class TestEnvironmentOverrides:
    def test_nested_override_uses_a_double_underscore(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("BEACON_RETRIEVAL__QUERY_COUNT", "10")
        get_settings.cache_clear()
        assert get_settings().retrieval.query_count == 10

    def test_api_key_is_read_from_the_sdk_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-example")
        get_settings.cache_clear()
        assert get_settings().anthropic.api_key.get_secret_value() == "sk-ant-example"

    def test_the_key_does_not_leak_when_settings_are_printed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-appear")
        get_settings.cache_clear()
        assert "sk-ant-should-not-appear" not in repr(get_settings())

    def test_password_does_not_leak_into_repr(self) -> None:
        s = Settings()
        assert "beacon" not in repr(s.postgres.password)

    def test_dsn_is_async_by_default(self) -> None:
        assert Settings().postgres.dsn().startswith("postgresql+asyncpg://")


class TestRedisKeys:
    def test_key_patterns_are_the_documented_ones(self) -> None:
        # These are a cross-process contract: one gateway replica publishes to
        # chan:match:{id} and another reads it, so a typo here is a bug that
        # only appears under more than one replica.
        assert beacon_redis.key_rate_limit("abc") == "rl:abc"
        assert beacon_redis.key_idempotency("abc") == "idem:abc"
        assert beacon_redis.key_llm_cache("deadbeef") == "llm:deadbeef"
        assert beacon_redis.key_lock("ingest") == "lock:ingest"
        assert beacon_redis.key_conversation("c1") == "chat:c1"
        assert beacon_redis.chan_match("m1") == "chan:match:m1"
