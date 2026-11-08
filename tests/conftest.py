"""Shared test fixtures.

Settings are process-cached, so anything that changes the environment has to
clear the cache or it reads a stale object.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from beacon_core.settings import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test sees defaults, never the developer's .env or a real key."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
