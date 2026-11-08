"""The record/replay store on its own."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from anthropic.types import Message, Usage

from beacon_llm import FixtureMissingError, FixtureMode, FixtureStore, resolve_mode
from beacon_llm.fixtures import MODE_ENV_VAR


def _message() -> Message:
    return Message(
        id="msg_test",
        model="claude-haiku-5-5",
        role="assistant",
        type="message",
        stop_reason="end_turn",
        content=[{"type": "text", "text": '{"ok": true}'}],
        usage=Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=2032),
    )


def test_replay_is_the_default_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfigured process cannot spend money.

    This is the property that protects against a stray import in a test, or a
    developer running the eval harness without thinking about it.
    """
    monkeypatch.delenv(MODE_ENV_VAR, raising=False)
    assert resolve_mode() is FixtureMode.REPLAY
    assert not FixtureMode.REPLAY.may_call_api


@pytest.mark.parametrize("mode", list(FixtureMode))
def test_the_environment_variable_selects_the_mode(
    monkeypatch: pytest.MonkeyPatch, mode: FixtureMode
) -> None:
    monkeypatch.setenv(MODE_ENV_VAR, mode.value)
    assert resolve_mode() is mode
    assert resolve_mode(FixtureMode.LIVE) is FixtureMode.LIVE


def test_an_unrecognised_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """``BEACON_LLM_MODE=replay-ish`` must not fall back to spending money."""
    monkeypatch.setenv(MODE_ENV_VAR, "replay-ish")
    with pytest.raises(ValueError, match="not one of"):
        resolve_mode()


def test_round_trip_preserves_usage_exactly(tmp_path: Path) -> None:
    """A replayed ``usage`` must cost the same as the live one did.

    The recorded cache-read count is the whole value of a fixture for cost
    testing; losing it would make every replayed call look like a cache miss.
    """
    store = FixtureStore(tmp_path)
    store.save(
        "abc",
        message=_message(),
        request={"model": "claude-haiku-5-5"},
        latency_ms=12.5,
        request_id="req_1",
    )
    loaded = store.load("abc")
    assert loaded is not None
    assert loaded.usage.cache_read_input_tokens == 2032
    assert loaded.usage.input_tokens == 10
    assert loaded.stop_reason == "end_turn"


def test_the_saved_file_is_human_readable(tmp_path: Path) -> None:
    """One indented file per prompt hash, reviewable in a diff.

    The request is stored next to the response so a reviewer can see which
    prompt produced it; no credential is ever in the request body.
    """
    store = FixtureStore(tmp_path)
    path = store.save(
        "abc",
        message=_message(),
        request={"model": "claude-haiku-5-5", "system": [{"type": "text", "text": "hi"}]},
        latency_ms=1.0,
        request_id=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["key"] == "abc"
    assert payload["request"]["model"] == "claude-haiku-5-5"
    assert payload["recorded_at"]
    assert "api_key" not in path.read_text(encoding="utf-8")


def test_a_miss_returns_none_but_require_raises(tmp_path: Path) -> None:
    """``load`` is for record mode, ``require`` for replay.

    Replay has no path from a miss to the network, which is what makes "zero
    network calls in CI" a guarantee rather than a hope.
    """
    store = FixtureStore(tmp_path)
    assert store.load("nope") is None
    with pytest.raises(FixtureMissingError) as excinfo:
        store.require("nope")
    assert "nope" in str(excinfo.value)
    assert "BEACON_LLM_MODE=record" in str(excinfo.value)


def test_save_creates_the_directory(tmp_path: Path) -> None:
    store = FixtureStore(tmp_path / "deep" / "nested")
    store.save("abc", message=_message(), request={}, latency_ms=0.0, request_id=None)
    assert (tmp_path / "deep" / "nested" / "abc.json").is_file()
