"""Record/replay: JSON on disk, keyed by prompt hash.

Why this exists at all: the evaluation harness re-runs the pipeline on every
change, and a pipeline run is hundreds of Claude calls. Paying for those on
every CI run makes the test suite something you avoid running, which is the
beginning of the end of the eval harness. Recorded responses make the suite
free and, more usefully, deterministic - two runs over the same fixtures
produce identical judgments, so a diff in the eval report is a code change and
never model variance.

The mode is the whole safety property:

``replay`` (the default)
    Look the response up by prompt hash. A miss raises
    :class:`~beacon_llm.errors.FixtureMissingError`. **There is no code path
    from a miss to the network.** That is deliberate: a store that silently
    fell through would make "zero network calls in CI" a hope rather than a
    guarantee, and the failure mode would be a surprise bill.
``record``
    Replay on a hit; on a miss, call the API and write the fixture.
``live``
    Always call the API, never write. For the smoke tests that must observe
    real ``usage``.

Default-replay matters: anything that forgets to choose, including a stray
import in a test, cannot spend money.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from anthropic.types import Message

from beacon_llm.errors import FixtureMissingError

__all__ = ["FixtureMode", "FixtureStore", "resolve_mode"]

MODE_ENV_VAR = "BEACON_LLM_MODE"


class FixtureMode(StrEnum):
    REPLAY = "replay"
    RECORD = "record"
    LIVE = "live"

    @property
    def may_call_api(self) -> bool:
        return self is not FixtureMode.REPLAY


def resolve_mode(explicit: FixtureMode | str | None = None) -> FixtureMode:
    """Pick the mode: the argument, else ``$BEACON_LLM_MODE``, else replay.

    Replay is the default everywhere, so an unconfigured process is a free
    process.
    """
    raw = explicit if explicit is not None else os.environ.get(MODE_ENV_VAR, FixtureMode.REPLAY)
    try:
        return FixtureMode(raw)
    except ValueError:
        valid = ", ".join(m.value for m in FixtureMode)
        raise ValueError(f"{MODE_ENV_VAR}={raw!r} is not one of: {valid}") from None


@dataclass(frozen=True, slots=True)
class FixtureStore:
    """A directory of recorded responses, one JSON file per prompt hash.

    One file per fixture rather than one big index: a recorded response is then
    reviewable in a diff, and re-recording one prompt does not rewrite the rest.
    """

    directory: Path

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def load(self, key: str) -> Message | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Message.model_validate(payload["response"])

    def require(self, key: str) -> Message:
        message = self.load(key)
        if message is None:
            raise FixtureMissingError(key=key, path=str(self.path_for(key)))
        return message

    def save(
        self,
        key: str,
        *,
        message: Message,
        request: dict[str, Any],
        latency_ms: float,
        request_id: str | None,
    ) -> Path:
        """Write a fixture.

        ``request`` is stored next to the response purely so a human reading the
        file can tell which prompt produced it. It is never replayed and never
        contains credentials - the API key lives on the client, not in the
        request body.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(key)
        payload = {
            "key": key,
            "recorded_at": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "latency_ms": latency_ms,
            "request": request,
            "response": message.model_dump(mode="json"),
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path
