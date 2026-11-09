"""The seam between "a retrieval configuration" and "a ranked list of trials".

The retrieval service (agent B2) is being written in parallel with this
harness and does not exist yet, so nothing here imports it, calls it, or
assumes it is running. What the harness needs is one function - *given a topic
and a configuration, return ranked NCT ids* - and that is what
:class:`RetrievalBackend` is.

This is not a workaround for B2 being late; it is the better design. A metric
tested end to end against a live service is tested against whatever that
service happened to return, and the test cannot tell a broken metric from a
broken retriever. A metric tested against a **synthetic ranked list whose
correct score was computed by hand** is tested against arithmetic. So the unit
suite drives :class:`ScriptedBackend`, and the real system arrives later behind
the same protocol without a metric changing.

Three implementations ship here:

* :class:`ScriptedBackend` - a dict of topic id to ranked ids. What the tests
  use, and what a hand-computed example is expressed in.
* :class:`RunFileBackend` - reads standard **TREC run files**. This is the real
  bridge to B2 and the recommended one: B2 (or any other system, including a
  published baseline) writes ``topic Q0 doc rank score tag`` lines, and the
  harness scores them with no service running, no network and no coupling.
  ``trec_eval`` has worked this way for thirty years for exactly this reason.
* :class:`UnavailableBackend` - the default. Every entry point that has no
  backend gets this one, and it raises a message naming what is missing rather
  than returning an empty ranking that would score 0.0 and look like a
  measurement.

An HTTP backend that posts :class:`beacon_core.schemas.SearchRequest` to B2 is
deliberately **not** here: ``evals`` declares no HTTP client dependency, and
adding one is the orchestrator's call, not this agent's. The adapter is a dozen
lines once that is decided - see :data:`HTTP_ADAPTER_SKETCH`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from evals.trec import Topic

__all__ = [
    "HTTP_ADAPTER_SKETCH",
    "CallableBackend",
    "RetrievalBackend",
    "RetrievalUnavailableError",
    "Run",
    "RunFileBackend",
    "ScriptedBackend",
    "UnavailableBackend",
    "load_run_file",
    "write_run_file",
]

Run = dict[str, list[str]]
"""``{topic_id: [doc_id, ...]}`` in rank order, best first."""


class RetrievalUnavailableError(RuntimeError):
    """Raised when a run was requested but no retrieval source is wired up."""


@runtime_checkable
class RetrievalBackend(Protocol):
    """Anything that can rank trials for a topic under some configuration.

    ``config`` is an :class:`evals.ablations.RunConfig`; it is typed loosely
    here (as an object with the attributes a backend needs) only to keep this
    module free of a circular import. Implementations should read the fields
    they understand and ignore the rest.
    """

    name: str

    def search(self, topic: Topic, config: object) -> list[str]:
        """Ranked trial ids for one topic, best first, no duplicates."""
        ...


@dataclass(frozen=True, slots=True)
class UnavailableBackend:
    """The default backend: refuses, loudly, with a reason.

    It exists so that "no retrieval available" can never be confused with
    "retrieval returned nothing". An empty ranking scores a legitimate-looking
    0.0 on every metric and would land in a report as a measured number.
    """

    reason: str = (
        "no retrieval backend is wired up. The retrieval service (agent B2) is not "
        "available to this harness. Supply --runs <dir> with TREC run files, or "
        "inject a RetrievalBackend."
    )
    name: str = "unavailable"

    def search(self, topic: Topic, config: object) -> list[str]:
        raise RetrievalUnavailableError(
            f"cannot produce a run for topic {topic.topic_id}: {self.reason}"
        )


@dataclass(frozen=True, slots=True)
class ScriptedBackend:
    """A fixed, hand-written run. The unit suite's retriever.

    Args:
        rankings: ``{topic_id: [doc_id, ...]}``. A topic absent from the
            mapping raises rather than returning ``[]``, because a silently
            empty ranking is indistinguishable from a bad one in the scores.
    """

    rankings: Mapping[str, Sequence[str]]
    name: str = "scripted"

    def search(self, topic: Topic, config: object) -> list[str]:
        try:
            return list(self.rankings[topic.topic_id])
        except KeyError:
            raise RetrievalUnavailableError(
                f"scripted backend {self.name!r} has no ranking for topic "
                f"{topic.topic_id}; it covers {sorted(self.rankings)[:5]}..."
            ) from None


@dataclass(frozen=True, slots=True)
class CallableBackend:
    """Adapter so any function can be a backend.

    This is how the real retrieval service is wired in once it exists and once
    the HTTP dependency question is settled: write the twelve-line function,
    wrap it here, and no metric or report code changes.
    """

    fn: object
    name: str = "callable"

    def search(self, topic: Topic, config: object) -> list[str]:
        call = self.fn
        if not callable(call):
            raise TypeError(f"CallableBackend.fn must be callable, got {type(call).__name__}")
        result = call(topic, config)
        return [str(doc) for doc in result]


# ------------------------------------------------------------- run files ----


def load_run_file(path: Path) -> Run:
    """Parse a TREC run file into ``{topic_id: ranked doc ids}``.

    Format - six whitespace-separated columns, one per retrieved document::

        <topic_id> Q0 <doc_id> <rank> <score> <tag>

    The ranking is taken from the **score**, descending, not from the rank
    column: ``trec_eval`` does the same, and a run whose rank column disagrees
    with its scores (an off-by-one, or ranks restarted per shard) would
    otherwise be scored in an order no system ever produced. Ties are broken by
    document id ascending, which is deterministic rather than correct - a run
    with tied scores is already ill-defined, so the harness at least scores it
    the same way every time.

    Blank lines and ``#`` comments are skipped. Duplicate ``(topic, doc)``
    pairs raise: they inflate precision and recall and are always a bug in the
    producer.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: on a malformed line, an unparseable score, or a duplicate.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no run file at {path}")

    scored: dict[str, list[tuple[float, str]]] = {}
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(
                    f"{path}:{lineno}: expected 6 columns "
                    f"'topic Q0 doc rank score tag', got {len(parts)}: {line!r}"
                )
            topic_id, _q0, doc_id, _rank, score = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                value = float(score)
            except ValueError:
                raise ValueError(
                    f"{path}:{lineno}: score column {score!r} is not a number"
                ) from None
            key = (topic_id, doc_id)
            if key in seen:
                raise ValueError(
                    f"{path}:{lineno}: document {doc_id} appears twice for topic "
                    f"{topic_id}; a duplicated document inflates precision and recall"
                )
            seen.add(key)
            scored.setdefault(topic_id, []).append((value, doc_id))

    if not scored:
        raise ValueError(f"{path} contains no run lines")
    return {
        topic_id: [doc for _, doc in sorted(rows, key=lambda row: (-row[0], row[1]))]
        for topic_id, rows in scored.items()
    }


def write_run_file(path: Path, run: Mapping[str, Sequence[str]], *, tag: str) -> None:
    """Write a run in TREC format, so a produced run can be re-scored later.

    Scores are synthesised as ``len(ranking) - index`` purely to preserve the
    given order on a round trip; they are not similarity scores and nothing
    should read them as such.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for topic_id in sorted(run, key=lambda t: (len(t), t)):
            ranking = run[topic_id]
            for index, doc_id in enumerate(ranking):
                score = float(len(ranking) - index)
                handle.write(f"{topic_id} Q0 {doc_id} {index + 1} {score:.4f} {tag}\n")


@dataclass(frozen=True, slots=True)
class RunFileBackend:
    """Serves rankings from a directory of TREC run files.

    One file per ablation configuration, named ``<config name>.run``. This is
    how the ablation table gets measured without this harness ever holding a
    socket open: B2 writes the runs, the harness scores them.

    Args:
        directory: holds the ``.run`` files.
        suffix: file extension, default ``.run``.
    """

    directory: Path
    suffix: str = ".run"
    name: str = "run-files"

    def path_for(self, config_name: str) -> Path:
        return self.directory / f"{config_name}{self.suffix}"

    def available(self) -> tuple[str, ...]:
        """Configuration names this directory has runs for."""
        if not self.directory.is_dir():
            return ()
        return tuple(sorted(p.stem for p in self.directory.glob(f"*{self.suffix}")))

    def run_for(self, config_name: str) -> Run:
        """The whole run for one configuration.

        Raises:
            RetrievalUnavailableError: when that configuration has no run file,
                naming the ones that do exist. The ablation runner catches this
                and records the row as *not measured* instead of scoring an
                absent run as zero.
        """
        path = self.path_for(config_name)
        if not path.is_file():
            raise RetrievalUnavailableError(
                f"no run file for configuration {config_name!r} at {path}. "
                f"Available: {self.available() or '(none)'}"
            )
        return load_run_file(path)

    def search(self, topic: Topic, config: object) -> list[str]:
        config_name = str(getattr(config, "name", ""))
        run = self.run_for(config_name)
        try:
            return run[topic.topic_id]
        except KeyError:
            raise RetrievalUnavailableError(
                f"run file {self.path_for(config_name)} has no rows for topic "
                f"{topic.topic_id}"
            ) from None


HTTP_ADAPTER_SKETCH = '''\
# Wiring the real retrieval service in, once `evals` is allowed an HTTP client
# dependency (orchestrator's call - this agent may not edit evals/pyproject.toml
# to add one). Nothing else in the harness changes.
#
#   import httpx
#   from beacon_core.schemas import AgeSexFilter, RankedQuery, SearchRequest, SearchResponse
#
#   def search(topic, config):
#       request = config.to_search_request(queries=generate_queries(topic, config))
#       reply = httpx.post(f"{base_url}/search", json=request.model_dump(), timeout=60)
#       reply.raise_for_status()
#       return [c.nct_id for c in SearchResponse.model_validate(reply.json()).candidates]
#
#   backend = CallableBackend(fn=search, name="retrieval-service")
'''
"""Kept as a string, not as code, so that nothing here can accidentally open a
socket and so the missing dependency is documented rather than smuggled in."""
