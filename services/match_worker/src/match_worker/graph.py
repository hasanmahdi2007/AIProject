"""The graph: seven nodes, one fan-out, a checkpoint after every node.

```
START -> profile_patient -> generate_queries -> retrieve -> parse_rules
                                                               |
                                              Send, one per candidate trial
                                                               |
                                                          check_rules  (xN, parallel)
                                                               |
                                                            verify -> rank_and_explain -> END
```

LangGraph's job here is scheduling and persistence, and nothing else. It does
not own a prompt, a model, a retry policy or a cost number - those are in
:mod:`match_worker.prompts`, :mod:`match_worker.llm` and
:mod:`match_worker.backoff`, where they can be read and tested without the
framework.

What the checkpointer buys
--------------------------
A checkpoint is written after every node *and* for every completed ``Send``
task. So a worker killed during the twelve-trial mark resumes with the
judgments of the trials that finished and re-runs only the ones that did not.
That is the crash test, and it is the reason the state carries a merge reducer
on ``judgments``: a replayed write must update its trial rather than append a
second copy of it.

``thread_id`` is the ``match_id``. A redelivered Kafka message therefore
resumes the *same* thread instead of starting a second run of the same match -
the first half of idempotency, the unique constraints in
:mod:`match_worker.repository` being the second.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from beacon_core.logging import get_logger
from beacon_core.settings import Settings, get_settings
from match_worker.nodes import (
    check_rules,
    fan_out_trials,
    generate_queries,
    parse_rules,
    profile_patient,
    rank_and_explain,
    retrieve,
    verify,
)
from match_worker.state import GraphState

__all__ = [
    "CHECKPOINT_TYPES",
    "build_graph",
    "build_serde",
    "checkpointer_dsn",
    "compile_graph",
    "postgres_checkpointer",
]

log = get_logger(__name__)

CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = (
    ("beacon_core.schemas", "NoteSentence"),
    ("beacon_core.schemas", "NumberedNote"),
    ("beacon_core.schemas", "PatientProfile"),
    ("beacon_core.schemas", "RankedQuery"),
    ("beacon_core.schemas", "TrialCandidate"),
    ("beacon_core.schemas", "CriterionJudgment"),
    ("beacon_core.schemas", "TrialJudgment"),
    ("beacon_core.schemas", "RankedTrial"),
)
"""Every ``beacon_core`` model that reaches a checkpoint, named explicitly.

The checkpointer's serialiser reconstructs arbitrary classes by module path on
read, which is a code-execution path for anyone who can write to the
checkpoint table - so LangGraph warns on an unregistered type and will refuse
one in a later version. Listing the eight state models turns that warning into
a decision: these are the types a match's state is made of, and a new one has
to be added here deliberately.
"""


def build_serde() -> SerializerProtocol:
    """The checkpoint serialiser, restricted to the types above."""
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)


def build_graph() -> StateGraph[GraphState, None, GraphState, GraphState]:
    """The uncompiled graph, so a test can compile it with its own saver."""
    graph: StateGraph[GraphState, None, GraphState, GraphState] = StateGraph(GraphState)

    graph.add_node("profile_patient", profile_patient)
    graph.add_node("generate_queries", generate_queries)
    graph.add_node("retrieve", retrieve)
    graph.add_node("parse_rules", parse_rules)
    # ``check_rules`` is a Send target: it is handed a TrialTask payload, not
    # the run's state, so its signature cannot be the state type the rest of
    # the nodes share. The ignore marks that deliberate difference.
    graph.add_node("check_rules", check_rules)  # type: ignore[arg-type]
    graph.add_node("verify", verify)
    graph.add_node("rank_and_explain", rank_and_explain)

    graph.add_edge(START, "profile_patient")
    graph.add_edge("profile_patient", "generate_queries")
    graph.add_edge("generate_queries", "retrieve")
    graph.add_edge("retrieve", "parse_rules")
    # The fan-out. ``["check_rules", "verify"]`` names the possible targets so
    # the graph is still statically drawable: an empty fan-out (every trial
    # already judged, or a halted run) must fall through to ``verify`` rather
    # than dead-end.
    graph.add_conditional_edges("parse_rules", fan_out_trials, ["check_rules", "verify"])
    graph.add_edge("check_rules", "verify")
    graph.add_edge("verify", "rank_and_explain")
    graph.add_edge("rank_and_explain", END)
    return graph


def compile_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph[GraphState, None, GraphState, GraphState]:
    """Compile with a checkpointer. ``None`` gives an in-memory one.

    In-memory is for unit tests only. It satisfies the API and survives nothing
    - which is the whole distinction the crash test exists to demonstrate, so
    the integration tests pass a real Postgres saver explicitly.
    """
    return build_graph().compile(checkpointer=checkpointer or InMemorySaver(serde=build_serde()))


def checkpointer_dsn(settings: Settings | None = None) -> str:
    """The checkpointer's connection string.

    ``psycopg`` rather than ``asyncpg``: ``langgraph-checkpoint-postgres`` is
    built on psycopg 3 and cannot use the application's asyncpg engine. Both
    come from the one ``settings.postgres`` block, so there is still a single
    source of truth for host, port and credentials - and the port is 15432,
    because the dev machine runs its own PostgreSQL on 5432.
    """
    return (settings or get_settings()).postgres.dsn(driver="postgresql")


@asynccontextmanager
async def postgres_checkpointer(
    settings: Settings | None = None, *, setup: bool = True
) -> AsyncIterator[BaseCheckpointSaver]:
    """A durable checkpointer, with its tables created if they are missing.

    ``setup()`` is the library's own migration entry point and is idempotent;
    the checkpointer's three tables are *its* schema, not Beacon's, which is
    why they are not in the Alembic migration. Whether they should be is a
    judgement for the orchestrator and is raised in the PR.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    dsn = checkpointer_dsn(settings)
    async with AsyncPostgresSaver.from_conn_string(dsn, serde=build_serde()) as saver:
        if setup:
            await saver.setup()
        log.info("worker.checkpointer.ready", backend="postgres")
        yield saver
