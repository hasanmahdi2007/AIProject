"""Beacon match worker: the durable pipeline behind one match request.

Seven nodes over the frozen ``MatchState``, consuming ``match.requested`` from
Kafka, checkpointing to Postgres after every node, and surviving a kill:

```
profile_patient -> generate_queries -> retrieve -> parse_rules
                                                      |
                                      Send: one parallel task per trial
                                                      |
                                                 check_rules -> verify -> rank_and_explain
```

Five properties the package exists to hold, each of which is a real failure
mode rather than a nicety:

* **LangGraph orchestrates; it never owns a prompt or a model call.** Every
  Claude call goes through ``beacon_llm`` via
  :class:`match_worker.llm.LlmGateway`, so the cache ordering, the cost meter
  and the Haiku token guard stay in our code. See :mod:`match_worker.prompts`.
* **Offsets are committed after the checkpoint is durable, never before.** A
  worker killed mid-run has not committed, so the message is redelivered and
  the run resumes from its last checkpoint. See :mod:`match_worker.runner`.
* **Every write is idempotent.** Kafka is at-least-once, so the unique
  constraints in ``beacon_core.models`` are load-bearing and
  :mod:`match_worker.repository` is the only thing that touches them.
* **One trial's failure is not the run's failure.** Two attempts, then that
  trial is marked "could not evaluate" and the other nineteen finish. See
  :func:`match_worker.nodes.check_rules`.
* **A budget breach returns partial results, clearly marked.** Never a silent
  truncation. See :mod:`match_worker.budget` and :mod:`match_worker.state`.

Two files are deliberately someone else's:
:mod:`match_worker.rule_check_prompt` holds the rule-check prompt text and is
reserved for the project owner to rewrite, and ``services/retrieval`` owns
search - this package only calls its frozen HTTP contract.
"""

from __future__ import annotations

from match_worker.budget import BudgetExhausted, BudgetMeter
from match_worker.config import WorkerConfig
from match_worker.context import RunContext, to_config
from match_worker.graph import build_graph, compile_graph, postgres_checkpointer
from match_worker.repository import MatchStore, PostgresStore
from match_worker.retrieval_client import RetrievalClient
from match_worker.runner import MatchWorker, RunOutcome, run_worker
from match_worker.state import (
    BUDGET_EXHAUSTED,
    COULD_NOT_EVALUATE,
    PARTIAL_PREFIX,
    GraphState,
    TrialTask,
)

__version__ = "0.1.0"

__all__ = [
    "BUDGET_EXHAUSTED",
    "COULD_NOT_EVALUATE",
    "PARTIAL_PREFIX",
    "BudgetExhausted",
    "BudgetMeter",
    "GraphState",
    "MatchStore",
    "MatchWorker",
    "PostgresStore",
    "RetrievalClient",
    "RunContext",
    "RunOutcome",
    "TrialTask",
    "WorkerConfig",
    "__version__",
    "build_graph",
    "compile_graph",
    "postgres_checkpointer",
    "run_worker",
    "to_config",
]
