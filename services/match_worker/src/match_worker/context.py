"""The per-run context, passed through LangGraph's config rather than its state.

A node needs four things that are not data about the match: the store, the LLM
gateway, the retrieval client and the budget meter. None of them belongs in
``MatchState``, because state is checkpointed - a database session or an HTTP
client cannot be serialised, and a budget meter that *were* serialised would be
restored to the value it had at the last checkpoint instead of the value the
run has actually spent.

So they travel in ``config["configurable"]["ctx"]``, which LangGraph passes to
every node untouched and never persists. The split is the useful part: what has
to survive a crash is in the state, what has to be live is in the context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from beacon_core.settings import Settings
from match_worker.budget import BudgetMeter
from match_worker.config import WorkerConfig
from match_worker.llm import LlmGateway
from match_worker.repository import MatchStore
from match_worker.retrieval_client import RetrievalClient

__all__ = ["CONTEXT_KEY", "RunContext", "context_from", "to_config"]

CONTEXT_KEY = "ctx"


@dataclass
class RunContext:
    """Everything live that one run's nodes need."""

    match_id: str
    store: MatchStore
    llm: LlmGateway
    retrieval: RetrievalClient
    budget: BudgetMeter
    settings: Settings
    worker: WorkerConfig
    patient_id: str | None = None
    trace_id: str | None = None

    @property
    def model(self) -> str:
        """``cheap_model`` - Haiku 5.5 - for every node except the decision.

        Profiling a note, generating queries, verifying a citation and writing
        the summary are extraction and paraphrase, where a bigger model buys
        little and effort is the better knob.

        ``check_rules`` is the exception and uses :meth:`decision_model`: its
        output *is* the product. See that method for why the split is worth the
        money and what it costs.
        """
        return self.llm.model or self.settings.anthropic.cheap_model

    @property
    def decision_model(self) -> str:
        """The model for ``check_rules``, the eligibility decision - Sonnet 5.5.

        An explicit ``llm.model`` override still wins, so a test or an ablation
        that pins one model pins it everywhere and does not silently pay for
        Sonnet.

        This is about 95% of token spend, since it runs once per candidate
        trial - twenty calls per patient against one for every other node.
        Switch ``anthropic.decision_model`` to Haiku for cheap iteration.
        """
        return self.llm.model or self.settings.anthropic.decision_model


def to_config(
    ctx: RunContext, *, thread_id: str | None = None, extra: dict[str, Any] | None = None
) -> RunnableConfig:
    """The LangGraph config for one run.

    ``thread_id`` is the match ID. That is what makes the checkpointer able to
    find the run again after a crash, and it is also what makes a redelivered
    message resume the existing thread instead of starting a second one.
    """
    configurable: dict[str, Any] = {
        "thread_id": thread_id or ctx.match_id,
        CONTEXT_KEY: ctx,
        **(extra or {}),
    }
    return {
        "configurable": configurable,
        "recursion_limit": 64,
        # LangGraph's async executor turns this into a semaphore over the tasks
        # of one superstep, which is the only thing that actually limits the
        # fan-out: twenty ``Send`` tasks are otherwise all scheduled at once.
        # That matters for money, not just for load. The twenty rule checks
        # share a cached prompt prefix, and the first call is the one that
        # *writes* it - fire all twenty simultaneously and they all miss the
        # cache that none of them has written yet, so the run pays full input
        # price twenty times instead of once. Nothing fails; it just costs ten
        # times more. ``WorkerConfig.fanout_concurrency`` is deliberately small.
        "max_concurrency": ctx.worker.fanout_concurrency,
    }


def context_from(config: RunnableConfig) -> RunContext:
    """Pull the context out of a node's config, or fail loudly.

    Loudly because the alternative - a node quietly constructing its own
    client - is how a test that believes it is stubbed starts making real
    calls.
    """
    configurable = config.get("configurable") or {}
    ctx = configurable.get(CONTEXT_KEY)
    if not isinstance(ctx, RunContext):
        raise RuntimeError(
            "no RunContext in config['configurable']['ctx']; build the config with "
            "match_worker.context.to_config() so nodes cannot invent their own clients"
        )
    return ctx
