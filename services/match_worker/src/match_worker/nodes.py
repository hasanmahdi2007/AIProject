"""The seven nodes. LangGraph calls them; it does not decide anything in them.

Every node is a plain async function of state (or, for the fan-out target, of
its ``Send`` payload) returning a **partial** state update. No node builds a
prompt - that is :mod:`match_worker.prompts` and
:mod:`match_worker.rule_check_prompt` - and no node talks to Anthropic except
through :class:`match_worker.llm.LlmGateway`, so the cache ordering, the
budget and the audit row are not a call site's responsibility.

Three behaviours are worth reading before the code:

**One trial's failure is that trial's problem.** ``check_rules`` never raises.
A trial whose rule check fails twice comes back as a judgment marked
"could not evaluate" and the other nineteen finish. The only failures that
abort a run are the ones that make the run meaningless - no patient note, no
retrieval - and those are raised so the Kafka layer can retry the message and
eventually park it in the DLQ.

**A budget breach is an answer, not an error.** Any node that cannot afford its
next call stops, writes the partial marker into ``MatchState.error``, and lets
the run finish with what it has. The marker is a prefix on a field in the
frozen contract because the contract has no ``partial`` flag; see
:mod:`match_worker.state`.

**Everything a node writes, it can write twice.** Judgments are written by the
fan-out task that produced them, trial rows by the node that ranked them, and
both go through the unique constraints in ``beacon_core.models``. That is what
makes a redelivered message or a resumed run converge on the same rows instead
of doubling them.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send

from beacon_core.logging import get_logger
from beacon_core.schemas import (
    CriterionJudgment,
    CriterionKind,
    NumberedNote,
    PatientProfile,
    RankedQuery,
    RankedTrial,
    TrialCandidate,
    TrialJudgment,
)
from beacon_llm import build_cached_prompt
from match_worker.backoff import RetryPolicy, retry_async
from match_worker.budget import BudgetExhausted
from match_worker.context import RunContext, context_from
from match_worker.notes import to_numbered_note
from match_worker.outputs import (
    GeneratedQueries,
    TrialExplanation,
    TrialRuleCheck,
    TrialVerification,
)
from match_worker.progress import NODE_STEPS, publish_progress
from match_worker.prompts import (
    build_explain_prompt,
    build_profile_prompt,
    build_query_prompt,
    build_verify_prompt,
)
from match_worker.repository import TrialRules
from match_worker.rule_check_prompt import render_trial_rules, rule_check_instructions
from match_worker.rules import split_eligibility
from match_worker.state import (
    BUDGET_EXHAUSTED,
    PARTIAL_PREFIX,
    GraphState,
    TrialTask,
    could_not_evaluate,
    merge_judgments,
    partial_reasons,
)
from match_worker.verdict import citations_valid, derive_trial_judgment, final_score

__all__ = [
    "check_rules",
    "fan_out_trials",
    "generate_queries",
    "parse_rules",
    "profile_patient",
    "rank_and_explain",
    "retrieve",
    "verify",
]

log = get_logger(__name__)


# --------------------------------------------------------------- helpers --


def _halted(state: GraphState) -> bool:
    """True once a budget breach has turned this run into a partial answer.

    Checked at the top of every node rather than wired as a conditional edge
    for each one: the graph shape stays the seven-node pipeline it is described
    as, and ``rank_and_explain`` still runs - a partial answer has to be
    ranked, persisted and marked, or it is just a dropped job.
    """
    return bool(state.error and state.error.startswith(PARTIAL_PREFIX))


def _partial_marker(reason: str) -> str:
    return f"{PARTIAL_PREFIX} {reason}"


async def _announce(ctx: RunContext, node: str, message: str) -> None:
    """One ``match.progress`` event per node, into the outbox.

    Paired with ``current_step`` on the run row so the console and the database
    agree about where a run is, even between two checkpoints.
    """
    await ctx.store.set_current_step(ctx.match_id, NODE_STEPS[node][0])
    await publish_progress(
        ctx.store, match_id=ctx.match_id, node=node, message=message, trace_id=ctx.trace_id
    )


def _note_of(state: GraphState) -> NumberedNote:
    if state.note is None:
        raise RuntimeError("note is missing; profile_patient did not run")
    return state.note


# ------------------------------------------------------- 1. profile_patient --


async def profile_patient(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Number the note in code, then extract the profile with one Claude call.

    The numbering is deterministic and deliberately not the model's job: those
    IDs are what every judgment cites and what the verifier checks against, so
    a model that renumbered between two calls would manufacture citation
    failures that no prompt change could explain.
    """
    ctx = context_from(config)
    await ctx.store.mark_running(ctx.match_id, step="profile_patient")
    await _announce(ctx, "profile_patient", "numbering the note and extracting the profile")

    note = to_numbered_note(state.patient_note)
    if not note.sentences:
        raise ValueError(f"match {ctx.match_id}: patient note is empty, nothing to match")

    try:
        result = await ctx.llm.call(
            step="profile_patient",
            prompt=build_profile_prompt(note),
            output_model=PatientProfile,
            effort="medium",
        )
    except BudgetExhausted as exc:
        return {
            "note": note,
            "status": "running",
            "current_step": "profile_patient",
            "error": _partial_marker(str(exc)),
        }

    await ctx.store.save_patient_profile(
        ctx.patient_id,
        sentences=[s.model_dump(mode="json") for s in note.sentences],
        profile=result.parsed.model_dump(mode="json"),
    )
    log.info(
        "worker.node.profile_patient",
        match_id=ctx.match_id,
        sentences=len(note.sentences),
        age=result.parsed.age,
        sex=result.parsed.sex,
        conditions=len(result.parsed.conditions),
    )
    return {
        "note": note,
        "profile": result.parsed,
        "status": "running",
        "current_step": "profile_patient",
        "total_cost_usd": result.cost.cost_usd,
    }


# ------------------------------------------------------ 2. generate_queries --


def _fallback_queries(profile: PatientProfile | None, note: NumberedNote) -> list[RankedQuery]:
    """Deterministic queries, for when the model returns none usable.

    ``SearchRequest`` requires at least one query, and a run that dies here has
    done nothing for the patient. A query built from the profile's own terms is
    worse than a generated one and far better than nothing.
    """
    terms: list[tuple[str, Any]] = []
    if profile:
        terms += [(c, "diagnosis") for c in profile.conditions[:2]]
        terms += [(b, "biomarker") for b in profile.biomarkers[:1]]
        terms += [(t, "prior_therapy") for t in profile.prior_treatments[:1]]
    if not terms:
        terms = [(" ".join(s.text for s in note.sentences[:2])[:300] or "clinical trial", "other")]
    return [
        RankedQuery(rank=index, text=text, facet=facet)
        for index, (text, facet) in enumerate(terms, start=1)
    ]


async def generate_queries(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """N facet-diverse queries. ``N`` is ``settings.retrieval.query_count``.

    N comes from settings and is never a literal in the prompt, because the
    evaluation sweep tunes it over {1, 3, 5, 10} and N=1 doubles as the
    "does LLM query generation help at all?" ablation row.
    """
    ctx = context_from(config)
    if _halted(state):
        return {}
    count = ctx.settings.retrieval.query_count
    await _announce(ctx, "generate_queries", f"writing {count} search queries")
    note = _note_of(state)

    try:
        result = await ctx.llm.call(
            step="generate_queries",
            prompt=build_query_prompt(note, profile=state.profile or PatientProfile(), count=count),
            output_model=GeneratedQueries,
            effort="medium",
        )
        queries = [q for q in result.parsed.queries if q.text.strip()][:count]
        delta = result.cost.cost_usd
    except BudgetExhausted as exc:
        return {
            "queries": _fallback_queries(state.profile, note),
            "current_step": "generate_queries",
            "error": _partial_marker(str(exc)),
        }

    if not queries:
        log.warning("worker.node.generate_queries.empty", match_id=ctx.match_id)
        queries = _fallback_queries(state.profile, note)
    # Re-rank densely: the model occasionally skips or repeats a rank, and the
    # retrieval service uses rank 1 as "most important".
    queries = [
        RankedQuery(rank=index, text=query.text, facet=query.facet)
        for index, query in enumerate(queries, start=1)
    ]
    log.info(
        "worker.node.generate_queries",
        match_id=ctx.match_id,
        queries=[q.text for q in queries],
        facets=[q.facet for q in queries],
    )
    return {"queries": queries, "current_step": "generate_queries", "total_cost_usd": delta}


# --------------------------------------------------------------- 3. retrieve --


async def retrieve(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Ask the retrieval service. **This worker does not search.**

    ``services/retrieval`` owns hybrid search, RRF fusion, the age/sex filters
    and the reranker. Here there is one HTTP POST of the frozen
    ``SearchRequest`` and one validation of the frozen ``SearchResponse``. If
    retrieval is down the message is retried and eventually parked: a match
    with no candidates is not a partial answer, it is no answer.
    """
    from match_worker.retrieval_client import build_search_request

    ctx = context_from(config)
    if _halted(state):
        return {}
    await _announce(ctx, "retrieve", "searching trials")

    queries = state.queries or _fallback_queries(state.profile, _note_of(state))
    request = build_search_request(
        queries=queries, profile=state.profile, settings=ctx.settings.retrieval
    )
    policy = RetryPolicy(
        attempts=ctx.worker.retrieval_attempts,
        base_seconds=ctx.worker.message_retry_base_seconds,
        max_seconds=ctx.worker.message_retry_max_seconds,
    )
    response = await retry_async(
        lambda: ctx.retrieval.search(request), policy=policy, what="retrieval.search"
    )

    candidates: list[TrialCandidate] = []
    for index, candidate in enumerate(response.candidates, start=1):
        # The contract lets retrieval set the rank; fill it in when it did not,
        # so downstream tie-breaking has something stable to use.
        candidates.append(
            candidate
            if candidate.retrieval_rank is not None
            else candidate.model_copy(update={"retrieval_rank": index})
        )

    for candidate in candidates:
        await ctx.store.upsert_trial_match(
            ctx.match_id,
            {
                "nct_id": candidate.nct_id,
                "retrieval_rank": candidate.retrieval_rank,
                "rrf_score": candidate.rrf_score,
                "rerank_score": candidate.rerank_score,
            },
        )

    log.info("worker.node.retrieve", match_id=ctx.match_id, candidates=len(candidates))
    return {"candidates": candidates, "current_step": "retrieve"}


# ------------------------------------------------------------ 4. parse_rules --


async def parse_rules(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Split each candidate's eligibility text into individually judgeable rules.

    Lazily, and only for the trials that were retrieved: splitting 375,000
    trials' criteria up front would be work for the 374,980 nobody asked about.
    Already-split trials are reused from ``trial_criteria``, which is why the
    rows are keyed by a stable ``criterion_id``.

    A trial whose text yields no rules is marked "could not evaluate" here
    rather than passed on. A trial with no criteria would otherwise be a trial
    every patient trivially qualifies for, which is the worst possible answer.
    """
    ctx = context_from(config)
    if _halted(state):
        return {}
    nct_ids = [candidate.nct_id for candidate in state.candidates]
    await _announce(ctx, "parse_rules", f"parsing eligibility rules for {len(nct_ids)} trials")

    # Lazy and cached: a trial is split once, ever. Already-parsed trials come
    # back from trial_criteria and cost nothing for every later patient, which
    # is why this is its own node rather than part of check_rules.
    known = await ctx.store.load_trial_rules(nct_ids)
    missing = [nct_id for nct_id in nct_ids if nct_id not in known]
    texts = await ctx.store.load_eligibility_text(missing) if missing else {}
    for nct_id in missing:
        title, eligibility = texts.get(nct_id, ("", ""))
        parsed = split_eligibility(nct_id, eligibility)
        if not parsed:
            continue
        await ctx.store.save_parsed_criteria(nct_id, parsed)
        known[nct_id] = TrialRules(
            nct_id=nct_id,
            title=title,
            criteria=[
                {"criterion_id": c.criterion_id, "kind": c.kind, "text": c.text} for c in parsed
            ],
        )

    unparseable = [
        could_not_evaluate(nct_id, "no eligibility criteria could be parsed for this trial")
        for nct_id in nct_ids
        if nct_id not in known or not known[nct_id].criteria
    ]
    rule_count = sum(len(rules.criteria) for rules in known.values())
    log.info(
        "worker.node.parse_rules",
        match_id=ctx.match_id,
        trials_with_rules=len(known),
        rules=rule_count,
        unparseable=len(unparseable),
    )
    return {"current_step": "parse_rules", "judgments": unparseable}


def fan_out_trials(state: GraphState, config: RunnableConfig) -> list[Send] | list[str]:
    """One parallel ``check_rules`` task per candidate trial.

    ``Send`` rather than a loop inside one node, because the fan-out is what
    makes the twenty rule checks *independent*: LangGraph records each task's
    write separately, so a worker killed on trial twelve resumes with eleven
    judgments already in the checkpoint and re-runs only the ones that did not
    finish. A single node looping over twenty trials would redo all twenty.

    The parsed rules are re-read here rather than carried in state: they are
    rows in ``trial_criteria``, the read is one indexed query, and putting
    twenty trials' criteria into the checkpoint would copy them into every
    subsequent snapshot of the run.
    """
    ctx = context_from(config)
    if _halted(state):
        return ["verify"]
    judged = {judgment.nct_id for judgment in state.judgments}
    note = _note_of(state)
    sends: list[Send] = []
    for candidate in state.candidates:
        if candidate.nct_id in judged:
            # Either already judged (a resumed run) or unparseable (marked by
            # parse_rules). Both are answers; neither needs a Claude call.
            continue
        sends.append(
            Send(
                "check_rules",
                TrialTask(
                    match_id=ctx.match_id,
                    nct_id=candidate.nct_id,
                    title=candidate.title,
                    note=note,
                    criteria=[],
                ).model_dump(mode="json"),
            )
        )
    log.info("worker.node.fan_out", match_id=ctx.match_id, tasks=len(sends), skipped=len(judged))
    # An empty fan-out must still reach the rest of the pipeline. Returning []
    # from a conditional edge schedules nothing at all, which would end the run
    # with no verdicts written - exactly the case a resumed run hits when every
    # trial was already judged before the crash.
    return sends or ["verify"]


# ----------------------------------------------------------- 5. check_rules --


async def check_rules(payload: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Judge one trial's rules. Runs once per trial, in parallel, and never raises.

    The prompt is assembled in the one order that caches - standing
    instructions, then the numbered note with the cache breakpoint after it,
    then *this* trial's rules - so the first of twenty calls writes the shared
    prefix and the other nineteen read it at a tenth of the input price. The
    ordering is enforced by ``build_cached_prompt``, not by this call site.

    Two attempts, then the trial is marked "could not evaluate" and the run
    carries on with nineteen. A third attempt does not make a malformed trial
    parse, and aborting the run would throw away nineteen good answers to
    report one bad one.
    """
    ctx = context_from(config)
    task = TrialTask.model_validate(payload)
    note = task.note
    # One progress event for the whole fan-out, not one per trial: the event ID
    # is derived from (match_id, node), so all twenty tasks write the same ID
    # and nineteen of them hit the outbox's unique constraint and do nothing.
    await _announce(ctx, "check_rules", "judging eligibility rules")

    criteria = task.criteria
    if not criteria:
        rules = await ctx.store.load_trial_rules([task.nct_id])
        found = rules.get(task.nct_id)
        criteria = found.criteria if found else []
        title = found.title if found and found.title else task.title
    else:
        title = task.title
    if not criteria:
        return {"judgments": [could_not_evaluate(task.nct_id, "no parsed rules for this trial")]}

    # Already judged by an earlier attempt? Then the answer is in Postgres and
    # re-asking Claude would be paying twice for it. This is the durable half of
    # idempotency: the checkpoint can lose a finished task's write when the
    # process is killed outright, but the rows it committed are still there.
    already = await ctx.store.load_judgments(task.match_id, task.nct_id)
    if already and {row["criterion_id"] for row in already} >= {
        criterion["criterion_id"] for criterion in criteria
    }:
        by_criterion = {row["criterion_id"]: row for row in already}
        restored = [
            CriterionJudgment.model_validate(by_criterion[criterion["criterion_id"]])
            for criterion in criteria
        ]
        log.info(
            "worker.node.check_rules.restored",
            match_id=task.match_id,
            nct_id=task.nct_id,
            rules=len(restored),
            detail="judgments already in Postgres from an earlier attempt; no Claude call",
        )
        return {
            "judgments": [derive_trial_judgment(nct_id=task.nct_id, judgments=restored, note=note)]
        }

    prompt = build_cached_prompt(
        instructions=rule_check_instructions(),
        note=note,
        variable=render_trial_rules(nct_id=task.nct_id, title=title, criteria=criteria),
    )
    policy = RetryPolicy(
        attempts=ctx.worker.trial_attempts,
        base_seconds=ctx.worker.trial_retry_base_seconds,
        max_seconds=ctx.worker.trial_retry_max_seconds,
    )

    try:
        result = await retry_async(
            lambda: ctx.llm.call(
                step="check_rules",
                prompt=prompt,
                output_model=TrialRuleCheck,
                # The one node that does not run on cheap_model. Its output is
                # the product, so it gets settings.anthropic.decision_model
                # (Sonnet 5.5). An explicit llm.model override still wins, so
                # an ablation pinning one model does not silently pay for it.
                model=ctx.decision_model,
                effort="high",
            ),
            policy=policy,
            what=f"check_rules:{task.nct_id}",
        )
    except BudgetExhausted:
        return {"judgments": [could_not_evaluate(task.nct_id, BUDGET_EXHAUSTED)]}
    except Exception as exc:
        log.warning(
            "worker.node.check_rules.failed",
            match_id=task.match_id,
            nct_id=task.nct_id,
            attempts=policy.attempts,
            error=f"{type(exc).__name__}: {exc}",
        )
        return {
            "judgments": [
                could_not_evaluate(
                    task.nct_id,
                    f"rule check failed after {policy.attempts} attempts ({type(exc).__name__})",
                )
            ]
        }

    by_id = {item["criterion_id"]: item for item in criteria}
    returned = {judgment.criterion_id: judgment for judgment in result.parsed.judgments}
    aligned: list[CriterionJudgment] = []
    for criterion_id, criterion in by_id.items():
        judgment = returned.get(criterion_id)
        if judgment is None:
            # The model dropped a criterion. Abstaining on its behalf is the
            # only honest reading: nothing was judged, so nothing is confirmed.
            aligned.append(
                CriterionJudgment(
                    criterion_id=criterion_id,
                    kind=cast("CriterionKind", criterion["kind"]),
                    label="insufficient_info",
                    evidence_sentence_ids=[],
                    explanation="No judgment was returned for this criterion.",
                )
            )
            continue
        aligned.append(
            judgment.model_copy(
                update={
                    # The criterion's kind is a fact about the trial, not an
                    # opinion: a flipped kind inverts the verdict, so the
                    # parsed value always wins over the returned one.
                    "kind": criterion["kind"],
                    "evidence_sentence_ids": [
                        sid for sid in judgment.evidence_sentence_ids if sid in note.valid_ids
                    ],
                }
            )
        )

    hallucinated = sorted(set(returned) - set(by_id))
    if hallucinated:
        log.warning(
            "worker.node.check_rules.unknown_criteria",
            match_id=task.match_id,
            nct_id=task.nct_id,
            criterion_ids=hallucinated,
        )

    trial_judgment = derive_trial_judgment(nct_id=task.nct_id, judgments=aligned, note=note)
    await ctx.store.save_judgments(
        task.match_id,
        task.nct_id,
        [
            {
                "criterion_id": item.criterion_id,
                "kind": item.kind,
                "label": item.label,
                "evidence_sentence_ids": list(item.evidence_sentence_ids),
                "explanation": item.explanation,
                "citations_valid": citations_valid(item, note),
            }
            for item in aligned
        ],
    )
    log.info(
        "worker.node.check_rules",
        match_id=task.match_id,
        nct_id=task.nct_id,
        rules=len(aligned),
        category=trial_judgment.category,
        confirmed_fraction=trial_judgment.confirmed_fraction,
        cache_read_tokens=result.cost.cache_read_tokens,
        cost_usd=result.cost.cost_usd,
    )
    return {"judgments": [trial_judgment], "total_cost_usd": result.cost.cost_usd}


# ----------------------------------------------------------------- 6. verify --


async def verify(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Check every citation in code, then audit the consequential ones with Claude.

    The code check is the one that gates the verdict: a cited sentence ID that
    is not in the note means the label cannot be used to exclude a trial - see
    :mod:`match_worker.verdict`. It runs over every judgment, costs nothing and
    cannot be skipped.

    The LLM pass is the faithfulness *measurement*: does the cited sentence
    actually support the label? Its verdicts are written to
    ``criterion_judgments.verifier_verdict`` and deliberately do not change any
    label, so the number it produces is an observation of the pipeline rather
    than part of it.
    """
    ctx = context_from(config)
    if _halted(state):
        return {}
    await _announce(ctx, "verify", "checking cited evidence")
    note = _note_of(state)

    judgments = state.judgments
    invalid = sum(
        1 for trial in judgments for item in trial.judgments if not citations_valid(item, note)
    )
    if invalid:
        log.warning("worker.node.verify.invalid_citations", match_id=ctx.match_id, count=invalid)

    if not ctx.worker.verify_with_llm:
        return {"current_step": "verify"}

    judged = [trial for trial in judgments if trial.judgments]
    rules_by_trial = await ctx.store.load_trial_rules([trial.nct_id for trial in judged])
    semaphore = asyncio.Semaphore(ctx.worker.fanout_concurrency)
    breach: str | None = None
    spend = 0.0

    async def audit(trial: TrialJudgment) -> float:
        nonlocal breach
        texts = {
            criterion["criterion_id"]: criterion["text"]
            for criterion in (
                rules_by_trial[trial.nct_id].criteria if trial.nct_id in rules_by_trial else []
            )
        }
        described = [
            {
                "criterion_id": item.criterion_id,
                "kind": item.kind,
                "criterion_text": texts.get(item.criterion_id, "(criterion text unavailable)"),
                "label": item.label,
                "evidence_sentence_ids": list(item.evidence_sentence_ids),
                "explanation": item.explanation,
            }
            for item in trial.judgments
        ]
        async with semaphore:
            try:
                result = await ctx.llm.call(
                    step="verify",
                    prompt=build_verify_prompt(note, nct_id=trial.nct_id, judgments=described),
                    output_model=TrialVerification,
                    effort="low",
                )
            except BudgetExhausted as exc:
                breach = str(exc)
                return 0.0
            except Exception as exc:
                log.warning(
                    "worker.node.verify.failed",
                    match_id=ctx.match_id,
                    nct_id=trial.nct_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
                return 0.0

        verdicts = {verdict.criterion_id: verdict for verdict in result.parsed.verdicts}
        await ctx.store.save_judgments(
            ctx.match_id,
            trial.nct_id,
            [
                {
                    "criterion_id": item.criterion_id,
                    "kind": item.kind,
                    "label": item.label,
                    "evidence_sentence_ids": list(item.evidence_sentence_ids),
                    "explanation": item.explanation,
                    "citations_valid": citations_valid(item, note),
                    "verifier_verdict": (
                        verdicts[item.criterion_id].verdict
                        if item.criterion_id in verdicts
                        else None
                    ),
                    "verifier_note": (
                        verdicts[item.criterion_id].note if item.criterion_id in verdicts else None
                    ),
                }
                for item in trial.judgments
            ],
        )
        return result.cost.cost_usd

    for delta in await asyncio.gather(*(audit(trial) for trial in judged)):
        spend += delta

    update: dict[str, Any] = {"current_step": "verify", "total_cost_usd": spend}
    if breach is not None:
        update["error"] = _partial_marker(f"verification incomplete: {breach}")
    return update


# ------------------------------------------------------- 7. rank_and_explain --


async def rank_and_explain(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Rank in code, explain with Claude, then write the answer down.

    Always runs, including for a halted run: a partial answer still has to be
    ranked, persisted and marked partial. Only the summaries are skipped when
    there is no budget left for them - and a trial without a summary is
    missing prose, not a verdict.
    """
    ctx = context_from(config)
    halted = _halted(state)
    if not halted:
        await _announce(ctx, "rank_and_explain", "ranking and writing summaries")
    note = _note_of(state)

    by_nct = {candidate.nct_id: candidate for candidate in state.candidates}
    # Re-derive every verdict here rather than trusting what the fan-out wrote:
    # the citation check is part of the derivation, and this is the only place
    # that sees all of a trial's judgments at once after verification.
    derived = [
        (
            derive_trial_judgment(
                nct_id=judgment.nct_id,
                judgments=judgment.judgments,
                note=note,
                summary=judgment.summary,
            )
            if judgment.judgments
            else judgment
        )
        for judgment in state.judgments
    ]
    derived.sort(
        key=lambda judgment: final_score(
            judgment,
            retrieval_rank=(
                by_nct[judgment.nct_id].retrieval_rank if judgment.nct_id in by_nct else None
            ),
        ),
        reverse=True,
    )

    spend = 0.0
    budget_note: str | None = None
    explain_count = 0 if halted else ctx.worker.explain_top_k
    for position, judgment in enumerate(derived[:explain_count]):
        if not judgment.judgments or (judgment.summary or "").startswith("could not evaluate"):
            continue
        candidate = by_nct.get(judgment.nct_id)
        try:
            result = await ctx.llm.call(
                step="rank_and_explain",
                prompt=build_explain_prompt(
                    note,
                    nct_id=judgment.nct_id,
                    title=candidate.title if candidate else "",
                    category=judgment.category,
                    confirmed_fraction=judgment.confirmed_fraction,
                    judgments=[
                        {
                            "criterion_id": item.criterion_id,
                            "kind": item.kind,
                            "label": item.label,
                            "evidence_sentence_ids": list(item.evidence_sentence_ids),
                            "explanation": item.explanation,
                        }
                        for item in judgment.judgments
                    ],
                    missing_facts=judgment.missing_facts,
                ),
                output_model=TrialExplanation,
                effort="medium",
            )
        except BudgetExhausted as exc:
            budget_note = str(exc)
            break
        except Exception as exc:
            log.warning(
                "worker.node.explain.failed",
                match_id=ctx.match_id,
                nct_id=judgment.nct_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            continue
        spend += result.cost.cost_usd
        derived[position] = judgment.model_copy(update={"summary": result.parsed.summary.strip()})

    ranked = [
        RankedTrial(
            rank=index,
            candidate=by_nct.get(judgment.nct_id)
            or TrialCandidate(nct_id=judgment.nct_id, title=""),
            judgment=judgment,
            score=final_score(
                judgment,
                retrieval_rank=(
                    by_nct[judgment.nct_id].retrieval_rank if judgment.nct_id in by_nct else None
                ),
            ),
        )
        for index, judgment in enumerate(derived, start=1)
    ]

    for row in ranked:
        await ctx.store.upsert_trial_match(
            ctx.match_id,
            {
                "nct_id": row.candidate.nct_id,
                "retrieval_rank": row.candidate.retrieval_rank,
                "rrf_score": row.candidate.rrf_score,
                "rerank_score": row.candidate.rerank_score,
                "final_rank": row.rank,
                "final_score": row.score,
                "category": row.judgment.category,
                "confirmed_fraction": row.judgment.confirmed_fraction,
                "missing_facts": list(row.judgment.missing_facts),
                "summary": row.judgment.summary,
            },
        )

    reasons = partial_reasons(derived)
    if budget_note:
        reasons.append(f"summaries incomplete: {budget_note}")
    error: str | None = None
    if halted and state.error:
        error = state.error if not reasons else f"{state.error}; {'; '.join(reasons)}"
    elif reasons:
        error = _partial_marker("; ".join(reasons))

    log.info(
        "worker.node.rank_and_explain",
        match_id=ctx.match_id,
        ranked=len(ranked),
        eligible=sum(1 for row in ranked if row.judgment.category == "eligible"),
        excluded=sum(1 for row in ranked if row.judgment.category == "excluded"),
        partial=bool(error),
        cost_usd=round(ctx.budget.spent_usd, 6),
    )
    return {
        "judgments": merge_judgments([], derived),
        "ranked": ranked,
        "status": "completed",
        "current_step": "rank_and_explain",
        "error": error,
        "total_cost_usd": spend,
    }
