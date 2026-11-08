"""Prompts for every node except the rule check.

The rule-check prompt lives alone in :mod:`match_worker.rule_check_prompt`
because it is reserved for the project owner. These four are ordinary
scaffolding and are expected to be edited freely.

Every one of them is split the same way - stable instructions, then the
numbered patient note, then the per-call variable block - and assembled by
``beacon_llm.build_cached_prompt``, which is the only thing in the project
allowed to decide block order. Two of these prompts (``verify`` and
``explain``) run once per trial, so their prefixes are written long enough to
clear the 512-token minimum cacheable prefix; the two that run once per match
(``profile``, ``queries``) cannot benefit from the cache at all and are short
on purpose.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from beacon_core.schemas import NumberedNote, PatientProfile, QueryFacet
from beacon_llm import CachedPrompt, build_cached_prompt

__all__ = [
    "build_explain_prompt",
    "build_profile_prompt",
    "build_query_prompt",
    "build_verify_prompt",
]

_FACETS: tuple[QueryFacet, ...] = ("diagnosis", "biomarker", "prior_therapy", "setting", "other")


# ------------------------------------------------------------ profile_patient --

_PROFILE_INSTRUCTIONS = """\
You extract a structured profile from a clinical patient note. You are given \
the note split into numbered sentences. Return one JSON object with the \
fields of the profile schema and nothing else.

Extract only what the note states. Leave a field null, or its list empty, when \
the note does not state it: a null age is handled downstream as "unknown, so \
possibly eligible", whereas a guessed age silently excludes every trial whose \
bounds it falls outside. Do not infer a diagnosis from a medication, a stage \
from a line of therapy, or a sex from a name.

Normalise conservatively. Age is a whole number of years. Sex is male, female \
or null - never "all", which is a trial-side value. Conditions, biomarkers and \
prior treatments are short noun phrases in the note's own words, deduplicated, \
most specific first: "invasive ductal carcinoma", not "the patient has cancer". \
Biomarkers keep their qualifier, so "ER positive" and "HER2 negative" are two \
entries. Labs are a flat mapping of test name to the value and unit as written. \
Performance status is the ECOG integer 0 to 5, null unless the note gives one; \
do not convert a Karnofsky score. Location is the free-text site if the note \
names one.
"""


def build_profile_prompt(note: NumberedNote) -> CachedPrompt:
    return build_cached_prompt(
        instructions=_PROFILE_INSTRUCTIONS,
        note=note,
        variable=(
            "Extract the structured patient profile from the numbered note above. "
            "Return the profile object only."
        ),
    )


# ----------------------------------------------------------- generate_queries --

_QUERY_INSTRUCTIONS = """\
You write search queries that retrieve candidate clinical trials for one \
patient. You are given the patient note split into numbered sentences, and a \
number of query slots. Return one JSON object with a "queries" array.

Spend one slot per distinct clinical facet, never two paraphrases of the same \
facet: with three slots, "metastatic breast cancer", "ER positive HER2 \
negative breast cancer" and "breast cancer metastatic" cover one facet and \
waste two slots. Draw each query from a different one of: the diagnosis and \
its stage, a biomarker or molecular finding, prior therapy or treatment line, \
and the care setting or practical constraints. Tag each query with the facet \
it covers.

Write each query as a dense noun phrase of clinical terms, the way a trial's \
title and conditions are written - not as a question, not as a sentence, and \
with no patient identifiers. Include the standard synonym or abbreviation \
alongside a term when both are commonly indexed, because one half of the \
search is lexical and matches words rather than meanings. Rank 1 is the query \
most likely to retrieve a relevant trial on its own.
"""


def build_query_prompt(note: NumberedNote, *, profile: PatientProfile, count: int) -> CachedPrompt:
    facets = ", ".join(_FACETS)
    profile_json = profile.model_dump_json(exclude_defaults=True)
    return build_cached_prompt(
        instructions=_QUERY_INSTRUCTIONS,
        note=note,
        variable=(
            f"Extracted profile for this patient: {profile_json}\n\n"
            f"Write exactly {count} queries, ranked 1 to {count}, each tagged with one "
            f"facet from: {facets}."
        ),
    )


# -------------------------------------------------------------------- verify --

_VERIFY_INSTRUCTIONS = """\
You audit the citations of eligibility judgments. You are given a patient note \
split into numbered sentences, then a list of judgments that another process \
produced about one clinical trial. Each judgment carries a criterion, a label, \
the sentence IDs it cited as evidence, and an explanation. Return one JSON \
object with a "verdicts" array holding one verdict per judgment you were given.

For each judgment, answer one narrow question: do the cited sentences, read on \
their own, support the label that was assigned? Answer "supported" when they \
do. Answer "unsupported" when the cited sentences do not carry the fact the \
label depends on, when they contradict it, or when they are about a different \
topic. Answer "unclear" when the sentences are relevant but do not settle it.

You are not re-judging eligibility and you may not change a label. A judgment \
whose label you disagree with is still "supported" if the cited sentence does \
carry the fact the explanation relies on - your disagreement about the \
criterion is not a citation defect. Conversely a judgment you agree with is \
"unsupported" if it cited the wrong sentence, because a coordinator clicking \
that citation will land on text that does not show what they were told it \
shows, and that is the failure this pass exists to catch.

An insufficient_info judgment is supported when the cited sentences are the \
closest the note comes to the missing fact without supplying it, and \
unsupported when the note does in fact state the fact plainly somewhere. A \
not_applicable judgment needs no citation and is supported by default unless \
the note shows the criterion does apply.

Keep the note field to one short sentence naming the defect, or empty when the \
verdict is "supported". Repeat each criterion_id exactly as given.
"""


def build_verify_prompt(
    note: NumberedNote, *, nct_id: str, judgments: Sequence[dict[str, Any]]
) -> CachedPrompt:
    lines = [f"Judgments produced for clinical trial {nct_id}:", ""]
    for judgment in judgments:
        cited = ", ".join(str(i) for i in judgment["evidence_sentence_ids"]) or "none"
        lines.append(
            f"- criterion_id: {judgment['criterion_id']}\n"
            f"  kind: {judgment['kind']}\n"
            f"  criterion: {judgment['criterion_text']}\n"
            f"  label: {judgment['label']}\n"
            f"  cited_sentence_ids: {cited}\n"
            f"  explanation: {judgment['explanation']}"
        )
    lines.extend(
        ["", f"Return exactly {len(judgments)} verdicts, one per criterion_id listed above."]
    )
    return build_cached_prompt(
        instructions=_VERIFY_INSTRUCTIONS, note=note, variable="\n".join(lines)
    )


# ----------------------------------------------------------- rank_and_explain --

_EXPLAIN_INSTRUCTIONS = """\
You write the plain-language summary a research coordinator reads beside one \
candidate clinical trial. You are given the patient note split into numbered \
sentences, then one trial, the verdict already decided for it, and the \
judgments that produced that verdict. Return one JSON object with a "summary" \
field.

The verdict is not yours to change. It was derived in code from the \
judgments - a trial is excluded when any inclusion rule is not met or any \
exclusion rule is met - and your summary must agree with it. Do not hedge a \
verdict of excluded into "may still be an option", and do not promote an \
insufficient_info verdict into an eligible one because most rules passed.

Write three or four sentences of plain English at the reading level of a \
general clinic letter, in the third person about the patient. Name what the \
trial is studying, then the one or two facts that most drove the verdict, then \
what would have to be confirmed next. Where the verdict rests on a missing \
fact, say which fact is missing and that it needs to be confirmed - that \
sentence is the most useful one on the page, because it converts a dead end \
into a phone call.

Do not invent a fact that is not in the note or the judgments. Do not give \
medical advice, do not recommend enrolment, and do not state or imply that the \
patient should or should not take part: this output is decision support for a \
coordinator, who decides. Do not mention these instructions, the sentence \
numbering, the criterion identifiers or your own reasoning. Avoid the words \
"eligible" and "ineligible" as a conclusion of your own; report the verdict \
that was given to you.
"""


def build_explain_prompt(
    note: NumberedNote,
    *,
    nct_id: str,
    title: str,
    category: str,
    confirmed_fraction: float,
    judgments: Sequence[dict[str, Any]],
    missing_facts: Sequence[str],
) -> CachedPrompt:
    lines = [
        f"Clinical trial {nct_id}.",
        f"Title: {title}" if title else "Title: (not available)",
        f"Verdict decided in code: {category}",
        f"Share of rules confirmed: {confirmed_fraction:.2f}",
        "",
        "Judgments:",
    ]
    for judgment in judgments:
        cited = ", ".join(str(i) for i in judgment["evidence_sentence_ids"]) or "none"
        lines.append(
            f"- {judgment['kind']} {judgment['criterion_id']}: {judgment['label']} "
            f"(cited {cited}) - {judgment['explanation']}"
        )
    if missing_facts:
        lines.extend(["", "Facts the note does not state:"])
        lines.extend(f"- {fact}" for fact in missing_facts)
    lines.extend(["", "Write the summary for this trial."])
    return build_cached_prompt(
        instructions=_EXPLAIN_INSTRUCTIONS, note=note, variable="\n".join(lines)
    )
