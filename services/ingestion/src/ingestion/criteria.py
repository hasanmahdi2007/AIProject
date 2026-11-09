"""Structure-aware splitting of ClinicalTrials.gov eligibility text into rules.

Why not fixed-size chunks
-------------------------
A criterion is the unit a judgment is made about, and a judgment is only
meaningful if the rule it judges is whole. ``"Platelet count greater than or
equal to"`` cut from ``"...20,000 cells/uL"`` does not become a slightly worse
rule, it becomes a rule that cannot be true or false, and the LLM will still
confidently label it. So the splitter only ever cuts where the *author* put a
boundary: a section header, a bullet, a number, or a blank line.

What the corpus looks like
--------------------------
Measured on a 20,000-record sample of ``clinicaltrials/2021`` (2026-11-08),
classified by whether an inclusion header, an exclusion header and a bullet
marker were present::

    16,062   inclusion + exclusion headers, dash bullets      (80.3%)
     2,616   inclusion + exclusion headers, numbered          (13.1%)
       333   no headers, dash bullets
       303   inclusion header only, dash bullets
       300   both headers, no bullet markers at all
       164   no headers, no markers (free prose)
        36 / 20 / 19 / 11 / 10 / 5   other combinations
         5   no <criteria> element at all
         4   star/bullet-character markers

So roughly 93% of records are bulleted or numbered under explicit headers and
split cleanly; about 2.5% are free prose and fall back to paragraph splitting.
The real shapes this has to survive, all taken from the corpus:

* ``NCT00001231`` -- the section header is *itself* a bullet
  (``-  INCLUSION CRITERIA:``) and items nest three deep.
* ``NCT00000306`` -- ``Inclusion Criteria- Subject must:``, and a ``•``
  bullet mixed in among the dashes.
* ``NCT00000134`` -- lowercase ``inclusion criteria:`` with the first rule on
  the *same line* as the header.
* ``NCT00000244`` -- headers present, no bullets, one rule per paragraph.
* ``NCT00000163`` / ``NCT00000389`` -- no headers and no markers, just prose.
* ``NCT00000187`` -- the entire criteria text is
  ``"Please contact site for information."``.
* ``NCT00000180`` -- no ``<criteria>`` element; ``NCT00026039`` -- no
  ``<eligibility>`` element at all.

Nesting is folded, not flattened
--------------------------------
In ``NCT00001231`` the top-level item is ``Perimenopausal Subjects:`` and the
seven rules that qualify it are nested one level deeper. Emitting the parent on
its own would produce a "rule" that says nothing, and emitting the children on
their own would lose which population they apply to. Sub-bullets are therefore
folded into their parent item, keeping their markers as ``-`` separators, so
the stored criterion is self-contained.

Reporting
---------
Every criterion records how it was obtained in ``TrialCriterion.parser``:

* ``rules`` -- the author's own markers were found and used.
* ``unstructured`` -- no markers; split on blank lines, or kept whole.

``unstructured`` is deliberately *not* called ``llm_fallback``: no LLM is
called here, and labelling a regex fallback as an LLM one would make the
fallback-rate metric a lie. An LLM pass over the ``unstructured`` rows is the
natural next step and this column is what makes its value measurable.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

__all__ = [
    "DEFAULT_KIND",
    "MAX_CRITERIA_PER_TRIAL",
    "MAX_SUBHEADING_CHARS",
    "MIN_CRITERION_CHARS",
    "NON_CRITERIA_TEXTS",
    "PARSER_RULES",
    "PARSER_UNSTRUCTURED",
    "Criterion",
    "SplitResult",
    "criterion_id",
    "split_eligibility",
]

PARSER_RULES = "rules"
PARSER_UNSTRUCTURED = "unstructured"

DEFAULT_KIND = "inclusion"
"""Kind assigned to criteria that appear under no header.

``trial_criteria.kind`` is constrained to ``inclusion``/``exclusion``, so there
is no "unknown" to record. Un-headed criteria lists on ClinicalTrials.gov are
overwhelmingly inclusion lists ("Patients must have..."), and treating them as
inclusions is the safe direction: a mislabelled inclusion that the patient
fails yields *insufficient evidence of eligibility*, whereas a mislabelled
exclusion yields an active exclusion -- the error that silently deletes a
trial.
"""

MIN_CRITERION_CHARS = 3
"""Below this a fragment carries no judgeable content (``"-"``, ``"1."``)."""

MAX_SUBHEADING_CHARS = 120
"""Longest line that may be read as a sub-heading rather than as prose.

A sub-heading is a short colon-terminated label such as ``"Concurrent
Medication:"``. The limit is what stops a long sentence that happens to end in
a colon from being hoisted out of its criterion and prefixed onto the next
several."""

MAX_CRITERIA_PER_TRIAL = 300
"""Guard against a pathological record producing thousands of rows. The 99.9th
percentile of the corpus is far below this; the splitter reports when it trips
rather than silently truncating."""

NON_CRITERIA_TEXTS = frozenset(
    {
        "please contact site for information",
        "please contact site for more information",
        "please contact the site for information",
        "contact site for information",
        "n/a",
        "na",
        "none",
        "not applicable",
        "no",
        "see above",
        "see protocol",
        "see study protocol",
        "to be determined",
        "tbd",
        "inclusion criteria",
        "exclusion criteria",
        "eligibility criteria",
    }
)
"""Boilerplate that occupies the criteria field without stating a rule.

``"Please contact site for information."`` alone accounts for a visible slice
of the older NIH records. Keeping it would create a criterion that every
patient trivially fails to satisfy evidence for, dragging down the
confirmed-fraction of otherwise fine trials.
"""

# --- section headers --------------------------------------------------------
#
# A header line is short, contains an inclusion/exclusion keyword, and ends in
# a colon -- optionally with the first rule inline after it, and optionally
# bulleted itself. The length limits on the text before and after the keyword
# are what stop an ordinary rule that happens to use the word "exclusion" from
# being mistaken for a header.
_HEADER_RE = re.compile(
    r"""^
    [ \t]*
    (?:[-\u2013\u2014*\u2022\u00b7\u25cf]+[ \t]+)?        # header may be bulleted
    (?P<pre>[^:\n]{0,28}?)
    \b(?P<kind>inclusion|inclusions|include|included|includes
       |exclusion|exclusions|exclude|excluded|excludes
       |ineligibility|eligibility)\b
    (?P<post>[^:\n]{0,60}?)
    [ \t]*[:\uff1a][ \t]*
    (?P<rest>.*)$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_HEADER_NO_COLON_RE = re.compile(
    r"""^
    [ \t]*
    (?:[-\u2013\u2014*\u2022\u00b7\u25cf]+[ \t]+)?
    (?P<pre>[a-z()/&' \t\u2013-]{0,28}?)
    \b(?P<kind>inclusion|exclusion|ineligibility|eligibility)\b
    [ \t]*(?:criteria|criterion)?[ \t]*[.\u2013-]?[ \t]*
    $""",
    re.IGNORECASE | re.VERBOSE,
)
"""``Inclusion Criteria`` on a line of its own, with no colon.

The NIAID AIDS-trial records -- a large family in this corpus, e.g.
``NCT00000637`` -- write their headers this way. Requiring the colon made the
splitter miss them, which filed every exclusion rule in those trials as an
inclusion: the single worst error this module can make, because an inclusion
the patient fails reads as *insufficient evidence* while an exclusion they
match reads as *excluded*. Found by the first 1,000-record smoke run.

Safe because the match must consume the entire line: a line that is nothing
but the words "Exclusion Criteria" is not a rule under any reading.
"""

_HEADER_PRE_RE = re.compile(r"^[a-z()/&' \t\u2013-]*$", re.IGNORECASE)
"""The words allowed before the keyword (``Key``, ``Main``, ``Patient``, ...).

Digits or punctuation there mean the line is a rule that mentions exclusion,
not a header introducing one."""

_HEADER_NEGATIVE_RE = re.compile(r"\b(no|without|not|non|nil|neither)\s*$", re.IGNORECASE)
"""``"- No exclusion criteria:"`` is a rule, not a section header."""

_EXCLUSION_KEYWORDS = frozenset(
    {"exclusion", "exclusions", "exclude", "excluded", "excludes", "ineligibility"}
)
_NEUTRAL_KEYWORDS = frozenset({"eligibility"})
"""``"ELIGIBILITY CRITERIA:"`` wraps the whole block; it opens no new section
and must not survive as a rule of its own."""

# --- bullet markers ---------------------------------------------------------
_DASH_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-\u2013\u2014*\u2022\u00b7\u25cf\u25aa])[ \t]+(?=\S)"
)
_NUMBER_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>\(?\d{1,2}[.):])[ \t]+(?=\S)")
_ALPHA_RE = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>\(?[a-zA-Z][.)])[ \t]+(?=\S)")

_MARKER_STYLES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dash", _DASH_RE),
    ("number", _NUMBER_RE),
    ("alpha", _ALPHA_RE),
)

_MIN_ALPHA_MARKERS = 2
"""A single ``a)`` is far more likely to be prose than a lettered list."""


@dataclass(frozen=True, slots=True)
class Criterion:
    """One eligibility rule, ready to become a ``TrialCriterion`` row."""

    nct_id: str
    criterion_id: str
    kind: str
    position: int
    text: str
    parser: str


@dataclass(frozen=True, slots=True)
class SplitResult:
    """The criteria of one trial plus how the split went.

    ``marker_style`` and ``had_headers`` are what the ingest run aggregates
    into the reportable fallback rate.
    """

    criteria: tuple[Criterion, ...]
    had_headers: bool
    marker_style: str
    truncated: bool = False

    @property
    def parser_counts(self) -> Counter[str]:
        return Counter(c.parser for c in self.criteria)

    @property
    def is_fallback(self) -> bool:
        """True when no criterion came from the author's own markers."""
        return bool(self.criteria) and all(c.parser == PARSER_UNSTRUCTURED for c in self.criteria)


def criterion_id(nct_id: str, kind: str, sequence: int) -> str:
    """``NCT01234567:inc:03`` -- the citation key used in judgments."""
    short = "exc" if kind == "exclusion" else "inc"
    return f"{nct_id}:{short}:{sequence:02d}"


def _normalise(text: str) -> list[str]:
    """Split into lines with CR and non-breaking space removed, tabs expanded.

    Indentation is preserved, because it is the only signal of nesting depth.
    """
    cleaned = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u2028", "\n")
        .expandtabs(4)
    )
    return [line.rstrip() for line in cleaned.split("\n")]


def _header_kind(line: str) -> tuple[str | None, str] | None:
    """``(kind, inline_rest)`` if ``line`` is a section header, else ``None``.

    ``kind`` is ``None`` for a neutral wrapper such as ``ELIGIBILITY
    CRITERIA:``, which closes nothing and opens nothing.
    """
    if not line.strip():
        return None
    match = _HEADER_RE.match(line)
    rest = ""
    if match is not None:
        post = match["post"] or ""
        if "." in post:
            return None
        rest = match["rest"].strip()
    else:
        match = _HEADER_NO_COLON_RE.match(line)
        if match is None:
            return None
    pre = match["pre"] or ""
    if not _HEADER_PRE_RE.match(pre) or _HEADER_NEGATIVE_RE.search(pre):
        return None
    keyword = match["kind"].lower()
    if keyword in _NEUTRAL_KEYWORDS:
        return None, rest
    kind = "exclusion" if keyword in _EXCLUSION_KEYWORDS else "inclusion"
    return kind, rest


@dataclass(frozen=True, slots=True)
class _Section:
    kind: str
    lines: tuple[str, ...]


def _sections(lines: list[str]) -> list[_Section]:
    """Partition lines into header-delimited sections.

    Text before the first header becomes a section of :data:`DEFAULT_KIND`.
    """
    out: list[_Section] = []
    current_kind = DEFAULT_KIND
    buffer: list[str] = []

    def flush() -> None:
        if any(line.strip() for line in buffer):
            out.append(_Section(kind=current_kind, lines=tuple(buffer)))
        buffer.clear()

    for line in lines:
        header = _header_kind(line)
        if header is None:
            buffer.append(line)
            continue
        kind, rest = header
        if kind is None:
            # Neutral wrapper: drop the header line, keep any inline remainder.
            if rest:
                buffer.append(rest)
            continue
        flush()
        current_kind = kind
        buffer = [rest] if rest else []
    flush()
    return out


def _marker_lines(lines: tuple[str, ...], pattern: re.Pattern[str]) -> dict[int, tuple[int, int]]:
    """``{line_index: (indent_width, marker_end_offset)}`` for marker lines."""
    found: dict[int, tuple[int, int]] = {}
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match is not None:
            found[index] = (len(match["indent"]), match.end())
    return found


def _choose_markers(lines: tuple[str, ...]) -> tuple[str, dict[int, tuple[int, int]]]:
    """Pick the marker style this section actually uses.

    Dash beats number beats letter, because a dashed list containing ``"2. "``
    inside a wrapped line is common while the reverse is not. Lettered markers
    need at least :data:`_MIN_ALPHA_MARKERS` hits to be believed.
    """
    for style, pattern in _MARKER_STYLES:
        found = _marker_lines(lines, pattern)
        if not found:
            continue
        if style == "alpha" and len(found) < _MIN_ALPHA_MARKERS:
            continue
        return style, found
    return "none", {}


def _clean(text: str) -> str:
    """Collapse whitespace and strip dangling list punctuation."""
    collapsed = " ".join(text.split())
    return collapsed.strip(" ;,")


def _is_non_criterion(text: str) -> bool:
    key = re.sub(r"[^a-z0-9/ ]", "", text.lower()).strip()
    return len(text) < MIN_CRITERION_CHARS or key in NON_CRITERIA_TEXTS


def _items_from_markers(lines: tuple[str, ...], markers: dict[int, tuple[int, int]]) -> list[str]:
    """Group marked lines into items, folding genuine sub-lists into parents.

    The top level is the indent of the *first* marker in the section, not the
    shallowest one: ``NCT00002558`` has its list at one indent and then a
    stray shallower bullet further down, and taking the minimum would push
    everything before that stray line into a single item.

    A deeper marker is only folded into the open item when that item reads
    like a lead-in -- when its text so far ends in a colon, as
    ``"Perimenopausal Subjects:"`` does in ``NCT00001231``. Once a depth has
    been folded it keeps folding until a marker at the top level closes the
    group, so all seven children of that lead-in stay with it. Where the open
    item is *not* a lead-in, a deeper marker is a sibling the author merely
    indented inconsistently -- ``NCT00000306`` mixes a ``•`` at one indent
    with dashes at another inside the same exclusion list -- and splitting
    there is right.

    An unmarked line at or above the top level that ends in a colon is a
    *sub-heading*, and it is prefixed onto every item beneath it instead of
    becoming an item of its own. ``NCT00000637`` is the case that forces this:
    under ``Inclusion Criteria`` it writes ``Concurrent Medication:`` then
    ``Allowed:`` then the bullets, and later ``Excluded:`` then more bullets.
    Emitted alone, ``"Allowed:"`` is not a rule; glued to the end of the
    preceding bullet -- which is what happens if it is treated as an ordinary
    continuation -- it corrupts that rule instead. Prefixed, the stored
    criterion reads ``"Concurrent Medication: Allowed: Acetaminophen,
    ibuprofen, or aspirin..."``, which is judgeable and true to the source.

    A new sub-heading run replaces the previous one rather than nesting under
    it, because the records write sibling sub-headings at the same indent and
    there is no reliable signal to tell a sibling from a child.

    All other unmarked lines are continuations of the open item, which is how
    wrapped text rejoins.
    """
    first_indent = min(index for index in markers)
    top = markers[first_indent][0]
    items: list[list[str]] = []
    preamble: list[str] = []
    folded_depths: set[int] = set()
    context: list[str] = []
    context_open = False

    def prefixed(body: str) -> str:
        return f"{' '.join(context)} {body}" if context else body

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        marker = markers.get(index)
        if marker is None:
            indent = len(line) - len(line.lstrip())
            if indent <= top and stripped.endswith(":") and len(stripped) <= MAX_SUBHEADING_CHARS:
                if not context_open:
                    context.clear()
                    context_open = True
                context.append(stripped)
                continue
            piece = stripped
        else:
            indent, offset = marker
            body = line[offset:].strip()
            open_item = " ".join(items[-1]) if items else ""
            if indent <= top:
                items.append([prefixed(body)])
                folded_depths.clear()
                context_open = False
                continue
            if indent in folded_depths or open_item.endswith(":"):
                folded_depths.add(indent)
                # Keep a visible separator so the folded sub-structure survives.
                piece = f"- {body}"
            else:
                items.append([prefixed(body)])
                folded_depths.clear()
                context_open = False
                continue
        context_open = False
        if items:
            items[-1].append(piece)
        else:
            preamble.append(piece)

    out = [" ".join(preamble)] if preamble else []
    out.extend(" ".join(parts) for parts in items)
    return out


def _items_from_paragraphs(lines: tuple[str, ...]) -> list[str]:
    """Blank-line separated paragraphs, as the only remaining author boundary.

    If there is only one paragraph it is kept whole. That makes for a long
    criterion, which is a visible, honest problem; cutting it at a character
    count would make for several short ones that only look right.
    """
    paragraphs: list[list[str]] = []
    for line in lines:
        if line.strip():
            if not paragraphs or paragraphs[-1] == []:
                paragraphs.append([])
            paragraphs[-1].append(line.strip())
        elif paragraphs and paragraphs[-1]:
            paragraphs.append([])
    return [" ".join(parts) for parts in paragraphs if parts]


def split_eligibility(nct_id: str, text: str | None) -> SplitResult:
    """Split one trial's eligibility text into individual criteria.

    Args:
        nct_id: the trial's NCT ID, used to build each ``criterion_id``.
        text: the raw ``eligibility/criteria/textblock``, or ``None``.

    Returns:
        A :class:`SplitResult`. An empty ``criteria`` tuple is a legitimate
        outcome -- 863 records of the corpus have no eligibility element and a
        few hundred more say only "Please contact site for information."
    """
    if not text or not text.strip():
        return SplitResult(criteria=(), had_headers=False, marker_style="none")

    lines = _normalise(text)
    had_headers = any(_header_kind(line) is not None for line in lines)
    sections = _sections(lines)

    collected: list[tuple[str, str, str]] = []  # (kind, text, parser)
    styles: list[str] = []
    for section in sections:
        style, markers = _choose_markers(section.lines)
        styles.append(style)
        if markers:
            raw_items = _items_from_markers(section.lines, markers)
            parser = PARSER_RULES
        else:
            raw_items = _items_from_paragraphs(section.lines)
            parser = PARSER_UNSTRUCTURED
        for item in raw_items:
            cleaned = _clean(item)
            if not _is_non_criterion(cleaned):
                collected.append((section.kind, cleaned, parser))

    truncated = len(collected) > MAX_CRITERIA_PER_TRIAL
    collected = collected[:MAX_CRITERIA_PER_TRIAL]

    sequences: Counter[str] = Counter()
    criteria: list[Criterion] = []
    for position, (kind, body, parser) in enumerate(collected):
        sequences[kind] += 1
        criteria.append(
            Criterion(
                nct_id=nct_id,
                criterion_id=criterion_id(nct_id, kind, sequences[kind]),
                kind=kind,
                position=position,
                text=body,
                parser=parser,
            )
        )

    marker_style = next((s for s in styles if s != "none"), "none")
    return SplitResult(
        criteria=tuple(criteria),
        had_headers=had_headers,
        marker_style=marker_style,
        truncated=truncated,
    )
