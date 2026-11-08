"""Reading the TREC Clinical Trials corpus with the fields ingestion needs.

Why this module exists (and does not simply call ``evals.trec.iter_trials``)
----------------------------------------------------------------------------
``evals.trec`` is the contract for *retrieval evaluation*, and the document
shape it exposes is the one ``ir_datasets`` parses out of the collection::

    doc_id, title, condition, summary, detailed_description, eligibility

That is six fields. B1's contract needs eleven, and five of the missing ones
are the ones the whole filtering story rests on: ``interventions``, ``phase``,
``overall_status``, ``minimum_age``/``maximum_age`` and ``gender``.
``ir_datasets`` 0.6.3 drops them at parse time -- its ``eligibility`` field is
only ``eligibility/criteria/textblock``, with the sibling ``<gender>``,
``<minimum_age>`` and ``<maximum_age>`` elements discarded -- and it also
reports ``condition`` as an empty string for a large share of records that do
carry ``<condition>`` elements in the source XML.

So the structured pass reads the **original ClinicalTrials.gov XML**, which
``ir_datasets`` has already downloaded and which it keeps verbatim in the five
``docs_2021_*.zip`` archives under ``IR_DATASETS_HOME``. Nothing is downloaded
here and no second copy of the corpus is created: this module opens the same
cache files, located through ``evals.trec``'s own
:data:`~evals.trec.IR_DATASETS_HOME` and :data:`~evals.trec.CORPUS_ZIP_NAMES`
constants, so the two packages can never disagree about where the corpus is.

:func:`cross_check_against_ir_datasets` keeps the two readings honest by
comparing titles and eligibility text for a sample of records against
``evals.trec.iter_trials()``.

Ordering and resumability
-------------------------
Records are yielded in zip-member order, which is ascending NCT ID within
ascending archive -- deterministic across runs and across machines. That makes
``skip``/``limit`` stable, but the job does **not** rely on it for
resumability: Postgres' ``trials.indexed_at`` is the resume marker, because an
offset cannot survive the corpus being re-fetched.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from evals.trec import CORPUS_ZIP_NAMES, IR_DATASETS_HOME, corpus_is_cached

__all__ = [
    "CORPUS_DIR",
    "EXPECTED_TRIAL_COUNT",
    "MAX_STORED_LOCATIONS",
    "CorpusNotCachedError",
    "RawTrial",
    "count_members",
    "cross_check_against_ir_datasets",
    "iter_raw_trials",
    "parse_trial_xml",
    "read_trial_xml",
]

CORPUS_DIR = IR_DATASETS_HOME / "clinicaltrials"
"""Where ``ir_datasets`` puts the five collection archives."""

EXPECTED_TRIAL_COUNT = 375_580
"""``clinicaltrials/2021`` document count, as ``ir_datasets`` reports it.

Checked against the number of ``.xml`` members across the five archives so a
silently partial cache fails loudly instead of indexing a short corpus.
"""

MAX_STORED_LOCATIONS = 50
"""Cap on ``trials.locations``.

A handful of trials list several hundred sites. Retrieval does not filter on
location in v1, so the whole list is dead weight in every row payload; the
first 50 are kept for provenance and the rest dropped.
"""


class CorpusNotCachedError(RuntimeError):
    """Raised when the collection archives are not on disk.

    Ingestion refuses to start a multi-gigabyte download as a side effect of
    ``python -m ingestion.run``; the operator fetches the corpus deliberately,
    through ``evals.trec``.
    """


@dataclass(frozen=True, slots=True)
class RawTrial:
    """One ClinicalTrials.gov record, verbatim apart from whitespace stripping.

    Every field here is *as published*. Nothing is normalised, parsed or
    defaulted: ``raw_min_age="N/A"`` stays ``"N/A"`` so that
    :mod:`ingestion.parsers` is the only place where a value can be lost, and
    so a parser bug is reproducible from the stored row.
    """

    nct_id: str
    title: str
    conditions: tuple[str, ...]
    interventions: tuple[str, ...]
    phase: str | None
    status: str | None
    brief_summary: str | None
    detailed_description: str | None
    eligibility_text: str | None
    raw_gender: str | None
    raw_min_age: str | None
    raw_max_age: str | None
    locations: tuple[dict[str, Any], ...] = field(default=())


def _text(element: ET.Element | None) -> str | None:
    """``element``'s text, stripped, or ``None`` when absent or blank."""
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


def _find_text(root: ET.Element, path: str) -> str | None:
    return _text(root.find(path))


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    """Order-preserving de-duplication; blank entries dropped."""
    seen: dict[str, None] = {}
    for value in values:
        cleaned = value.strip()
        if cleaned:
            seen.setdefault(cleaned, None)
    return tuple(seen)


def _locations(root: ET.Element) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for node in root.findall("location")[:MAX_STORED_LOCATIONS]:
        entry = {
            "facility": _find_text(node, "facility/name"),
            "city": _find_text(node, "facility/address/city"),
            "state": _find_text(node, "facility/address/state"),
            "country": _find_text(node, "facility/address/country"),
        }
        if any(entry.values()):
            out.append(entry)
    return tuple(out)


def parse_trial_xml(data: bytes | str, *, nct_id: str | None = None) -> RawTrial:
    """Parse one ``<clinical_study>`` document into a :class:`RawTrial`.

    Args:
        data: the XML document.
        nct_id: fallback identifier, used when the document has no
            ``<id_info><nct_id>`` element (the archives always have one; the
            argument exists so a hand-written test fixture need not).

    Raises:
        ValueError: when no NCT ID can be determined, since ``nct_id`` is the
            natural key for both stores and guessing one would silently merge
            two trials.
    """
    root = ET.fromstring(data) if isinstance(data, bytes) else ET.fromstring(data)

    resolved = _find_text(root, "id_info/nct_id") or nct_id
    if not resolved:
        raise ValueError("clinical_study document carries no id_info/nct_id")

    # brief_title is the human-facing one-liner and the most discriminative
    # text in the record; official_title is a formal restatement and only used
    # when brief_title is missing.
    title = _find_text(root, "brief_title") or _find_text(root, "official_title") or ""

    eligibility = root.find("eligibility")
    return RawTrial(
        nct_id=resolved,
        title=title,
        conditions=_dedupe([e.text or "" for e in root.findall("condition")]),
        interventions=_dedupe(
            [e.findtext("intervention_name") or "" for e in root.findall("intervention")]
        ),
        phase=_find_text(root, "phase"),
        status=_find_text(root, "overall_status"),
        brief_summary=_find_text(root, "brief_summary/textblock"),
        detailed_description=_find_text(root, "detailed_description/textblock"),
        eligibility_text=(
            None if eligibility is None else _find_text(eligibility, "criteria/textblock")
        ),
        raw_gender=None if eligibility is None else _find_text(eligibility, "gender"),
        raw_min_age=None if eligibility is None else _find_text(eligibility, "minimum_age"),
        raw_max_age=None if eligibility is None else _find_text(eligibility, "maximum_age"),
        locations=_locations(root),
    )


def _require_cache() -> None:
    if not corpus_is_cached():
        raise CorpusNotCachedError(
            f"The TREC collection archives are not in {CORPUS_DIR}. Expected "
            f"{', '.join(CORPUS_ZIP_NAMES)}. Fetch the corpus first (it is ~1.8 GB "
            "compressed) with `uv run python -c \"from evals.trec import iter_trials; "
            'next(iter_trials(limit=1))"` and re-run ingestion.'
        )


def _archives() -> Iterator[tuple[Path, zipfile.ZipFile]]:
    for name in CORPUS_ZIP_NAMES:
        path = CORPUS_DIR / name
        with zipfile.ZipFile(path) as archive:
            yield path, archive


def count_members() -> int:
    """Number of ``.xml`` members across the five archives.

    Reads only each zip's central directory, so it is cheap (about 20 s for
    375,580 entries) and does not decompress anything.
    """
    _require_cache()
    total = 0
    for _path, archive in _archives():
        total += sum(1 for name in archive.namelist() if name.endswith(".xml"))
    return total


def read_trial_xml(nct_id: str) -> bytes:
    """The raw XML for one trial, found by scanning the archives' directories.

    Linear in the number of members, so this is for spot checks and the
    hand-check report -- never for the ingest loop, which streams instead.

    Raises:
        KeyError: if no archive holds that NCT ID.
    """
    _require_cache()
    suffix = f"/{nct_id}.xml"
    for _path, archive in _archives():
        for name in archive.namelist():
            if name.endswith(suffix):
                return archive.read(name)
    raise KeyError(f"{nct_id} is not in the cached TREC collection")


def iter_raw_trials(*, skip: int = 0, limit: int | None = None) -> Iterator[RawTrial]:
    """Stream :class:`RawTrial` records from the cached collection archives.

    A generator by design: the corpus is 375,580 records and must never be
    materialised. Members are visited in archive order, which is ascending NCT
    ID, so ``skip`` is reproducible.

    Args:
        skip: records to pass over before yielding.
        limit: stop after yielding this many records; ``None`` means all.

    Raises:
        CorpusNotCachedError: if the archives are not downloaded.
        ValueError: on a negative ``skip`` or a non-positive ``limit``.
    """
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive or None, got {limit}")
    _require_cache()

    seen = 0
    yielded = 0
    for _path, archive in _archives():
        for info in archive.infolist():
            if not info.filename.endswith(".xml"):
                continue
            seen += 1
            if seen <= skip:
                continue
            nct_id = info.filename.rsplit("/", 1)[-1].removesuffix(".xml")
            yield parse_trial_xml(archive.read(info), nct_id=nct_id)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def cross_check_against_ir_datasets(sample: int = 200) -> dict[str, int]:
    """Compare this reader against ``evals.trec.iter_trials`` on a sample.

    The two readings must agree on identity and on the text they both claim to
    carry; where they disagree (``condition``, and every field ``ir_datasets``
    drops) the XML is authoritative, which is the reason this module exists.

    Returns a dict of counters: ``checked``, ``id_mismatch``,
    ``title_mismatch``, ``eligibility_mismatch`` and ``irds_condition_empty``
    (how often ``ir_datasets`` reported no condition while the XML had one --
    the measurement that justifies re-parsing).
    """
    from evals.trec import iter_trials

    counts = dict.fromkeys(
        ("checked", "id_mismatch", "title_mismatch", "eligibility_mismatch", "irds_condition_empty"),
        0,
    )
    by_id = {raw.nct_id: raw for raw in iter_raw_trials(limit=sample)}
    for doc in iter_trials(limit=sample):
        raw = by_id.get(doc.doc_id)
        if raw is None:
            counts["id_mismatch"] += 1
            continue
        counts["checked"] += 1
        if doc.title.strip() != raw.title:
            counts["title_mismatch"] += 1
        left = " ".join((doc.eligibility or "").split())
        right = " ".join((raw.eligibility_text or "").split())
        if left != right:
            counts["eligibility_mismatch"] += 1
        if not doc.condition.strip() and raw.conditions:
            counts["irds_condition_empty"] += 1
    return counts
