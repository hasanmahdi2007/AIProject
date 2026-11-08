"""TREC Clinical Trials access for the 2021 and 2022 tracks, via ``ir_datasets``.

Three entry points: :func:`load_topics` (patient notes), :func:`load_qrels`
(graded judgements) and :func:`iter_trials` (the document collection, lazily).

Why the dataset IDs are constants with a provenance comment
-----------------------------------------------------------
A wrong-but-importable dataset ID is a silent wrong-corpus bug that would
invalidate every number in the project, so the IDs below were **discovered by
enumerating the registry**, not guessed. See :data:`TREC_CT_2021` and
:data:`TREC_CT_2022`.

``IR_DATASETS_HOME``
--------------------
``ir_datasets`` resolves its cache from its own ``IR_DATASETS_HOME``
environment variable, which Beacon's settings object knows nothing about. Worse,
it reads it **at import time**: ``ir_datasets/datasets/clinicaltrials.py`` calls
``ir_datasets.util.home_path()`` while registering the dataset, so an env var set
after ``import ir_datasets`` is ignored for download paths. This module therefore
sets the env var from ``beacon_core.settings`` *before* importing ``ir_datasets``
(hence the deliberate non-top-level import below), which is the only ordering in
which the two configuration systems agree.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from beacon_core.settings import get_settings

__all__ = [
    "CORPUS_ZIP_NAMES",
    "DEFAULT_IR_DATASETS_HOME",
    "IR_DATASETS_HOME",
    "POSIX_FALLBACK_IR_DATASETS_HOME",
    "QRELS_2022_MD5",
    "QRELS_2022_URL",
    "RELEVANCE_LABELS",
    "TREC_CT_2021",
    "TREC_CT_2022",
    "TREC_CT_CORPUS",
    "Qrels",
    "Topic",
    "Trial",
    "corpus_is_cached",
    "dataset_id",
    "iter_trials",
    "load_qrels",
    "load_topics",
    "main",
]

logger = logging.getLogger(__name__)

DEFAULT_IR_DATASETS_HOME = Path("C:/Users/hasan/ir_datasets")
"""Fallback when ``BEACON_IR_DATASETS_HOME`` is unset, on this developer's box.

Deliberately outside the repo: the corpus is ~1.79 GB compressed (~2.6 GB
cached, measured 8 Nov) and must never land in
a git worktree. Overridden by ``BEACON_IR_DATASETS_HOME`` in ``.env``.
"""

POSIX_FALLBACK_IR_DATASETS_HOME = Path.home() / "ir_datasets"
"""Used when :data:`DEFAULT_IR_DATASETS_HOME` is not absolute on this platform.

CI runs on ubuntu-latest, where ``Path("C:/Users/hasan/ir_datasets")`` is a
*relative* path -- using it would create a directory literally named ``C:``
inside the repo checkout. ``~/ir_datasets`` is the platform-correct equivalent:
absolute, outside the repo, and still not the ``~/.ir_datasets`` default that
ir_datasets would otherwise pick on its own.
"""


def _configure_ir_datasets_home() -> Path:
    """Point ``ir_datasets`` at the directory Beacon's settings nominate.

    Must run before ``import ir_datasets``; see the module docstring.
    """
    configured = get_settings().ir_datasets_home
    if configured is not None:
        home = Path(configured)
    else:
        home = DEFAULT_IR_DATASETS_HOME
        if not home.is_absolute():
            home = POSIX_FALLBACK_IR_DATASETS_HOME
        logger.warning(
            "BEACON_IR_DATASETS_HOME is unset; defaulting ir_datasets cache to %s. "
            "Set BEACON_IR_DATASETS_HOME in .env to override (needs ~3 GB free).",
            home,
        )
    home.mkdir(parents=True, exist_ok=True)
    os.environ["IR_DATASETS_HOME"] = str(home)
    return home


IR_DATASETS_HOME = _configure_ir_datasets_home()

import ir_datasets  # noqa: E402  (must follow _configure_ir_datasets_home; see docstring)
from ir_datasets.formats import TrecQrels  # noqa: E402
from ir_datasets.util import Download, RequestsDownload  # noqa: E402

# --------------------------------------------------------------- dataset IDs --
#
# HOW THESE WERE CONFIRMED (ir_datasets 0.6.3, enumerated 2026-11-08).
#
# 1. Enumerated the registry -- `sorted(ir_datasets.registry._registered)` --
#    790 entries, of which exactly nine contain "clinical":
#        clinicaltrials
#        clinicaltrials/2017              clinicaltrials/2017/trec-pm-2017
#        clinicaltrials/2017/trec-pm-2018 clinicaltrials/2019
#        clinicaltrials/2019/trec-pm-2019 clinicaltrials/2021
#        clinicaltrials/2021/trec-ct-2021 clinicaltrials/2021/trec-ct-2022
#    The trec-pm-* subsets are the Precision Medicine track, not Clinical
#    Trials; the two trec-ct-* entries are the tracks we want.
#
# 2. NOTE THE TRAP: the 2022 track is "clinicaltrials/2021/trec-ct-2022", NOT
#    "clinicaltrials/2022/...". There is no clinicaltrials/2022 corpus at all.
#    TREC CT 2022 re-used the April-2021 ClinicalTrials.gov snapshot, and
#    ir_datasets models that by hanging both tracks off the 2021 collection
#    (verified in ir_datasets/datasets/clinicaltrials.py, where both
#    subsets['2021/trec-ct-2021'] and subsets['2021/trec-ct-2022'] are
#    constructed from the same `collection21` object). Hence TREC_CT_CORPUS is
#    shared and iter_trials() takes no year.
#
# 3. Verified each entry's reported metadata, then re-verified by actually
#    fetching the (small) topics and qrels:
#        clinicaltrials/2021              docs 375,580
#        clinicaltrials/2021/trec-ct-2021 docs 375,580; queries 75;
#                                         qrels 35,832 {0: 24243, 1: 6019, 2: 5570}
#        clinicaltrials/2021/trec-ct-2022 docs 375,580; queries 50; qrels NONE
#    Actual fetch matched the metadata exactly: 75 topics / 35,832 qrels for
#    2021 and 50 topics for 2022.
#
# 4. Provenance of the files ir_datasets downloads is decisive -- these are the
#    official track files, not a lookalike corpus:
#        2021 topics  http://www.trec-cds.org/topics2021.xml
#        2021 qrels   https://trec.nist.gov/data/trials/qrels2021.txt
#        2022 topics  https://www.trec-cds.org/topics2022.xml
#    and the qrels definitions ir_datasets ships for trec-ct-2021 are
#    {0: 'Not Relevant', 1: 'Excluded', 2: 'Eligible'} -- the TREC CT grading
#    scheme, which no other registry entry uses.

TREC_CT_CORPUS = "clinicaltrials/2021"
"""The April-2021 ClinicalTrials.gov snapshot: 375,580 trials, shared by both
tracks. Not year-specific -- see point 2 above."""

TREC_CT_2021 = "clinicaltrials/2021/trec-ct-2021"
"""TREC 2021 Clinical Trials track: 75 topics, 35,832 graded judgements."""

TREC_CT_2022 = "clinicaltrials/2021/trec-ct-2022"
"""TREC 2022 Clinical Trials track: 50 topics. ir_datasets 0.6.3 ships no
qrels for this subset; :func:`load_qrels` fetches them from NIST instead."""

_DATASET_IDS: dict[int, str] = {2021: TREC_CT_2021, 2022: TREC_CT_2022}

RELEVANCE_LABELS: dict[int, str] = {0: "Not Relevant", 1: "Excluded", 2: "Eligible"}
"""The TREC CT graded scheme, as ir_datasets reports it for trec-ct-2021.

Grades are **ordinal, not boolean**: 2 (eligible) > 1 (excluded) > 0 (not
relevant). "Excluded" means the trial is topically right for this patient but
the patient fails an eligibility criterion, which is a far better retrieval
result than an unrelated trial -- so graded nDCG must see the integer. Never
collapse this to a bool.
"""

QRELS_2022_URL = "https://trec.nist.gov/data/trials/qrels2022.txt"
"""Official TREC 2022 CT judgements.

ir_datasets 0.6.3 registers trec-ct-2022 with topics but **no** qrels (see
``subsets['2021/trec-ct-2022']`` in its clinicaltrials.py -- it is built with no
``TrecQrels`` argument, and ``has_qrels()`` returns False). The file does exist
at NIST, in the same directory as the 2021 qrels that ir_datasets itself uses,
in the same 4-column TREC format. We fetch it through ir_datasets' own download
and parsing machinery so it caches under IR_DATASETS_HOME like everything else.

Verified on 2026-11-08: downloads 666 kB and parses to 35,394 judgements over
all 50 topics, graded {0: 28419, 1: 3036, 2: 3939} -- the same three-grade
scheme as 2021. The md5 below is pinned so a silently changed or truncated
file fails loudly rather than skewing the 2022 numbers.
"""

QRELS_2022_MD5 = "64946c7c3dfc2f36dcb0e098a434fc9a"
"""md5 of the 2022 qrels file as fetched on 2026-11-08 (reported by ir_datasets)."""

Qrels = dict[str, dict[str, int]]
"""``{topic_id: {doc_id: grade}}`` -- the shape ``ranx.Qrels.from_dict`` wants."""


@dataclass(frozen=True, slots=True)
class Topic:
    """A TREC CT topic: a free-text synthetic patient note."""

    topic_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Trial:
    """One ClinicalTrials.gov record, as the TREC collection provides it."""

    doc_id: str
    title: str
    condition: str
    summary: str
    detailed_description: str
    eligibility: str


def dataset_id(year: int) -> str:
    """The verified ``ir_datasets`` ID for a track year.

    Raises:
        ValueError: if ``year`` is not a track Beacon supports.
    """
    try:
        return _DATASET_IDS[year]
    except KeyError:
        supported = ", ".join(str(y) for y in sorted(_DATASET_IDS))
        raise ValueError(
            f"No TREC Clinical Trials track for {year}; supported: {supported}"
        ) from None


def load_topics(year: int) -> list[Topic]:
    """Topics (patient notes) for a track year, in track order.

    Small download (tens of KB); safe in tests.
    """
    dataset = ir_datasets.load(dataset_id(year))
    return [Topic(topic_id=q.query_id, text=q.text.strip()) for q in dataset.queries_iter()]


def load_qrels(year: int) -> Qrels:
    """Graded judgements for a track year as ``{topic_id: {doc_id: grade}}``.

    Grades are kept as integers 2/1/0 (eligible/excluded/not relevant); see
    :data:`RELEVANCE_LABELS` for why collapsing them to a boolean is wrong.

    Small download (under 1 MB); safe in tests.
    """
    qrels: Qrels = {}
    for qrel in _qrels_iter(year):
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    return qrels


class _QrelRecord(Protocol):
    """The shape of an ``ir_datasets`` qrel tuple (which is untyped)."""

    query_id: str
    doc_id: str
    relevance: int


def _qrels_iter(year: int) -> Iterator[_QrelRecord]:
    """Yield raw qrel records, falling back to NIST where ir_datasets has none."""
    dataset = ir_datasets.load(dataset_id(year))
    if dataset.has_qrels():
        records: Iterator[_QrelRecord] = dataset.qrels_iter()
        yield from records
        return
    logger.info(
        "ir_datasets ships no qrels for %s; fetching the official file from %s",
        dataset_id(year),
        QRELS_2022_URL,
    )
    yield from _nist_qrels(year).qrels_iter()


def _nist_qrels(year: int) -> TrecQrels:
    """A ``TrecQrels`` over the official NIST judgements, cached under our home."""
    if year != 2022:
        raise ValueError(f"No NIST qrels fallback configured for {year}")
    cache_path = IR_DATASETS_HOME / "clinicaltrials" / "trec-ct-2022" / "qrels.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    download = Download(
        [RequestsDownload(QRELS_2022_URL)],
        cache_path=str(cache_path),
        expected_md5=QRELS_2022_MD5,
    )
    return TrecQrels(download, RELEVANCE_LABELS)


CORPUS_ZIP_NAMES = tuple(f"docs_2021_{part}.zip" for part in range(1, 6))
"""The five cache files ``ir_datasets`` downloads for the 2021 collection.

Names taken from ``ir_datasets/etc/downloads.json`` (keys ``docs/2021/1``..``5``,
``cache_path`` fields), which fetch
``http://www.trec-cds.org/2021_data/ClinicalTrials.2021-04-27.partN.zip``.
Together about 1.79 GB compressed -- the brief's "tens of GB" is pessimistic,
though ir_datasets then builds its own store alongside them, so still size the
volume generously.
"""


def corpus_is_cached() -> bool:
    """Whether all five collection archives are already downloaded.

    Lets a caller (or a test) decide whether :func:`iter_trials` will stream
    from disk or kick off a ~1.8 GB download. Partial ``.zip.tmp`` files do not
    count as present.
    """
    base = IR_DATASETS_HOME / "clinicaltrials"
    return all((base / name).is_file() for name in CORPUS_ZIP_NAMES)


def iter_trials(*, skip: int = 0, limit: int | None = None) -> Iterator[Trial]:
    """Lazily yield trials from the 375,580-document collection.

    Never materialises the collection: this is a generator, and callers should
    keep it that way. Downloads ~1.79 GB on first use, so it is excluded
    from CI and any test touching it is marked ``@pytest.mark.integration``.

    Args:
        skip: number of records to pass over first, so an interrupted
            ingestion job can resume from its last committed offset instead of
            re-reading from the start. Uses ``ir_datasets``' sliced ``docs_iter``,
            which seeks where the backing format allows it.
        limit: stop after yielding this many records. ``None`` means all.

    Raises:
        ValueError: on a negative ``skip`` or a non-positive ``limit``.
    """
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive or None, got {limit}")

    dataset = ir_datasets.load(TREC_CT_CORPUS)
    docs = dataset.docs_iter()
    stop = None if limit is None else skip + limit
    if skip or stop is not None:
        docs = docs[skip:stop]
    for doc in docs:
        yield Trial(
            doc_id=doc.doc_id,
            title=doc.title,
            condition=doc.condition,
            summary=doc.summary,
            detailed_description=doc.detailed_description,
            eligibility=doc.eligibility,
        )


# ----------------------------------------------------------------------- CLI --


def _print_first_topic(year: int) -> None:
    topics = load_topics(year)
    qrels = load_qrels(year)
    topic = topics[0]
    judged = qrels.get(topic.topic_id, {})
    by_grade = Counter(judged.values())

    print(f"dataset id      : {dataset_id(year)}")
    print(f"corpus id       : {TREC_CT_CORPUS}")
    print(f"IR_DATASETS_HOME: {IR_DATASETS_HOME}")
    print(f"topics          : {len(topics)}")
    print(f"qrels (all)     : {sum(len(v) for v in qrels.values())} over {len(qrels)} topics")
    print()
    print(f"--- topic {topic.topic_id} ---")
    print(topic.text)
    print()
    print(f"qrels for topic {topic.topic_id}: {len(judged)}")
    for grade, count in sorted(by_grade.items(), reverse=True):
        print(f"  {grade} ({RELEVANCE_LABELS.get(grade, '?')}): {count}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m evals.trec --topic 2021 --first``."""
    parser = argparse.ArgumentParser(
        prog="python -m evals.trec",
        description="Inspect the TREC Clinical Trials topics and qrels Beacon evaluates on.",
    )
    parser.add_argument(
        "--topic",
        type=int,
        choices=sorted(_DATASET_IDS),
        default=2021,
        help="track year (default: 2021)",
    )
    parser.add_argument(
        "--first",
        action="store_true",
        help="print the first topic of that year plus its qrels count",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not args.first:
        parser.error("nothing to do: pass --first")
    _print_first_topic(args.topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
