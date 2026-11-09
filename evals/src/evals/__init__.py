"""Beacon evaluation harness.

Every number that reaches ``README.md`` or a resume comes out of this package,
so the modules are arranged around one idea: *a number is worthless without
the conditions it was measured under*.

.. code-block:: text

    trec.py        TREC 2021/2022 topics, graded qrels, the document corpus
    split.py       the tune/held-out barrier - 2022 needs a capability token
    metrics.py     Recall@100, graded nDCG@10, P@10, MRR, + a ranx cross-check
    bootstrap.py   percentile and paired bootstrap CIs over topics
    backends.py    the retrieval seam: TREC run files, no live service needed
    ablations.py   the ablation table and the query-count sweep, config-driven
    judge.py       the Opus judge interface and Cohen's kappa (no live calls)
    provenance.py  corpus coverage: what index a number was measured against
    report.py      one command that regenerates the whole report

Regenerating everything::

    uv run python -m evals.report --runs <dir-of-trec-run-files> \\
        --indexed-trials <count>

Two deliberate pieces of friction, both documented where they live:

* measuring the held-out TREC 2022 split requires a written reason
  (:func:`evals.split.unlock_held_out`), and the reason is printed in the
  report;
* a result whose corpus coverage is partial *or unestablished* is marked
  unpublishable (:class:`evals.provenance.CorpusCoverage`), because a recall
  figure measured against a third of the collection is not comparable to any
  published number.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
