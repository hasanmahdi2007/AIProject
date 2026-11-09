"""Length-sorted reranker batching, tested without loading a model.

The optimisation itself is measured in the PR (1.7-2.5x, and bit-identical
scores). What is tested here is the part that can go *silently* wrong: the index
bookkeeping. Reordering documents for the model and failing to restore the
permutation would return a perfectly plausible list of scores attached to the
wrong trials - no exception, no shape change, just a quietly wrong ranking.

A fake model is injected through the ``RerankModel`` protocol, so none of this
needs the 280 MB of ONNX weights.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from retrieval.encoders import CrossEncoder


@dataclass
class LengthScoringModel:
    """Scores each document by its own length, and records what it was sent.

    Scoring by length is the trick that makes the permutation checkable: the
    expected score for ``documents[i]`` is ``len(documents[i])``, whatever order
    the model happened to see them in. If the restore step is wrong, the
    assertion fails with two numbers that say exactly which document got whose
    score.
    """

    seen: list[list[str]] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64) -> Iterable[float]:
        materialised = list(documents)
        self.seen.append(materialised)
        self.batch_sizes.append(batch_size)
        return [float(len(document)) for document in materialised]


DOCUMENTS = [
    "x" * 300,
    "x" * 10,
    "x" * 3000,
    "x" * 50,
    "x" * 700,
]


class TestLengthSortedBatching:
    def test_the_model_receives_documents_shortest_first(self) -> None:
        model = LengthScoringModel()

        CrossEncoder(model=model).score("q", list(DOCUMENTS))

        lengths = [len(document) for document in model.seen[0]]
        assert lengths == sorted(lengths)
        assert lengths == [10, 50, 300, 700, 3000]

    def test_scores_come_back_in_the_callers_order(self) -> None:
        """The whole point. Each score must land on the document it was for."""
        model = LengthScoringModel()

        scores = CrossEncoder(model=model).score("q", list(DOCUMENTS))

        assert scores == [300.0, 10.0, 3000.0, 50.0, 700.0]

    def test_no_document_is_dropped_or_duplicated(self) -> None:
        model = LengthScoringModel()
        documents = list(DOCUMENTS)

        scores = CrossEncoder(model=model).score("q", documents)

        assert len(scores) == len(documents)
        assert sorted(model.seen[0]) == sorted(documents)

    def test_equal_length_documents_are_handled(self) -> None:
        """``sorted`` is stable, so ties keep their arrival order - and distinct
        documents of equal length must still each get their own score."""
        model = LengthScoringModel()
        documents = ["aaa", "bbb", "c", "ddd"]

        scores = CrossEncoder(model=model).score("q", documents)

        assert scores == [3.0, 3.0, 1.0, 3.0]
        assert model.seen[0] == ["c", "aaa", "bbb", "ddd"]

    def test_a_single_document_is_unchanged(self) -> None:
        model = LengthScoringModel()

        assert CrossEncoder(model=model).score("q", ["only"]) == [4.0]

    def test_no_documents_does_not_call_the_model(self) -> None:
        """A fused set can be empty when a filter admits nothing; that is not an
        error and must not cost a forward pass."""
        model = LengthScoringModel()

        assert CrossEncoder(model=model).score("q", []) == []
        assert model.seen == []

    def test_the_measured_batch_size_is_passed_through(self) -> None:
        """8, because sorted-and-narrow was the fastest combination measured -
        in arrival order a wider batch was strictly *slower* (57 s -> 116 s at
        batch 32), since a wider batch more often contains the outlier that
        everything else is padded up to."""
        model = LengthScoringModel()

        CrossEncoder(model=model).score("q", list(DOCUMENTS))

        assert model.batch_sizes == [8]

    def test_the_batch_size_is_overridable(self) -> None:
        model = LengthScoringModel()

        CrossEncoder(model=model, batch_size=32).score("q", list(DOCUMENTS))

        assert model.batch_sizes == [32]

    def test_the_query_reaches_the_model_unchanged(self) -> None:
        """Only the documents are reordered. Sorting is per-call, so the query
        is the same for every pair in it."""
        captured: list[str] = []

        @dataclass
        class QueryCapturingModel:
            def rerank(
                self, query: str, documents: Iterable[str], batch_size: int = 64
            ) -> Iterable[float]:
                captured.append(query)
                return [0.0 for _ in documents]

        CrossEncoder(model=QueryCapturingModel()).score("the patient query", ["a", "bb"])

        assert captured == ["the patient query"]
