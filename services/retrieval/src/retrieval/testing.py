"""Fixture collections and fakes, so the tests are honest and self-contained.

Two reasons this is in ``src/`` rather than ``tests/``: a fixture collection is
a small piece of infrastructure worth reusing from an eval or a REPL, and the
repo bans a second ``conftest.py``.

The integration tests build their own collections rather than querying the real
``trials`` one. That is not only about not depending on B1's 375K ingest
finishing - it is about the tests being able to *state* what they prove. The
overfiltering test needs a filter of known selectivity (exactly 10%), and the
NULL-safety test needs a trial whose ``min_age_years`` is known to be absent.
Against the production collection both of those would be "whatever the corpus
happens to contain this week", which is not a test.

Two fixtures, because they prove different things and would interfere:

:func:`create_scale_collection`
    ~1,200 points with random vectors and payloads tuned so that exactly 10% of
    them pass an age-45 filter. Random vectors are the point: the ranking is
    meaningless, which is what makes the *count* the only thing the test can be
    reading. No model is loaded, so it is fast.

:func:`create_semantic_collection`
    ~24 hand-written trials with real embeddings and deliberately awkward
    age/sex payloads - absent bounds, a male-only trial, a paediatric cap. Small
    enough to reason about by hand, real enough that search actually works.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.model_description import BaseModelDescription
from fastembed.common.utils import define_cache_dir
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.conversions.common_types import QueryResponse

from beacon_core.schemas import Sex
from beacon_core.settings import QdrantSettings
from retrieval.hydrate import TrialText

__all__ = [
    "FixtureTrial",
    "INTEGRATION_HINT",
    "SCALE_FILTER_AGE",
    "SCALE_SELECTIVITY",
    "SEMANTIC_TRIALS",
    "StubCrossEncoder",
    "StubDenseEncoder",
    "StubQdrantClient",
    "StubSparseEncoder",
    "create_scale_collection",
    "create_semantic_collection",
    "missing_models",
    "point_id",
    "semantic_text_store",
]

INTEGRATION_HINT = "Start the stack: docker compose -f deploy/compose.yaml up -d --wait"

SCALE_FILTER_AGE = 45
"""The patient age the scale fixture is tuned around."""

SCALE_SELECTIVITY = 0.10
"""Share of scale-fixture points that pass an age-:data:`SCALE_FILTER_AGE`
filter. 10% is the brief's scenario: with ``hnsw_ef=40``, post-filtering a
top-40 beam leaves about four survivors."""


def point_id(nct_id: str) -> str:
    """uuid5 of the NCT ID - the same scheme B1's store uses.

    Qdrant point IDs must be an unsigned integer or a UUID, so an NCT ID cannot
    be one directly. Deriving it makes upserts idempotent by NCT ID.
    """
    return str(uuid5(NAMESPACE_URL, nct_id))


@dataclass(frozen=True, slots=True)
class FixtureTrial:
    """One synthetic trial: the text that gets embedded plus its filter facts."""

    nct_id: str
    text: str
    min_age_years: float | None = None
    max_age_years: float | None = None
    sex: Sex | None = None
    phase: str | None = "Phase 2"
    status: str | None = "Recruiting"
    conditions: list[str] = field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        """Exactly the six keys B1 writes, including explicit ``None``s."""
        return {
            "nct_id": self.nct_id,
            "min_age_years": self.min_age_years,
            "max_age_years": self.max_age_years,
            "sex": self.sex,
            "phase": self.phase,
            "status": self.status,
        }


# --------------------------------------------------------------------------
# The semantic fixture. Ages and sexes are chosen so that a 45-year-old female
# patient is the interesting case: some trials exclude her on positive evidence,
# some admit her, and several have unknown bounds and must survive.
# --------------------------------------------------------------------------

SEMANTIC_TRIALS: tuple[FixtureTrial, ...] = (
    # --- the NULL cases: these must survive every age and sex filter ---
    FixtureTrial(
        nct_id="NCTFIX0001",
        text=(
            "Lung cancer. A study of osimertinib in EGFR-mutant non-small cell lung "
            "carcinoma. Interventions: osimertinib. Adults with advanced NSCLC and an "
            "activating EGFR exon 19 deletion or L858R mutation."
        ),
        min_age_years=None,  # unknown: "N/A" on the source record
        max_age_years=None,
        sex=None,
        conditions=["Non-Small Cell Lung Carcinoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0002",
        text=(
            "Lung cancer. Pembrolizumab plus platinum doublet chemotherapy in metastatic "
            "non-small cell lung cancer with PD-L1 expression. Interventions: "
            "pembrolizumab, carboplatin, pemetrexed."
        ),
        min_age_years=18.0,
        max_age_years=None,  # one bound known, one not
        sex="all",
        conditions=["Non-Small Cell Lung Carcinoma", "Carcinoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0003",
        text=(
            "Lung cancer. Durvalumab consolidation after concurrent chemoradiotherapy in "
            "unresectable stage III non-small cell lung cancer. Interventions: durvalumab."
        ),
        min_age_years=None,
        max_age_years=80.0,
        sex=None,
        conditions=["Non-Small Cell Lung Carcinoma"],
    ),
    # --- positive evidence of ineligibility for a 45-year-old ---
    FixtureTrial(
        nct_id="NCTFIX0010",
        text=(
            "Lung cancer in older adults. A geriatric assessment-guided study of "
            "immunotherapy in non-small cell lung carcinoma in patients aged 65 and over. "
            "Interventions: atezolizumab."
        ),
        min_age_years=65.0,
        max_age_years=None,
        sex="all",
        conditions=["Non-Small Cell Lung Carcinoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0011",
        text=(
            "Paediatric lung tumours. A study of targeted therapy in children and "
            "adolescents with thoracic malignancies including pleuropulmonary blastoma. "
            "Interventions: crizotinib."
        ),
        min_age_years=1.0,
        max_age_years=17.0,
        sex="all",
        conditions=["Pleuropulmonary Blastoma"],
    ),
    # --- sex restrictions ---
    FixtureTrial(
        nct_id="NCTFIX0020",
        text=(
            "Prostate cancer. Androgen deprivation plus abiraterone in metastatic "
            "castration-sensitive prostate carcinoma with lung metastases. Interventions: "
            "abiraterone acetate, leuprolide."
        ),
        min_age_years=18.0,
        max_age_years=None,
        sex="male",
        conditions=["Prostate Carcinoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0021",
        text=(
            "Ovarian cancer. Niraparib maintenance in platinum-sensitive recurrent "
            "high-grade serous ovarian carcinoma. Interventions: niraparib."
        ),
        min_age_years=18.0,
        max_age_years=None,
        sex="female",
        conditions=["Ovarian Carcinoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0022",
        text=(
            "Breast cancer. A study of an oral CDK4/6 inhibitor with endocrine therapy in "
            "hormone receptor-positive HER2-negative metastatic breast cancer. "
            "Interventions: ribociclib, letrozole."
        ),
        min_age_years=None,
        max_age_years=None,
        sex="female",
        conditions=["Breast Carcinoma"],
    ),
    # --- distractors on other diseases, so relevance ordering has work to do ---
    FixtureTrial(
        nct_id="NCTFIX0030",
        text=(
            "Type 2 diabetes. A randomised trial of semaglutide versus placebo on "
            "cardiovascular outcomes in type 2 diabetes mellitus. Interventions: "
            "semaglutide."
        ),
        min_age_years=18.0,
        max_age_years=75.0,
        sex="all",
        conditions=["Type 2 Diabetes Mellitus"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0031",
        text=(
            "Rheumatoid arthritis. A study of a JAK inhibitor in moderate to severe "
            "rheumatoid arthritis after methotrexate failure. Interventions: upadacitinib."
        ),
        min_age_years=18.0,
        max_age_years=None,
        sex="all",
        conditions=["Rheumatoid Arthritis"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0032",
        text=(
            "Heart failure. Sacubitril-valsartan in heart failure with reduced ejection "
            "fraction. Interventions: sacubitril, valsartan."
        ),
        min_age_years=18.0,
        max_age_years=None,
        sex="all",
        conditions=["Heart Failure"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0033",
        text=(
            "Major depressive disorder. Esketamine nasal spray in treatment-resistant "
            "depression. Interventions: esketamine."
        ),
        min_age_years=18.0,
        max_age_years=64.0,
        sex="all",
        conditions=["Major Depressive Disorder"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0034",
        text=(
            "Colorectal cancer. FOLFOXIRI plus bevacizumab in previously untreated "
            "metastatic colorectal adenocarcinoma. Interventions: fluorouracil, "
            "oxaliplatin, irinotecan, bevacizumab."
        ),
        min_age_years=18.0,
        max_age_years=70.0,
        sex="all",
        conditions=["Colorectal Carcinoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0035",
        text=(
            "Multiple myeloma. Daratumumab with lenalidomide and dexamethasone in newly "
            "diagnosed multiple myeloma ineligible for transplant. Interventions: "
            "daratumumab, lenalidomide."
        ),
        min_age_years=None,
        max_age_years=None,
        sex=None,
        conditions=["Multiple Myeloma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0036",
        text=(
            "Asthma. An anti-IL-5 monoclonal antibody in severe eosinophilic asthma. "
            "Interventions: mepolizumab."
        ),
        min_age_years=12.0,
        max_age_years=None,
        sex="all",
        conditions=["Asthma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0037",
        text=(
            "Chronic lymphocytic leukaemia. Venetoclax plus obinutuzumab in previously "
            "untreated chronic lymphocytic leukaemia. Interventions: venetoclax."
        ),
        min_age_years=18.0,
        max_age_years=None,
        sex="all",
        conditions=["Chronic Lymphocytic Leukemia"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0038",
        text=(
            "Glioblastoma. Tumour treating fields with temozolomide in newly diagnosed "
            "glioblastoma multiforme. Interventions: temozolomide."
        ),
        min_age_years=22.0,
        max_age_years=None,
        sex="all",
        conditions=["Glioblastoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0039",
        text=(
            "Melanoma. Adjuvant nivolumab versus ipilimumab in resected stage III or IV "
            "melanoma. Interventions: nivolumab, ipilimumab."
        ),
        min_age_years=15.0,
        max_age_years=None,
        sex="all",
        conditions=["Melanoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0040",
        text=(
            "Atrial fibrillation. Apixaban versus warfarin for stroke prevention in "
            "non-valvular atrial fibrillation. Interventions: apixaban."
        ),
        min_age_years=18.0,
        max_age_years=None,
        sex="all",
        conditions=["Atrial Fibrillation"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0041",
        text=(
            "Crohn disease. Risankizumab induction and maintenance in moderately to "
            "severely active Crohn disease. Interventions: risankizumab."
        ),
        min_age_years=16.0,
        max_age_years=80.0,
        sex="all",
        conditions=["Crohn Disease"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0042",
        text=(
            "Alzheimer disease. An anti-amyloid monoclonal antibody in early symptomatic "
            "Alzheimer disease. Interventions: lecanemab."
        ),
        min_age_years=50.0,
        max_age_years=90.0,
        sex="all",
        conditions=["Alzheimer Disease"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0043",
        text=(
            "Sickle cell disease. Gene therapy with a modified autologous haematopoietic "
            "stem cell product in severe sickle cell disease. Interventions: exagamglogene "
            "autotemcel."
        ),
        min_age_years=12.0,
        max_age_years=35.0,
        sex="all",
        conditions=["Sickle Cell Disease"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0044",
        text=(
            "Hepatocellular carcinoma. Atezolizumab plus bevacizumab in unresectable "
            "hepatocellular carcinoma. Interventions: atezolizumab, bevacizumab."
        ),
        min_age_years=18.0,
        max_age_years=None,
        sex="all",
        conditions=["Hepatocellular Carcinoma"],
    ),
    FixtureTrial(
        nct_id="NCTFIX0045",
        text=(
            "Pulmonary fibrosis. Nintedanib in progressive fibrosing interstitial lung "
            "disease. Interventions: nintedanib."
        ),
        min_age_years=40.0,
        max_age_years=None,
        sex="all",
        conditions=["Idiopathic Pulmonary Fibrosis"],
    ),
)


def semantic_text_store(
    trials: Sequence[FixtureTrial] = SEMANTIC_TRIALS,
) -> Mapping[str, TrialText]:
    """The text side of the semantic fixture, for a ``MappingTrialTextStore``.

    ``overview_text`` is the same string that was embedded, exactly as in
    production, so the reranker reads what the retriever ranked.
    """
    return {
        trial.nct_id: TrialText(
            nct_id=trial.nct_id,
            title=trial.text.split(".")[0],
            conditions=list(trial.conditions),
            interventions=[],
            overview_text=trial.text,
        )
        for trial in trials
    }


# --------------------------------------------------------------------------
# collection construction
# --------------------------------------------------------------------------


async def _recreate(
    client: AsyncQdrantClient,
    name: str,
    settings: QdrantSettings,
    *,
    force_hnsw: bool,
) -> None:
    """Create the collection with the production vector configuration.

    ``force_hnsw`` sets both thresholds to 1 KB so Qdrant actually builds and
    uses an HNSW graph at fixture scale. Without it, a 1,200-point collection
    sits under the default 20 MB ``indexing_threshold``, Qdrant's planner
    answers every query by brute force, and the overfiltering test would be
    measuring exact search while claiming to measure a graph walk.
    """
    if await client.collection_exists(name):
        await client.delete_collection(name)
    await client.create_collection(
        collection_name=name,
        vectors_config={
            settings.dense_vector_name: models.VectorParams(
                size=settings.dense_dim,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            settings.sparse_vector_name: models.SparseVectorParams(
                # The same IDF declaration production uses. Without it the
                # sparse branch of the fixture would behave differently from the
                # sparse branch of production, which would make the fixture a
                # test of something else.
                modifier=models.Modifier.IDF
                if settings.sparse_idf_modifier
                else models.Modifier.NONE
            )
        },
        hnsw_config=(
            models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=1)
            if force_hnsw
            else None
        ),
        optimizers_config=(
            models.OptimizersConfigDiff(indexing_threshold=1) if force_hnsw else None
        ),
    )
    # The payload indexes the filters need. Float for the bounds, keyword for
    # sex - the same types B1 creates, and the same ones the integration test
    # asserts are present on the production collection.
    await client.create_payload_index(
        collection_name=name,
        field_name="min_age_years",
        field_schema=models.PayloadSchemaType.FLOAT,
    )
    await client.create_payload_index(
        collection_name=name,
        field_name="max_age_years",
        field_schema=models.PayloadSchemaType.FLOAT,
    )
    await client.create_payload_index(
        collection_name=name,
        field_name="sex",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )


async def create_semantic_collection(
    client: AsyncQdrantClient,
    name: str,
    settings: QdrantSettings,
    *,
    dense_vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[models.SparseVector],
    trials: Sequence[FixtureTrial] = SEMANTIC_TRIALS,
) -> None:
    """Upsert :data:`SEMANTIC_TRIALS` with real embeddings.

    The vectors are passed in rather than computed here so the caller owns model
    loading - the test fixture loads the two bi-encoders once per session
    instead of once per collection.
    """
    if not (len(dense_vectors) == len(sparse_vectors) == len(trials)):
        raise ValueError("dense_vectors, sparse_vectors and trials must be the same length")
    await _recreate(client, name, settings, force_hnsw=False)
    await client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=point_id(trial.nct_id),
                vector={
                    settings.dense_vector_name: list(dense),
                    settings.sparse_vector_name: sparse,
                },
                payload=trial.payload(),
            )
            for trial, dense, sparse in zip(trials, dense_vectors, sparse_vectors, strict=True)
        ],
        wait=True,
    )


async def create_scale_collection(
    client: AsyncQdrantClient,
    name: str,
    settings: QdrantSettings,
    *,
    size: int = 1200,
    seed: int = 20261109,
) -> int:
    """Build a collection of ``size`` random points with a 10%-selective filter.

    Payloads are assigned so that exactly ``round(size * SCALE_SELECTIVITY)``
    points pass an age-:data:`SCALE_FILTER_AGE` filter: the rest carry
    ``min_age_years = 65``, which is *positive evidence* that a 45-year-old is
    ineligible. No point carries a ``None`` age, deliberately - NULL-safety is
    the semantic fixture's job, and a NULL here would survive the filter and
    muddy the selectivity this fixture's whole purpose is to pin down.

    Returns:
        The number of points that pass the filter, so the test asserts against a
        computed value rather than a second copy of the arithmetic.
    """
    await _recreate(client, name, settings, force_hnsw=True)
    rng = random.Random(seed)
    eligible = round(size * SCALE_SELECTIVITY)

    points: list[models.PointStruct] = []
    for index in range(size):
        nct_id = f"NCTSCALE{index:06d}"
        # Interleave rather than block the eligible ones, so they are not
        # clustered in insertion order - segment layout should not be able to
        # flatter the result.
        is_eligible = index % (size // eligible) == 0
        trial = FixtureTrial(
            nct_id=nct_id,
            text="",
            min_age_years=18.0 if is_eligible else 65.0,
            max_age_years=100.0,
            sex="all",
        )
        points.append(
            models.PointStruct(
                id=point_id(nct_id),
                vector={
                    # Cosine: Qdrant normalises on insert, so an un-normalised
                    # Gaussian draw is fine and keeps the fixture dependency-free.
                    settings.dense_vector_name: [
                        rng.gauss(0.0, 1.0) for _ in range(settings.dense_dim)
                    ],
                    # A handful of token slots with random weights: enough for
                    # the sparse branch to return a ranking, meaningless as
                    # relevance, which is the point.
                    settings.sparse_vector_name: models.SparseVector(
                        indices=sorted(rng.sample(range(5_000), 12)),
                        values=[rng.uniform(0.2, 1.0) for _ in range(12)],
                    ),
                },
                payload=trial.payload(),
            )
        )

    # Batched: one 1,200-point upsert of 384-float vectors is a ~2 MB request
    # body, which Qdrant accepts but which makes a failure harder to localise.
    for start in range(0, len(points), 200):
        await client.upsert(collection_name=name, points=points[start : start + 200], wait=True)

    return sum(1 for point in points if (point.payload or {})["min_age_years"] == 18.0)


# --------------------------------------------------------------------------
# Test doubles. The unit suite must pass with no containers and no ONNX
# weights, so everything the service talks to has a stub here.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StubDenseEncoder:
    """A :class:`~retrieval.encoders.DenseQueryEncoder` that embeds nothing.

    The vector's *content* is irrelevant to every unit test: the stub Qdrant
    client below returns canned rankings regardless of what it is asked, so what
    is under test is the fusion arithmetic and the request shape, not
    similarity. Returning a fixed vector of the right length keeps the request
    valid without loading 130 MB of weights.
    """

    dim: int = 384

    def encode_query(self, text: str) -> list[float]:
        # Not all zeros: a zero vector is degenerate for cosine, and a stub that
        # happens to be degenerate invites a confusing failure if one ever
        # reaches a real Qdrant.
        return [0.1] * self.dim


@dataclass(frozen=True, slots=True)
class StubSparseEncoder:
    """A :class:`~retrieval.encoders.SparseQueryEncoder` with a fixed vector."""

    def encode_query(self, text: str) -> models.SparseVector:
        return models.SparseVector(indices=[1, 2, 3], values=[1.0, 1.0, 1.0])


@dataclass
class StubCrossEncoder:
    """A :class:`~retrieval.encoders.PairScorer` driven by a lookup table.

    ``scores`` maps a substring of the document to the score to return, so a
    test can say "the trial mentioning melanoma must rerank to the top" without
    a model. Documents matching nothing score ``default``.
    """

    scores: Mapping[str, float]
    default: float = 0.0
    calls: list[tuple[str, int]] = field(default_factory=list)

    def score(self, query: str, documents: list[str]) -> list[float]:
        self.calls.append((query, len(documents)))
        out: list[float] = []
        for document in documents:
            out.append(
                next((v for k, v in self.scores.items() if k in document), self.default)
            )
        return out


class StubQdrantClient:
    """Returns canned ranked lists, one per request, in request order.

    It is positional on purpose. :class:`~retrieval.search.HybridSearcher`
    builds ``2N`` requests as dense, sparse, dense, sparse... and pairs the
    responses back up by index; a stub keyed on anything else could not catch a
    regression that transposed the two branches, which is exactly the kind of
    bug that produces plausible-looking wrong rankings.

    ``requests`` records what it was asked, so a test can assert the filter was
    attached to *both* branches of *every* query.
    """

    def __init__(
        self,
        branches: Sequence[Sequence[str]] = (),
        payloads: Mapping[str, dict[str, Any]] | None = None,
        *,
        info: models.CollectionInfo | None = None,
        exists: bool = True,
    ) -> None:
        self.branches = [list(branch) for branch in branches]
        self.payloads = dict(payloads or {})
        self.requests: list[models.QueryRequest] = []
        self.info = info
        self.exists = exists

    async def collection_exists(self, collection_name: str, **kwargs: Any) -> bool:
        return self.exists

    async def get_collection(
        self, collection_name: str, **kwargs: Any
    ) -> models.CollectionInfo:
        if self.info is None:
            raise AssertionError("StubQdrantClient was not given a CollectionInfo")
        return self.info

    async def query_batch_points(
        self,
        collection_name: str,
        requests: Sequence[models.QueryRequest],
        **kwargs: Any,
    ) -> list[QueryResponse]:
        self.requests.extend(requests)
        if len(requests) != len(self.branches):
            raise AssertionError(
                f"stub has {len(self.branches)} canned branches but was sent "
                f"{len(requests)} requests"
            )
        return [
            QueryResponse(
                points=[
                    models.ScoredPoint(
                        id=point_id(nct_id),
                        version=0,
                        # A descending synthetic score. RRF must not read it -
                        # the two branches are deliberately on different scales
                        # in the tests, so a score-based fusion would give a
                        # different answer than the rank-based one asserted.
                        score=float(len(branch) - position),
                        payload=self.payloads.get(nct_id, {"nct_id": nct_id}),
                    )
                    for position, nct_id in enumerate(branch)
                ]
            )
            for branch in self.branches
        ]


def stub_collection_info(
    *,
    dense_dim: int | None = 384,
    dense_name: str = "dense",
    sparse_name: str | None = "sparse",
    sparse_modifier: models.Modifier | None = models.Modifier.IDF,
    payload_fields: Sequence[str] = ("min_age_years", "max_age_years", "sex", "nct_id"),
    points: int = 375_580,
    unnamed_vector: bool = False,
) -> models.CollectionInfo:
    """A ``CollectionInfo`` describing a collection that may be misconfigured.

    ``model_construct`` skips validation, which is what makes this possible at
    all: a real ``CollectionInfo`` demands a full ``HnswConfig`` and
    ``OptimizersConfig`` that the verifier never reads, and inventing them would
    put 30 lines of irrelevant defaults between the test and what it is testing.
    """
    vectors: models.VectorsConfig | None
    if unnamed_vector:
        vectors = models.VectorParams(
            size=dense_dim or 384, distance=models.Distance.COSINE
        )
    elif dense_dim is None:
        vectors = {}
    else:
        vectors = {
            dense_name: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        }

    sparse_vectors = (
        None
        if sparse_name is None
        else {sparse_name: models.SparseVectorParams(modifier=sparse_modifier)}
    )
    params = models.CollectionParams(vectors=vectors, sparse_vectors=sparse_vectors)
    return models.CollectionInfo.model_construct(
        status=models.CollectionStatus.GREEN,
        optimizer_status=models.OptimizersStatusOneOf.OK,
        points_count=points,
        segments_count=1,
        config=models.CollectionConfig.model_construct(params=params),
        payload_schema={
            name: models.PayloadIndexInfo.model_construct(
                data_type=models.PayloadSchemaType.FLOAT, points=points
            )
            for name in payload_fields
        },
    )


# --------------------------------------------------------------------------
# model availability, so integration tests skip instead of downloading 400 MB
# --------------------------------------------------------------------------


def missing_models(settings: QdrantSettings) -> list[str]:
    """Names of the configured models whose weights are not already cached.

    CI runs ``pytest -m integration`` against a live compose stack, on a fresh
    runner with a cold FastEmbed cache. A test that constructed a real encoder
    there would pull ~400 MB of ONNX weights - ``bge-reranker-base`` alone is
    ~280 MB - and either blow the job's timeout or quietly turn the job into a
    download benchmark. So the model-dependent tests consult this and skip,
    while the tests that need only Qdrant - including the overfiltering test,
    which uses random vectors precisely so that it needs no model - run for
    real.

    The cache directory is inspected rather than the model being loaded,
    because attempting the load *is* the download.
    """
    wanted: list[tuple[str, Sequence[BaseModelDescription]]] = [
        (settings.dense_model, TextEmbedding._list_supported_models()),
        (settings.sparse_model, SparseTextEmbedding._list_supported_models()),
        (settings.reranker_model, TextCrossEncoder._list_supported_models()),
    ]
    cache = define_cache_dir()
    absent: list[str] = []
    for name, supported in wanted:
        match = next((d for d in supported if d.model.lower() == name.lower()), None)
        if match is None:
            absent.append(f"{name} (not a FastEmbed model)")
            continue
        repo = match.sources.hf
        # huggingface_hub names a snapshot folder models--<org>--<repo>.
        if repo is None or not (cache / f"models--{repo.replace('/', '--')}").is_dir():
            absent.append(name)
    return absent
