"""Real messy records from the TREC corpus, frozen so tests need no corpus.

Why they are literals in a module and not read from disk
--------------------------------------------------------
The parsers and the splitter are only worth testing against records that
actually exist: a hand-invented ``"Inclusion Criteria:\\n- foo"`` proves the
splitter handles the shape its author imagined. Every record below was pulled
from ``clinicaltrials/2021`` (the April-2021 ClinicalTrials.gov snapshot, 2.6 GB
on disk) with its ``eligibility`` fields and ``criteria/textblock`` copied
verbatim, including the ``\\r\\n`` line endings and the odd ``\\u2022``.

They are frozen here as literals rather than loaded from
``IR_DATASETS_HOME`` because CI has no corpus: a test that needed the 1.8 GB
download would be marked ``integration``, get excluded from the fast run, and
then never catch the regression it exists for.

This module is named with a leading underscore and lives inside the package for
the reason the operating protocol spells out: the repo allows exactly one
``conftest.py`` (at ``tests/conftest.py``), so shared test material goes in a
uniquely-named package module -- the pattern ``beacon_llm/_testing.py``
established.

The records are copied byte for byte, so many of the literals below are far
longer than the project's 100-column limit. Reflowing them would mean editing
the data, so E501 is suppressed for this file alone -- nowhere else.
"""
# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass

from ingestion.corpus import RawTrial

__all__ = [
    "AGE_CASES",
    "REAL_RECORDS",
    "SEX_CASES",
    "RealRecord",
    "WordTokenizer",
    "as_raw_trial",
    "record",
]


@dataclass(frozen=True, slots=True)
class RealRecord:
    """One record's fields exactly as the source XML has them."""

    nct_id: str
    title: str
    conditions: tuple[str, ...]
    interventions: tuple[str, ...]
    phase: str | None
    status: str | None
    raw_gender: str | None
    raw_min_age: str | None
    raw_max_age: str | None
    criteria: str | None


def record(nct_id: str) -> RealRecord:
    """Look a frozen record up, with a useful error when the id is a typo."""
    try:
        return REAL_RECORDS[nct_id]
    except KeyError:
        raise KeyError(
            f"{nct_id} is not one of the frozen records: {sorted(REAL_RECORDS)}"
        ) from None


def as_raw_trial(nct_id: str, *, brief_summary: str | None = None) -> RawTrial:
    """A frozen record as the :class:`~ingestion.corpus.RawTrial` the pipeline sees."""
    fixture = record(nct_id)
    return RawTrial(
        nct_id=fixture.nct_id,
        title=fixture.title,
        conditions=fixture.conditions,
        interventions=fixture.interventions,
        phase=fixture.phase,
        status=fixture.status,
        brief_summary=brief_summary,
        detailed_description=None,
        eligibility_text=fixture.criteria,
        raw_gender=fixture.raw_gender,
        raw_min_age=fixture.raw_min_age,
        raw_max_age=fixture.raw_max_age,
    )


AGE_CASES: tuple[tuple[str | None, float | None], ...] = (
    # (raw value as the corpus writes it, expected years or None)
    ("18 Years", 18.0),
    ("1 Year", 1.0),
    ("6 Months", 0.5),
    ("2 Months", round(2 / 12, 6)),
    ("20 Weeks", round(20 * 7 / 365.25, 6)),
    ("90 Days", round(90 / 365.25, 6)),
    ("24 Hours", round(24 / (365.25 * 24), 6)),
    ("30 Minutes", round(30 / (365.25 * 24 * 60), 6)),
    ("N/A", None),
    (None, None),
    # Shapes the corpus does not contain but a future snapshot might. Each must
    # yield None rather than a plausible-looking number.
    ("", None),
    ("unknown", None),
    ("65 Years and older", None),
    ("18-65 Years", None),
    ("at least 18 Years", None),
    ("adult", None),
    ("18", None),
    ("eighteen years", None),
)
"""Age strings and the only acceptable answers.

The 2026-11-08 full-corpus scan found every one of the 749,434 populated
``minimum_age``/``maximum_age`` values to be either ``N/A`` or
``<number> <unit>``, with units years/months/weeks/days/hours/minutes. The
trailing cases are the near-misses that must fail closed to ``None``:
``"65 Years and older"`` is the one the brief calls out, and a loose parser
would read it as a 65-year *maximum*.
"""

SEX_CASES: tuple[tuple[str | None, str | None], ...] = (
    ("All", "all"),
    ("Female", "female"),
    ("Male", "male"),
    ("all", "all"),
    ("Both", "all"),
    (None, None),
    ("", None),
    ("Other", None),
)
"""``gender`` values and their mapping to the ``trials.sex`` check constraint."""


class WordTokenizer:
    """A whitespace tokenizer shaped like ``tokenizers.Tokenizer``.

    Lets the overview truncation logic be tested without loading a 130 MB ONNX
    model: it adds two special tokens and reports character offsets, which is
    the whole interface :class:`~ingestion.overview.OverviewBuilder` uses.
    """

    class _Encoding:
        def __init__(self, ids: list[int], offsets: list[tuple[int, int]]) -> None:
            self.ids = ids
            self.offsets = offsets

    def encode(self, text: str) -> WordTokenizer._Encoding:
        offsets: list[tuple[int, int]] = [(0, 0)]
        ids: list[int] = [101]
        position = 0
        for word in text.split():
            start = text.index(word, position)
            end = start + len(word)
            offsets.append((start, end))
            ids.append(len(ids))
            position = end
        offsets.append((0, 0))
        ids.append(102)
        return WordTokenizer._Encoding(ids, offsets)


REAL_RECORDS: dict[str, RealRecord] = {
    # dash bullets under both headers, headers inline-colon - the common 80% shape
    "NCT00000102": RealRecord(
        nct_id="NCT00000102",
        title="Congenital Adrenal Hyperplasia: Calcium Channels as Therapeutic Targets",
        conditions=("Congenital Adrenal Hyperplasia",),
        interventions=("Nifedipine",),
        phase="Phase 1/Phase 2",
        status="Completed",
        raw_gender="All",
        raw_min_age="14 Years",
        raw_max_age="35 Years",
        criteria="\n        Inclusion Criteria:\r\n\r\n          -  diagnosed with Congenital Adrenal Hyperplasia (CAH)\r\n\r\n          -  normal ECG during baseline evaluation\r\n\r\n        Exclusion Criteria:\r\n\r\n          -  history of liver disease, or elevated liver function tests\r\n\r\n          -  history of cardiovascular disease\r\n      ",
    ),
    # dash bullets, long inclusion and exclusion lists
    "NCT00000105": RealRecord(
        nct_id="NCT00000105",
        title="Vaccination With Tetanus and KLH to Assess Immune Responses.",
        conditions=("Cancer",),
        interventions=("Intracel KLH Vaccine", "Biosyn KLH", "Montanide ISA51", "Tetanus toxoid"),
        phase=None,
        status="Terminated",
        raw_gender="All",
        raw_min_age="18 Years",
        raw_max_age="N/A",
        criteria="\n        Inclusion Criteria:\r\n\r\n          -  Patients must have a diagnosis of cancer of any histologic type.\r\n\r\n          -  Patients must have a Karnofsky performance status great or equal to 70%.\r\n\r\n          -  Patients must have an expected survival for at least four months.\r\n\r\n          -  Normal healthy volunteers to serve as control for this study.\r\n\r\n          -  All patients must sign informed consent approved by the Committee on the Use of Human\r\n             Subjects at the University of Minnesota\r\n\r\n        Exclusion Criteria:\r\n\r\n          -  Pregnant or lactating women. Females of child-bearing potential will be asked to take\r\n             a pregnancy test before receiving vaccines.\r\n\r\n          -  Serious intercurrent medical illnesses which would interfere with the ability of the\r\n             patient to carry out the follow-up monitoring program.\r\n\r\n          -  Immunization should not be administered during the course of any febrile illness or\r\n             acute infection.\r\n\r\n          -  Hypersensitivity to any component of the vaccine, including Thimerosal, a mercury\r\n             derivative.\r\n\r\n          -  The occurrence of any type of neurologic symptoms to tetanus vaccine in th past.\r\n\r\n          -  Patients with a history of seafood allergy are excluded from receiving KLH.\r\n\r\n          -  Subjects who have had tetanus toxoid within the last 7 years are not eligible for\r\n             tetanus vaccine component of this protocol.\r\n      ",
    ),
    # numbered list, no headers at all, continuation lines indented
    "NCT00000162": RealRecord(
        nct_id="NCT00000162",
        title="Branch Vein Occlusion Study",
        conditions=(
            "Macular Degeneration",
            "Neovascularization, Pathologic",
            "Vitreous Hemorrhage",
        ),
        interventions=(
            "Macular Argon Laser Photocoagulation",
            "Peripheral Scatter Argon Laser Photocoagulation",
            "Scatter Argon Laser Photocoagulation",
        ),
        phase="Phase 3",
        status="Completed",
        raw_gender="All",
        raw_min_age="N/A",
        raw_max_age="N/A",
        criteria="\n        Patients with three types of diagnoses were accepted:\r\n\r\n          1. major BVO without neovascularization;\r\n\r\n          2. major BVO with neovascularization;\r\n\r\n          3. BVO with macular edema and reduced vision. All patients must have had onset of signs\r\n             and/or symptoms of BVO less than 18 months before the initial visit, vision of 5/200\r\n             or better, and sufficient clarity of the ocular media to permit confirmation of the\r\n             condition with fundus photography. Other eligibility criteria apply to each of the\r\n             three major groups as well as special cases such as the occurrence of bilateral\r\n             disease.\r\n      ",
    ),
    # both headers, no bullet markers - one rule per paragraph
    "NCT00000244": RealRecord(
        nct_id="NCT00000244",
        title="Effects of Dynorphin 1-13 on Heroin Addiction - 1",
        conditions=("Opioid-Related Disorders", "Substance Withdrawal Syndrome"),
        interventions=("Dynorphin 1 - 13",),
        phase="Phase 2",
        status="Completed",
        raw_gender="All",
        raw_min_age="18 Years",
        raw_max_age="55 Years",
        criteria="\n        Inclusion Criteria:\r\n\r\n        opiate addict between the ages of 18-55\r\n\r\n        Exclusion Criteria:\r\n\r\n        Regular abuse of other drugs, unstable medical conditions\r\n      ",
    ),
    # 'Inclusion Criteria- Subject must:' header, a bullet mixed among dashes
    "NCT00000306": RealRecord(
        nct_id="NCT00000306",
        title="Dextroamphetamine as Adjunct in Cocaine/Opiate Dependent Patients - 3",
        conditions=("Cocaine-Related Disorders", "Opioid-Related Disorders"),
        interventions=("Dextroamphetamine",),
        phase="Phase 2",
        status="Completed",
        raw_gender="All",
        raw_min_age="18 Years",
        raw_max_age="45 Years",
        criteria="\n        Inclusion Criteria- Subject must:\r\n\r\n          -  Exhibit cocaine and opiate dependence through a DSM-IV diagnosis as determined by\r\n             SCID.\r\n\r\n          -  Be between 18 and 45 years old\r\n\r\n          -  Women must agree to use contraception\r\n\r\n          -  Have an EKG that has been confirmed by a cardiologist\r\n\r\n          -  Give a cocaine positive urine during screening - Present with evidence of opiate\r\n             withdrawal\r\n\r\n        Exclusion Criteria- Subject must not:\r\n\r\n        • Have a serious medical illness including, but not limited to the following: Hypertension\r\n        Significant heart disease Clinically significant cardiovascular abnormality Angina Hepatic,\r\n        renal, or gastrointestinal disorders that could result in an altered metabolism or\r\n        excretion of study agent\r\n\r\n          -  Have any Axis I disorder that is not related to drug use\r\n\r\n          -  Have current dependence on any psychoactive disorder other than nicotine\r\n\r\n          -  Be on probation or parole for reasons other than those related to drug charges (ASI)\r\n\r\n          -  Be pregnant or lactating\r\n\r\n          -  Have been in any outside treatment in 3 months\r\n      ",
    ),
    # no headers, single dash bullet wrapped over two lines
    "NCT00000389": RealRecord(
        nct_id="NCT00000389",
        title="Treatment for Anxiety in Children",
        conditions=(
            "Obsessive-Compulsive Disorder",
            "Anxiety Disorders",
            "Generalized Anxiety Disorder",
            "Social Phobia",
            "Separation Anxiety",
        ),
        interventions=("Fluvoxamine",),
        phase="Phase 3",
        status="Completed",
        raw_gender="All",
        raw_min_age="6 Years",
        raw_max_age="17 Years",
        criteria="\n        - Patients must have: DSM -IV diagnosis of generalized anxiety disorder, social phobia, or\r\n        separation anxiety.\r\n      ",
    ),
    # the header is itself a bullet; items nest three levels deep
    "NCT00001231": RealRecord(
        nct_id="NCT00001231",
        title="Perimenopause-Related Mood and Behavioral Disorders",
        conditions=("Mood Disorder",),
        interventions=(),
        phase=None,
        status="Recruiting",
        raw_gender="Female",
        raw_min_age="40 Years",
        raw_max_age="65 Years",
        criteria="\n        -  INCLUSION CRITERIA:\r\n\r\n          -  Perimenopausal Subjects:\r\n\r\n               -  History within the last one year of at least one month with perimenopause-related\r\n                  mood or behavioral disturbances of at least moderate severity - that is,\r\n                  disturbances that are distinct in appearance and associated with a notable degree\r\n                  of functional impairment;\r\n\r\n               -  Age 40-65;\r\n\r\n               -  History of the onset of menstrual irregularity during the past six months but not\r\n                  greater than one year of amenorrhea (i.e., not postmenopausal);\r\n\r\n               -  Biological evidence of a deterioration of normal ovarian activity, specifically,\r\n                  plasma FSH levels persistently elevated (> 14IU/L) (as recommended by the Stages\r\n                  of Reproductive Aging Workshop criteria (106)) drawn at two week intervals over a\r\n                  period of eight weeks;\r\n\r\n               -  No prior estrogen replacement therapy for treatment of perimenopausal physical or\r\n                  emotional symptoms within the last six months; and\r\n\r\n               -  In good medical health;\r\n\r\n               -  Subjects must be competent to comprehend the purpose of the screening process and\r\n                  to provide written informed consent and be willing to participate in NIMH IRB\r\n                  approved research protocols.\r\n\r\n          -  A control group of age-matched perimenopausal women who meet patient selection\r\n             criteria with the exception of the presence of mood or behavioral disorders will also\r\n             be sought.\r\n\r\n          -  Premenopausal women will be women who meet the following inclusion criteria for the\r\n             longitudinal study participants:\r\n\r\n               -  Regular menstrual cycle function (21-34 days);\r\n\r\n               -  Absence of current mood or behavioral disturbance as determined by a structured\r\n                  diagnostic interview;\r\n\r\n               -  Plasma gonadotropin levels in pre-perimenopausal range (<14 IU/L);\r\n\r\n               -  In good medical health; and\r\n\r\n               -  Medication free.\r\n\r\n        EXCLUSION CRITERIA:\r\n\r\n          -  Primary general medical illness (i.e., appears to be causing the mood disorder);\r\n\r\n          -  Current antidepressant therapy (since this is a screening protocol for subsequent\r\n             treatment studies in which participants must be untreated). Antidepressants will not\r\n             be withheld from participants who need or want them;\r\n\r\n          -  Current alcohol or substance abuse or dependence (excluding nicotine) of sufficient\r\n             magnitude to require independent, concurrent treatment intervention (e.g., antabuse or\r\n             opiate treatment, but not including self-help groups);\r\n\r\n          -  Pregnant or lactating women;\r\n\r\n          -  Subjects who are unable to provide informed consent.\r\n\r\n          -  NIMH employees, staff and their immediate family members will be excluded from the\r\n             study per NIMH policy.\r\n      ",
    ),
    # the entire criteria text is 'Please contact site for information.'
    "NCT00000187": RealRecord(
        nct_id="NCT00000187",
        title="Ritanserin in Treatment of Cocaine Dependence - 1",
        conditions=("Cocaine-Related Disorders",),
        interventions=("Ritanserin",),
        phase="Phase 2",
        status="Completed",
        raw_gender="Male",
        raw_min_age="28 Years",
        raw_max_age="47 Years",
        criteria="\n        Please contact site for information.\r\n      ",
    ),
    # min and max age both N/A; female-only
    "NCT00000104": RealRecord(
        nct_id="NCT00000104",
        title="Does Lead Burden Alter Neuropsychological Development?",
        conditions=("Lead Poisoning",),
        interventions=("ERP measures of attention and memory",),
        phase=None,
        status="Completed",
        raw_gender="Female",
        raw_min_age="N/A",
        raw_max_age="N/A",
        criteria="\n        Inclusion Criteria:\r\n\r\n          -  Pregnant mothers of the Phillips neighborhood in Minneapolis, Minnesota. Subject\r\n             recruitment will take place in local clinics which serve pregnant women and offspring\r\n      ",
    ),
    # paediatric: '2 Months' to '1 Year'
    "NCT00000363": RealRecord(
        nct_id="NCT00000363",
        title="Acute Otitis Media: Adjuvant Therapy to Improve Outcome",
        conditions=("Otitis Media",),
        interventions=("Antihistamine", "Corticosteroid"),
        phase="Phase 3",
        status="Completed",
        raw_gender="All",
        raw_min_age="2 Months",
        raw_max_age="1 Year",
        criteria="\n        Inclusion Criteria:\r\n\r\n          -  Have had two documented prior episodes of acute otitis media.\r\n\r\n          -  Have no current middle ear effusion (fluid).\r\n\r\n        Exclusion Criteria:\r\n\r\n          -  Have chronic otitis media.\r\n\r\n          -  Have acute otitis media in addition to chronic otitis media.\r\n      ",
    ),
    # max age in weeks, no min age; free prose with no markers
    "NCT00000163": RealRecord(
        nct_id="NCT00000163",
        title="Congenital Esotropia Observational Study (CEOS)",
        conditions=("Esotropia",),
        interventions=(),
        phase=None,
        status="Completed",
        raw_gender="All",
        raw_min_age="N/A",
        raw_max_age="20 Weeks",
        criteria="\n        Neurologically and developmentally normal infants with congenital esotropia who are between\r\n        4 and 20 weeks of age are eligible for CEOS.\r\n      ",
    ),
    # no <eligibility> element at all - every eligibility field absent
    "NCT00026039": RealRecord(
        nct_id="NCT00026039",
        title="Women's Estrogen for Stroke Trial (WEST)",
        conditions=("Stroke",),
        interventions=("estrogen",),
        phase="Phase 3",
        status="Completed",
        raw_gender=None,
        raw_min_age=None,
        raw_max_age=None,
        criteria=None,
    ),
    # <eligibility> present but no <criteria> element
    "NCT00000180": RealRecord(
        nct_id="NCT00000180",
        title="AIT-082 Phase 1B Study",
        conditions=("Memory Disorders",),
        interventions=("AIT-082",),
        phase="Phase 1",
        status="Completed",
        raw_gender="All",
        raw_min_age="N/A",
        raw_max_age="N/A",
        criteria=None,
    ),
    # lowercase headers with the first rule on the same line
    "NCT00000134": RealRecord(
        nct_id="NCT00000134",
        title="Studies of the Ocular Complications of AIDS (SOCA)--Cytomegalovirus Retinitis Retreatment Trial (CRRT)",
        conditions=(
            "HIV Infections",
            "Acquired Immunodeficiency Syndrome",
            "Cytomegalovirus Retinitis",
        ),
        interventions=("Ganciclovir", "Foscarnet"),
        phase="Phase 3",
        status="Completed",
        raw_gender="All",
        raw_min_age="18 Years",
        raw_max_age="N/A",
        criteria="\n        inclusion criteria: Males and females eligible for the CRRT must have been age 18 years or\r\n        older and have had AIDS and CMV retinitis. They must have had active CMV despite a minimum\r\n        of 28 days of previous treatment with an anti-CMV drug. Furthermore, they must have had an\r\n        absolute neutrophil count greater than or equal to 500 cells/µL, platelet count greater\r\n        than or equal to 20,000 cells/µL, and a serum creatinine < 2.5 mg/dL in order to tolerate\r\n        the drug regimens.\r\n\r\n        exclusion criteria: history of intolerance to ganciclovir or foscarnet, history of therapy\r\n        involving the combination of foscarnet and ganciclovir, unwillingness to practice\r\n        appropriate birth control, active drug or alcohol abuse, media opacity, retinal detachment\r\n        not scheduled for surgical repair\r\n      ",
    ),
    # bullets interleaved with unbulleted continuation paragraphs
    "NCT00002558": RealRecord(
        nct_id="NCT00002558",
        title="Combination Chemotherapy Plus Peripheral Stem Cell Transplantation in Treating Patients With Germ Cell Tumors",
        conditions=("Extragonadal Germ Cell Tumor", "Ovarian Cancer", "Testicular Germ Cell Tumor"),
        interventions=(
            "filgrastim",
            "carboplatin",
            "etoposide",
            "ifosfamide",
            "paclitaxel",
            "peripheral blood stem cell transplantation",
        ),
        phase="Phase 1/Phase 2",
        status="Completed",
        raw_gender="All",
        raw_min_age="15 Years",
        raw_max_age="N/A",
        criteria="\n        Inclusion criteria:\r\n\r\n          -  Male/Female with histologically confirmed GCT with review by the Department of\r\n             Pathology at this center.\r\n\r\n          -  Patients with advanced GCT, including patients with:\r\n\r\n        measurable or evaluable disease,\r\n\r\n          -  patients with only elevated serum tumor markers (AFP and/or HCG), or\r\n\r\n          -  patients with known residual disease after postchemotherapy surgery. Eligible patients\r\n             must have established clinical resistance to cisplatin by their failure to achieve a\r\n             durable CR to a cisplatin-based regimen.\r\n\r\n          -  Prior treatment limited to ≤ 6 prior cycles (≤ four cycles preferred) of cisplatin.\r\n             (GROUP A)\r\n\r\n          -  Prior therapy > 6 cycles of cisplatin. (GROUP B)\r\n\r\n          -  Therapy must have been discontinued at least 3 weeks before entry onto protocol.\r\n\r\n          -  Patients must have one or more unfavorable prognostic factors for achieving a CR to\r\n             cisplatin-based salvage therapy. These are:\r\n\r\n          -  Extragonadal primary site.\r\n\r\n          -  Testis/ovarian primary site with the best response of an IR to first-line therapy, or\r\n             a partial response with normal tumor markers of six months or less in duration.\r\n\r\n          -  Prior treatment with ifosfamide-containing therapy\r\n\r\n          -  General medical condition sufficient to allow for general anesthesia at the time of\r\n             pheresis catheter placement.\r\n\r\n          -  Patients must have negative serology for Human Immunodeficiency Virus.\r\n\r\n          -  Laboratory criteria for protocol entry:\r\n\r\n        WBC ≥ 3000/ul Platelets 3 100,000/ul Cr Clearance > 50 cc/min*\r\n\r\n        * (unless renal dysfunction is due to tumor obstructing the ureters in which case\r\n        eligibility will be determined by the Principal Investigator).\r\n\r\n          -  Age ≥ 15 years.\r\n\r\n          -  Signed informed consent.\r\n\r\n        Exclusion Criteria:\r\n\r\n          -  Presence of active infection\r\n\r\n          -  Concurrent treatment with chemotherapy or\r\n\r\n          -  Inability to comply with the treatment protocol or to undergo the specified follow-up\r\n             tests for safety or effectiveness.\r\n\r\n          -  Prior high-dose therapy with AuBMT.\r\n\r\n          -  Patients must have recovered from recent surgery.\r\n      ",
    ),
}
