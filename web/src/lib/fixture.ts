/**
 * A committed, realistic fixture run.
 *
 * The retrieval service and the match worker are still being built, so no real
 * end-to-end run exists yet. This fixture is shaped exactly like what the
 * gateway will emit (`beacon_core.events.MatchCompleted.results` plus the
 * numbered note and profile the worker produces), which makes the console
 * demonstrable today and switchable to live data with no rewrite.
 *
 * Properties it deliberately has, because they are what the UI must prove:
 *   - every `evidence_sentence_ids` entry resolves to a real `sentence_id`;
 *   - all four `CriterionLabel` values appear;
 *   - `insufficient_info` appears both with and without a cited sentence;
 *   - one rule cites two sentences, so multi-sentence highlighting is visible;
 *   - three of the four `EligibilityCategory` values appear.
 *
 * The patient is synthetic. It is not a real person and never was.
 */

import type {
  MatchResultBundle,
  MatchProgressEvent,
  NumberedNote,
  PatientProfile,
  RankedTrial,
} from "./types";

export const FIXTURE_MATCH_ID = "fixture-9f1c2d6a-demo";

/** The raw note, exactly as it would be pasted into the box. */
export const FIXTURE_NOTE_TEXT = [
  "A 67-year-old woman presents with a three-month history of progressive dyspnea and a 10-pound unintentional weight loss.",
  "CT of the chest shows a 4.2 cm right upper lobe mass with ipsilateral hilar lymphadenopathy.",
  "Biopsy confirms invasive adenocarcinoma of the lung.",
  "Next-generation sequencing on the biopsy specimen reports an EGFR exon 19 deletion.",
  "PD-L1 tumor proportion score is 15 percent.",
  "She completed four cycles of carboplatin and pemetrexed eighteen months ago for stage IIIA disease.",
  "She has never received an EGFR tyrosine kinase inhibitor.",
  "Her ECOG performance status is 1.",
  "Serum creatinine is 0.9 mg/dL and total bilirubin is 0.6 mg/dL.",
  "She has a remote history of treated hepatitis B and no known brain metastases.",
].join(" ");

/**
 * The note as the worker numbers it. Sentence IDs are 1-based and are the only
 * thing a `CriterionJudgment` may cite.
 */
export const FIXTURE_NOTE: NumberedNote = {
  sentences: [
    {
      sentence_id: 1,
      text: "A 67-year-old woman presents with a three-month history of progressive dyspnea and a 10-pound unintentional weight loss.",
    },
    {
      sentence_id: 2,
      text: "CT of the chest shows a 4.2 cm right upper lobe mass with ipsilateral hilar lymphadenopathy.",
    },
    {
      sentence_id: 3,
      text: "Biopsy confirms invasive adenocarcinoma of the lung.",
    },
    {
      sentence_id: 4,
      text: "Next-generation sequencing on the biopsy specimen reports an EGFR exon 19 deletion.",
    },
    { sentence_id: 5, text: "PD-L1 tumor proportion score is 15 percent." },
    {
      sentence_id: 6,
      text: "She completed four cycles of carboplatin and pemetrexed eighteen months ago for stage IIIA disease.",
    },
    {
      sentence_id: 7,
      text: "She has never received an EGFR tyrosine kinase inhibitor.",
    },
    { sentence_id: 8, text: "Her ECOG performance status is 1." },
    {
      sentence_id: 9,
      text: "Serum creatinine is 0.9 mg/dL and total bilirubin is 0.6 mg/dL.",
    },
    {
      sentence_id: 10,
      text: "She has a remote history of treated hepatitis B and no known brain metastases.",
    },
  ],
};

export const FIXTURE_PROFILE: PatientProfile = {
  age: 67,
  sex: "female",
  conditions: [
    "Non-small cell lung cancer",
    "Lung adenocarcinoma, previously treated stage IIIA",
  ],
  biomarkers: ["EGFR exon 19 deletion", "PD-L1 TPS 15%"],
  prior_treatments: ["Carboplatin", "Pemetrexed"],
  labs: {
    "Serum creatinine": "0.9 mg/dL",
    "Total bilirubin": "0.6 mg/dL",
  },
  performance_status: 1,
  location: null,
};

const TRIALS: RankedTrial[] = [
  {
    rank: 1,
    score: 0.912,
    candidate: {
      nct_id: "NCT05123456",
      title:
        "Osimertinib With or Without Platinum-Pemetrexed in EGFR-Mutated Advanced Non-Small Cell Lung Cancer",
      conditions: ["Non-Small Cell Lung Carcinoma", "EGFR Gene Mutation"],
      interventions: ["Osimertinib", "Carboplatin", "Pemetrexed"],
      phase: "Phase 3",
      status: "Recruiting",
      min_age_years: 18,
      max_age_years: null,
      sex: "all",
      dense_score: 0.834,
      sparse_score: 12.41,
      rrf_score: 0.0317,
      rerank_score: 0.912,
      retrieval_rank: 1,
    },
    judgment: {
      nct_id: "NCT05123456",
      category: "likely_eligible",
      confirmed_fraction: 0.667,
      missing_facts: [
        "Whether disease is measurable by RECIST 1.1",
        "Date definitive radiotherapy was completed",
      ],
      summary:
        "This trial tests an EGFR-targeted pill, alone or with chemotherapy, in people whose lung cancer carries an EGFR mutation. The note confirms the diagnosis, the EGFR exon 19 deletion and a performance status the trial allows, and nothing in the note rules the patient out. Two details are still missing before eligibility could be settled.",
      judgments: [
        {
          criterion_id: "INC-1",
          kind: "inclusion",
          label: "met",
          evidence_sentence_ids: [3],
          explanation:
            "Histologically confirmed non-small cell lung cancer is required. The biopsy result in the note establishes invasive adenocarcinoma of the lung.",
        },
        {
          criterion_id: "INC-2",
          kind: "inclusion",
          label: "met",
          evidence_sentence_ids: [4],
          explanation:
            "An activating EGFR mutation is required. The sequencing result reports an exon 19 deletion, which is one of the listed activating mutations.",
        },
        {
          criterion_id: "INC-3",
          kind: "inclusion",
          label: "met",
          evidence_sentence_ids: [8],
          explanation:
            "ECOG performance status 0 or 1 is required. The note records ECOG 1.",
        },
        {
          criterion_id: "INC-4",
          kind: "inclusion",
          label: "insufficient_info",
          evidence_sentence_ids: [],
          explanation:
            "At least one lesion measurable by RECIST 1.1 is required. The note describes a 4.2 cm mass but does not say it was assessed as measurable under RECIST 1.1, so this rule cannot be settled from the note.",
        },
        {
          criterion_id: "INC-5",
          kind: "inclusion",
          label: "insufficient_info",
          evidence_sentence_ids: [6],
          explanation:
            "A four-week washout since the last anticancer therapy is required. The note dates the last chemotherapy at eighteen months ago, but gives no date for the end of radiotherapy, so the washout cannot be confirmed.",
        },
        {
          criterion_id: "EXC-1",
          kind: "exclusion",
          label: "not_met",
          evidence_sentence_ids: [7],
          explanation:
            "Prior treatment with an EGFR tyrosine kinase inhibitor is an exclusion. The note states the patient has never received one, so the exclusion does not fire.",
        },
        {
          criterion_id: "EXC-2",
          kind: "exclusion",
          label: "not_applicable",
          evidence_sentence_ids: [1],
          explanation:
            "This exclusion concerns participants who are pregnant or breastfeeding at screening and applies only to participants of childbearing potential. The note describes a 67-year-old woman, so the criterion does not apply.",
        },
      ],
    },
  },
  {
    rank: 2,
    score: 0.617,
    candidate: {
      nct_id: "NCT04987654",
      title:
        "Pembrolizumab Plus Platinum Doublet Chemotherapy in PD-L1-Positive Advanced Non-Small Cell Lung Cancer",
      conditions: ["Non-Small Cell Lung Carcinoma", "PD-L1 Positive"],
      interventions: ["Pembrolizumab", "Carboplatin", "Pemetrexed"],
      phase: "Phase 2",
      status: "Recruiting",
      min_age_years: 18,
      max_age_years: 85,
      sex: "all",
      dense_score: 0.771,
      sparse_score: 9.07,
      rrf_score: 0.0294,
      rerank_score: 0.617,
      retrieval_rank: 3,
    },
    judgment: {
      nct_id: "NCT04987654",
      category: "insufficient_info",
      confirmed_fraction: 0.4,
      missing_facts: [
        "Current disease stage (stage IV versus recurrent)",
        "Hepatitis B virus DNA viral load",
        "Hepatitis C antibody status",
      ],
      summary:
        "This trial adds an immunotherapy drug to standard chemotherapy for people whose tumour expresses PD-L1. The note's PD-L1 score clears the trial's threshold, but two things the trial needs to know are absent from the note: the current stage of disease, and whether the treated hepatitis B is still active. The system is not guessing at either, so this trial stays undecided rather than being ranked up or dropped.",
      judgments: [
        {
          criterion_id: "INC-1",
          kind: "inclusion",
          label: "met",
          evidence_sentence_ids: [3],
          explanation:
            "Histologically confirmed non-squamous non-small cell lung cancer is required, and the biopsy in the note reports adenocarcinoma.",
        },
        {
          criterion_id: "INC-2",
          kind: "inclusion",
          label: "insufficient_info",
          evidence_sentence_ids: [],
          explanation:
            "Stage IV or recurrent disease is required. The note's only staging statement is historical, and no current stage is given, so this rule cannot be settled.",
        },
        {
          criterion_id: "INC-3",
          kind: "inclusion",
          label: "met",
          evidence_sentence_ids: [5],
          explanation:
            "A PD-L1 tumour proportion score of at least 1 percent is required. The note reports 15 percent.",
        },
        {
          criterion_id: "INC-4",
          kind: "inclusion",
          label: "met",
          evidence_sentence_ids: [9],
          explanation:
            "Adequate renal and hepatic function is required. Creatinine of 0.9 mg/dL and total bilirubin of 0.6 mg/dL are both inside the stated limits.",
        },
        {
          criterion_id: "EXC-1",
          kind: "exclusion",
          label: "insufficient_info",
          evidence_sentence_ids: [10],
          explanation:
            "Active hepatitis B is an exclusion. The note records a remote treated hepatitis B history but no HBV DNA result, so whether the infection is active cannot be determined from the note.",
        },
        {
          criterion_id: "EXC-2",
          kind: "exclusion",
          label: "not_met",
          evidence_sentence_ids: [10],
          explanation:
            "Untreated central nervous system metastases are an exclusion. The note states there are no known brain metastases.",
        },
      ],
    },
  },
  {
    rank: 3,
    score: 0.208,
    candidate: {
      nct_id: "NCT03456789",
      title:
        "Consolidation Durvalumab After Definitive Chemoradiotherapy in Unresectable Stage III Non-Small Cell Lung Cancer",
      conditions: ["Non-Small Cell Lung Carcinoma", "Stage III"],
      interventions: ["Durvalumab"],
      phase: "Phase 3",
      status: "Active, not recruiting",
      min_age_years: 18,
      max_age_years: null,
      sex: "all",
      dense_score: 0.688,
      sparse_score: 7.52,
      rrf_score: 0.0241,
      rerank_score: 0.208,
      retrieval_rank: 7,
    },
    judgment: {
      nct_id: "NCT03456789",
      category: "excluded",
      confirmed_fraction: 0.5,
      missing_facts: [],
      summary:
        "This trial is for people who have just finished chemoradiotherapy for stage III disease that cannot be removed by surgery. The note describes someone who finished that kind of treatment eighteen months ago and has since progressed, and the trial excludes anyone who has already had platinum chemotherapy for this cancer. Two rules point the same way, so the trial is ruled out.",
      judgments: [
        {
          criterion_id: "INC-1",
          kind: "inclusion",
          label: "met",
          evidence_sentence_ids: [3],
          explanation:
            "Histologically confirmed non-small cell lung cancer is required, and the biopsy in the note confirms it.",
        },
        {
          criterion_id: "INC-2",
          kind: "inclusion",
          label: "not_met",
          evidence_sentence_ids: [2, 6],
          explanation:
            "Unresectable stage III disease that has just completed definitive concurrent chemoradiotherapy is required. The note describes a new 4.2 cm mass after chemotherapy completed eighteen months ago, which is not the required clinical situation.",
        },
        {
          criterion_id: "EXC-1",
          kind: "exclusion",
          label: "met",
          evidence_sentence_ids: [6],
          explanation:
            "Any prior platinum-based chemotherapy for this malignancy is an exclusion. The note documents four cycles of carboplatin and pemetrexed.",
        },
        {
          criterion_id: "EXC-2",
          kind: "exclusion",
          label: "not_applicable",
          evidence_sentence_ids: [],
          explanation:
            "This exclusion concerns prior solid organ transplantation, which the trial assesses only for participants on chronic immunosuppression. Nothing in the note places the patient in that group, so the criterion does not apply.",
        },
      ],
    },
  },
];

export const FIXTURE_BUNDLE: MatchResultBundle = {
  match_id: FIXTURE_MATCH_ID,
  status: "completed",
  results: TRIALS,
  note: FIXTURE_NOTE,
  profile: FIXTURE_PROFILE,
  total_cost_usd: 0.0412,
  duration_ms: 18420,
};

/**
 * The progress frames a real run would publish, in order. Replayed by the demo
 * so the live-progress view is driven by the same reducer as a real stream.
 */
export const FIXTURE_PROGRESS: MatchProgressEvent[] = [
  {
    event_type: "match.progress",
    match_id: FIXTURE_MATCH_ID,
    step: "profile_patient",
    pct: 8,
    message: "Numbered the note into 10 sentences and extracted the profile.",
  },
  {
    event_type: "match.progress",
    match_id: FIXTURE_MATCH_ID,
    step: "generate_queries",
    pct: 20,
    message: "Generated 3 queries across diagnosis, biomarker and prior therapy.",
  },
  {
    event_type: "match.progress",
    match_id: FIXTURE_MATCH_ID,
    step: "retrieve",
    pct: 42,
    message: "Hybrid search returned 20 candidates after reranking.",
  },
  {
    event_type: "match.progress",
    match_id: FIXTURE_MATCH_ID,
    step: "parse_rules",
    pct: 54,
    message: "Split the eligibility text of 3 trials into 17 numbered rules.",
  },
  {
    event_type: "match.progress",
    match_id: FIXTURE_MATCH_ID,
    step: "check_rules",
    pct: 72,
    message: "Judged 17 eligibility rules against the numbered note.",
  },
  {
    event_type: "match.progress",
    match_id: FIXTURE_MATCH_ID,
    step: "verify",
    pct: 86,
    message: "All cited sentence IDs resolve to real sentences in the note.",
  },
  {
    event_type: "match.progress",
    match_id: FIXTURE_MATCH_ID,
    step: "rank_and_explain",
    pct: 100,
    message: "Ranked 3 trials and wrote plain-language summaries.",
  },
];
