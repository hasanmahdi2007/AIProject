/**
 * TypeScript mirrors of `beacon_core.schemas` and `beacon_core.events`.
 *
 * These are transcriptions, not a second source of truth. The Python models in
 * `libs/beacon-core/src/beacon_core/schemas.py` are frozen and authoritative;
 * if one of them changes, this file is edited to follow, never the reverse.
 *
 * Nothing here computes anything. The console renders the fields the API sends.
 */

// ------------------------------------------------------------ vocabularies --

export type Sex = "male" | "female" | "all";

export type CriterionKind = "inclusion" | "exclusion";

/**
 * `insufficient_info` is a success state: it is how the system abstains instead
 * of guessing. The UI must present it as a neutral outcome, never as an error.
 */
export type CriterionLabel =
  | "met"
  | "not_met"
  | "insufficient_info"
  | "not_applicable";

export type EligibilityCategory =
  | "eligible"
  | "likely_eligible"
  | "insufficient_info"
  | "excluded";

export type MatchStatus = "queued" | "running" | "completed" | "failed";

/** The seven worker nodes, in the order the contract declares them. */
export type PipelineStep =
  | "profile_patient"
  | "generate_queries"
  | "retrieve"
  | "parse_rules"
  | "check_rules"
  | "verify"
  | "rank_and_explain";

/**
 * Declaration order of the pipeline, as the contract declares it.
 *
 * `Record<PipelineStep, string>` below is what keeps this honest: if the
 * contract gains an eighth step, `npm run build` fails on the missing label
 * rather than quietly rendering a six-step rail for a seven-step run.
 */
export const PIPELINE_STEPS: readonly PipelineStep[] = [
  "profile_patient",
  "generate_queries",
  "retrieve",
  "parse_rules",
  "check_rules",
  "verify",
  "rank_and_explain",
];

export const STEP_LABELS: Record<PipelineStep, string> = {
  profile_patient: "Profile the patient note",
  generate_queries: "Generate search queries",
  retrieve: "Retrieve trial candidates",
  parse_rules: "Parse each trial's eligibility rules",
  check_rules: "Check eligibility rules",
  verify: "Verify every citation",
  rank_and_explain: "Rank and explain",
};

// ------------------------------------------------------------ patient side --

export interface NoteSentence {
  sentence_id: number;
  text: string;
}

export interface NumberedNote {
  sentences: NoteSentence[];
}

export interface PatientProfile {
  age: number | null;
  sex: Sex | null;
  conditions: string[];
  biomarkers: string[];
  prior_treatments: string[];
  labs: Record<string, string>;
  /** ECOG 0-5. */
  performance_status: number | null;
  location: string | null;
}

// -------------------------------------------------------------- trial side --

export interface TrialCandidate {
  nct_id: string;
  title: string;
  conditions: string[];
  interventions: string[];
  phase: string | null;
  status: string | null;
  min_age_years: number | null;
  max_age_years: number | null;
  sex: Sex | null;
  dense_score: number | null;
  sparse_score: number | null;
  rrf_score: number | null;
  rerank_score: number | null;
  retrieval_rank: number | null;
}

export interface CriterionJudgment {
  criterion_id: string;
  kind: CriterionKind;
  label: CriterionLabel;
  /** Points at `NumberedNote.sentences[].sentence_id` for the same request. */
  evidence_sentence_ids: number[];
  explanation: string;
}

export interface TrialJudgment {
  nct_id: string;
  judgments: CriterionJudgment[];
  category: EligibilityCategory;
  confirmed_fraction: number;
  missing_facts: string[];
  summary: string | null;
}

export interface RankedTrial {
  rank: number;
  candidate: TrialCandidate;
  judgment: TrialJudgment;
  score: number;
}

// --------------------------------------------------------- gateway wire API --

/** 202 body of `POST /v1/matches`. */
export interface MatchAccepted {
  match_id: string;
  patient_id: string;
  status: MatchStatus;
  event_id: string;
}

/** First SSE frame (`event: status`) of `GET /v1/matches/{id}/events`. */
export interface MatchStatusSnapshot {
  match_id: string;
  status: MatchStatus;
  current_step: PipelineStep | null;
}

/** `event: progress` payload, i.e. `beacon_core.events.MatchProgress`. */
export interface MatchProgressEvent {
  event_type: "match.progress";
  match_id: string;
  step: PipelineStep;
  pct: number;
  message: string;
}

/** `event: completed` payload, i.e. `beacon_core.events.MatchCompleted`. */
export interface MatchCompletedEvent {
  event_type: "match.completed";
  match_id: string;
  status: MatchStatus;
  results: RankedTrial[];
  trial_count: number;
  total_cost_usd: number;
  duration_ms: number;
  error: string | null;
}

/** `event: error` payload framed by `gateway.sse.format_sse`. */
export interface MatchErrorEvent {
  code: string;
  message: string;
  match_id?: string;
}

/** `event: done` payload. */
export interface MatchDoneEvent {
  match_id: string;
  status: MatchStatus;
}

/**
 * Everything one finished run needs to be *displayed*.
 *
 * `MatchCompleted` carries `results` only. The numbered note and the patient
 * profile are produced by the worker and stored in Postgres, but as of the
 * merged gateway no route returns them, so the console treats both as optional
 * and says so on screen when they are absent rather than re-deriving them.
 * See `docs`/the C4 PR for the precise gap.
 */
export interface MatchResultBundle {
  match_id: string;
  status: MatchStatus;
  results: RankedTrial[];
  note?: NumberedNote;
  profile?: PatientProfile;
  total_cost_usd?: number;
  duration_ms?: number;
}
