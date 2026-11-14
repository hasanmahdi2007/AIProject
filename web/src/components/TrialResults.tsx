"use client";

import {
  CRITERION_LABELS,
  ELIGIBILITY_CATEGORIES,
  KIND_LABELS,
  formatPercent,
  formatFinalScore,
  formatScore,
} from "@/lib/labels";
import type { CriterionJudgment, RankedTrial } from "@/lib/types";

export interface TrialResultsProps {
  results: RankedTrial[];
  openNctId: string | null;
  onToggleTrial: (nctId: string) => void;
  /** `${nct_id}::${criterion_id}` of the rule whose evidence is highlighted. */
  selectedRuleKey: string | null;
  onSelectRule: (nctId: string, judgment: CriterionJudgment) => void;
  /**
   * The sentence IDs the numbered note actually contains, or `null` when the
   * run returned no note. Used only to flag a citation that points at a
   * sentence that does not exist: invalid citations are a measured metric in
   * this project, so one is shown as a problem rather than quietly dropped.
   */
  validSentenceIds: ReadonlySet<number> | null;
}

export function ruleKey(nctId: string, criterionId: string): string {
  return `${nctId}::${criterionId}`;
}

/** "NCT03641313:inc:01" reads as "inc:01" inside its own trial's table. */
function shortRuleId(nctId: string, criterionId: string): string {
  return criterionId.startsWith(`${nctId}:`) ? criterionId.slice(nctId.length + 1) : criterionId;
}

function Chip({ className, text }: { className: string; text: string }) {
  return <span className={`tag ${className}`}>{text}</span>;
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 20 20"
      aria-hidden
      className={`h-5 w-5 shrink-0 transition-transform duration-200 ${open ? "rotate-180" : ""}`}
    >
      <path d="M5 7.5l5 5 5-5" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function RuleTable({
  trial,
  selectedRuleKey,
  onSelectRule,
  validSentenceIds,
}: {
  trial: RankedTrial;
  selectedRuleKey: string | null;
  onSelectRule: TrialResultsProps["onSelectRule"];
  validSentenceIds: ReadonlySet<number> | null;
}) {
  const nctId = trial.candidate.nct_id;

  return (
    <table className="w-full max-sm:block sm:table-fixed border-separate border-spacing-0 text-left text-[14px]">
      <caption className="block px-4 pb-2 sm:table-caption text-left text-[14px] text-nhs-mid">
        Click a rule to highlight the sentence of the note it was decided on.
      </caption>
      <thead className="max-sm:hidden">
        <tr className="text-[13px] text-nhs-mid">
          <th scope="col" className="w-20 pb-1.5 pl-4 font-bold">
            Rule
          </th>
          <th scope="col" className="w-24 pb-1.5 font-bold">
            Kind
          </th>
          <th scope="col" className="w-36 pb-1.5 font-bold">
            Verdict
          </th>
          <th scope="col" className="pb-1.5 pr-4 font-bold">
            Why, and the sentence it cites
          </th>
        </tr>
      </thead>
      <tbody className="max-sm:block">
        {trial.judgment.judgments.map((judgment) => {
          const key = ruleKey(nctId, judgment.criterion_id);
          const selected = key === selectedRuleKey;
          const chip = CRITERION_LABELS[judgment.label];
          const cited = judgment.evidence_sentence_ids;
          const unresolved =
            validSentenceIds === null
              ? []
              : cited.filter((id) => !validSentenceIds.has(id));

          return (
            <tr
              key={key}
              onClick={() => onSelectRule(nctId, judgment)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelectRule(nctId, judgment);
                }
              }}
              tabIndex={0}
              role="button"
              aria-pressed={selected}
              aria-label={`Rule ${judgment.criterion_id}, ${chip.text}. ${
                cited.length > 0
                  ? `Cites sentence ${cited.join(", ")}.`
                  : "Cites no sentence."
              }`}
              className={`cursor-pointer align-top transition-colors max-sm:block max-sm:border-t max-sm:border-nhs-line max-sm:px-4 max-sm:py-3 ${
                selected ? "evidence-on" : "hover:bg-nhs-pale"
              }`}
            >
              <td className="font-mono text-[13px] text-nhs-mid max-sm:mr-3 max-sm:inline-block sm:border-t sm:border-nhs-line sm:py-3 sm:pl-4 sm:pr-2" title={judgment.criterion_id}>
                {shortRuleId(nctId, judgment.criterion_id)}
              </td>
              <td className="max-sm:mr-3 max-sm:inline-block sm:border-t sm:border-nhs-line sm:py-3 sm:pr-2">
                <span
                  className={`text-[13px] font-bold ${
                    judgment.kind === "inclusion" ? "text-nhs-dark" : "text-nhs-purple"
                  }`}
                >
                  {KIND_LABELS[judgment.kind]}
                </span>
              </td>
              <td className="max-sm:inline-block sm:border-t sm:border-nhs-line sm:py-3 sm:pr-2">
                <Chip className={chip.className} text={chip.text} />
              </td>
              <td className="max-sm:mt-2 max-sm:block sm:border-t sm:border-nhs-line sm:py-3 sm:pr-4">
                <p className="leading-6 text-nhs-ink">{judgment.explanation}</p>
                <p className="mt-1.5 text-[13px]">
                  {cited.length > 0 ? (
                    <span
                      className={`tag ${
                        selected ? "bg-nhs-aqua text-nhs-ink" : "bg-nhs-aquapale text-[#00445E]"
                      }`}
                    >
                      Evidence: sentence {cited.join(", ")}
                      {selected ? " — highlighted in the note" : ""}
                    </span>
                  ) : (
                    <span className="text-nhs-mid">
                      No sentence in the note addresses this rule.
                    </span>
                  )}
                </p>
                {unresolved.length > 0 ? (
                  <p className="mt-1.5 rounded bg-[#FBE7E5] px-2 py-1 text-[13px] font-bold text-nhs-red">
                    Invalid citation: sentence {unresolved.join(", ")} is not in
                    the numbered note, so this verdict cannot be checked.
                  </p>
                ) : null}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function TrialCard({
  trial,
  isOpen,
  onToggleTrial,
  selectedRuleKey,
  onSelectRule,
  validSentenceIds,
}: {
  trial: RankedTrial;
  isOpen: boolean;
  onToggleTrial: TrialResultsProps["onToggleTrial"];
  selectedRuleKey: string | null;
  onSelectRule: TrialResultsProps["onSelectRule"];
  validSentenceIds: ReadonlySet<number> | null;
}) {
  const { candidate, judgment } = trial;
  const category = ELIGIBILITY_CATEGORIES[judgment.category];
  const band = category.accent ?? "bg-nhs-mid text-white";
  const missing = judgment.missing_facts.length;
  const compact = judgment.category === "excluded" && !isOpen;

  return (
    <li
      className={`card overflow-hidden transition-shadow ${
        isOpen ? "ring-4 ring-nhs-aqua" : "hover:ring-2 hover:ring-white/70"
      }`}
    >
      <button
        type="button"
        onClick={() => onToggleTrial(candidate.nct_id)}
        aria-expanded={isOpen}
        className="block w-full text-left"
      >
        <span className={`flex items-center gap-3 px-4 font-bold ${compact ? "py-1 text-[13px]" : "py-1.5 text-[14px]"} ${band}`}>
          <span className="tabular">Rank {trial.rank}</span>
          <span aria-hidden className="opacity-60">·</span>
          <span>{category.text}</span>
          <span className="ml-auto tabular">score {formatFinalScore(trial.score)}</span>
        </span>
        <span className="flex items-start gap-4 px-4 pb-4 pt-3">
          <span className="min-w-0 flex-1">
            <span className="flex flex-wrap items-center gap-2">
              <code className="font-mono text-[14px] font-bold text-nhs-blue">
                {candidate.nct_id}
              </code>
              {candidate.phase ? (
                <span className="tag bg-nhs-pale text-nhs-ink">{candidate.phase}</span>
              ) : null}
              {candidate.status ? (
                <span className="tag bg-nhs-pale font-semibold text-nhs-mid">{candidate.status}</span>
              ) : null}
            </span>
            <span className={`mt-1 block font-bold text-nhs-ink ${compact ? "truncate text-[15px] leading-6" : "text-[16px] leading-6"}`}>
              {candidate.title}
            </span>
            <span className="mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px] text-nhs-mid">
              <span aria-hidden className="h-1.5 w-28 overflow-hidden rounded-full bg-nhs-line">
                <span
                  className="block h-full rounded-full bg-nhs-green"
                  style={{ width: formatPercent(judgment.confirmed_fraction) }}
                />
              </span>
              <span className="tabular">
                rules confirmed {formatPercent(judgment.confirmed_fraction)} ·{" "}
                {judgment.judgments.length} rules judged
                {missing > 0 ? ` · ${missing} missing fact${missing === 1 ? "" : "s"}` : ""}
              </span>
            </span>
          </span>
          <span className="mt-1 text-nhs-blue">
            <Chevron open={isOpen} />
          </span>
        </span>
      </button>

      {isOpen ? (
        <div className="divide-y divide-nhs-line border-t border-nhs-line">
          {judgment.summary ? (
            <div className="px-4 py-4">
              <h3 className="heading text-[15px]">What this trial is, in plain language</h3>
              <p className="mt-1 max-w-[72ch] text-[15px] leading-7 text-nhs-ink">{judgment.summary}</p>
            </div>
          ) : null}

          <div className="py-4">
            <h3 className="heading px-4 text-[15px]">Eligibility rules</h3>
            <div className="mt-2">
              <RuleTable
                trial={trial}
                selectedRuleKey={selectedRuleKey}
                onSelectRule={onSelectRule}
                validSentenceIds={validSentenceIds}
              />
            </div>
          </div>

          <div className="px-4 py-4">
            <h3 className="heading text-[15px]">Missing facts</h3>
            {missing > 0 ? (
              <>
                <p className="mt-1 text-[14px] leading-6 text-nhs-mid">
                  These would settle the rules the system abstained on. They are
                  gaps in the note, not errors.
                </p>
                <ul className="mt-2 grid gap-1.5 text-[14px] text-nhs-ink">
                  {judgment.missing_facts.map((fact) => (
                    <li key={fact} className="flex gap-2.5 rounded bg-nhs-pale px-3 py-2 leading-6">
                      <span aria-hidden className="mt-2.5 h-1.5 w-1.5 shrink-0 rounded-full bg-nhs-blue" />
                      {fact}
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="mt-1 text-[14px] text-nhs-mid">
                None. Every rule was settled from the note.
              </p>
            )}
          </div>

          <dl className="grid grid-cols-2 gap-x-6 gap-y-3 px-4 py-4 text-[14px] text-nhs-ink sm:grid-cols-4">
            <div>
              <dt className="text-[13px] font-bold text-nhs-mid">Retrieval rank</dt>
              <dd className="tabular">{candidate.retrieval_rank ?? "—"}</dd>
            </div>
            <div>
              <dt className="text-[13px] font-bold text-nhs-mid">RRF</dt>
              <dd className="tabular">{formatScore(candidate.rrf_score)}</dd>
            </div>
            <div>
              <dt className="text-[13px] font-bold text-nhs-mid">Rerank</dt>
              <dd className="tabular">{formatScore(candidate.rerank_score)}</dd>
            </div>
            <div>
              <dt className="text-[13px] font-bold text-nhs-mid">Age / sex bounds</dt>
              <dd>
                {candidate.min_age_years ?? "any"}–
                {candidate.max_age_years ?? "any"} ·{" "}
                {candidate.sex ?? "unknown"}
              </dd>
            </div>
          </dl>
        </div>
      ) : null}
    </li>
  );
}

/**
 * The ranked answer. Rank, score and verdict are printed as the API returned
 * them; this component sorts nothing and decides nothing.
 */
export function TrialResults({
  results,
  openNctId,
  onToggleTrial,
  selectedRuleKey,
  onSelectRule,
  validSentenceIds,
}: TrialResultsProps) {
  if (results.length === 0) {
    return (
      <section className="card p-5">
        <p className="text-[14px] text-nhs-mid">
          No trials to show yet. Submit a note, or load the sample run.
        </p>
      </section>
    );
  }

  // Split where the API's order first reaches an excluded trial. Nothing is
  // re-sorted: the split only adds space between the two runs of the list.
  const firstExcluded = results.findIndex((trial) => trial.judgment.category === "excluded");
  const groups =
    firstExcluded > 0 ? [results.slice(0, firstExcluded), results.slice(firstExcluded)] : [results];
  const card = (trial: RankedTrial) => (
    <TrialCard
      key={trial.candidate.nct_id}
      trial={trial}
      isOpen={openNctId === trial.candidate.nct_id}
      onToggleTrial={onToggleTrial}
      selectedRuleKey={selectedRuleKey}
      onSelectRule={onSelectRule}
      validSentenceIds={validSentenceIds}
    />
  );

  return (
    <section>
      <h2 className="mb-3 text-[20px] font-bold text-white">
        Ranked trials ({results.length})
      </h2>
      {groups.map((group, index) => (
        <ul
          key={index}
          className={`space-y-3 ${index > 0 ? "mt-8 border-t border-white/30 pt-8" : ""}`}
        >
          {group.map(card)}
        </ul>
      ))}
    </section>
  );
}
