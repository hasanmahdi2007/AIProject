"use client";

import {
  CRITERION_LABELS,
  ELIGIBILITY_CATEGORIES,
  KIND_LABELS,
  formatPercent,
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
}

export function ruleKey(nctId: string, criterionId: string): string {
  return `${nctId}::${criterionId}`;
}

function Chip({ className, text }: { className: string; text: string }) {
  return (
    <span
      className={`inline-block whitespace-nowrap rounded px-2 py-0.5 text-xs font-medium ring-1 ${className}`}
    >
      {text}
    </span>
  );
}

function RuleTable({
  trial,
  selectedRuleKey,
  onSelectRule,
}: {
  trial: RankedTrial;
  selectedRuleKey: string | null;
  onSelectRule: TrialResultsProps["onSelectRule"];
}) {
  const nctId = trial.candidate.nct_id;

  return (
    <table className="w-full table-fixed border-separate border-spacing-0 text-left text-sm">
      <caption className="pb-2 text-left text-xs text-slate-500">
        Click a rule to highlight the sentence of the note it was decided on.
      </caption>
      <thead>
        <tr className="text-xs uppercase tracking-wide text-slate-500">
          <th scope="col" className="w-20 pb-1 font-medium">
            Rule
          </th>
          <th scope="col" className="w-24 pb-1 font-medium">
            Kind
          </th>
          <th scope="col" className="w-32 pb-1 font-medium">
            Verdict
          </th>
          <th scope="col" className="pb-1 font-medium">
            Why, and the sentence it cites
          </th>
        </tr>
      </thead>
      <tbody>
        {trial.judgment.judgments.map((judgment) => {
          const key = ruleKey(nctId, judgment.criterion_id);
          const selected = key === selectedRuleKey;
          const chip = CRITERION_LABELS[judgment.label];
          const cited = judgment.evidence_sentence_ids;

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
              className={`cursor-pointer align-top transition-colors ${
                selected
                  ? "bg-amber-50 outline outline-2 -outline-offset-2 outline-amber-400"
                  : "hover:bg-slate-50"
              }`}
            >
              <td className="border-t border-slate-100 py-2 pr-2 font-mono text-xs text-slate-700">
                {judgment.criterion_id}
              </td>
              <td className="border-t border-slate-100 py-2 pr-2 text-xs text-slate-600">
                {KIND_LABELS[judgment.kind]}
              </td>
              <td className="border-t border-slate-100 py-2 pr-2">
                <Chip className={chip.className} text={chip.text} />
              </td>
              <td className="border-t border-slate-100 py-2">
                <p className="leading-6 text-slate-700">
                  {judgment.explanation}
                </p>
                <p className="mt-1 text-xs">
                  {cited.length > 0 ? (
                    <span
                      className={
                        selected
                          ? "font-semibold text-amber-800"
                          : "text-sky-700"
                      }
                    >
                      Evidence: sentence {cited.join(", ")}
                      {selected ? " — highlighted in the note" : ""}
                    </span>
                  ) : (
                    <span className="text-slate-500">
                      No sentence in the note addresses this rule.
                    </span>
                  )}
                </p>
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
}: {
  trial: RankedTrial;
  isOpen: boolean;
  onToggleTrial: TrialResultsProps["onToggleTrial"];
  selectedRuleKey: string | null;
  onSelectRule: TrialResultsProps["onSelectRule"];
}) {
  const { candidate, judgment } = trial;
  const category = ELIGIBILITY_CATEGORIES[judgment.category];

  return (
    <li className="rounded-lg border border-slate-200 bg-white">
      <button
        type="button"
        onClick={() => onToggleTrial(candidate.nct_id)}
        aria-expanded={isOpen}
        className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-slate-50"
      >
        <span className="w-6 shrink-0 pt-0.5 text-sm font-semibold text-slate-400">
          {trial.rank}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2">
            <code className="text-xs font-semibold text-slate-900">
              {candidate.nct_id}
            </code>
            <Chip className={category.className} text={category.text} />
            {candidate.phase ? (
              <span className="text-xs text-slate-500">{candidate.phase}</span>
            ) : null}
            {candidate.status ? (
              <span className="text-xs text-slate-500">{candidate.status}</span>
            ) : null}
          </span>
          <span className="mt-1 block text-sm font-medium leading-6 text-slate-900">
            {candidate.title}
          </span>
          <span className="mt-1 block text-xs text-slate-500">
            score {formatScore(trial.score)} · rules confirmed{" "}
            {formatPercent(judgment.confirmed_fraction)} ·{" "}
            {judgment.judgments.length} rules judged
            {judgment.missing_facts.length > 0
              ? ` · ${judgment.missing_facts.length} missing fact${
                  judgment.missing_facts.length === 1 ? "" : "s"
                }`
              : ""}
          </span>
        </span>
        <span aria-hidden className="shrink-0 pt-1 text-xs text-slate-400">
          {isOpen ? "Close" : "Open"}
        </span>
      </button>

      {isOpen ? (
        <div className="space-y-4 border-t border-slate-200 px-4 py-4">
          {judgment.summary ? (
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                What this trial is, in plain language
              </h3>
              <p className="mt-1 text-sm leading-6 text-slate-800">
                {judgment.summary}
              </p>
            </div>
          ) : null}

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Eligibility rules
            </h3>
            <div className="mt-2">
              <RuleTable
                trial={trial}
                selectedRuleKey={selectedRuleKey}
                onSelectRule={onSelectRule}
              />
            </div>
          </div>

          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500">
              Missing facts
            </h3>
            {judgment.missing_facts.length > 0 ? (
              <>
                <p className="mt-1 text-sm text-slate-600">
                  These would settle the rules the system abstained on. They are
                  gaps in the note, not errors.
                </p>
                <ul className="mt-1 list-disc space-y-0.5 pl-5 text-sm text-slate-800">
                  {judgment.missing_facts.map((fact) => (
                    <li key={fact}>{fact}</li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="mt-1 text-sm text-slate-600">
                None. Every rule was settled from the note.
              </p>
            )}
          </div>

          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 border-t border-slate-100 pt-3 text-xs text-slate-600 sm:grid-cols-4">
            <div>
              <dt className="text-slate-500">Retrieval rank</dt>
              <dd>{candidate.retrieval_rank ?? "—"}</dd>
            </div>
            <div>
              <dt className="text-slate-500">RRF</dt>
              <dd>{formatScore(candidate.rrf_score)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Rerank</dt>
              <dd>{formatScore(candidate.rerank_score)}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Age / sex bounds</dt>
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
}: TrialResultsProps) {
  if (results.length === 0) {
    return (
      <section className="rounded-lg border border-slate-200 bg-white p-4">
        <p className="text-sm text-slate-600">
          No trials to show yet. Submit a note, or load the sample run.
        </p>
      </section>
    );
  }

  return (
    <section>
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
        Ranked trials ({results.length})
      </h2>
      <ul className="space-y-2">
        {results.map((trial) => (
          <TrialCard
            key={trial.candidate.nct_id}
            trial={trial}
            isOpen={openNctId === trial.candidate.nct_id}
            onToggleTrial={onToggleTrial}
            selectedRuleKey={selectedRuleKey}
            onSelectRule={onSelectRule}
          />
        ))}
      </ul>
    </section>
  );
}
