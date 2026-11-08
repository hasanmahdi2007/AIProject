import Link from "next/link";
import { CRITERION_LABELS, ELIGIBILITY_CATEGORIES } from "@/lib/labels";
import { PIPELINE_STEPS, STEP_LABELS } from "@/lib/types";
import type { CriterionLabel, EligibilityCategory } from "@/lib/types";

const CRITERION_ORDER: CriterionLabel[] = [
  "met",
  "not_met",
  "insufficient_info",
  "not_applicable",
];

const CATEGORY_ORDER: EligibilityCategory[] = [
  "eligible",
  "likely_eligible",
  "insufficient_info",
  "excluded",
];

export default function AboutPage() {
  return (
    <div className="space-y-8">
      <section className="space-y-3">
        <h1 className="text-2xl font-semibold tracking-tight">
          How to read this console
        </h1>
        <p className="max-w-3xl text-sm leading-6 text-slate-700">
          Beacon takes a free-text patient note, searches a clinical-trial
          index, and then judges each trial&apos;s eligibility rules one at a
          time against the note. The console exists to make that judgement
          checkable: every rule verdict carries the sentence it was decided on,
          and clicking the rule highlights that sentence in the note.
        </p>
        <p className="max-w-3xl text-sm leading-6 text-slate-700">
          Nothing on this page is computed in the browser. Ranks, scores,
          verdicts and summaries are rendered exactly as the API returns them.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">
          &ldquo;Insufficient info&rdquo; is an answer, not a bug
        </h2>
        <p className="max-w-3xl text-sm leading-6 text-slate-700">
          When a note does not settle a rule, Beacon says so instead of
          guessing. That verdict is drawn in neutral grey on purpose: an
          abstention is the designed outcome, and it feeds the missing-facts
          list, which names the specific facts that would settle the trial. A
          confident wrong answer would be worse than an honest gap.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Rule verdicts</h2>
        <dl className="max-w-3xl divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
          {CRITERION_ORDER.map((label) => {
            const chip = CRITERION_LABELS[label];
            return (
              <div key={label} className="flex gap-4 p-3">
                <dt className="w-40 shrink-0">
                  <span
                    className={`inline-block rounded px-2 py-0.5 text-xs font-medium ring-1 ${chip.className}`}
                  >
                    {chip.text}
                  </span>
                </dt>
                <dd className="text-sm leading-6 text-slate-700">
                  {chip.meaning}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Trial verdicts</h2>
        <dl className="max-w-3xl divide-y divide-slate-200 rounded-lg border border-slate-200 bg-white">
          {CATEGORY_ORDER.map((category) => {
            const chip = ELIGIBILITY_CATEGORIES[category];
            return (
              <div key={category} className="flex gap-4 p-3">
                <dt className="w-40 shrink-0">
                  <span
                    className={`inline-block rounded px-2 py-0.5 text-xs font-medium ring-1 ${chip.className}`}
                  >
                    {chip.text}
                  </span>
                </dt>
                <dd className="text-sm leading-6 text-slate-700">
                  {chip.meaning}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">The pipeline you watch</h2>
        <ol className="max-w-3xl list-decimal space-y-1 pl-6 text-sm leading-6 text-slate-700">
          {PIPELINE_STEPS.map((step) => (
            <li key={step}>
              <span className="font-medium">{STEP_LABELS[step]}</span>{" "}
              <code className="text-xs text-slate-500">{step}</code>
            </li>
          ))}
        </ol>
        <p className="max-w-3xl text-sm leading-6 text-slate-700">
          Submitting a note returns immediately with a match id; the answer
          arrives over a server-sent event stream as the pipeline advances.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Limits</h2>
        <ul className="max-w-3xl list-disc space-y-1 pl-6 text-sm leading-6 text-slate-700">
          <li>
            The patients are synthetic benchmark cases. The system has no
            clinical validity and has never been evaluated on real patients.
          </li>
          <li>
            A trial being ranked first means the note matched its rules well,
            not that the patient should enrol.
          </li>
          <li>
            Beacon only reads the note it is given. A fact absent from the note
            is unknown to it, however obvious it is to a clinician.
          </li>
        </ul>
      </section>

      <Link
        href="/"
        className="inline-block text-sm text-sky-700 hover:underline"
      >
        &larr; Back to the console
      </Link>
    </div>
  );
}
