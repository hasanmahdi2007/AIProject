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
        <h1 className="text-[32px] font-extrabold leading-10 tracking-tight text-white">
          How to read this console
        </h1>
        <p className="max-w-3xl text-[16px] leading-7 text-white/90">
          Beacon takes a free-text patient note, searches a clinical-trial
          index, and then judges each trial&apos;s eligibility rules one at a
          time against the note. The console exists to make that judgement
          checkable: every rule verdict carries the sentence it was decided on,
          and clicking the rule highlights that sentence in the note.
        </p>
        <p className="max-w-3xl text-[16px] leading-7 text-white/90">
          Nothing on this page is computed in the browser. Ranks, scores,
          verdicts and summaries are rendered exactly as the API returns them.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-[22px] font-bold text-white">
          &ldquo;Insufficient info&rdquo; is an answer, not a bug
        </h2>
        <p className="max-w-3xl text-[16px] leading-7 text-white/90">
          When a note does not settle a rule, Beacon says so instead of
          guessing. That verdict is drawn in neutral grey on purpose: an
          abstention is the designed outcome, and it feeds the missing-facts
          list, which names the specific facts that would settle the trial. A
          confident wrong answer would be worse than an honest gap.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-[22px] font-bold text-white">Rule verdicts</h2>
        <dl className="card max-w-3xl divide-y divide-nhs-line">
          {CRITERION_ORDER.map((label) => {
            const chip = CRITERION_LABELS[label];
            return (
              <div key={label} className="flex flex-col gap-2 p-4 sm:flex-row sm:gap-4">
                <dt className="shrink-0 sm:w-40">
                  <span
                    className={`tag ${chip.className}`}
                  >
                    {chip.text}
                  </span>
                </dt>
                <dd className="text-[15px] leading-7 text-nhs-ink">
                  {chip.meaning}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>

      <section className="space-y-3">
        <h2 className="text-[22px] font-bold text-white">Trial verdicts</h2>
        <dl className="card max-w-3xl divide-y divide-nhs-line">
          {CATEGORY_ORDER.map((category) => {
            const chip = ELIGIBILITY_CATEGORIES[category];
            return (
              <div key={category} className="flex flex-col gap-2 p-4 sm:flex-row sm:gap-4">
                <dt className="shrink-0 sm:w-40">
                  <span
                    className={`tag ${chip.className}`}
                  >
                    {chip.text}
                  </span>
                </dt>
                <dd className="text-[15px] leading-7 text-nhs-ink">
                  {chip.meaning}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>

      <section className="space-y-3">
        <h2 className="text-[22px] font-bold text-white">The pipeline you watch</h2>
        <ol className="max-w-3xl list-decimal space-y-1 pl-6 text-[16px] leading-7 text-white/90">
          {PIPELINE_STEPS.map((step) => (
            <li key={step}>
              <span className="font-bold text-white">{STEP_LABELS[step]}</span>{" "}
              <code className="rounded bg-nhs-dark px-1.5 py-0.5 font-mono text-[13px] text-white">{step}</code>
            </li>
          ))}
        </ol>
        <p className="max-w-3xl text-[16px] leading-7 text-white/90">
          Submitting a note returns immediately with a match id; progress
          arrives over a server-sent event stream as the pipeline advances. The
          match id is the durable handle on the answer — the console reads the
          finished run back by id, so a reload, a late connect or a second
          person opening the same id all get the same result.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-[22px] font-bold text-white">Limits</h2>
        <ul className="max-w-3xl list-disc space-y-1 pl-6 text-[16px] leading-7 text-white/90">
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
        className="inline-block text-[16px] font-bold text-white underline underline-offset-4 hover:no-underline"
      >
        &larr; Back to the console
      </Link>
    </div>
  );
}
