"use client";

import { PIPELINE_STEPS, STEP_LABELS } from "@/lib/types";
import type { MatchStatus, PipelineStep } from "@/lib/types";

export interface ProgressRailProps {
  status: MatchStatus;
  currentStep: PipelineStep | null;
  pct: number;
  message: string;
  matchId: string | null;
  /** Steps the stream has reported, in arrival order. */
  seen: PipelineStep[];
}

const STATUS_TEXT: Record<MatchStatus, string> = {
  queued: "Queued — accepted, not started",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
};

const STATUS_TAG: Record<MatchStatus, string> = {
  queued: "bg-nhs-pale text-nhs-ink ring-1 ring-inset ring-nhs-line",
  running: "bg-nhs-blue text-white",
  completed: "bg-nhs-green text-white",
  failed: "bg-nhs-red text-white",
};

function Check() {
  return (
    <svg viewBox="0 0 16 16" className="h-4 w-4" aria-hidden>
      <path
        d="M3.5 8.5l3 3 6-7"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * Live pipeline progress, driven entirely by the `status` and `progress` SSE
 * frames. The step order is the `PipelineStep` vocabulary from the contract;
 * which step is current, and the percentage, come from the stream.
 */
export function ProgressRail({
  status,
  currentStep,
  pct,
  message,
  matchId,
  seen,
}: ProgressRailProps) {
  const done = new Set(seen);

  return (
    <section className="card p-5 sm:p-6">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h2 className="heading">Pipeline</h2>
        <span className={`tag ${STATUS_TAG[status]}`}>{STATUS_TEXT[status]}</span>
        {matchId ? (
          <code className="ml-auto rounded bg-nhs-pale px-2 py-0.5 font-mono text-[13px] text-nhs-ink">
            match {matchId}
          </code>
        ) : null}
      </div>

      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-nhs-line">
        <div
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          className={`h-full rounded-full transition-[width] duration-700 ease-out ${
            status === "failed" ? "bg-nhs-red" : status === "completed" ? "bg-nhs-green" : "bg-nhs-blue"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <ol className="mt-5 grid grid-cols-1 gap-x-2 gap-y-3 sm:grid-cols-2 lg:grid-cols-7">
        {PIPELINE_STEPS.map((step, index) => {
          const isCurrent = step === currentStep && status === "running";
          const isDone = status === "completed" || (done.has(step) && !isCurrent);
          const isFailedHere = status === "failed" && step === currentStep;
          return (
            <li key={step} className="flex items-center gap-2.5 lg:flex-col lg:items-start lg:gap-2">
              <span
                aria-hidden
                className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-[13px] font-bold tabular ${
                  isFailedHere
                    ? "bg-nhs-red text-white"
                    : isCurrent
                      ? "bg-nhs-blue text-white ring-4 ring-nhs-aqua/50"
                      : isDone
                        ? "bg-nhs-green text-white"
                        : "bg-nhs-pale text-nhs-mid ring-1 ring-inset ring-nhs-line"
                }`}
              >
                {isDone && !isFailedHere ? <Check /> : index + 1}
              </span>
              <span
                className={`text-[14px] leading-5 ${
                  isFailedHere
                    ? "font-bold text-nhs-red"
                    : isCurrent
                      ? "font-bold text-nhs-blue"
                      : isDone
                        ? "text-nhs-ink"
                        : "text-nhs-mid"
                }`}
              >
                {STEP_LABELS[step]}
              </span>
            </li>
          );
        })}
      </ol>

      {message ? (
        <p className="mt-5 rounded bg-nhs-pale px-4 py-2.5 text-[14px] leading-6 text-nhs-ink">
          {message}
        </p>
      ) : null}
    </section>
  );
}
