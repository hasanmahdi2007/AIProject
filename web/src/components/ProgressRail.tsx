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
    <section className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Pipeline
        </h2>
        <span className="text-sm font-medium text-slate-900">
          {STATUS_TEXT[status]}
        </span>
        {matchId ? (
          <code className="ml-auto text-xs text-slate-500">
            match {matchId}
          </code>
        ) : null}
      </div>

      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          className={`h-full transition-all duration-500 ${
            status === "failed" ? "bg-rose-500" : "bg-sky-600"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <ol className="mt-4 space-y-1.5">
        {PIPELINE_STEPS.map((step) => {
          const isCurrent = step === currentStep && status === "running";
          const isDone =
            status === "completed" || (done.has(step) && !isCurrent);
          return (
            <li key={step} className="flex items-center gap-2 text-sm">
              <span
                aria-hidden
                className={`inline-block h-2 w-2 shrink-0 rounded-full ${
                  isCurrent
                    ? "animate-pulse bg-sky-600"
                    : isDone
                      ? "bg-emerald-500"
                      : "bg-slate-300"
                }`}
              />
              <span
                className={
                  isCurrent
                    ? "font-semibold text-slate-900"
                    : isDone
                      ? "text-slate-700"
                      : "text-slate-400"
                }
              >
                {STEP_LABELS[step]}
              </span>
            </li>
          );
        })}
      </ol>

      {message ? (
        <p className="mt-3 border-t border-slate-100 pt-3 text-sm text-slate-600">
          {message}
        </p>
      ) : null}
    </section>
  );
}
