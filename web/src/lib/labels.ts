/**
 * Presentation only: how each contract vocabulary value is worded and coloured.
 *
 * No value here is computed from patient or trial data. The API decides what a
 * criterion's label and a trial's category are; this file decides what colour
 * that label is drawn in.
 *
 * `insufficient_info` is deliberately slate/neutral in both tables. It is not
 * amber, not red, and carries no warning glyph: it is the system abstaining,
 * which is a designed outcome, not a failure.
 */

import type { CriterionLabel, EligibilityCategory } from "./types";

export interface Chip {
  text: string;
  className: string;
  /** One line explaining what the value means, for a tooltip or caption. */
  meaning: string;
}

export const CRITERION_LABELS: Record<CriterionLabel, Chip> = {
  met: {
    text: "Met",
    className: "bg-emerald-100 text-emerald-900 ring-emerald-300",
    meaning: "The note contains evidence that satisfies this rule.",
  },
  not_met: {
    text: "Not met",
    className: "bg-rose-100 text-rose-900 ring-rose-300",
    meaning: "The note contains evidence that contradicts this rule.",
  },
  insufficient_info: {
    text: "Insufficient info",
    className: "bg-slate-100 text-slate-700 ring-slate-300",
    meaning:
      "The note does not settle this rule, so the system abstains instead of guessing. This is a normal outcome.",
  },
  not_applicable: {
    text: "Not applicable",
    className: "bg-sky-50 text-sky-900 ring-sky-200",
    meaning: "This rule does not apply to this patient.",
  },
};

export const ELIGIBILITY_CATEGORIES: Record<EligibilityCategory, Chip> = {
  eligible: {
    text: "Eligible",
    className: "bg-emerald-600 text-white ring-emerald-700",
    meaning: "Every rule the system could judge was satisfied.",
  },
  likely_eligible: {
    text: "Likely eligible",
    className: "bg-emerald-100 text-emerald-900 ring-emerald-300",
    meaning: "No rule was violated, but some rules are still unsettled.",
  },
  insufficient_info: {
    text: "Insufficient info",
    className: "bg-slate-100 text-slate-700 ring-slate-300",
    meaning:
      "Too much is unknown to place this trial either way. The missing-facts list says exactly what would settle it.",
  },
  excluded: {
    text: "Excluded",
    className: "bg-rose-100 text-rose-900 ring-rose-300",
    meaning:
      "An inclusion rule was not met, or an exclusion rule was met.",
  },
};

export const KIND_LABELS: Record<"inclusion" | "exclusion", string> = {
  inclusion: "Inclusion",
  exclusion: "Exclusion",
};

/** Formats a float the API already computed. Rounding is not re-ranking. */
export function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(3);
}

export function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}
