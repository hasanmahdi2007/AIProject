/**
 * Presentation only: how each contract vocabulary value is worded and coloured.
 *
 * No value here is computed from patient or trial data. The API decides what a
 * criterion's label and a trial's category are; this file decides what colour
 * that label is drawn in.
 *
 * `insufficient_info` is deliberately neutral grey in both tables. It is
 * not amber, not red, and carries no warning glyph: it is the system
 * abstaining, which is a designed outcome, not a failure.
 */

import type { CriterionLabel, EligibilityCategory } from "./types";

export interface Chip {
  text: string;
  className: string;
  /** One line explaining what the value means, for a tooltip or caption. */
  meaning: string;
  /** The trial card's header band, for trial categories only. */
  accent?: string;
}

export const CRITERION_LABELS: Record<CriterionLabel, Chip> = {
  met: {
    text: "Met",
    className: "bg-nhs-green text-white",
    meaning: "The note contains evidence that satisfies this rule.",
  },
  not_met: {
    text: "Not met",
    className: "bg-nhs-red text-white",
    meaning: "The note contains evidence that contradicts this rule.",
  },
  insufficient_info: {
    text: "Insufficient info",
    className: "bg-nhs-unknown text-nhs-ink ring-1 ring-inset ring-nhs-mid",
    meaning:
      "The note does not settle this rule, so the system abstains instead of guessing. This is a normal outcome.",
  },
  not_applicable: {
    text: "Not applicable",
    className: "bg-nhs-pale text-nhs-ink ring-1 ring-inset ring-nhs-line",
    meaning: "This rule does not apply to this patient.",
  },
};

export const ELIGIBILITY_CATEGORIES: Record<EligibilityCategory, Chip> = {
  eligible: {
    text: "Eligible",
    className: "bg-nhs-green text-white",
    meaning: "Every rule the system could judge was satisfied.",
    accent: "bg-nhs-green text-white",
  },
  likely_eligible: {
    text: "Likely eligible",
    className: "bg-nhs-lightgreen text-nhs-ink",
    meaning: "No rule was violated, but some rules are still unsettled.",
    accent: "bg-nhs-lightgreen text-nhs-ink",
  },
  insufficient_info: {
    text: "Insufficient info",
    className: "bg-nhs-unknown text-nhs-ink ring-1 ring-inset ring-nhs-mid",
    meaning:
      "Too much is unknown to place this trial either way. The missing-facts list says exactly what would settle it.",
    accent: "bg-nhs-unknown text-nhs-ink",
  },
  excluded: {
    text: "Excluded",
    className: "bg-nhs-red text-white",
    meaning:
      "An inclusion rule was not met, or an exclusion rule was met.",
    accent: "bg-nhs-red text-white",
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

/**
 * The ranking score, which the API computes on a 0-5 scale: eligible trials
 * land in 3.75-5, likely eligible 2.5-3.75, insufficient info 1.25-2.5 and
 * excluded 0-1.25 (see `match_worker/verdict.py`).
 */
export function formatFinalScore(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(2)} / 5`;
}

export function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}
