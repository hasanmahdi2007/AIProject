"use strict";

/**
 * Renders the real components against a real `GET /v1/matches/{id}` body.
 *
 * The fixture checks prove the console renders *a* correct payload. These prove
 * it renders the payload the gateway actually sends, which is a different
 * claim: the two diverged once already - a completed run keeps `current_step`
 * set to the last node that ran, where the fixture nulls it, and a progress
 * rail that read "in progress" off that field would show a finished run as
 * stuck on step 7 forever.
 *
 * Read-only. Point it at a run with `VERIFY_MATCH_ID`; it skips cleanly when
 * the run does not exist, so this is safe to run against any gateway.
 */

const { GATEWAY_BASE, group, listItems, load, render } = require("./harness");

const MATCH_ID = process.env.VERIFY_MATCH_ID ?? "demo-completed-run";

const CONTRACT_FIELDS = [
  "match_id",
  "status",
  "current_step",
  "note",
  "profile",
  "results",
  "trial_count",
  "total_cost_usd",
  "duration_ms",
  "error",
];

function highlightedIds(html) {
  return listItems(html)
    .map((item, index) => (item.includes("evidence-on") ? index + 1 : null))
    .filter((value) => value !== null);
}

async function run(root) {
  const g = group("live payload rendering");
  const React = require("react");
  const { NotePanel } = load(root, "components/NotePanel.js");
  const { TrialResults, ruleKey } = load(root, "components/TrialResults.js");
  const { ProfileCard } = load(root, "components/ProfileCard.js");
  const { ProgressRail } = load(root, "components/ProgressRail.js");
  const { ELIGIBILITY_CATEGORIES } = load(root, "lib/labels.js");
  const types = load(root, "lib/types.js");

  const response = await fetch(
    `${GATEWAY_BASE}/v1/matches/${encodeURIComponent(MATCH_ID)}`,
  );
  if (response.status === 404) {
    g.skip(
      "live payload rendering",
      `no run "${MATCH_ID}" on ${GATEWAY_BASE} - set VERIFY_MATCH_ID to a run that exists`,
    );
    return g.state;
  }
  if (!response.ok) {
    g.check(false, `GET /v1/matches/${MATCH_ID}`, `HTTP ${response.status}`);
    return g.state;
  }

  const live = await response.json();
  const keys = Object.keys(live);
  g.check(
    CONTRACT_FIELDS.every((f) => keys.includes(f)) && keys.length === CONTRACT_FIELDS.length,
    `the live body has exactly the ${CONTRACT_FIELDS.length} MatchResult fields`,
    `got [${keys.join(",")}]`,
  );
  g.check(
    live.trial_count === live.results.length,
    `trial_count (${live.trial_count}) equals results.length (${live.results.length})`,
  );
  g.check(
    live.current_step === null || types.PIPELINE_STEPS.includes(live.current_step),
    `current_step is a contract step or null (got ${JSON.stringify(live.current_step)})`,
  );

  const note = live.note;
  const valid = note ? new Set(note.sentences.map((s) => s.sentence_id)) : null;
  g.log(
    `status=${live.status} step=${live.current_step} trials=${live.results.length} ` +
      `note=${note ? `${note.sentences.length} sentences` : "null"}`,
  );

  // ------------------------------------------- the progress rail, as served --
  const rail = render(
    React.createElement(ProgressRail, {
      status: live.status,
      currentStep: live.current_step,
      pct: live.status === "completed" ? 100 : 0,
      message: "",
      matchId: live.match_id,
      seen: [],
    }),
  );
  g.check(rail.includes(live.match_id), "the rail shows the live match id");
  if (live.status === "completed") {
    // The divergence named in the module docstring: current_step is still set.
    g.check(
      !rail.includes("animate-pulse"),
      "a completed run shows no step in progress, although current_step is set",
    );
    g.check(
      (rail.match(/bg-emerald-500/g) || []).length === types.PIPELINE_STEPS.length,
      `all ${types.PIPELINE_STEPS.length} steps read as done on a completed run`,
    );
  } else if (live.status === "queued") {
    g.check(!rail.includes("animate-pulse"), "nothing pulses before the first progress frame");
    g.check(
      (rail.match(/bg-slate-300/g) || []).length === types.PIPELINE_STEPS.length,
      `all ${types.PIPELINE_STEPS.length} steps read as not yet started`,
    );
  }

  // ---------------------------------------------- a run with nothing in it yet --
  if (note === null) {
    g.log("this run has no numbered note yet - checking the in-flight rendering");
    const pending = render(
      React.createElement(NotePanel, {
        note: null,
        highlighted: [1],
        caption: null,
        rawFallback: "the pasted note",
      }),
    );
    g.check(
      !pending.includes("Invalid citation"),
      "a live null note raises no invalid-citation alert (not-yet, not a defect)",
    );
    g.check(
      pending.includes("has not split this note into sentences yet"),
      "the live null note is explained as an in-flight state",
    );
    g.check(!pending.includes('role="alert"'), "nothing on the in-flight page is an alert");

    const empty = render(
      React.createElement(TrialResults, {
        results: live.results,
        openNctId: null,
        onToggleTrial: () => {},
        selectedRuleKey: null,
        onSelectRule: () => {},
        validSentenceIds: null,
      }),
    );
    g.check(
      empty.includes("No trials to show yet"),
      "an empty live result list reads as 'not yet', not 'none matched'",
    );
    g.check(
      render(React.createElement(ProfileCard, { profile: live.profile })).includes(
        "Nothing is wrong with the run",
      ),
      "the empty profile card says the run is fine, not broken",
    );
    return g.state;
  }

  // -------------------------------------------------- a run with real results --
  const unresolvable = [];
  const abstentions = [];
  let citations = 0;
  for (const trial of live.results) {
    for (const judgment of trial.judgment.judgments) {
      if (judgment.label === "insufficient_info" && judgment.evidence_sentence_ids.length === 0) {
        abstentions.push(`${trial.candidate.nct_id}/${judgment.criterion_id}`);
      }
      for (const id of judgment.evidence_sentence_ids) {
        citations += 1;
        if (!valid.has(id)) {
          unresolvable.push(`${trial.candidate.nct_id}/${judgment.criterion_id} -> ${id}`);
        }
      }
    }
  }
  g.log(`${citations} live citations; uncited abstentions: [${abstentions.join(", ")}]`);
  g.check(
    unresolvable.length === 0,
    "every live citation resolves to a sentence of the live note",
    unresolvable.join("; "),
  );
  g.check(
    live.results.every(
      (t) => t.candidate.dense_score === null && t.candidate.sparse_score === null,
    ),
    "dense_score and sparse_score are null on every live candidate, as the console assumes",
  );
  g.check(
    new Set(live.results.map((t) => t.rank)).size === live.results.length &&
      live.results.every((t) => t.rank >= 1),
    "live ranks are unique and at least 1",
  );

  // Every cited rule must highlight exactly what it cites, on live data.
  for (const trial of live.results) {
    for (const judgment of trial.judgment.judgments) {
      if (judgment.evidence_sentence_ids.length === 0) continue;
      const html = render(
        React.createElement(NotePanel, {
          note,
          highlighted: judgment.evidence_sentence_ids,
          caption: null,
          rawFallback: "",
        }),
      );
      g.check(
        JSON.stringify(highlightedIds(html)) === JSON.stringify(judgment.evidence_sentence_ids),
        `${trial.candidate.nct_id}/${judgment.criterion_id} highlights exactly sentence ${judgment.evidence_sentence_ids.join(", ")}`,
      );
      g.check(
        !html.includes("Invalid citation"),
        `${trial.candidate.nct_id}/${judgment.criterion_id} raises no alert on live data`,
      );
    }
  }

  // Abstentions that cite nothing must stay neutral.
  for (const reference of abstentions) {
    const [nctId, criterionId] = reference.split("/");
    const judgment = live.results
      .find((t) => t.candidate.nct_id === nctId)
      .judgment.judgments.find((j) => j.criterion_id === criterionId);
    const html = render(
      React.createElement(NotePanel, {
        note,
        highlighted: judgment.evidence_sentence_ids,
        caption: null,
        rawFallback: "",
      }),
    );
    g.check(
      !html.includes("Invalid citation") && !html.includes("evidence-on"),
      `${reference}: an uncited abstention highlights nothing and raises no alert`,
    );
  }

  // One rule table per trial, against the live note.
  for (const trial of live.results) {
    const html = render(
      React.createElement(TrialResults, {
        results: live.results,
        openNctId: trial.candidate.nct_id,
        onToggleTrial: () => {},
        selectedRuleKey: null,
        onSelectRule: () => {},
        validSentenceIds: valid,
      }),
    );
    const chip = ELIGIBILITY_CATEGORIES[trial.judgment.category];
    g.check(
      html.includes(`>${chip.text}<`),
      `${trial.candidate.nct_id}: verdict chip reads "${chip.text}" (${trial.judgment.category})`,
    );
    g.check(
      !html.includes("Invalid citation"),
      `${trial.candidate.nct_id}: no citation flagged against the live note`,
    );
    for (const judgment of trial.judgment.judgments) {
      const rows = html.split("<tr").filter((row) => row.includes(`>${judgment.criterion_id}<`));
      g.check(
        rows.length === 1,
        `${trial.candidate.nct_id}/${judgment.criterion_id}: exactly one rule row`,
      );
    }
  }

  // Selected-row wiring on a live rule.
  const cited = live.results
    .flatMap((t) => t.judgment.judgments.map((j) => [t, j]))
    .find(([, j]) => j.evidence_sentence_ids.length > 0);
  if (cited) {
    const [trial, judgment] = cited;
    const html = render(
      React.createElement(TrialResults, {
        results: live.results,
        openNctId: trial.candidate.nct_id,
        onToggleTrial: () => {},
        selectedRuleKey: ruleKey(trial.candidate.nct_id, judgment.criterion_id),
        onSelectRule: () => {},
        validSentenceIds: valid,
      }),
    );
    g.check(html.includes('aria-pressed="true"'), "the selected live rule row reports itself pressed");
    g.check(
      html.includes(`Evidence: sentence ${judgment.evidence_sentence_ids.join(", ")}`),
      "the live rule row names the sentences it cites",
    );
  }

  if (live.profile) {
    const html = render(React.createElement(ProfileCard, { profile: live.profile }));
    g.check(
      live.profile.age === null || html.includes(String(live.profile.age)),
      `the live profile renders (age ${live.profile.age}, ${live.profile.sex})`,
    );
    for (const biomarker of live.profile.biomarkers) {
      g.check(html.includes(biomarker), `live biomarker rendered: ${biomarker}`);
    }
  }

  return g.state;
}

module.exports = { run };
