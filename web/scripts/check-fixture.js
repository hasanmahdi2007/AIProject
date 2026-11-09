"use strict";

/**
 * Asserts the committed fixture still has the properties the console's
 * features are demonstrated with.
 *
 * The fixture is not decoration: it is what makes the console reviewable while
 * the pipeline is incomplete, and a quiet edit that broke one of these would
 * make a feature look broken that is not. In particular every citation must
 * resolve - a fixture with a dangling `evidence_sentence_id` would trip the
 * console's own invalid-citation alert and make the demo look like a defect.
 */

const { group, load } = require("./harness");

function run(root) {
  const g = group("fixture integrity");
  const { FIXTURE_RESULT, FIXTURE_PROGRESS } = load(root, "lib/fixture.js");
  const types = load(root, "lib/types.js");

  const note = FIXTURE_RESULT.note;
  g.check(note !== null, "the fixture carries a numbered note");
  if (note === null) return g.state;

  const valid = new Set(note.sentences.map((s) => s.sentence_id));
  g.log(`note: ${note.sentences.length} sentences, ids ${[...valid].join(",")}`);
  g.check(
    FIXTURE_RESULT.trial_count === FIXTURE_RESULT.results.length,
    `trial_count (${FIXTURE_RESULT.trial_count}) equals results.length (${FIXTURE_RESULT.results.length})`,
  );

  const unresolvable = [];
  const labels = new Set();
  const categories = new Set();
  let rules = 0;
  let citations = 0;

  for (const trial of FIXTURE_RESULT.results) {
    categories.add(trial.judgment.category);
    for (const judgment of trial.judgment.judgments) {
      rules += 1;
      labels.add(judgment.label);
      for (const id of judgment.evidence_sentence_ids) {
        citations += 1;
        if (!valid.has(id)) {
          unresolvable.push(
            `${trial.candidate.nct_id}/${judgment.criterion_id} -> ${id}`,
          );
        }
      }
    }
  }
  g.log(`${FIXTURE_RESULT.results.length} trials, ${rules} rules, ${citations} citations`);

  g.check(
    unresolvable.length === 0,
    "every evidence_sentence_id resolves to a real sentence",
    unresolvable.join("; "),
  );

  // All four verdicts must appear, or the rule table has an unexercised state.
  for (const label of ["met", "not_met", "insufficient_info", "not_applicable"]) {
    g.check(labels.has(label), `a rule labelled ${label} is present`);
  }

  // Abstention has two shapes and the console words them differently.
  const abstentions = FIXTURE_RESULT.results.flatMap((trial) =>
    trial.judgment.judgments.filter((j) => j.label === "insufficient_info"),
  );
  g.check(
    abstentions.some((j) => j.evidence_sentence_ids.length === 0),
    "an insufficient_info rule that cites nothing (pure abstention)",
  );
  g.check(
    abstentions.some((j) => j.evidence_sentence_ids.length > 0),
    "an insufficient_info rule that still cites a sentence",
  );

  g.check(
    FIXTURE_RESULT.results.some((trial) =>
      trial.judgment.judgments.some((j) => j.evidence_sentence_ids.length > 1),
    ),
    "a rule citing more than one sentence, for the multi-sentence highlight",
  );

  g.log(`categories: ${[...categories].join(", ")}`);
  g.check(categories.has("insufficient_info"), "a trial whose verdict is insufficient_info");
  g.check(categories.has("excluded"), "a trial whose verdict is excluded");
  g.check(
    FIXTURE_RESULT.results.some((trial) => trial.judgment.missing_facts.length > 0),
    "at least one trial carries a missing-facts list",
  );

  // The gateway can never populate these two, so the fixture must not either -
  // otherwise the console could grow a dependency on a field live data lacks.
  g.check(
    FIXTURE_RESULT.results.every(
      (trial) =>
        trial.candidate.dense_score === null && trial.candidate.sparse_score === null,
    ),
    "dense_score and sparse_score are null, as live data always has them",
  );

  const replayed = FIXTURE_PROGRESS.map((p) => p.step);
  g.log(`progress replay: ${replayed.join(" -> ")}`);
  g.check(
    JSON.stringify(replayed) === JSON.stringify([...types.PIPELINE_STEPS]),
    `the replay covers all ${types.PIPELINE_STEPS.length} pipeline steps in contract order`,
  );
  g.check(
    Object.keys(types.STEP_LABELS).length === types.PIPELINE_STEPS.length,
    "every pipeline step has a display label",
  );

  return g.state;
}

module.exports = { run };
