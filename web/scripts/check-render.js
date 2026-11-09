"use strict";

/**
 * Renders the real components and asserts the evidence highlight.
 *
 * The highlight is the feature that makes Beacon's reasoning checkable rather
 * than a black box, so "it compiles" is not evidence that it works. These
 * checks render `NotePanel` and `TrialResults` with `react-dom/server` and
 * inspect which sentence actually carries the highlight class.
 *
 * What they cannot cover: the click itself. `app/page.tsx` turns a click into
 * the `highlighted` prop, and that wiring is checked by the type checker only.
 * Everything downstream of the prop is checked here.
 */

const { group, listItems, load, render } = require("./harness");

function highlightedIds(html) {
  return listItems(html)
    .map((item, index) => (item.includes("evidence-on") ? index + 1 : null))
    .filter((value) => value !== null);
}

function run(root) {
  const g = group("component rendering");
  const React = require("react");
  const { NotePanel } = load(root, "components/NotePanel.js");
  const { TrialResults, ruleKey } = load(root, "components/TrialResults.js");
  const { FIXTURE_RESULT } = load(root, "lib/fixture.js");

  const note = FIXTURE_RESULT.note;
  const valid = new Set(note.sentences.map((s) => s.sentence_id));
  const trial = FIXTURE_RESULT.results[0];

  const panel = (props) => render(React.createElement(NotePanel, props));
  const table = (props) => render(React.createElement(TrialResults, props));

  // ------------------------------------------------------ the highlight --
  const single = trial.judgment.judgments.find(
    (j) => j.evidence_sentence_ids.length === 1,
  );
  const singleHtml = panel({
    note,
    highlighted: single.evidence_sentence_ids,
    caption: "test",
    rawFallback: "",
  });
  g.check(
    listItems(singleHtml).length === note.sentences.length,
    `all ${note.sentences.length} sentences render`,
  );
  g.check(
    JSON.stringify(highlightedIds(singleHtml)) ===
      JSON.stringify(single.evidence_sentence_ids),
    `rule ${single.criterion_id} highlights exactly sentence ${single.evidence_sentence_ids.join(", ")}`,
  );
  g.check(
    singleHtml.includes('aria-current="true"'),
    "the highlighted sentence is marked aria-current",
  );
  g.check(
    singleHtml.includes(
      note.sentences.find((s) => s.sentence_id === single.evidence_sentence_ids[0]).text,
    ),
    "the highlighted sentence's own text is on the page",
  );

  const multi = FIXTURE_RESULT.results
    .flatMap((t) => t.judgment.judgments)
    .find((j) => j.evidence_sentence_ids.length > 1);
  g.check(
    JSON.stringify(
      highlightedIds(panel({ note, highlighted: multi.evidence_sentence_ids, caption: null, rawFallback: "" })),
    ) === JSON.stringify(multi.evidence_sentence_ids),
    `a two-sentence citation highlights both [${multi.evidence_sentence_ids.join(",")}]`,
  );

  const idle = panel({ note, highlighted: [], caption: null, rawFallback: "" });
  g.check(!idle.includes("evidence-on"), "nothing is highlighted before a rule is clicked");
  g.check(idle.includes("Click any rule"), "the panel invites the click when idle");

  // --------------------------------- invalid citation vs a note not yet --
  const bogus = panel({ note, highlighted: [9999], caption: null, rawFallback: "" });
  g.check(
    bogus.includes("Invalid citation") && bogus.includes("9999"),
    "a citation outside the note raises the invalid-citation alert, naming the id",
  );
  g.check(bogus.includes('role="alert"'), "the invalid-citation alert is a live region");

  // The distinction that matters most: a null note is a not-yet, and flagging
  // it would paint every in-flight run as a wall of bad citations.
  const pending = panel({
    note: null,
    highlighted: [9999],
    caption: null,
    rawFallback: "the pasted note",
  });
  g.check(
    !pending.includes("Invalid citation"),
    "a NULL note raises no invalid-citation alert (not-yet, not a defect)",
  );
  g.check(
    pending.includes("has not split this note into sentences yet"),
    "a null note is explained as an in-flight state",
  );
  g.check(pending.includes("the pasted note"), "the pasted note shows while unnumbered");

  // ------------------------------------------------------- the rule table --
  const open = table({
    results: FIXTURE_RESULT.results,
    openNctId: trial.candidate.nct_id,
    onToggleTrial: () => {},
    selectedRuleKey: ruleKey(trial.candidate.nct_id, single.criterion_id),
    onSelectRule: () => {},
    validSentenceIds: valid,
  });
  g.check(open.includes(trial.candidate.nct_id), "the open trial's NCT id renders");
  g.check(
    (open.match(/role="button"/g) || []).length >= trial.judgment.judgments.length,
    "every rule row is an activatable control",
  );
  g.check(open.includes('aria-pressed="true"'), "the selected rule row reports itself pressed");
  g.check(
    open.includes("highlighted in the note"),
    "the selected rule row says its evidence is highlighted",
  );
  g.check(
    open.includes("No sentence in the note addresses this rule."),
    "a rule citing nothing is explained neutrally, not as an error",
  );
  for (const text of ["Met", "Not met", "Insufficient info", "Not applicable"]) {
    g.check(open.includes(`>${text}<`), `the "${text}" verdict chip renders`);
  }
  g.check(
    open.includes("gaps in the note, not errors"),
    "missing facts are framed as gaps, not failures",
  );
  g.check(!open.includes("Invalid citation"), "no citation is flagged for an all-valid fixture");

  // Told the note holds only sentences 1-3, the rules citing 4+ must flag,
  // selected or not.
  const narrowed = table({
    results: FIXTURE_RESULT.results,
    openNctId: trial.candidate.nct_id,
    onToggleTrial: () => {},
    selectedRuleKey: null,
    onSelectRule: () => {},
    validSentenceIds: new Set([1, 2, 3]),
  });
  g.check(
    narrowed.includes("Invalid citation"),
    "the rule table flags a citation missing from the note, unselected",
  );

  const unknown = table({
    results: FIXTURE_RESULT.results,
    openNctId: trial.candidate.nct_id,
    onToggleTrial: () => {},
    selectedRuleKey: null,
    onSelectRule: () => {},
    validSentenceIds: null,
  });
  g.check(
    !unknown.includes("Invalid citation"),
    "with no note available the rule table flags nothing (not-yet, not a defect)",
  );

  return g.state;
}

module.exports = { run };
