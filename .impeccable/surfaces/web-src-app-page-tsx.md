---
version: 1
slug: "web-src-app-page-tsx"
primary_target: "web/src/app/page.tsx"
related_targets: ["web/src/app/about/page.tsx"]
---

Scope: the Beacon console (/) and "How to read this" (/about). Visitor mode: Operate.
Audience: interviewers and engineers judging an AI-engineering portfolio; they paste a note, watch the run, read the cited results.
Job: understand within one look that every verdict is checked rule by rule and cites the note sentence it relies on.
Constraints: keep all copy, behaviour, aria semantics and the layout-rendered disclaimer; restyle only.
User feedback: rejected white-dominant screens twice; chose NHS Blue in round 3 and asked to make the sketch "a bit better, it's kinda messy".

## Direction contract

THESIS: A drenched NHS-blue working frame where results read as a calm ranked list of care cards and every rule carries a reference tag to its note sentence; it refuses the white SaaS dashboard and the dark neon AI console.

OWN-WORLD: Page ground NHS blue #005EB8, header NHS dark blue #003087, panels pale grey #E8EDEE (white-free; only inputs are white), ink #212B32, verdict colours eligible green #007F3B, likely eligible light green #78BE20, insufficient info neutral grey #AEB7BD (an abstention, never a warning), excluded red #D5281B, aqua #41B6E6 for citations and selection, warm yellow #FFB81C only for the disclaimer; Noto Sans, bold headings, tabular figures; square-ish 4px corners, no gradients, no glow, no glass.

STORY: The visitor pastes a note, watches seven steps complete, sees the patient summarised, then scans a ranked list where verdict colour, rank and score line up; opening a trial shows each rule with its label and a note reference tag that lights the cited sentence on the left.

FIRST VIEWPORT: Dark-blue header bar with the Beacon wordmark and nav; the disclaimer as a yellow-edged warning callout; then a two-column grid on the blue ground: left (4/12) the note panel, sticky; right (8/12) the run status, then results as one vertical list of care-card rows (colour band with verdict word, rank, NCT id and title, rules-confirmed bar, score), excluded trials grouped after the open ones.

FORM: NHS Blue, grounded candidate 2 of the safer round-3 lineup (user-chosen). seed:763b2de6c9f73206

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
