---
name: Beacon console
description: Clinical-trial matching with its reasoning shown, on a drenched NHS-blue working frame.
colors:
  nhs-blue: "#005EB8"
  nhs-dark: "#003087"
  nhs-panel: "#E8EDEE"
  nhs-pale: "#D8DDE0"
  nhs-line: "#AEB7BD"
  nhs-unknown: "#AEB7BD"
  nhs-ink: "#212B32"
  nhs-mid: "#4C6272"
  white: "#FFFFFF"
  nhs-green: "#007F3B"
  green-hover: "#00662F"
  green-disabled: "#7FA58F"
  nhs-greendark: "#00401E"
  nhs-lightgreen: "#78BE20"
  nhs-red: "#D5281B"
  red-pale: "#FBE7E5"
  nhs-purple: "#330072"
  nhs-aqua: "#41B6E6"
  nhs-aquapale: "#D2EEFA"
  aqua-ink: "#00445E"
  nhs-warm: "#FFB81C"
  warm-pale: "#FFF8E1"
  nhs-focus: "#FFEB3B"
typography:
  display:
    fontFamily: "Noto Sans, system-ui, Segoe UI, sans-serif"
    fontSize: "40px"
    fontWeight: 800
    lineHeight: "48px"
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "Noto Sans, system-ui, Segoe UI, sans-serif"
    fontSize: "22px"
    fontWeight: 700
    lineHeight: "32px"
  title:
    fontFamily: "Noto Sans, system-ui, Segoe UI, sans-serif"
    fontSize: "17px"
    fontWeight: 700
    lineHeight: "24px"
  body:
    fontFamily: "Noto Sans, system-ui, Segoe UI, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: "24px"
  body-small:
    fontFamily: "Noto Sans, system-ui, Segoe UI, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: "24px"
  label:
    fontFamily: "Noto Sans, system-ui, Segoe UI, sans-serif"
    fontSize: "13px"
    fontWeight: 700
    lineHeight: "20px"
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, Consolas, monospace"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: "20px"
rounded:
  sm: "2px"
  base: "4px"
  md: "6px"
  full: "9999px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "20px"
  xl: "24px"
  2xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.nhs-green}"
    textColor: "{colors.white}"
    typography: "{typography.title}"
    rounded: "{rounded.base}"
    padding: "10px 20px"
  button-primary-hover:
    backgroundColor: "{colors.green-hover}"
  button-primary-disabled:
    backgroundColor: "{colors.green-disabled}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.nhs-blue}"
    typography: "{typography.body}"
    rounded: "{rounded.base}"
    padding: "8px 16px"
  button-secondary-hover:
    backgroundColor: "{colors.nhs-blue}"
    textColor: "{colors.white}"
  field:
    backgroundColor: "{colors.white}"
    textColor: "{colors.nhs-ink}"
    rounded: "{rounded.base}"
    padding: "8px 12px"
  card:
    backgroundColor: "{colors.nhs-panel}"
    textColor: "{colors.nhs-ink}"
    rounded: "{rounded.md}"
    padding: "20px"
  tag-met:
    backgroundColor: "{colors.nhs-green}"
    textColor: "{colors.white}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "2px 8px"
  tag-likely:
    backgroundColor: "{colors.nhs-lightgreen}"
    textColor: "{colors.nhs-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "2px 8px"
  tag-insufficient:
    backgroundColor: "{colors.nhs-unknown}"
    textColor: "{colors.nhs-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "2px 8px"
  tag-not-met:
    backgroundColor: "{colors.nhs-red}"
    textColor: "{colors.white}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "2px 8px"
  tag-neutral:
    backgroundColor: "{colors.nhs-pale}"
    textColor: "{colors.nhs-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "2px 8px"
  tag-citation:
    backgroundColor: "{colors.nhs-aquapale}"
    textColor: "{colors.aqua-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.base}"
    padding: "2px 8px"
  tag-citation-selected:
    backgroundColor: "{colors.nhs-aqua}"
    textColor: "{colors.nhs-ink}"
  header:
    backgroundColor: "{colors.nhs-dark}"
    textColor: "{colors.white}"
  nav-link:
    textColor: "{colors.white}"
    typography: "{typography.title}"
    padding: "10px 12px 8px"
  disclaimer:
    backgroundColor: "{colors.warm-pale}"
    textColor: "{colors.nhs-ink}"
    typography: "{typography.body-small}"
    padding: "12px 16px"
---

# Design System: Beacon console

## Overview

**Creative North Star: "The Blue Working Frame"**

The whole page is drenched in NHS blue, and the work sits on it as pale grey panels: a note to paste, a pipeline to watch, a ranked list of care cards to read. Nothing floats on white. The blue is the room, the panels are the paper on the desk, and every other colour on screen means something specific: a verdict, a citation, a warning, or where the keyboard is.

The register is a public-service working tool, not a SaaS dashboard and not a dark AI console. Type is plain Noto Sans, bold where it labels and regular where it explains, with sentence-case headings and tabular figures for every rank, score and percentage. Density is moderate: panels carry generous 20 to 24px padding, while the trial list is tight enough to scan twenty results by colour band, rank and score in one pass.

The system's signature is the citation link: a rule in a trial card carries a pale-aqua reference tag, and selecting it lights the cited sentence in the numbered note with the same aqua. Aqua is spent on nothing else, which is why the eye finds it instantly.

**Key Characteristics:**
- NHS-blue page ground, NHS dark-blue header, pale-grey panels; white only inside inputs.
- Verdict colour is semantic and fixed: green eligible / met, light green likely eligible, neutral grey insufficient info, red excluded / not met.
- Aqua is reserved for citations and the selected rule or trial.
- NHS.UK frontend conventions carried verbatim: the green button with a pressed bottom edge, and the yellow focus state with a dark underline.
- Square-ish 4px corners; no gradients, no glow, no glass.

## Colors

A saturated institutional blue field with a strictly semantic accent set; colour is information, never decoration.

### Primary
- **NHS Blue** (nhs-blue): the page ground behind everything; also IDs and links on panels, sentence-number badges, the running pipeline bar, and the secondary button's stroke.
- **NHS Dark Blue** (nhs-dark): the header bar, headings on panels, the wordmark text, inclusion-rule labels, and the cost tag.
### Secondary (verdict palette)
- **NHS Green** (nhs-green): eligible, met, completed, and the primary action. Hover deepens to Green Hover; disabled washes to Green Disabled.
- **NHS Dark Green** (nhs-greendark): only the primary button's pressed edge.
- **NHS Light Green** (nhs-lightgreen): likely eligible; always carries ink text, never white.
- **NHS Red** (nhs-red): not met, excluded, failed runs, invalid citations. Its pale tint (red-pale) backs error callouts and the error code chip.
- **NHS Purple** (nhs-purple): exclusion-rule labels and prior-treatment chips.

### Tertiary (citation and attention)
- **NHS Aqua** (nhs-aqua): the selected citation tag, the inset stroke of the evidence highlight, the selected sentence badge, and the ring on the open trial card.
- **Aqua Pale** (nhs-aquapale): the evidence highlight fill and the resting citation tag, with Aqua Ink text.
- **Warm Yellow** (nhs-warm): the disclaimer's 4px top edge, over Warm Pale.
- **Focus Yellow** (nhs-focus): keyboard focus fill and text selection.

### Neutral
- **Panel Grey** (nhs-panel): every card and panel surface.
- **Recessed Grey** (nhs-pale): recessed areas inside panels (the run-lookup tray, status messages, missing-fact rows, neutral tags, hover on rows).
- **Hairline Grey** (nhs-line): dividers on panels and the empty track of progress bars.
- **Abstention Grey** (nhs-unknown): the insufficient-info band and tag. Same value as the hairline on purpose: it reads as absence of a verdict, not a signal.
- **Ink** (nhs-ink): all text on panels.
- **Mid Slate** (nhs-mid): secondary text, captions, column headings.
- **White**: text on the blue ground and on solid verdict colours; the only surface allowed to be white is an input.

### Named Rules
**The Never-White Rule.** Panels are Panel Grey on the blue ground. Only text inputs and textareas are white; no card, section or modal is.

**The Abstention Rule.** Insufficient info is neutral grey with ink text and a thin slate inset ring. Never amber, never red, never a warning glyph. It is a designed outcome, not a failure.

**The Reserved Aqua Rule.** Aqua means "this is cited" or "this is selected". It does not decorate, label or highlight anything else.

**The One Warning Edge Rule.** Warm yellow belongs to the disclaimer. It is the only standing warning on the page.

## Typography

**Body Font:** Noto Sans (400, 600, 700, 800), with system-ui and Segoe UI fallback
**Label/Mono Font:** JetBrains Mono, with ui-monospace and Consolas fallback

**Character:** One humanist sans doing all the work, in the NHS.UK manner: heavy and tight for the page title, bold for every heading and label, plain for prose. The mono face is a data voice only: NCT ids, rule ids, match ids, lab values, the gateway URL.

### Hierarchy
- **Display** (800, 40px / 48px, tracking -0.025em): the page title on the blue ground, white. The about page steps it down to 32px / 40px.
- **Headline** (700, 20 to 22px): section headings on the blue ground in white ("Ranked trials", about-page sections), and the patient-summary lead sentence in ink on its panel.
- **Title** (700, 17px / 24px): panel headings in NHS Dark Blue; 15px inside an open trial card. Primary button and nav links use 16px bold.
- **Body** (400, 15px / 24px): default text; long explanations open up to 28px line height and cap at 72ch. Intro paragraphs on blue are 16 to 17px at 90% white.
- **Body Small** (400, 14px / 24px): captions, rule explanations, missing facts, callout copy.
- **Label** (700, 13px / 20px): tags, column headings, definition-list terms, step numbers.
- **Mono** (400 to 700, 13 to 14px): identifiers and raw data only.

### Named Rules
**The Sentence-Case Heading Rule.** Headings are bold sentence case. There is no tracked, uppercase label above a heading anywhere in the system.

**The Tabular Figures Rule.** Every rank, score, percentage, count, cost and duration uses tabular numerals so columns of numbers align.

## Layout

A single centred container (max 1280px) with 16px side gutters, 24px from 640px. The header and disclaimer share that container; main content starts 32px below the disclaimer and ends with 64px of bottom room.

The console is one column on small screens and becomes a two-column grid from 1024px, split 5 : 7 with a 24px gap. The left column holds the note input panel and, once a run has a result, the numbered note, which sticks 16px from the top so the cited sentence stays visible while the right column scrolls. The right column stacks pipeline, callouts, patient summary and the ranked list, 24px apart.

The ranked list is a single vertical stack of care cards 12px apart. Excluded trials run after the open ones, separated by 32px and a 30%-white hairline; the split never re-sorts. Inside cards, padding is 16px; inside panels, 20px rising to 24px from 640px. The rule table collapses from a fixed-column table into stacked blocks below 640px. The pipeline steps run 1 column, then 2 from 640px, then a 7-across row from 1024px.

## Elevation & Depth

Depth is almost flat: panels lift off the blue ground with one soft, blue-tinted shadow, and that is the only ambient shadow in the system. Everything inside a panel is tonal (recessed grey trays, hairlines). The other two shadows are not elevation at all; they are hard, zero-blur edges borrowed from NHS.UK frontend: the primary button's pressed edge and the focus underline.

### Shadow Vocabulary
- **Panel lift** (`box-shadow: 0 1px 2px rgba(0, 32, 80, 0.18), 0 10px 28px -14px rgba(0, 32, 80, 0.45)`): every card, the disclaimer, and standalone callouts on the blue ground.
- **Button edge** (`box-shadow: 0 4px 0 #00401E`, pressed `0 2px 0 #00401E`): the primary button only.
- **Focus underline** (`box-shadow: 0 4px 0 #212B32` under a 3px yellow outline): every focusable element.
- **Evidence stroke** (`box-shadow: inset 0 0 0 2px #41B6E6`): the highlighted note sentence and the selected rule row.

### Named Rules
**The One Lift Rule.** Panels get the panel lift and nothing else does. No glow, no layered elevations, no backdrop blur.

## Shapes

Square-ish and calm. Controls, tags, fields and rows use a 4px corner; cards and panels a slightly softer 6px; the wordmark plate is nearly square at 2px. Full rounding is reserved for progress bars, pipeline step dots and missing-fact bullets. Edges that carry meaning are thick and straight: a 4px top edge marks a callout's kind (warm for the disclaimer, red for errors), a 4px bottom border marks the current nav link, and 2px strokes outline fields and the secondary button.

## Components

### Buttons
Tactile and plain, straight from NHS.UK.
- **Shape:** gently squared (4px).
- **Primary:** NHS Green with white 16px bold text, 10px by 20px padding, sitting on a 4px NHS Dark Green edge. Hover deepens the green; pressing moves the button down 2px and shortens the edge to 2px. Disabled washes to Green Disabled with no edge.
- **Secondary:** transparent with a 2px NHS Blue stroke and blue 15px bold text, 8px by 16px. Hover fills blue with white text. Disabled turns the stroke hairline grey and the text slate.
- **Focus:** the system focus (below).

### Chips / Tags
- **Style:** 13px bold, 4px corners, 2px by 8px, no wrapping. Solid verdict colours for verdicts and run status; Recessed Grey for phase, status and queued; Aqua Pale with Aqua Ink for a citation.
- **State:** a citation tag turns solid aqua when its rule is selected. Insufficient info and not applicable add a 1px inset ring (slate and hairline respectively) so the grey tags hold their shape on a grey panel.
- **Profile chips:** 14px semibold, tinted by kind (pale blue conditions, aqua-pale biomarkers, pale purple prior treatments, mono labs on Recessed Grey).

### Cards / Containers
- **Corner Style:** 6px.
- **Background:** Panel Grey, ink text.
- **Shadow Strategy:** the panel lift.
- **Border:** none outside; Hairline Grey dividers inside.
- **Internal Padding:** 20px, 24px from 640px; trial cards 16px.

### Inputs / Fields
- **Style:** white, 2px ink stroke, 4px corners, slate placeholder, NHS Blue caret. Match-id input is mono 13px with a sans placeholder.
- **Focus:** 3px Focus Yellow outline plus a 1px inset ink ring.

### Navigation
Header bar in NHS Dark Blue. The wordmark "Beacon" is 22px extrabold dark blue on a white 2px-cornered plate, followed by a white tagline. Nav links are 16px bold white at 85% opacity with a 4px transparent bottom border; hover brings a 50% white border, the current page a solid white border and full white text.

### Focus
One focus state for everything: 3px Focus Yellow outline with no offset and a 4px dark ink underline. Text selection uses the same yellow with ink text.

### Trial care card (signature)
A full-width band in the verdict colour carries rank, verdict word and score (13 to 14px bold, tabular). Below it: the NCT id in bold mono NHS Blue, phase and status tags, the title in 16px bold ink, and a 112px rules-confirmed bar (green on hairline) with its counts. A chevron in NHS Blue turns 180 degrees in 200ms when open. Hover adds a 2px 70%-white ring; the open card wears a 4px aqua ring. Excluded cards collapse to a compact band and a single truncated title line until opened.

### Evidence highlight (signature)
The numbered note lists each sentence with a 24px NHS Blue number badge. Selecting a rule fills its cited sentences with Aqua Pale under a 2px aqua inset stroke, switches the badge to solid aqua with ink text, sets the sentence semibold, and scrolls the first one to centre (instantly under reduced motion). The selected rule row in the trial card wears the same highlight, so both ends of the citation match.

### Disclaimer
A calm warning callout, not an alarm: Warm Pale fill, 4px Warm Yellow top edge, panel lift, 14px ink text with bold lead-in. Rendered from the layout on every page.

### Pipeline rail
Seven numbered 28px dots: pending is Recessed Grey with a hairline ring, current is NHS Blue with a 4px half-aqua ring, done is NHS Green with a check, failed is NHS Red. An 8px bar above fills blue while running (700ms ease-out width transition), green on completion, red on failure.

## Do's and Don'ts

### Do:
- **Do** put every panel in Panel Grey on the NHS Blue ground, with the panel lift and 6px corners.
- **Do** draw verdicts only in their fixed colours: green eligible / met, light green with ink text for likely eligible, neutral grey with an inset slate ring for insufficient info, red excluded / not met.
- **Do** use the NHS.UK primary button: green, white bold text, `0 4px 0 #00401E` edge that shortens to 2px when pressed.
- **Do** use the single NHS focus state (3px yellow outline, 4px dark underline) on every focusable element.
- **Do** set ids and raw data in JetBrains Mono and every number in tabular figures.
- **Do** write headings as bold sentence case in NHS Dark Blue on panels, white on the blue ground.
- **Do** honour reduced motion: transitions collapse and the note scrolls without smoothing.

### Don't:
- **Don't** make a card, panel or section white; only inputs are white.
- **Don't** colour insufficient info amber or red, or give it a warning glyph.
- **Don't** use aqua for anything other than citations and the selected rule or trial.
- **Don't** use warm yellow outside the disclaimer.
- **Don't** add gradients, glow, glass or backdrop blur.
- **Don't** put a tracked uppercase label above a heading.
- **Don't** add a second ambient shadow or a hard offset shadow beyond the button edge and the focus underline.
