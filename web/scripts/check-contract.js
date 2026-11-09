"use strict";

/**
 * Diffs the TypeScript mirrors in `src/lib/types.ts` against the Pydantic
 * models they transcribe, field name by field name.
 *
 * This is the check the other two cannot replace. `beacon_core.schemas` is
 * frozen and authoritative, `types.ts` is a hand transcription of it, and a
 * field renamed or removed on the Python side stays *valid TypeScript*: the
 * JSON still parses, the property is simply `undefined`, and the console
 * renders a blank panel. Neither `tsc` nor `next build` can see that, because
 * nothing in the TypeScript is wrong - it is only wrong about the server.
 *
 * So this walks both sides and insists they have exactly the same field names,
 * and that every `Literal` member is in the matching TS union. It found nothing
 * the day it was written. That is what a guard looks like before the day it
 * earns its keep.
 */

const fs = require("node:fs");
const path = require("node:path");
const { REPO_ROOT, WEB_ROOT, group } = require("./harness");

const CONTRACT_SOURCE = path.join(
  REPO_ROOT,
  "libs/beacon-core/src/beacon_core/schemas.py",
);
const GATEWAY_SOURCE = path.join(
  REPO_ROOT,
  "services/gateway/src/gateway/schemas.py",
);
const TYPES_SOURCE = path.join(WEB_ROOT, "src/lib/types.ts");

/** Field names of one Pydantic class body. */
function pythonFields(source, className) {
  const start = source.indexOf(`class ${className}(`);
  if (start === -1) throw new Error(`no class ${className} in the contract`);
  const rest = source.slice(start);
  const nextClass = rest.slice(1).search(/\nclass \w+\(/);
  const body = nextClass === -1 ? rest : rest.slice(0, nextClass + 1);

  const names = [];
  let inDocstring = false;
  for (const line of body.split("\n")) {
    // Prose inside a docstring can look exactly like an annotation - the real
    // PatientProfile docstring contains "numbers: every numeric eligibility
    // decision...", which a naive regex reports as a ninth field. Track the
    // quoting instead of guessing.
    const quoteCount = (line.match(/"""/g) || []).length;
    const wasInDocstring = inDocstring;
    if (quoteCount % 2 === 1) inDocstring = !inDocstring;
    if (wasInDocstring || quoteCount > 0) continue;

    const match = /^ {4}([a-z_][a-z0-9_]*)\s*:\s*[^=\n]/i.exec(line);
    if (match && !match[1].startsWith("model_")) names.push(match[1]);
  }
  return names;
}

/** Field names of one exported TS interface. */
function tsFields(source, name) {
  const start = source.indexOf(`export interface ${name} {`);
  if (start === -1) throw new Error(`no interface ${name} in types.ts`);
  const body = source.slice(start, start + source.slice(start).indexOf("\n}"));

  const names = [];
  for (const line of body.split("\n")) {
    const match = /^ {2}([a-z_][a-z0-9_]*)\??\s*:/i.exec(line);
    if (match) names.push(match[1]);
  }
  return names;
}

/** Members of a Python `Literal[...]` alias. */
function pythonLiteral(source, name) {
  const match = new RegExp(`${name} = Literal\\[([^\\]]*)\\]`, "s").exec(source);
  if (!match) throw new Error(`no Literal ${name} in the contract`);
  return [...match[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]);
}

/** Members of the matching TS union, in declaration order. */
function tsUnion(source, name) {
  const start = source.indexOf(`export type ${name} =`);
  if (start === -1) throw new Error(`no type ${name} in types.ts`);
  const block = source.slice(start);
  const declaration = block.slice(0, block.indexOf(";"));
  return [...declaration.matchAll(/"([^"]+)"/g)].map((m) => m[1]);
}

/** Members of a `readonly` array constant, in order. */
function tsConstArray(source, name) {
  const start = source.indexOf(name);
  if (start === -1) throw new Error(`no constant ${name} in types.ts`);
  const block = source.slice(start);
  return [...block.slice(0, block.indexOf("];")).matchAll(/"([^"]+)"/g)].map(
    (m) => m[1],
  );
}

/** Classes to compare, as [python source key, class name]. */
const DOMAIN_MODELS = [
  "NoteSentence",
  "NumberedNote",
  "PatientProfile",
  "TrialCandidate",
  "CriterionJudgment",
  "TrialJudgment",
  "RankedTrial",
];

const WIRE_MODELS = ["MatchAccepted", "MatchStatusSnapshot", "MatchResult"];

const VOCABULARIES = [
  "Sex",
  "CriterionKind",
  "CriterionLabel",
  "EligibilityCategory",
  "MatchStatus",
  "PipelineStep",
];

function run() {
  const g = group("contract mirrors");

  const missing = [CONTRACT_SOURCE, GATEWAY_SOURCE].filter(
    (file) => !fs.existsSync(file),
  );
  if (missing.length > 0) {
    g.skip(
      "contract mirrors",
      `Python contracts not found at ${missing.join(", ")} - run this from a full checkout`,
    );
    return g.state;
  }

  const contract = fs.readFileSync(CONTRACT_SOURCE, "utf8");
  const gateway = fs.readFileSync(GATEWAY_SOURCE, "utf8");
  const types = fs.readFileSync(TYPES_SOURCE, "utf8");

  const compare = (label, python, ts) => {
    const absent = python.filter((f) => !ts.includes(f));
    const invented = ts.filter((f) => !python.includes(f));
    g.check(
      absent.length === 0 && invented.length === 0,
      `${label} (${python.length} fields)`,
      `missing in TS: [${absent}] / not in the contract: [${invented}]`,
    );
  };

  for (const name of DOMAIN_MODELS) {
    compare(name, pythonFields(contract, name), tsFields(types, name));
  }
  for (const name of WIRE_MODELS) {
    compare(name, pythonFields(gateway, name), tsFields(types, name));
  }

  for (const name of VOCABULARIES) {
    const python = pythonLiteral(contract, name);
    const ts = tsUnion(types, name);
    g.check(
      JSON.stringify(python) === JSON.stringify(ts),
      `${name} union matches the contract (${python.length} values)`,
      `contract [${python}] vs TS [${ts}]`,
    );
  }

  // Order matters here, not just membership: the progress rail draws the steps
  // in this order, so a reordered contract would render a misleading pipeline.
  const steps = pythonLiteral(contract, "PipelineStep");
  g.check(
    JSON.stringify(steps) === JSON.stringify(tsConstArray(types, "PIPELINE_STEPS")),
    `PIPELINE_STEPS lists the contract's ${steps.length} steps in order`,
    `contract [${steps}]`,
  );

  return g.state;
}

module.exports = { run };
