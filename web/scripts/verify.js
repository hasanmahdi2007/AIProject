"use strict";

/**
 * `npm run verify` - the offline checks.
 * `npm run verify:live` - the same, plus the two that need a gateway.
 *
 * Why this exists next to `npm run build`: the build proves the console
 * compiles, not that it is still right about the server. The contract check is
 * the one that cannot be replaced by a type checker - `beacon_core.schemas` is
 * frozen Python mirrored by hand in `src/lib/types.ts`, and a field renamed on
 * the Python side leaves valid TypeScript that renders `undefined`.
 *
 * The live checks skip cleanly with one explanatory line and a zero exit when
 * no gateway is reachable. An absent optional dependency is not a failure, and
 * a verification script that stack-traces instead of skipping teaches people
 * to ignore it.
 */

const {
  BUILD_DIR,
  GATEWAY_BASE,
  compile,
  gatewayReachable,
  installAlias,
} = require("./harness");

const fs = require("node:fs");

const LIVE = process.argv.includes("--live");

async function main() {
  const root = compile();
  installAlias(root);

  const states = [];
  const section = (title) => console.log(`\n--- ${title} ---`);

  section("contract mirrors vs the frozen Pydantic models");
  states.push(require("./check-contract").run());

  section("fixture integrity");
  states.push(require("./check-fixture").run(root));

  section("component rendering and the evidence highlight");
  states.push(require("./check-render").run(root));

  if (LIVE) {
    const reachable = await gatewayReachable();
    if (!reachable) {
      section("live checks");
      console.log(`  SKIP  live checks (no gateway at ${GATEWAY_BASE})`);
      console.log(
        "        start one, or set NEXT_PUBLIC_GATEWAY_URL, then rerun npm run verify:live",
      );
    } else {
      section(`live payload rendering (${GATEWAY_BASE})`);
      states.push(await require("./check-live").run(root));

      section(`api client (${GATEWAY_BASE})`);
      states.push(await require("./check-client").run(root));
    }
  }

  const passed = states.reduce((sum, s) => sum + s.passed, 0);
  const failed = states.reduce((sum, s) => sum + s.failed, 0);
  const skipped = states.reduce((sum, s) => sum + s.skipped, 0);

  console.log(
    `\n${failed === 0 ? "OK" : "FAILED"}: ${passed} passed, ${failed} failed, ${skipped} skipped` +
      (LIVE ? "" : " (offline checks only; npm run verify:live adds the gateway checks)"),
  );
  if (failed > 0) {
    for (const state of states) {
      for (const label of state.failures) console.log(`  failed: ${label}`);
    }
  }

  fs.rmSync(BUILD_DIR, { recursive: true, force: true });
  // Set the code rather than exiting: an immediate exit while libuv still holds
  // an async handle from the live checks aborts the process on Windows.
  process.exitCode = failed === 0 ? 0 : 1;
}

main().catch((error) => {
  console.error(`\nverify crashed: ${error && error.stack}`);
  process.exitCode = 1;
});
