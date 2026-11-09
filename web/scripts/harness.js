"use strict";

/**
 * Shared plumbing for `npm run verify`.
 *
 * Three jobs, none of which needs a dependency the console does not already
 * have: compile the TypeScript to CommonJS so plain Node can require it, map
 * the `@/` alias that `tsc` leaves in the emitted JS, and count assertions.
 *
 * Deliberately no test framework. The whole point of these checks is that they
 * run with `tsc` and `react-dom/server`, both already installed, so a reviewer
 * can run them without installing anything.
 */

const fs = require("node:fs");
const path = require("node:path");
const Module = require("node:module");
const { spawnSync } = require("node:child_process");

const WEB_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_ROOT, "..");
const BUILD_DIR = path.join(WEB_ROOT, ".verify");

/** Where the compiled mirror of `src/` lands. */
const COMPILED_SRC = path.join(BUILD_DIR, "src");

/**
 * Compiles `src/` to CommonJS once per run.
 *
 * `tsc` is invoked through its own bin rather than `npx` so this behaves the
 * same on Windows and in CI, and resolves the exact TypeScript the project
 * pinned instead of whatever `npx` would fetch.
 */
function compile({ quiet = false } = {}) {
  if (!quiet) process.stdout.write("compiling src/ for verification... ");
  fs.rmSync(BUILD_DIR, { recursive: true, force: true });

  const tsc = require.resolve("typescript/bin/tsc");
  const result = spawnSync(
    process.execPath,
    [
      tsc,
      "-p",
      path.join(WEB_ROOT, "tsconfig.json"),
      "--noEmit",
      "false",
      "--outDir",
      BUILD_DIR,
      "--module",
      "commonjs",
      "--target",
      "es2022",
      "--moduleResolution",
      "node",
      "--jsx",
      "react-jsx",
      "--esModuleInterop",
      "--declaration",
      "false",
      "--incremental",
      "false",
    ],
    { cwd: WEB_ROOT, encoding: "utf8" },
  );

  if (result.status !== 0) {
    if (!quiet) process.stdout.write("FAILED\n");
    console.error(result.stdout || "");
    console.error(result.stderr || "");
    throw new Error("tsc could not compile src/ for verification");
  }
  if (!quiet) process.stdout.write("ok\n");
  return COMPILED_SRC;
}

/**
 * Teaches `require` the `@/` alias.
 *
 * `tsc` type-checks the alias but emits it verbatim, so the compiled files ask
 * for `@/lib/labels` at runtime. Installed once per process.
 */
let aliasInstalled = false;
function installAlias(root) {
  if (aliasInstalled) return;
  aliasInstalled = true;
  const original = Module._resolveFilename;
  Module._resolveFilename = function (request, ...rest) {
    if (request.startsWith("@/")) {
      return original.call(this, path.join(root, request.slice(2)), ...rest);
    }
    return original.call(this, request, ...rest);
  };
}

/** Requires a module from the compiled tree. */
function load(root, relative) {
  return require(path.join(root, relative));
}

/** One check group: counts passes, failures and skips, and prints as it goes. */
function group(title) {
  const state = { title, passed: 0, failed: 0, skipped: 0, failures: [] };

  return {
    state,
    log(message) {
      console.log(`      ${message}`);
    },
    check(ok, label, detail) {
      if (ok) {
        state.passed += 1;
        console.log(`  PASS  ${label}`);
      } else {
        state.failed += 1;
        state.failures.push(label);
        console.log(`  FAIL  ${label}${detail ? ` -> ${detail}` : ""}`);
      }
      return ok;
    },
    skip(label, why) {
      state.skipped += 1;
      console.log(`  SKIP  ${label}${why ? ` (${why})` : ""}`);
    },
  };
}

/** Renders a React element to static markup. */
function render(element) {
  const { renderToStaticMarkup } = require("react-dom/server");
  return renderToStaticMarkup(element);
}

/** The `<li>` blocks of a rendered list, in document order. */
function listItems(html) {
  return html
    .split("<li")
    .slice(1)
    .map((chunk) => `<li${chunk}`);
}

const GATEWAY_BASE =
  process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

/**
 * Is a gateway answering? Used to skip the live checks cleanly.
 *
 * A developer running the live checks with nothing listening must get one
 * clear "skipped" line and a zero exit, never a stack trace: an unreachable
 * optional dependency is not a test failure.
 */
async function gatewayReachable(base = GATEWAY_BASE, timeoutMs = 2000) {
  try {
    const response = await fetch(`${base}/health`, {
      signal: AbortSignal.timeout(timeoutMs),
    });
    // Health reports 503 when a dependency is down, but the gateway itself is
    // clearly up and able to serve the routes these checks exercise.
    return response.status === 200 || response.status === 503;
  } catch {
    return false;
  }
}

module.exports = {
  BUILD_DIR,
  COMPILED_SRC,
  GATEWAY_BASE,
  REPO_ROOT,
  WEB_ROOT,
  compile,
  gatewayReachable,
  group,
  installAlias,
  listItems,
  load,
  render,
};
