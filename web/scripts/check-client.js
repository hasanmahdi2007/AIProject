"use strict";

/**
 * Drives `src/lib/api.ts` itself against a running gateway.
 *
 * Not curl: the console's own client, so URL building, the 404 branch and the
 * SSE event dispatch are exercised rather than reasoned about.
 *
 * Read-only by default. `createMatch` inserts a patient, a match run and an
 * outbox row, so the checks that need a new run only run when
 * `VERIFY_ALLOW_WRITES=1` - a verification script must not quietly write to
 * whichever database the gateway happens to point at.
 */

const { GATEWAY_BASE, group, load } = require("./harness");

const MATCH_ID = process.env.VERIFY_MATCH_ID ?? "demo-completed-run";
const ALLOW_WRITES = process.env.VERIFY_ALLOW_WRITES === "1";

const SAMPLE_NOTE =
  "A 61-year-old man presents with progressive dyspnea over two months. " +
  "Biopsy confirms squamous cell carcinoma of the lung. " +
  "He has received no prior systemic therapy. His ECOG performance status is 1.";

/** Waits for the first SSE frame, or gives up. Never hangs. */
function firstFrame(api, matchId, timeoutMs = 10000) {
  return new Promise((resolve) => {
    const frames = [];
    const stop = api.subscribeToMatch(matchId, {
      onStatus: (payload) => frames.push(["status", payload]),
      onProgress: (payload) => frames.push(["progress", payload]),
      onCompleted: (payload) => frames.push(["completed", payload]),
      onError: (payload) => frames.push(["error", payload]),
      onDone: (payload) => frames.push(["done", payload]),
    });
    const started = Date.now();
    const tick = setInterval(() => {
      if (frames.length > 0 || Date.now() - started > timeoutMs) {
        clearInterval(tick);
        stop();
        resolve(frames);
      }
    }, 200);
  });
}

async function run(root) {
  const g = group("api client against a live gateway");
  const api = load(root, "lib/api.js");

  g.check(
    api.GATEWAY_BASE === GATEWAY_BASE,
    `the client resolved its base URL to ${api.GATEWAY_BASE}`,
  );

  // ----------------------------------------------------- the read path --
  let result = null;
  try {
    result = await api.getMatchResult(MATCH_ID);
  } catch (error) {
    if (error instanceof api.MatchNotFoundError) {
      g.skip("getMatchResult", `no run "${MATCH_ID}" on ${GATEWAY_BASE}`);
    } else {
      g.check(false, "getMatchResult", String(error && error.message));
    }
  }

  if (result) {
    g.check(result.match_id === MATCH_ID, "getMatchResult returns the run it asked for");
    g.check(
      Array.isArray(result.results) && typeof result.status === "string",
      `getMatchResult parsed the live body (status=${result.status}, ${result.results.length} trials)`,
    );
    g.check(
      result.trial_count === result.results.length,
      "trial_count agrees with the rows returned",
    );
    // The reason this route exists: a reload must get the same answer.
    const again = await api.getMatchResult(MATCH_ID);
    g.check(
      JSON.stringify(again) === JSON.stringify(result),
      "reading the same run twice returns an identical body (repeatable GET)",
    );
    if (result.note) {
      g.check(
        result.note.sentences.length > 0,
        `the body carries the ${result.note.sentences.length}-sentence note the highlight needs`,
      );
    }
  }

  // --------------------------------------------------------- the 404 path --
  let notFound = null;
  try {
    await api.getMatchResult("no-such-run-verify-check");
  } catch (error) {
    notFound = error;
  }
  g.check(
    notFound instanceof api.MatchNotFoundError,
    `an unknown id raises MatchNotFoundError (got ${notFound && notFound.name})`,
  );
  g.check(
    notFound !== null && /No match run/.test(notFound.message),
    "the 404 reads 'no such run', never 'no trials matched'",
  );

  // An id that needs encoding must stay one path segment.
  let traversed = null;
  try {
    await api.getMatchResult("a b/../health");
  } catch (error) {
    traversed = error;
  }
  g.check(
    traversed instanceof api.MatchNotFoundError,
    "an id needing URL encoding is encoded, not path-traversed into another route",
  );

  // ------------------------------------------------- the write path, opt-in --
  if (!ALLOW_WRITES) {
    g.skip(
      "createMatch and subscribeToMatch",
      "writes a patient, a match run and an outbox row - set VERIFY_ALLOW_WRITES=1 against a disposable database",
    );
    return g.state;
  }

  const accepted = await api.createMatch(SAMPLE_NOTE);
  g.check(
    typeof accepted.match_id === "string" && accepted.match_id.length > 0,
    `createMatch parsed the 202 body (match ${accepted.match_id})`,
  );
  g.check(accepted.status === "queued", `a new run comes back queued (got "${accepted.status}")`);
  g.check(
    typeof accepted.event_id === "string" && accepted.event_id.length > 0,
    "the 202 carries the outbox event id",
  );

  const fresh = await api.getMatchResult(accepted.match_id);
  g.check(
    fresh.status === "queued" && fresh.note === null && fresh.results.length === 0,
    "the new run is readable immediately, as queued with a null note",
  );

  if (typeof EventSource === "undefined") {
    g.skip(
      "subscribeToMatch",
      "no global EventSource in this Node build - rerun with node --experimental-eventsource",
    );
    return g.state;
  }

  const frames = await firstFrame(api, accepted.match_id);
  g.log(`SSE frames: ${JSON.stringify(frames)}`);
  const status = frames.find(([name]) => name === "status");
  g.check(status !== undefined, "subscribeToMatch received the opening status frame");
  g.check(
    status !== undefined && status[1].match_id === accepted.match_id,
    "the status frame names the run subscribed to",
  );
  g.check(
    !frames.some(([name]) => name === "error"),
    "no error frame on a healthy live stream",
  );

  return g.state;
}

module.exports = { run };
