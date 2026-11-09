"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { NotePanel } from "@/components/NotePanel";
import { ProfileCard } from "@/components/ProfileCard";
import { ProgressRail } from "@/components/ProgressRail";
import { TrialResults, ruleKey } from "@/components/TrialResults";
import {
  GATEWAY_BASE,
  MatchNotFoundError,
  createMatch,
  getMatchResult,
  subscribeToMatch,
} from "@/lib/api";
import {
  FIXTURE_MATCH_ID,
  FIXTURE_NOTE_TEXT,
  FIXTURE_PROGRESS,
  FIXTURE_RESULT,
} from "@/lib/fixture";
import { isTerminal } from "@/lib/types";
import type {
  CriterionJudgment,
  MatchCompletedEvent,
  MatchErrorEvent,
  MatchProgressEvent,
  MatchResult,
  MatchStatus,
  MatchStatusSnapshot,
  PipelineStep,
} from "@/lib/types";

// ------------------------------------------------------------------ state --

interface RunState {
  matchId: string | null;
  started: boolean;
  status: MatchStatus;
  currentStep: PipelineStep | null;
  pct: number;
  message: string;
  seen: PipelineStep[];
  /** The run's answer as `GET /v1/matches/{id}` returned it. */
  result: MatchResult | null;
  /** True while `result` came from the SSE frame and not from the route. */
  resultIsProvisional: boolean;
  error: MatchErrorEvent | null;
  loadError: string | null;
  loading: boolean;
  /** The note as pasted, shown only until the server numbers it. */
  submittedNote: string;
}

type RunAction =
  | { type: "reset" }
  | { type: "submitted"; matchId: string; note: string }
  | { type: "watching"; matchId: string }
  | { type: "status"; payload: MatchStatusSnapshot }
  | { type: "progress"; payload: MatchProgressEvent }
  | { type: "streamCompleted"; payload: MatchCompletedEvent }
  | { type: "loading" }
  | { type: "loaded"; result: MatchResult }
  | { type: "loadFailed"; message: string }
  | { type: "failed"; payload: MatchErrorEvent };

const INITIAL: RunState = {
  matchId: null,
  started: false,
  status: "queued",
  currentStep: null,
  pct: 0,
  message: "",
  seen: [],
  result: null,
  resultIsProvisional: false,
  error: null,
  loadError: null,
  loading: false,
  submittedNote: "",
};

/**
 * One reducer for the live SSE stream, the result route and the fixture
 * replay, so the demo exercises the same code path a real run uses. It records
 * what the gateway said and derives no verdicts.
 */
function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "reset":
      return INITIAL;
    case "submitted":
      return {
        ...INITIAL,
        started: true,
        matchId: action.matchId,
        status: "queued",
        message: "Accepted. The answer arrives over the event stream.",
        submittedNote: action.note,
      };
    case "watching":
      return { ...INITIAL, started: true, matchId: action.matchId };
    case "status":
      return {
        ...state,
        matchId: action.payload.match_id,
        status: action.payload.status,
        currentStep: action.payload.current_step,
      };
    case "progress":
      return {
        ...state,
        status: "running",
        currentStep: action.payload.step,
        pct: action.payload.pct,
        message: action.payload.message,
        seen: state.seen.includes(action.payload.step)
          ? state.seen
          : [...state.seen, action.payload.step],
      };
    // The `completed` frame is an optimisation: it shows the ranked trials the
    // instant they exist, but it carries no note, so it cannot support the
    // evidence highlight. `loaded` replaces it with the route's answer.
    case "streamCompleted":
      return {
        ...state,
        matchId: action.payload.match_id,
        status: action.payload.status,
        currentStep: null,
        pct: 100,
        message: `${action.payload.results.length} trials ranked.`,
        result: {
          match_id: action.payload.match_id,
          status: action.payload.status,
          current_step: null,
          note: null,
          profile: null,
          results: action.payload.results,
          trial_count: action.payload.trial_count,
          total_cost_usd: action.payload.total_cost_usd,
          duration_ms: action.payload.duration_ms,
          error: action.payload.error,
        },
        resultIsProvisional: true,
        error: null,
      };
    case "loading":
      return { ...state, loading: true, loadError: null };
    case "loaded":
      return {
        ...state,
        loading: false,
        loadError: null,
        matchId: action.result.match_id,
        status: action.result.status,
        currentStep: action.result.current_step,
        pct: isTerminal(action.result.status) ? 100 : state.pct,
        result: action.result,
        resultIsProvisional: false,
        error:
          action.result.status === "failed" && action.result.error
            ? { code: "run_failed", message: action.result.error }
            : state.error,
      };
    case "loadFailed":
      return { ...state, loading: false, loadError: action.message };
    case "failed":
      return { ...state, status: "failed", error: action.payload };
    default:
      return state;
  }
}

// ------------------------------------------------------------------- page --

export default function ConsolePage() {
  const [note, setNote] = useState("");
  const [lookupId, setLookupId] = useState("");
  const [run, dispatch] = useReducer(runReducer, INITIAL);
  const [openNctId, setOpenNctId] = useState<string | null>(null);
  const [selectedRuleKey, setSelectedRuleKey] = useState<string | null>(null);
  const [highlighted, setHighlighted] = useState<number[]>([]);
  const [caption, setCaption] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const unsubscribe = useRef<(() => void) | null>(null);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  const clearPending = useCallback(() => {
    unsubscribe.current?.();
    unsubscribe.current = null;
    timers.current.forEach(clearTimeout);
    timers.current = [];
  }, []);

  useEffect(() => clearPending, [clearPending]);

  const clearSelection = useCallback(() => {
    setSelectedRuleKey(null);
    setHighlighted([]);
    setCaption(null);
  }, []);

  const resetRun = useCallback(() => {
    clearPending();
    clearSelection();
    setOpenNctId(null);
    setSubmitError(null);
    dispatch({ type: "reset" });
  }, [clearPending, clearSelection]);

  /** Reads the run's answer from the result route, the source of truth. */
  const load = useCallback(async (matchId: string): Promise<MatchResult | null> => {
    dispatch({ type: "loading" });
    try {
      const result = await getMatchResult(matchId);
      dispatch({ type: "loaded", result });
      return result;
    } catch (error) {
      dispatch({
        type: "loadFailed",
        message:
          error instanceof MatchNotFoundError
            ? error.message
            : `Could not read the run. ${error instanceof Error ? error.message : String(error)}`,
      });
      return null;
    }
  }, []);

  /** The key interaction: a rule's citation becomes a highlight in the note. */
  const handleSelectRule = useCallback(
    (nctId: string, judgment: CriterionJudgment) => {
      const key = ruleKey(nctId, judgment.criterion_id);
      if (key === selectedRuleKey) {
        clearSelection();
        return;
      }
      setSelectedRuleKey(key);
      setHighlighted(judgment.evidence_sentence_ids);
      setCaption(
        judgment.evidence_sentence_ids.length > 0
          ? `${nctId} · rule ${judgment.criterion_id} was decided on sentence ${judgment.evidence_sentence_ids.join(", ")}, highlighted below.`
          : `${nctId} · rule ${judgment.criterion_id} cites no sentence: nothing in the note addresses it, which is why the system abstained.`,
      );
    },
    [selectedRuleKey, clearSelection],
  );

  const handleToggleTrial = useCallback(
    (nctId: string) => {
      setOpenNctId((current) => (current === nctId ? null : nctId));
      clearSelection();
    },
    [clearSelection],
  );

  /** Follows one run to its end, then re-reads it from the result route. */
  const watch = useCallback(
    (matchId: string) => {
      unsubscribe.current = subscribeToMatch(matchId, {
        onStatus: (payload) => dispatch({ type: "status", payload }),
        onProgress: (payload) => dispatch({ type: "progress", payload }),
        onCompleted: (payload) => {
          dispatch({ type: "streamCompleted", payload });
          void load(matchId);
        },
        onDone: () => void load(matchId),
        onError: (payload) => dispatch({ type: "failed", payload }),
      });
    },
    [load],
  );

  const submit = useCallback(async () => {
    const text = note.trim();
    if (text.length === 0) return;
    resetRun();
    setBusy(true);
    try {
      const accepted = await createMatch(text);
      dispatch({ type: "submitted", matchId: accepted.match_id, note: text });
      setLookupId(accepted.match_id);
      watch(accepted.match_id);
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [note, resetRun, watch]);

  /**
   * Opens a run by id. This is the case the result route exists for: a reload,
   * a late connect or a second viewer. If the run is still going, the stream is
   * joined from wherever it has got to.
   */
  const openExisting = useCallback(async () => {
    const id = lookupId.trim();
    if (id.length === 0) return;
    resetRun();
    setBusy(true);
    dispatch({ type: "watching", matchId: id });
    const result = await load(id);
    if (result && !isTerminal(result.status)) watch(id);
    if (result) setOpenNctId(result.results[0]?.candidate.nct_id ?? null);
    setBusy(false);
  }, [lookupId, resetRun, load, watch]);

  /** Replays the committed fixture through the same reducer as a live run. */
  const loadSample = useCallback(() => {
    resetRun();
    setNote(FIXTURE_NOTE_TEXT);
    setLookupId(FIXTURE_MATCH_ID);
    dispatch({
      type: "submitted",
      matchId: FIXTURE_MATCH_ID,
      note: FIXTURE_NOTE_TEXT,
    });
    FIXTURE_PROGRESS.forEach((payload, index) => {
      timers.current.push(
        setTimeout(
          () => dispatch({ type: "progress", payload }),
          350 * (index + 1),
        ),
      );
    });
    timers.current.push(
      setTimeout(() => {
        dispatch({ type: "loaded", result: FIXTURE_RESULT });
        setOpenNctId(FIXTURE_RESULT.results[0]?.candidate.nct_id ?? null);
      }, 350 * (FIXTURE_PROGRESS.length + 1)),
    );
  }, [resetRun]);

  const result = run.result;
  const results = result?.results ?? [];
  const noteForPanel = result?.note ?? null;
  // Null while no note exists, so an unresolvable citation is only ever
  // reported against a note that is actually present.
  const validSentenceIds = noteForPanel
    ? new Set(noteForPanel.sentences.map((sentence) => sentence.sentence_id))
    : null;
  const ranksProvisional = result !== null && !isTerminal(result.status);

  return (
    <div className="space-y-5">
      <section className="rounded-lg border border-slate-200 bg-white p-4">
        <label
          htmlFor="note"
          className="text-sm font-semibold uppercase tracking-wide text-slate-500"
        >
          Synthetic patient note
        </label>
        <p className="mt-1 text-sm text-slate-600">
          Paste a synthetic benchmark note. Do not paste anything about a real
          person: this is a demo surface with no safeguards for identifiable
          data.
        </p>
        <textarea
          id="note"
          value={note}
          onChange={(event) => setNote(event.target.value)}
          rows={6}
          spellCheck={false}
          placeholder="A 67-year-old woman presents with..."
          className="mt-2 w-full resize-y rounded border border-slate-300 p-3 font-mono text-sm leading-6 text-slate-900 outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500"
        />
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={submit}
            disabled={busy || note.trim().length === 0}
            className="rounded bg-sky-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-800 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {busy ? "Working…" : "Match trials"}
          </button>
          <button
            type="button"
            onClick={loadSample}
            className="rounded border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-800 hover:bg-slate-50"
          >
            Load sample run
          </button>
          {run.started ? (
            <button
              type="button"
              onClick={resetRun}
              className="rounded border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-50"
            >
              Clear
            </button>
          ) : null}
          <span className="ml-auto text-xs text-slate-500">
            gateway <code>{GATEWAY_BASE}</code>
          </span>
        </div>

        <div className="mt-3 flex flex-wrap items-end gap-2 border-t border-slate-100 pt-3">
          <div className="min-w-0 flex-1">
            <label
              htmlFor="matchId"
              className="text-xs font-medium uppercase tracking-wide text-slate-500"
            >
              Or open a run by id
            </label>
            <input
              id="matchId"
              value={lookupId}
              onChange={(event) => setLookupId(event.target.value)}
              spellCheck={false}
              placeholder="match id from an earlier submission"
              className="mt-1 w-full rounded border border-slate-300 px-2 py-1.5 font-mono text-xs text-slate-900 outline-none focus:border-sky-500 focus:ring-1 focus:ring-sky-500"
            />
          </div>
          <button
            type="button"
            onClick={openExisting}
            disabled={busy || lookupId.trim().length === 0}
            className="rounded border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-800 hover:bg-slate-50 disabled:cursor-not-allowed disabled:text-slate-400"
          >
            {run.loading ? "Reading…" : "Open run"}
          </button>
        </div>

        {submitError ? (
          <p className="mt-3 rounded border border-rose-300 bg-rose-50 p-3 text-sm text-rose-900">
            Could not submit the note. {submitError}
          </p>
        ) : null}

        <p className="mt-3 border-t border-slate-100 pt-3 text-xs leading-5 text-slate-500">
          <strong className="font-semibold text-slate-600">
            Sample run vs live run.
          </strong>{" "}
          &ldquo;Load sample run&rdquo; replays a committed fixture through the
          same event handling and the same result shape a live run uses, so the
          console is demonstrable while the retrieval service and match worker
          are still being built. &ldquo;Match trials&rdquo; and &ldquo;Open
          run&rdquo; call the real gateway.
        </p>
      </section>

      {run.started ? (
        <>
          <ProgressRail
            status={run.status}
            currentStep={run.currentStep}
            pct={run.pct}
            message={run.message}
            matchId={run.matchId}
            seen={run.seen}
          />

          {run.loadError ? (
            <section className="rounded-lg border border-amber-300 bg-amber-50 p-4">
              <h2 className="text-sm font-semibold text-amber-900">
                Could not read this run
              </h2>
              <p className="mt-1 text-sm text-amber-900">{run.loadError}</p>
              {run.matchId ? (
                <button
                  type="button"
                  onClick={() => void load(run.matchId as string)}
                  className="mt-2 rounded border border-amber-400 bg-white px-2 py-1 text-xs font-medium text-amber-900 hover:bg-amber-100"
                >
                  Try again
                </button>
              ) : null}
            </section>
          ) : null}

          {run.error ? (
            <section className="rounded-lg border border-rose-300 bg-rose-50 p-4">
              <h2 className="text-sm font-semibold text-rose-900">
                The run failed
              </h2>
              <p className="mt-1 text-sm text-rose-900">{run.error.message}</p>
              <code className="mt-1 block text-xs text-rose-700">
                {run.error.code}
              </code>
            </section>
          ) : null}

          {result ? (
            <>
              {ranksProvisional ? (
                <p className="rounded-lg border border-sky-200 bg-sky-50 px-4 py-2 text-sm text-sky-900">
                  This run is still in flight. Trials whose rules have not been
                  checked yet are listed after the ranked ones and report{" "}
                  <strong className="font-semibold">insufficient info</strong>,
                  so the order and the verdicts below can still change. Neither
                  is final until the run completes.
                </p>
              ) : null}

              {run.resultIsProvisional ? (
                <p className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2 text-sm text-slate-700">
                  Showing the ranked trials from the completion event while the
                  full run is read back. The patient note arrives with that
                  read, so evidence highlighting turns on a moment later.
                </p>
              ) : null}

              <ProfileCard profile={result.profile} />
              <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                <div className="lg:sticky lg:top-4 lg:self-start">
                  <NotePanel
                    note={noteForPanel}
                    highlighted={highlighted}
                    caption={caption}
                    rawFallback={run.submittedNote}
                  />
                </div>
                <TrialResults
                  results={results}
                  openNctId={openNctId}
                  onToggleTrial={handleToggleTrial}
                  selectedRuleKey={selectedRuleKey}
                  onSelectRule={handleSelectRule}
                  validSentenceIds={validSentenceIds}
                />
              </div>
              <p className="text-xs text-slate-500">
                {result.trial_count} trial
                {result.trial_count === 1 ? "" : "s"} returned
                {result.duration_ms !== null
                  ? ` · ran in ${Math.round(result.duration_ms)} ms`
                  : ""}{" "}
                · model cost ${result.total_cost_usd.toFixed(4)} · match{" "}
                <code>{result.match_id}</code>
              </p>
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
