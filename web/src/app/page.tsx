"use client";

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { NotePanel } from "@/components/NotePanel";
import { ProfileCard } from "@/components/ProfileCard";
import { ProgressRail } from "@/components/ProgressRail";
import { TrialResults, ruleKey } from "@/components/TrialResults";
import { GATEWAY_BASE, createMatch, subscribeToMatch } from "@/lib/api";
import {
  FIXTURE_BUNDLE,
  FIXTURE_MATCH_ID,
  FIXTURE_NOTE_TEXT,
  FIXTURE_PROGRESS,
} from "@/lib/fixture";
import type {
  CriterionJudgment,
  MatchCompletedEvent,
  MatchErrorEvent,
  MatchProgressEvent,
  MatchResultBundle,
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
  bundle: MatchResultBundle | null;
  error: MatchErrorEvent | null;
  /** The note as pasted, kept only to show something while the run is live. */
  submittedNote: string;
}

type RunAction =
  | { type: "reset" }
  | { type: "submitted"; matchId: string; note: string }
  | { type: "status"; payload: MatchStatusSnapshot }
  | { type: "progress"; payload: MatchProgressEvent }
  | { type: "completed"; bundle: MatchResultBundle }
  | { type: "failed"; payload: MatchErrorEvent };

const INITIAL: RunState = {
  matchId: null,
  started: false,
  status: "queued",
  currentStep: null,
  pct: 0,
  message: "",
  seen: [],
  bundle: null,
  error: null,
  submittedNote: "",
};

/**
 * One reducer for both the live SSE stream and the fixture replay, so the
 * demo exercises exactly the code path a real run uses. It only records what
 * the events say; it derives no verdicts.
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
    case "completed":
      return {
        ...state,
        matchId: action.bundle.match_id,
        status: action.bundle.status,
        currentStep: null,
        pct: 100,
        message: `${action.bundle.results.length} trials ranked.`,
        bundle: action.bundle,
        error: null,
      };
    case "failed":
      return { ...state, status: "failed", error: action.payload };
    default:
      return state;
  }
}

// ------------------------------------------------------------------- page --

export default function ConsolePage() {
  const [note, setNote] = useState("");
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

  /** Replays the committed fixture through the same reducer as a live run. */
  const loadSample = useCallback(() => {
    resetRun();
    setNote(FIXTURE_NOTE_TEXT);
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
        dispatch({ type: "completed", bundle: FIXTURE_BUNDLE });
        setOpenNctId(FIXTURE_BUNDLE.results[0]?.candidate.nct_id ?? null);
      }, 350 * (FIXTURE_PROGRESS.length + 1)),
    );
  }, [resetRun]);

  const submit = useCallback(async () => {
    const text = note.trim();
    if (text.length === 0) return;
    resetRun();
    setBusy(true);
    try {
      const accepted = await createMatch(text);
      dispatch({ type: "submitted", matchId: accepted.match_id, note: text });
      unsubscribe.current = subscribeToMatch(accepted.match_id, {
        onStatus: (payload) => dispatch({ type: "status", payload }),
        onProgress: (payload) => dispatch({ type: "progress", payload }),
        onCompleted: (payload: MatchCompletedEvent) =>
          dispatch({
            type: "completed",
            bundle: {
              match_id: payload.match_id,
              status: payload.status,
              results: payload.results,
              total_cost_usd: payload.total_cost_usd,
              duration_ms: payload.duration_ms,
            },
          }),
        onError: (payload) => dispatch({ type: "failed", payload }),
      });
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [note, resetRun]);

  const bundle = run.bundle;
  const results = bundle?.results ?? [];
  const validSentenceIds = bundle?.note
    ? new Set(bundle.note.sentences.map((sentence) => sentence.sentence_id))
    : null;

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
            {busy ? "Submitting…" : "Match trials"}
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
          same event handling a live run uses, so the console is demonstrable
          while the retrieval service and match worker are still being built.
          &ldquo;Match trials&rdquo; calls the real gateway. A live run&apos;s{" "}
          <code>completed</code> event carries the ranked trials but not the
          numbered note, so until the gateway exposes it, evidence highlighting
          is only available on the sample run.
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

          {bundle ? (
            <>
              <ProfileCard profile={bundle.profile} />
              <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                <div className="lg:sticky lg:top-4 lg:self-start">
                  <NotePanel
                    note={bundle.note}
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
              {bundle.duration_ms !== undefined ? (
                <p className="text-xs text-slate-500">
                  Run finished in {Math.round(bundle.duration_ms)} ms
                  {bundle.total_cost_usd !== undefined
                    ? ` · model cost $${bundle.total_cost_usd.toFixed(4)}`
                    : ""}
                  .
                </p>
              ) : null}
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
