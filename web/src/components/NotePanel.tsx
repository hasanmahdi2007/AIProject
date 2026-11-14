"use client";

import { useEffect, useRef } from "react";
import type { NumberedNote } from "@/lib/types";

export interface NotePanelProps {
  /** `null` means the worker has not split the note yet. Not a defect. */
  note: NumberedNote | null;
  /** Sentence IDs cited by the currently selected rule. */
  highlighted: number[];
  /** What the selection is, in words, shown above the note. */
  caption: string | null;
  /** The raw pasted text, shown when the server has not numbered it yet. */
  rawFallback?: string;
}

/**
 * The numbered patient note, with the cited sentences highlighted.
 *
 * This is the half of the console that makes the other half checkable. The
 * sentence numbers in the gutter are the same IDs a `CriterionJudgment` cites,
 * so a reader can verify a citation by eye without trusting the UI.
 */
export function NotePanel({
  note,
  highlighted,
  caption,
  rawFallback,
}: NotePanelProps) {
  const firstRef = useRef<HTMLLIElement | null>(null);
  const key = highlighted.join(",");

  useEffect(() => {
    if (key === "") return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    firstRef.current?.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
  }, [key]);

  const active = new Set(highlighted);
  const sentences = note?.sentences ?? [];
  const known = new Set(sentences.map((s) => s.sentence_id));

  // Only a note that EXISTS can have an unresolvable citation. A null note is
  // a not-yet - the worker has not reached `profile_patient` - and flagging
  // every citation as invalid in that window would turn a normal in-flight
  // state into a wall of false defects.
  const unresolved =
    note === null ? [] : highlighted.filter((id) => !known.has(id));

  return (
    <section className="card overflow-hidden">
      <header className="border-b border-nhs-line px-5 pb-3 pt-4">
        <h2 className="heading">Patient note, as the system numbered it</h2>
        <p className={`mt-1 text-[14px] leading-6 ${caption ? "font-semibold text-[#00445E]" : "text-nhs-mid"}`}>
          {caption ?? "Click any rule in a trial to highlight the sentence it was decided on."}
        </p>
      </header>

      {sentences.length > 0 ? (
        <ol className="space-y-1 p-3">
          {sentences.map((sentence) => {
            const on = active.has(sentence.sentence_id);
            const isFirst = on && sentence.sentence_id === highlighted[0];
            return (
              <li
                key={sentence.sentence_id}
                ref={isFirst ? firstRef : null}
                aria-current={on ? "true" : undefined}
                className={`flex gap-3 rounded px-3 py-2.5 transition-colors duration-200 ${
                  on ? "evidence-on" : "hover:bg-nhs-pale"
                }`}
              >
                <span
                  className={`mt-px flex h-6 min-w-6 shrink-0 select-none items-center justify-center rounded px-1.5 text-[13px] font-bold tabular ${
                    on ? "bg-nhs-aqua text-nhs-ink" : "bg-nhs-blue text-white"
                  }`}
                >
                  {sentence.sentence_id}
                </span>
                <span className={`text-[15px] leading-6 ${on ? "font-semibold text-nhs-ink" : "text-nhs-ink"}`}>
                  {sentence.text}
                </span>
              </li>
            );
          })}
        </ol>
      ) : (
        <div className="space-y-2 px-5 py-4">
          <p className="text-[14px] leading-6 text-nhs-mid">
            {note === null
              ? "The worker has not split this note into sentences yet, so citations cannot be resolved against it. This is a normal in-flight state, not a problem with the run."
              : "The run returned a note containing no sentences, so there is nothing for a citation to point at."}{" "}
            Numbering is done by the server; the console does not split the note
            itself, because a different split would give different sentence IDs
            from the ones the rules cite.
          </p>
          {rawFallback ? (
            <p className="whitespace-pre-wrap rounded bg-nhs-pale p-3.5 text-[15px] leading-6 text-nhs-ink">
              {rawFallback}
            </p>
          ) : null}
        </div>
      )}

      {unresolved.length > 0 ? (
        <div role="alert" className="border-t-4 border-nhs-red bg-[#FBE7E5] px-5 py-3.5">
          <p className="text-[14px] font-bold text-nhs-red">
            Invalid citation: sentence{" "}
            {unresolved.length === 1 ? "" : "s "}
            {unresolved.join(", ")} {unresolved.length === 1 ? "does" : "do"} not
            exist in this note.
          </p>
          <p className="mt-1 text-[14px] leading-6 text-nhs-ink">
            The selected rule cites {unresolved.length === 1 ? "an" : ""}{" "}
            sentence {unresolved.length === 1 ? "id" : "ids"} outside the
            numbered note, so the evidence cannot be shown and the rule&apos;s
            verdict is not checkable. The verifier node is supposed to catch
            this before a result is returned; seeing it here means a citation
            escaped verification.
          </p>
        </div>
      ) : null}
    </section>
  );
}
