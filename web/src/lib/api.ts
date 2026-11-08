/**
 * The only place the console talks to the gateway.
 *
 * Two calls, both exactly as the merged gateway defines them:
 *   POST /v1/matches                  -> 202 MatchAccepted
 *   GET  /v1/matches/{id}/events      -> SSE: status | progress | completed | error | done
 *
 * The SSE frames are typed by `gateway.sse.format_sse`, so this file dispatches
 * on the event *name* and never sniffs the body.
 */

import type {
  MatchAccepted,
  MatchCompletedEvent,
  MatchDoneEvent,
  MatchErrorEvent,
  MatchProgressEvent,
  MatchStatusSnapshot,
} from "./types";

export const GATEWAY_BASE =
  process.env.NEXT_PUBLIC_GATEWAY_URL ?? "http://localhost:8000";

export interface MatchEventHandlers {
  onStatus?: (snapshot: MatchStatusSnapshot) => void;
  onProgress?: (progress: MatchProgressEvent) => void;
  onCompleted?: (completed: MatchCompletedEvent) => void;
  onError?: (error: MatchErrorEvent) => void;
  onDone?: (done: MatchDoneEvent) => void;
}

export async function createMatch(patientNote: string): Promise<MatchAccepted> {
  const response = await fetch(`${GATEWAY_BASE}/v1/matches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patient_note: patientNote }),
  });

  if (!response.ok) {
    // 422 bodies are FastAPI validation detail; surface them verbatim.
    const detail = await response.text();
    throw new Error(
      `POST /v1/matches returned ${response.status}: ${detail.slice(0, 400)}`,
    );
  }
  return (await response.json()) as MatchAccepted;
}

/**
 * Subscribes to one run's event stream. Returns an unsubscribe function.
 *
 * `done` and `error` are terminal and mutually exclusive per the gateway's own
 * contract, so the source is closed on either and no timeout is needed.
 */
export function subscribeToMatch(
  matchId: string,
  handlers: MatchEventHandlers,
): () => void {
  const source = new EventSource(
    `${GATEWAY_BASE}/v1/matches/${encodeURIComponent(matchId)}/events`,
  );

  const parse = <T,>(raw: string): T | null => {
    try {
      return JSON.parse(raw) as T;
    } catch {
      return null;
    }
  };

  const bind = <T,>(name: string, handler: ((payload: T) => void) | undefined) => {
    source.addEventListener(name, (event) => {
      if (!handler) return;
      const payload = parse<T>((event as MessageEvent<string>).data);
      if (payload !== null) handler(payload);
    });
  };

  bind<MatchStatusSnapshot>("status", handlers.onStatus);
  bind<MatchProgressEvent>("progress", handlers.onProgress);
  bind<MatchCompletedEvent>("completed", handlers.onCompleted);
  bind<MatchErrorEvent>("error", handlers.onError);
  bind<MatchDoneEvent>("done", (payload) => {
    handlers.onDone?.(payload);
    source.close();
  });

  // A transport-level failure (gateway down, 404, connection dropped). The
  // gateway's typed `error` event is a different thing and handled above.
  source.onerror = () => {
    if (source.readyState === EventSource.CLOSED) return;
    handlers.onError?.({
      code: "stream_disconnected",
      message:
        "The event stream closed before the run finished. The run may still be going on the server.",
      match_id: matchId,
    });
    source.close();
  };

  return () => source.close();
}
