"""One match run, in its own process, so the test can kill it for real.

Run by ``test_crash_integration.py`` twice: once to be killed mid-fan-out, once
to finish the job from the checkpoint it left behind. It is a separate file
because an in-process ``raise`` cannot demonstrate what this has to - a
``SIGKILL`` gives the process no chance to flush, close a transaction or write
a final row, which is exactly the situation the checkpoint exists for.

What is real here: Postgres, the ``AsyncPostgresSaver`` checkpointer, the
``PostgresStore`` and its unique constraints, the whole seven-node graph, the
``beacon_llm`` client, and an HTTP round trip for retrieval.

What is stubbed: the retrieval *service* - another agent owns it and it is not
written yet - which is served by a throwaway HTTP server in this process
speaking the frozen ``SearchResponse`` contract; and Claude, through
``beacon_llm``'s replay mode against the fixtures the parent test authored.
Replay has no code path to the network, so this makes no API calls.

Usage::

    python crash_child.py --match-id <id> --retrieval-port <port>

Environment: ``BEACON_LLM_FIXTURES_DIR`` points at the seeded fixtures,
``BEACON_WORKER_THROTTLE_SECONDS`` slows the run enough to be killed at a
reproducible point. Prints one line of JSON per milestone, flushed, so the
parent can watch progress on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from beacon_core.db import get_sessionmaker
from beacon_core.events import MatchRequested
from beacon_core.settings import get_settings
from beacon_llm import AsyncLlmClient
from match_worker._testing import search_response
from match_worker.compat import run as run_async
from match_worker.config import WorkerConfig
from match_worker.graph import compile_graph, postgres_checkpointer
from match_worker.repository import PostgresStore
from match_worker.retrieval_client import RetrievalClient
from match_worker.runner import MatchWorker


def _say(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


class _RetrievalHandler(BaseHTTPRequestHandler):
    """``POST /search`` returning the frozen contract. Nothing else."""

    def do_POST(self) -> None:
        if self.path != "/search":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = search_response().model_dump_json().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: Any) -> None:
        return None


def _serve_retrieval(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), _RetrievalHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


async def _run(match_id: str, port: int) -> int:
    settings = get_settings()
    worker_config = WorkerConfig(relay_enabled=False)
    # See the note in runner.run_worker: get_sessionmaker is lru_cached and
    # Settings is unhashable, so it is called with no argument.
    store = PostgresStore(sessionmaker=get_sessionmaker())

    async with postgres_checkpointer(settings) as saver:
        worker = MatchWorker(
            store=store,
            llm_client=AsyncLlmClient(settings=settings),
            graph=compile_graph(saver),
            settings=settings,
            worker=worker_config,
            retrieval=RetrievalClient(base_url=f"http://127.0.0.1:{port}"),
        )
        _say("child.ready", match_id=match_id, throttle=worker_config.throttle_seconds)
        outcome = await worker.run_match(MatchRequested(match_id=match_id))
        _say(
            "child.finished",
            match_id=match_id,
            status=outcome.status,
            resumed=outcome.resumed,
            already_complete=outcome.already_complete,
            trials=outcome.trials,
            partial=outcome.partial,
            cost_usd=outcome.cost_usd,
            checkpoint_id=outcome.checkpoint_id,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--retrieval-port", type=int, required=True)
    args = parser.parse_args()

    server = _serve_retrieval(args.retrieval_port)
    try:
        # Not asyncio.run: psycopg cannot use Windows' default loop.
        return run_async(_run(args.match_id, args.retrieval_port))
    finally:
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
