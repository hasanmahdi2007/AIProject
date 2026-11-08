"""``python -m match_worker`` - run the worker against the compose stack.

Ports come from ``beacon_core.settings``, which already defaults to the ones
``deploy/compose.yaml`` publishes (Postgres 15432, Redis 16379), so a normal
local run needs no overrides.
"""

from __future__ import annotations

import argparse
import asyncio

from beacon_core.logging import configure_logging, get_logger
from beacon_core.settings import get_settings
from match_worker.config import WorkerConfig
from match_worker.runner import run_worker

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="match_worker", description=__doc__)
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="handle N messages and exit; omit to run until interrupted",
    )
    parser.add_argument(
        "--no-relay",
        action="store_true",
        help="do not drain the outbox to Kafka from this process",
    )
    parser.add_argument(
        "--max-cost-usd", type=float, default=None, help="override the per-match dollar budget"
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging()
    worker = WorkerConfig()
    if args.no_relay:
        worker = worker.model_copy(update={"relay_enabled": False})
    if args.max_cost_usd is not None:
        worker = worker.model_copy(update={"max_cost_usd": args.max_cost_usd})

    try:
        handled = asyncio.run(
            run_worker(settings=settings, worker=worker, max_messages=args.max_messages)
        )
    except KeyboardInterrupt:
        log.info("worker.interrupted")
        return 0
    log.info("worker.exit", handled=handled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
