# ADR 0002 — Kafka (Redpanda) for the job queue, not Redis Streams

- **Status:** accepted
- **Date:** 2026-11-08
- **Affects:** `deploy/compose.yaml`, `beacon_core.kafka`, `beacon_core.events`, `services/gateway`, `services/match_worker`

## Context

`POST /v1/matches` returns **202 Accepted** with a `match_id` immediately; the
actual work is a multi-step LangGraph pipeline that takes tens of seconds and
spends real money on Claude calls. Something has to carry the job from the
gateway to a worker, and that something has to survive a worker being killed
mid-run — which is a scripted part of the demo, not a hypothetical.

The requirements, in the order that decided the choice:

1. **Durability.** An accepted job must not vanish because a process died.
2. **Redelivery with offsets committed after the checkpoint**, so a crash
   resumes rather than loses.
3. **Consumer-group scaling**: add worker replicas, get throughput.
4. **Retry with delay plus a dead-letter queue**, so one poisonous job does not
   block the others.
5. **Replay.** The evaluation harness wants to re-consume `match.completed` for
   a whole benchmark run.

Redis is *already* in the stack for rate limiting, the exact-match LLM cache,
`lock:ingest` and the `chan:match:{id}` pub/sub that feeds SSE. So Redis
Streams is the option that adds no new container.

## Options considered

| Option | Verdict |
|---|---|
| **FastAPI `BackgroundTasks`** | Rejected outright. The work lives in the gateway process; a restart loses every in-flight job and the gateway's latency becomes the pipeline's latency. |
| **Celery + Redis broker** | Rejected. A task queue, not an event log: no replay, no ordered per-job event stream, and the progress-event path would need a second mechanism anyway. |
| **Redis Streams** | Rejected, but it is the close call — see below. |
| **Kafka via Redpanda** | **Chosen.** |
| Kafka via Apache Kafka + KRaft | Rejected for local dev only on footprint: Redpanda is one container, no JVM, no ZooKeeper, and speaks the same protocol. The code is Kafka code either way. |

### Why not Redis Streams, honestly

Redis Streams is a real queue and it should not be dismissed with a hand-wave.
It has consumer groups, `XACK`, `XPENDING` and `XAUTOCLAIM`, which together give
at-least-once delivery and recovery of messages whose consumer died. For a
smaller system it would be the right answer.

What ruled it out here:

- **Durability.** The compose file runs Redis with `--appendonly yes
  --appendfsync everysec`, so up to one second of acknowledged writes can be
  lost on an unclean stop. That is the correct trade-off for rate-limit counters
  and a response cache; it is the wrong one for an accepted job.
- **Retention is trimming, not a log.** A stream is bounded by `MAXLEN`/`MINID`.
  Re-reading a whole benchmark run's `match.completed` events later means
  keeping the stream from ever being trimmed, which is a different data-lifecycle
  promise than a log with a retention window and committed offsets.
- **The event-log parts are hand-built.** Partitioning, offset semantics,
  delayed retry and a DLQ all exist in Redis Streams only as things you write.
  In Kafka they are the model.
- **It would also put the job queue and the hot-path cache in the same process.**
  A worker's bulk consumption competing with the gateway's rate-limit `Lua`
  script on one Redis is an avoidable coupling.

## Decision

Use **Kafka, served locally by Redpanda**, for six topics, fixed in
`beacon_core.events`:

| Topic | Purpose |
|---|---|
| `match.requested` | new jobs; partition key = `match_id` |
| `match.progress` | step-by-step updates for the live console |
| `match.completed` | finished results, for evaluation and analytics |
| `match.retry` | delayed retry |
| `match.dlq` | parked after `KafkaSettings.max_retries` (3) failures |
| `trials.updated` | phase 2: changed trials, re-indexed incrementally |

Two rules are encoded in the factories in `beacon_core.kafka` rather than left
to each caller:

- **`enable_auto_commit=False`.** The offset is committed *after* the LangGraph
  checkpoint is written, never before. This is the whole mechanism by which a
  killed worker's job is redelivered instead of silently dropped.
- **The partition key is always `match_id`**, so every event belonging to one
  job lands on one partition and stays ordered.

Redis stays, for exactly the things it is good at: `rl:{api_key}`, `idem:{key}`,
`llm:{sha256}`, `lock:ingest`, `chan:match:{id}`. The answer to "why both?" is
hot, short-lived, overwriteable data versus a durable log.

## Consequences

- **One more container, and it is not free** — five backing services instead of
  four. Redpanda runs with `--smp=1
  --memory=1G --mode=dev-container`. On an 8 GB laptop that is a real slice.
- **Two listeners are required, and getting this wrong is a classic.** A Kafka
  client is told where to reconnect by the *broker's advertised address*, so
  bootstrapping successfully proves nothing. The compose file binds
  `internal://0.0.0.0:9092` advertised as `redpanda:9092` for sibling
  containers, and `external://0.0.0.0:19092` advertised as `localhost:9092`
  published on host 9092. Advertise `redpanda:9092` to the host and host clients
  fail on an unresolvable name; advertise `localhost:9092` to containers and
  each one reconnects to itself.
- **At-least-once means every consumer must be idempotent.** A redelivered
  message must not duplicate results, so the schema carries the uniqueness:
  `uq_trial_matches_run_nct`, `uq_criterion_judgments_identity`,
  `uq_trial_criteria_nct_criterion`, `uq_outbox_event_id`. Idempotency is a
  database constraint, not a convention.
- **It forces the transactional outbox.** Committing the `match_runs` row and
  then publishing to Kafka are two operations that can half-fail. So
  `gateway.router.create_match` writes the `match_runs` row *and* the `outbox`
  row in **one** transaction, and a relay publishes from `outbox`. A queued job
  and its event cannot desynchronise. This is extra machinery that a single
  Redis `XADD` inside the same connection would not have needed, and it is
  listed here as a cost of the choice.
- Debugging needs `rpk`. The compose health check already uses it
  (`rpk cluster health`), because a bare TCP probe passes while the controller
  still has no leader and every produce call fails.
