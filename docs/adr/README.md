# Architecture Decision Records

One page each: **context**, **options considered**, **the decision**, **its
consequences**. They exist so that every structural choice in Beacon has a
written reason that survives being asked about six months later.

| ADR | Decision | The one-line reason |
|---|---|---|
| [0001](0001-qdrant-vs-pgvector.md) | Qdrant alongside Postgres, not pgvector | a real BM25 baseline — **not** scale; 375,580 vectors is small |
| [0002](0002-kafka-vs-redis-streams.md) | Kafka (Redpanda), not Redis Streams | durable log with offsets committed after the checkpoint |
| [0003](0003-langgraph-raw-sdk-vs-langchain.md) | LangGraph + the raw Anthropic SDK, not LangChain | cache ordering, raw `usage` and structured outputs all fail silently through a wrapper |
| [0004](0004-trial-level-vs-rule-level-indexing.md) | One point per trial, overview text only | the answer unit is a trial, and `bge-small` truncates at 512 tokens |
| [0005](0005-model-choice-and-cost.md) | Haiku for dev, Sonnet for the headline, Opus as judge only | ~95% of spend is one stage, and Haiku has a 5× cliff at 100K prompt tokens |
| [0006](0006-2021-2022-tuning-split.md) | Tune on TREC CT 2021, report on held-out 2022 | the only honest generalisation claim available |

**No performance number in these ADRs is invented.** Anything not yet measured
appears as an explicit placeholder in braces, e.g. `{recall@100}`; see the
placeholder index in the [README](../../README.md#placeholder-index).
