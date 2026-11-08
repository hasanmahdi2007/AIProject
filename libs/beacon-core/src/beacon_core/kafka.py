"""Kafka producer/consumer factories and trace-header helpers.

Stubs in Wave 0: the shapes are fixed here so the gateway (A2) and the worker
(C1) cannot disagree about offsets, partition keys or headers.

Two rules the factories encode:

* ``enable_auto_commit=False``. The offset is committed *after* the checkpoint
  is written, never before, which is what makes a crashed worker's job get
  redelivered instead of lost.
* The partition key is always ``match_id``, so one job's events stay ordered.
"""

from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from beacon_core.settings import Settings, get_settings

__all__ = [
    "TRACE_HEADER",
    "build_consumer",
    "build_producer",
    "headers_from_trace",
    "trace_from_headers",
]

TRACE_HEADER = "traceparent"
"""W3C trace context, carried through the broker so one trace spans
gateway -> Kafka -> worker -> Claude."""


def build_producer(settings: Settings | None = None) -> AIOKafkaProducer:
    cfg = settings or get_settings()
    return AIOKafkaProducer(
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: None if k is None else str(k).encode("utf-8"),
        enable_idempotence=True,
        acks="all",
    )


def build_consumer(*topics: str, settings: Settings | None = None) -> AIOKafkaConsumer:
    cfg = settings or get_settings()
    return AIOKafkaConsumer(
        *topics,
        bootstrap_servers=cfg.kafka.bootstrap_servers,
        group_id=cfg.kafka.consumer_group,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )


def headers_from_trace(traceparent: str | None) -> list[tuple[str, bytes]]:
    if not traceparent:
        return []
    return [(TRACE_HEADER, traceparent.encode("utf-8"))]


def trace_from_headers(headers: Any) -> str | None:
    for name, value in headers or ():
        if name == TRACE_HEADER and value:
            decoded: str = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            return decoded
    return None
