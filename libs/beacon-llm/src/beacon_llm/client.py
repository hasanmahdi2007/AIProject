"""The seam. Every Claude call in Beacon goes through :meth:`LlmClient.call`.

One function, so that the five things that are easy to get wrong are got right
once instead of at every call site:

1. **Structured outputs** go through ``output_config.format`` with a JSON
   schema derived from a Pydantic model. Not ``output_format=`` - see the note
   on that below - and not a forced ``tool_choice``, which Opus 5.5 and
   Sonnet 5.5 do not support.
2. **Cache ordering** is not a parameter. ``call()`` takes a
   :class:`~beacon_llm.prompt.CachedPrompt`, which can only render its blocks
   in the one order that caches.
3. **The token guard** runs before the request leaves the process, on every
   path including replay, so a batching regression fails in CI.
4. **Cost** is metered from the returned ``usage`` by
   :mod:`beacon_llm.cost` and every field
   ``beacon_core.models.LlmCall`` wants comes back on the result.
5. **Thinking** is ``{"type": "adaptive"}``, always. One path, no per-model
   branch - see :mod:`beacon_llm.capabilities` for the Models API output that
   makes a branch impossible. ``output_config.effort`` is the only dial, and it
   is passed per call site.

A note on ``output_format`` versus ``output_config.format``
-----------------------------------------------------------
``messages.parse()`` in anthropic 2.6.0 accepts both. ``output_format=`` is an
SDK-side convenience that takes a Pydantic *type*, converts it with
``transform_schema``, and merges the result into ``output_config["format"]``
before the request goes out; passing a schema *dict* to it raises ``TypeError``
telling you to use ``output_config`` instead. ``messages.create()`` has no
``output_format`` parameter at all. So ``output_config.format`` is the real
parameter and the other is a wrapper over it.

This module therefore does the schema transform itself and sends
``output_config.format``, which also means the Pydantic validation is ours: a
response that does not validate raises
:class:`~beacon_llm.errors.StructuredOutputError` with the raw text attached,
rather than surfacing as ``parsed_output is None`` and being read as an empty
result somewhere downstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib._parse._transform import transform_schema
from anthropic.types import JSONOutputFormatParam, Message, OutputConfigParam
from pydantic import BaseModel, ValidationError

from beacon_core.logging import get_logger
from beacon_core.settings import Settings, get_settings
from beacon_llm.capabilities import Effort, assert_supports
from beacon_llm.cost import CostBreakdown, price_usage
from beacon_llm.errors import RefusalError, StructuredOutputError
from beacon_llm.fixtures import FixtureMode, FixtureStore, resolve_mode
from beacon_llm.guard import (
    ApiTokenCounter,
    HeuristicTokenCounter,
    TokenCounter,
    enforce_prompt_budget,
)
from beacon_llm.pricing import MIN_CACHEABLE_PREFIX_TOKENS
from beacon_llm.prompt import CachedPrompt

__all__ = ["ADAPTIVE_THINKING", "AsyncLlmClient", "CallResult", "LlmClient"]

log = get_logger(__name__)

ADAPTIVE_THINKING: dict[str, str] = {"type": "adaptive"}
"""The one and only thinking configuration this library sends.

``budget_tokens`` is absent because there is nowhere to put it: all four
current models report ``thinking.types.enabled.supported == false``, and
``enabled`` was the only shape that carried a budget. Sonnet 5.5 additionally
reports ``disabled.supported == false``, so thinking cannot be switched off
there either. One dict, every call.
"""


@dataclass(frozen=True, slots=True)
class CallResult[OutputT: BaseModel]:
    """A validated answer plus everything needed to audit what it cost."""

    parsed: OutputT
    message: Message
    cost: CostBreakdown
    model: str
    effort: Effort | None
    prompt_hash: str
    latency_ms: float
    stop_reason: str | None
    request_id: str | None
    served_from_fixture: bool

    def audit_row(
        self,
        *,
        match_run_id: str | None = None,
        step: str | None = None,
        trace_id: str | None = None,
        batch_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Keyword arguments for ``beacon_core.models.LlmCall``.

        A plain dict rather than the ORM object, so this package needs no
        session and no SQLAlchemy import on the hot path. ``test_audit_row``
        asserts every key is a real column and that ``LlmCall(**row)``
        constructs, which keeps the two in step without the coupling.
        """
        return {
            "match_run_id": match_run_id,
            "step": step,
            "model": self.model,
            "effort": self.effort,
            "prompt_hash": self.prompt_hash,
            "input_tokens": self.cost.input_tokens,
            "output_tokens": self.cost.output_tokens,
            "cache_read_tokens": self.cost.cache_read_tokens,
            "cache_write_tokens": self.cost.cache_write_tokens,
            "cost_usd": self.cost.cost_usd,
            "billed_tier": self.cost.billed_tier,
            "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
            "request_id": self.request_id,
            "batch_id": batch_id,
            "trace_id": trace_id,
            "error": error,
        }


@dataclass(frozen=True, slots=True)
class _Plan:
    """A request, fully formed, not yet sent.

    Built identically on the sync and async paths so the two cannot drift: the
    only thing that differs downstream is the ``await``.
    """

    model: str
    effort: Effort | None
    max_tokens: int
    output_config: OutputConfigParam
    system: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    prompt_hash: str
    request_summary: dict[str, Any]


def _schema_param(output_model: type[BaseModel]) -> JSONOutputFormatParam:
    """The Pydantic model as a json_schema output format.

    ``transform_schema`` is the SDK's own converter - the same one
    ``output_format=`` would have used - so the schema sent here is byte for
    byte what the supported path would send.
    """
    return {"type": "json_schema", "schema": transform_schema(output_model.model_json_schema())}


def _plan(
    *,
    prompt: CachedPrompt,
    output_model: type[BaseModel],
    model: str,
    effort: Effort | None,
    max_tokens: int,
) -> _Plan:
    output_config: OutputConfigParam = {"format": _schema_param(output_model)}
    if effort is not None:
        output_config["effort"] = effort

    system = list(prompt.system_blocks())
    messages = list(prompt.messages())
    prompt_hash = prompt.fingerprint()

    if prompt.maybe_too_short_to_cache:
        log.warning(
            "llm.cache_prefix_may_be_too_short",
            estimated_prefix_tokens=prompt.estimated_prefix_tokens,
            minimum=MIN_CACHEABLE_PREFIX_TOKENS,
            prompt_hash=prompt_hash,
            detail=(
                "a cache_control breakpoint below the minimum cacheable prefix is "
                "accepted, ignored, and reports no error; check usage.cache_read_input_tokens"
            ),
        )

    return _Plan(
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        output_config=output_config,
        system=[dict(block) for block in system],
        messages=[dict(message) for message in messages],
        prompt_hash=prompt_hash,
        request_summary={
            "model": model,
            "max_tokens": max_tokens,
            "effort": effort,
            "thinking": ADAPTIVE_THINKING,
            "output_schema": output_model.__name__,
            "system": [dict(block) for block in system],
            "messages": [dict(message) for message in messages],
        },
    )


def _validate[OutputT: BaseModel](
    *, message: Message, output_model: type[OutputT], model: str, request_id: str | None
) -> OutputT:
    """Turn the response into an instance of ``output_model``, or raise.

    Thinking blocks are skipped; only ``text`` blocks carry the JSON. A
    ``refusal`` stop reason is raised as its own type because retrying the same
    prompt will refuse again - it is an outcome, not a transport blip.
    """
    if message.stop_reason == "refusal":
        raise RefusalError(model=model, request_id=request_id)

    text = "".join(block.text for block in message.content if block.type == "text")

    if message.stop_reason in {"max_tokens", "model_context_window_exceeded"}:
        raise StructuredOutputError(
            model_name=output_model.__name__,
            detail=(
                f"response stopped on {message.stop_reason!r}, so the JSON is truncated; "
                f"raise max_tokens or shrink the request"
            ),
            raw=text,
        )
    if not text.strip():
        raise StructuredOutputError(
            model_name=output_model.__name__,
            detail="response contained no text block",
            raw=text,
        )
    try:
        return output_model.model_validate_json(text)
    except ValidationError as exc:
        raise StructuredOutputError(
            model_name=output_model.__name__, detail=str(exc), raw=text
        ) from exc


def _result[OutputT: BaseModel](
    *,
    plan: _Plan,
    message: Message,
    parsed: OutputT,
    prompt: CachedPrompt,
    latency_ms: float,
    served_from_fixture: bool,
) -> CallResult[OutputT]:
    cost = price_usage(plan.model, message.usage, cache_ttl=prompt.cache_ttl)
    result = CallResult(
        parsed=parsed,
        message=message,
        cost=cost,
        model=plan.model,
        effort=plan.effort,
        prompt_hash=plan.prompt_hash,
        latency_ms=latency_ms,
        stop_reason=message.stop_reason,
        request_id=getattr(message, "_request_id", None),
        served_from_fixture=served_from_fixture,
    )
    log.info(
        "llm.call",
        model=result.model,
        effort=result.effort,
        prompt_hash=result.prompt_hash,
        billed_tier=cost.billed_tier,
        prompt_tokens=cost.prompt_tokens,
        cache_read_tokens=cost.cache_read_tokens,
        cache_write_tokens=cost.cache_write_tokens,
        output_tokens=cost.output_tokens,
        cost_usd=cost.cost_usd,
        latency_ms=round(result.latency_ms, 1),
        served_from_fixture=served_from_fixture,
    )
    return result


@dataclass
class _Base:
    settings: Settings = field(default_factory=get_settings)
    mode: FixtureMode = field(default_factory=resolve_mode)
    fixtures: FixtureStore = field(init=False)

    def __post_init__(self) -> None:
        self.fixtures = FixtureStore(self.settings.llm_fixtures_dir)

    @property
    def _api_key(self) -> str:
        return self.settings.anthropic.api_key.get_secret_value()

    def _guard(self, plan: _Plan, counter: TokenCounter) -> int:
        """Count the prompt and refuse if it crosses a cliff.

        Runs on every path. The counter differs - exact from the API when live,
        character-based when replaying - but the ceiling is enforced either way,
        which is the point: the regression this guards against is a code change,
        and code changes are reviewed against the replay suite.
        """
        prompt_tokens = counter.count(
            model=plan.model,
            system=plan.system,  # type: ignore[arg-type]
            messages=plan.messages,  # type: ignore[arg-type]
            output_config=plan.output_config,
        )
        enforce_prompt_budget(
            model=plan.model,
            prompt_tokens=prompt_tokens,
            ceiling=self.settings.anthropic.haiku_prompt_token_ceiling,
        )
        return prompt_tokens

    def _from_fixture[OutputT: BaseModel](
        self, plan: _Plan, prompt: CachedPrompt, output_model: type[OutputT]
    ) -> CallResult[OutputT] | None:
        """Serve this call from disk, or return ``None`` to go to the network.

        Shared by both clients, because replay reads a file either way and
        duplicating the mode logic is how the two transports would drift apart.

        In replay the guard runs *before* the lookup: an over-ceiling prompt
        must fail as over-ceiling, and it will never have a fixture, so a
        lookup-first order would report the wrong problem.
        """
        if self.mode is FixtureMode.LIVE:
            return None

        if self.mode is FixtureMode.REPLAY:
            self._guard(plan, HeuristicTokenCounter())
            message = self.fixtures.require(plan.prompt_hash)
        else:
            cached = self.fixtures.load(plan.prompt_hash)
            if cached is None:
                return None
            self._guard(plan, HeuristicTokenCounter())
            message = cached

        parsed = _validate(
            message=message, output_model=output_model, model=plan.model, request_id=None
        )
        return _result(
            plan=plan,
            message=message,
            parsed=parsed,
            prompt=prompt,
            latency_ms=0.0,
            served_from_fixture=True,
        )


@dataclass
class LlmClient(_Base):
    """Synchronous entry point.

    ``mode`` defaults to ``$BEACON_LLM_MODE`` and then to replay, so a client
    constructed with no arguments in a test cannot reach the network.
    """

    _client: Anthropic | None = None

    @property
    def raw(self) -> Anthropic:
        """The underlying SDK client, created lazily.

        Lazily because replay must work with no API key set at all: building the
        client eagerly would raise on a key-less CI machine even though no call
        is ever made.
        """
        if self._client is None:
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def call[OutputT: BaseModel](
        self,
        *,
        prompt: CachedPrompt,
        output_model: type[OutputT],
        model: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
        verify_capabilities: bool = True,
    ) -> CallResult[OutputT]:
        """Make one structured call.

        :param prompt: Built by :func:`~beacon_llm.prompt.build_cached_prompt`.
            Its block order is what makes the prefix cacheable.
        :param output_model: Pydantic model the answer must validate as; its
            schema goes in ``output_config.format``.
        :param effort: ``output_config.effort`` for this call site.
        :param verify_capabilities: Check the model against the Models API once
            per process. Skipped automatically in replay.
        """
        settings = self.settings.anthropic
        plan = _plan(
            prompt=prompt,
            output_model=output_model,
            model=model or settings.cheap_model,
            effort=effort,
            max_tokens=max_tokens or settings.max_tokens,
        )

        replayed = self._from_fixture(plan, prompt, output_model)
        if replayed is not None:
            return replayed

        if verify_capabilities:
            assert_supports(self._api_key, plan.model, effort=plan.effort)

        self._guard(plan, ApiTokenCounter(self.raw))

        started = time.perf_counter()
        message = self.raw.messages.parse(
            model=plan.model,
            max_tokens=plan.max_tokens,
            system=plan.system,  # type: ignore[arg-type]
            messages=plan.messages,  # type: ignore[arg-type]
            output_config=plan.output_config,
            thinking=ADAPTIVE_THINKING,  # type: ignore[arg-type]
        )
        latency_ms = (time.perf_counter() - started) * 1000
        request_id = getattr(message, "_request_id", None)

        if self.mode is FixtureMode.RECORD:
            self.fixtures.save(
                plan.prompt_hash,
                message=message,
                request=plan.request_summary,
                latency_ms=latency_ms,
                request_id=request_id,
            )

        parsed = _validate(
            message=message, output_model=output_model, model=plan.model, request_id=request_id
        )
        return _result(
            plan=plan,
            message=message,
            parsed=parsed,
            prompt=prompt,
            latency_ms=latency_ms,
            served_from_fixture=False,
        )


@dataclass
class AsyncLlmClient(_Base):
    """Async entry point, for the LangGraph worker and the gateway.

    Shares the plan, guard, validation, pricing and fixture logic with
    :class:`LlmClient`; only the transport differs. Replay is synchronous
    anyway - it reads a file - so the async path short-circuits without
    touching an event loop boundary.
    """

    _client: AsyncAnthropic | None = None

    @property
    def raw(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def call[OutputT: BaseModel](
        self,
        *,
        prompt: CachedPrompt,
        output_model: type[OutputT],
        model: str | None = None,
        effort: Effort | None = None,
        max_tokens: int | None = None,
        verify_capabilities: bool = True,
    ) -> CallResult[OutputT]:
        settings = self.settings.anthropic
        plan = _plan(
            prompt=prompt,
            output_model=output_model,
            model=model or settings.cheap_model,
            effort=effort,
            max_tokens=max_tokens or settings.max_tokens,
        )

        replayed = self._from_fixture(plan, prompt, output_model)
        if replayed is not None:
            return replayed

        if verify_capabilities:
            assert_supports(self._api_key, plan.model, effort=plan.effort)

        prompt_tokens = (
            await self.raw.messages.count_tokens(
                model=plan.model,
                system=plan.system,  # type: ignore[arg-type]
                messages=plan.messages,  # type: ignore[arg-type]
                output_config=plan.output_config,
            )
        ).input_tokens
        enforce_prompt_budget(
            model=plan.model,
            prompt_tokens=prompt_tokens,
            ceiling=self.settings.anthropic.haiku_prompt_token_ceiling,
        )

        started = time.perf_counter()
        message = await self.raw.messages.parse(
            model=plan.model,
            max_tokens=plan.max_tokens,
            system=plan.system,  # type: ignore[arg-type]
            messages=plan.messages,  # type: ignore[arg-type]
            output_config=plan.output_config,
            thinking=ADAPTIVE_THINKING,  # type: ignore[arg-type]
        )
        latency_ms = (time.perf_counter() - started) * 1000
        request_id = getattr(message, "_request_id", None)

        if self.mode is FixtureMode.RECORD:
            self.fixtures.save(
                plan.prompt_hash,
                message=message,
                request=plan.request_summary,
                latency_ms=latency_ms,
                request_id=request_id,
            )

        parsed = _validate(
            message=message, output_model=output_model, model=plan.model, request_id=request_id
        )
        return _result(
            plan=plan,
            message=message,
            parsed=parsed,
            prompt=prompt,
            latency_ms=latency_ms,
            served_from_fixture=False,
        )
