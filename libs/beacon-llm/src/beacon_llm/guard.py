"""The prompt-token guard: count first, then decide whether to call at all.

Haiku 5.5 prices 5x higher above 100,000 prompt tokens, and the threshold
counts cache reads and cache writes alongside fresh input. The consequence that
matters for Beacon: the obvious-looking optimisation "stop making twenty calls
per patient, batch all twenty trials into one" turns twenty cheap requests into
one request on the far side of the cliff. Nothing about that fails. It just
costs five times more per token, on a prompt that is also twenty times bigger.

So the count happens *before* the request leaves the process, and crossing the
ceiling raises :class:`~beacon_llm.errors.PromptTooLargeError`. The ceiling is
``settings.anthropic.haiku_prompt_token_ceiling``, kept distinct from the
pricing cliff in :mod:`beacon_llm.pricing` so that policy and price can move
independently.

Two counters, one interface:

* :class:`ApiTokenCounter` asks ``client.messages.count_tokens`` - exact, free,
  but a network round trip. Used on the live path.
* :class:`HeuristicTokenCounter` divides characters by four - offline, so the
  guard still runs under replay and in CI, where a network call is forbidden.

The guard therefore runs on *every* path, including replay. That is what makes
a batching regression a test failure rather than a surprise on the invoice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from anthropic import Anthropic
from anthropic.types import MessageParam, OutputConfigParam, TextBlockParam

from beacon_llm.errors import PromptTooLargeError
from beacon_llm.pricing import pricing_for
from beacon_llm.prompt import APPROX_CHARS_PER_TOKEN

__all__ = [
    "ApiTokenCounter",
    "HeuristicTokenCounter",
    "TokenCounter",
    "enforce_prompt_budget",
]


@runtime_checkable
class TokenCounter(Protocol):
    """Counts the prompt tokens of a request before it is sent."""

    def count(
        self,
        *,
        model: str,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        output_config: OutputConfigParam,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class HeuristicTokenCounter:
    """Offline estimate: total characters divided by ``chars_per_token``.

    Used wherever the network is off limits. It is an estimate and is never
    used for billing - billing reads the ``usage`` the API returns. Its only job
    is to make the ceiling enforceable without a round trip, which is what lets
    the guard's test run in CI.

    The JSON schema in ``output_config`` is counted too: it is part of the
    prompt the API prices, and for a schema-heavy call it is not negligible.
    """

    chars_per_token: int = APPROX_CHARS_PER_TOKEN

    def count(
        self,
        *,
        model: str,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        output_config: OutputConfigParam,
    ) -> int:
        chars = sum(len(block.get("text", "")) for block in system)
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                chars += len(content)
            else:
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        chars += len(str(block.get("text", "")))
        output_format = output_config.get("format")
        if output_format is not None:
            chars += len(repr(output_format.get("schema", {})))
        return chars // self.chars_per_token


@dataclass(frozen=True, slots=True)
class ApiTokenCounter:
    """Exact count from ``client.messages.count_tokens``.

    The endpoint is free but it is a network call, so this counter must not be
    reachable from replay mode. It is given the same ``system``, ``messages``
    and ``output_config`` the real request will carry, because anything omitted
    here is tokens the guard does not see.
    """

    client: Anthropic

    def count(
        self,
        *,
        model: str,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        output_config: OutputConfigParam,
    ) -> int:
        result = self.client.messages.count_tokens(
            model=model,
            system=system,
            messages=messages,
            output_config=output_config,
        )
        return result.input_tokens


def enforce_prompt_budget(*, model: str, prompt_tokens: int, ceiling: int) -> None:
    """Refuse a request whose prompt would cross a pricing cliff.

    Only models that *have* a cliff are checked: there is no point refusing an
    Opus call at 100,001 tokens, where the rate does not change. For Haiku 5.5
    the ceiling is a hard stop.
    """
    pricing = pricing_for(model)
    if pricing.prompt_token_cliff is None:
        return
    if prompt_tokens > ceiling:
        raise PromptTooLargeError(model=model, prompt_tokens=prompt_tokens, ceiling=ceiling)
