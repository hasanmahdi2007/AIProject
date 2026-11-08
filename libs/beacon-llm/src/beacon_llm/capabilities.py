"""What a model can do, asked of the Models API rather than hardcoded.

A capability table written into the source is wrong the moment a model ships.
``GET /v1/models/{id}`` returns a ``capabilities`` object, so this module asks
and caches the answer for the process.

What it confirmed on 2026-11-08 for all of Opus 5.5, Sonnet 5.5 and Haiku 5.5,
and why the single thinking path in :mod:`beacon_llm.client` is correct:

* ``capabilities.thinking.types.enabled.supported`` is **false** on all three -
  so the old ``thinking={"type": "enabled", "budget_tokens": N}`` shape is not
  merely discouraged, it is unavailable. There is nowhere to send
  ``budget_tokens``.
* ``capabilities.thinking.types.adaptive.supported`` is **true** on all three.
* ``capabilities.thinking.types.disabled.supported`` is **false** on Sonnet 5.5
  and Opus 5.5 - Sonnet cannot have thinking turned off at all.
* ``capabilities.effort`` supports ``low``/``medium``/``high``/``xhigh``/``max``
  on all three.

Those four facts together mean one code path: ``thinking={"type": "adaptive"}``
on every call, with ``output_config.effort`` as the only dial. A per-model
branch would have nothing to branch on.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from anthropic import Anthropic
from anthropic.types import ModelInfo

from beacon_llm.errors import CapabilityError

__all__ = ["Effort", "assert_supports", "model_info"]

Effort = Literal["low", "medium", "high", "xhigh", "max"]
"""``output_config.effort``, chosen per call site.

The dial the pipeline actually turns: ``low`` for mechanical extraction,
``high`` for the rule check where a wrong judgment is a clinical error.
"""


@lru_cache(maxsize=16)
def model_info(api_key: str, model: str) -> ModelInfo:
    """``ModelInfo`` for ``model``, fetched once per process.

    Keyed on the API key as well as the model so that two clients with
    different credentials do not share a cache entry. Uses its own short-lived
    sync client, which keeps the probe identical for sync and async callers -
    it runs once at startup, not per call.
    """
    return Anthropic(api_key=api_key).models.retrieve(model)


def assert_supports(api_key: str, model: str, *, effort: Effort | None) -> None:
    """Check structured outputs and the requested effort level against the API.

    Called once per model per process on the live path, and never on the replay
    path - a recorded fixture is proof enough that the call was once valid, and
    replay must make zero network calls.
    """
    try:
        info = model_info(api_key, model)
    except Exception as exc:  # pragma: no cover - network dependent
        raise CapabilityError(
            f"could not read capabilities for {model!r} from the Models API: {exc}"
        ) from exc

    if info.lifecycle == "retired":
        raise CapabilityError(f"{model} is retired; inference requests naming it fail")

    caps = info.capabilities
    if caps is None:
        # An older API version, or a model the catalogue has no capability data
        # for. Silence beats a false negative here: the point of this probe is
        # to catch a model that cannot do what the call site asked, not to block
        # a call because the catalogue is quiet.
        return

    if not caps.structured_outputs.supported:
        raise CapabilityError(
            f"{model} does not support structured outputs, which every Beacon call needs"
        )
    if not caps.thinking.types.adaptive.supported:
        raise CapabilityError(
            f"{model} does not support adaptive thinking; beacon_llm has exactly one "
            f"thinking path and this model is outside it"
        )
    if effort is not None:
        if not caps.effort.supported:
            raise CapabilityError(
                f"{model} does not support the effort parameter (asked for {effort!r})"
            )
        level = getattr(caps.effort, effort, None)
        if level is None or not level.supported:
            raise CapabilityError(f"{model} does not support effort level {effort!r}")
